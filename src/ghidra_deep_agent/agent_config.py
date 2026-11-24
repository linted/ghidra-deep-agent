"""Config-driven agent definitions (models) loaded from TOML.

Two agents are declared in ``agents.toml`` (path overridable via ``AGENT_CONFIG``):
the ``[main]`` coordinator and the single ``[worker]`` (general-purpose)
sub-agent. Each has an optional ``model`` / ``max_tokens`` / ``reasoning_effort``;
the worker also takes
an optional ``exclude`` list of tool names. This module loads and validates that
file and turns it into the objects ``create_deep_agent`` expects.

Both agents get the full tool set (minus the globally ``WITHHELD_TOOLS``). The
analysis/repair workflows that used to be per-sub-agent system prompts now live in
on-demand **skills** under ``skills/`` (see ``cli.py``); the model reads a skill's
``SKILL.md`` when it needs that workflow. Read-only modes (plan/ask) drop every
``MUTATION_TOOLS`` entry from both the coordinator and the worker, which is what
makes those graphs read-only *by construction*.

What stays in code (not expressible in TOML): the agents' middleware — our
``ArgumentValidationMiddleware`` is a Python object, attached here — and the
system prompts (see prompt.py). The worker's tool set is the live tool set, so a
renamed/absent Ghidra MCP tool simply isn't present rather than crashing.
"""

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from deepagents import SubAgent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.compaction import build_tuned_summarization_middleware
from ghidra_deep_agent.defaults import config_path
from ghidra_deep_agent.models import build_model
from ghidra_deep_agent.prompt import WORKER_SYSTEM_PROMPT
from ghidra_deep_agent.resilience import (
    build_model_resilience_middleware,
    build_tool_retry_middleware,
)
from ghidra_deep_agent.validation import create_argument_validation_middleware

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_CONFIG_FILENAME = "agents.toml"

# The single general-purpose sub-agent's name. Overrides deepagents' auto-added
# general-purpose sub-agent by matching its name.
WORKER_NAME = "general-purpose"

# Every Ghidra tool that mutates the program/project (or the knowledge base). A
# read-only context (plan/ask mode) drops these from its tool set entirely.
# GhidrAssistMCP 3.x exposes each read and write operation as its own tool, so
# "read-only" is then "everything else", and newly added *read* tools are
# auto-covered.
MUTATION_TOOLS = frozenset(
    {
        # symbols
        "rename_function",
        "rename_label",
        "rename_local_variable",
        "rename_global_variable",
        "rename_symbols_batch",
        # comments / bookmarks
        "set_comment",
        "set_function_comment",
        "remove_comment",
        "set_bookmark",
        "remove_bookmark",
        # variables / prototypes / types / structs
        "set_local_variable_type",
        "set_function_prototype",
        "set_data_type",
        "create_enum",
        "create_typedef",
        "delete_data_type",
        "create_structure",
        "replace_structure",
        "merge_structure",
        "set_structure_field",
        "name_structure_gap",
        "infer_structure_from_variable",
        "rename_structure_field",
        # listing / bytes
        "create_function",
        "create_data_variable",
        "disassemble_bytes",  # converts undefined bytes to instructions / clears units
        "patch_instructions",
        "patch_bytes",
        # programs / project
        "open_program",
        "close_program",
        "import_file",
        "export_program",
        "delete_project_file",
        "delete_project_folder",
        "create_project",
        "open_project",
        "save_project",
        "close_project",
        # analysis
        "set_analysis_options",
        "reset_analysis_options",
        "save_analysis_preset",
        "apply_analysis_preset",
        "delete_analysis_preset",
        "cancel_analysis",
        # Ghidra scripts
        "create_script",
        "delete_script",
        "run_script",
        "run_script_with_file_output",  # run_script + deliver files to sandbox
        # local script-driven tools
        "recover_prototypes",  # commits recovered prototypes
        "apply_switch_override",  # writes jump-table overrides
        "deobfuscate_cff",  # rewrites flattened control flow (apply=True)
        # knowledge base
        "save_knowledge",
        "update_knowledge",
    }
)

# Read-only write-scope text appended to a read-only worker's system prompt, so
# the model is told plainly it cannot write (the tools are also physically
# absent). Normal mode appends nothing — the worker simply has the write tools.
_READ_ONLY_SECTION = """
## Your write scope in this run
You are STRICTLY READ-ONLY. You cannot mutate the Ghidra program in any way and
cannot write the knowledge base: no renames, retypes, comments, bookmarks,
prototypes, types, structs, patches, or scripts. Do not ask for those tools.

State the concrete changes you would recommend (renames, retypes, comments,
prototypes, knowledge-base entries) in your summary instead of applying them, so
the coordinator can fold them into its plan or answer.
""".strip()

# Tools withheld from every agent. Filtered out of the full tool set once at
# startup (cli.py), before any per-agent selection.
#   ``analyze_program`` runs full Ghidra Auto Analysis over the whole program; the
#     expected workflow is to analyze in the Ghidra GUI first, so the agent
#     triggering (or re-triggering) it is both slow and rarely wanted.
#   ``get_task_status`` is polled *internally* by ``AsyncTaskMiddleware`` to
#     resolve async submission stubs (see async_tasks.py); the model must never
#     receive it as a callable tool, or it starts manual polling — the very
#     context-bloating spin-loop the middleware exists to prevent. Withholding is
#     safe because the middleware and ``recover_prototypes`` look the tool up in
#     the *raw* MCP tool list (cli.py), upstream of this filter. ``list_tasks`` and
#     ``cancel_task`` are the rest of that job API and are withheld with it.
#   ``create_project`` / ``open_project`` / ``close_project`` replace or close the
#     active Ghidra project. A session is bound to one program in it, so every
#     later tool call would fail; that is the human's call, not an agent's.
WITHHELD_TOOLS = frozenset(
    {
        "analyze_program",
        "get_task_status",
        "list_tasks",
        "cancel_task",
        "create_project",
        "open_project",
        "close_project",
    }
)


class ModelResolver(Protocol):
    """Resolve a model string (and optional output cap) to a model instance.

    Also callable with just the string, so it satisfies resilience.py's
    one-argument ``ModelResolver`` for fallback specs.
    """

    def __call__(
        self,
        model: str | None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str | BaseChatModel: ...


@dataclass(frozen=True)
class AgentConfig:
    """The full agent configuration parsed from ``agents.toml``."""

    main_model: str | None
    worker_model: str | None
    worker_exclude: tuple[str, ...]
    default_model: str | None
    main_max_tokens: int | None = None
    worker_max_tokens: int | None = None
    default_max_tokens: int | None = None
    main_reasoning_effort: str | None = None
    worker_reasoning_effort: str | None = None
    default_reasoning_effort: str | None = None


# --- TOML loading / validation -------------------------------------------------


def _default_config_path() -> Path:
    """Resolve the config path: ``AGENT_CONFIG`` env, else repo-root TOML."""
    return config_path("AGENT_CONFIG", _CONFIG_FILENAME)


def _opt_str(table: Mapping[str, Any], key: str, where: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string if set")
    return value


def _opt_int(table: Mapping[str, Any], key: str, where: str) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}: '{key}' must be a positive integer if set")
    return value


def _opt_str_list(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    """A list-of-strings field that defaults to empty when absent."""
    value = table.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where}: '{key}' must be a list of strings")
    return tuple(value)


def _table(raw: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{path}: [{key}] must be a table")
    return value


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Load and validate the agent configuration from TOML.

    Raises:
        ValueError: if the file is missing, not valid TOML, or has an ill-typed
            field.
    """
    path = path or _default_config_path()
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Agent config not found at {path}. Set AGENT_CONFIG or create "
            f"{_CONFIG_FILENAME} at the repo root."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Agent config {path} is not valid TOML: {exc}") from exc

    default_model = _opt_str(raw, "default", str(path))
    default_max_tokens = _opt_int(raw, "max_tokens", str(path))

    main_raw = _table(raw, "main", path)
    worker_raw = _table(raw, "worker", path)

    return AgentConfig(
        main_model=_opt_str(main_raw, "model", f"{path} [main]"),
        main_max_tokens=_opt_int(main_raw, "max_tokens", f"{path} [main]"),
        worker_model=_opt_str(worker_raw, "model", f"{path} [worker]"),
        worker_max_tokens=_opt_int(worker_raw, "max_tokens", f"{path} [worker]"),
        worker_exclude=_opt_str_list(worker_raw, "exclude", f"{path} [worker]"),
        default_model=default_model,
        default_max_tokens=default_max_tokens,
        main_reasoning_effort=_opt_str(main_raw, "reasoning_effort", f"{path} [main]"),
        worker_reasoning_effort=_opt_str(
            worker_raw, "reasoning_effort", f"{path} [worker]"
        ),
        default_reasoning_effort=_opt_str(raw, "reasoning_effort", str(path)),
    )


# --- Model resolution ----------------------------------------------------------


def _model_spec(model: str | None, default_model: str | None) -> str:
    """Resolve a model string: entry -> TOML default -> MODEL env -> built-in."""
    return model or default_model or os.environ.get("MODEL", _DEFAULT_MODEL)


def resolve_model_spec(model: str | None, config: AgentConfig) -> str:
    """The model string an agent will use (for display/logging)."""
    return _model_spec(model, config.default_model)


def make_model_resolver(
    default_model: str | None,
    default_max_tokens: int | None = None,
    default_reasoning_effort: str | None = None,
) -> ModelResolver:
    """Return a cached resolver building each distinct model string once.

    ``max_tokens`` resolves like the model string: per-call value (a config
    entry's ``max_tokens``) -> TOML top-level ``max_tokens`` -> ``MODEL_MAX_TOKENS``
    env -> unset (provider default; unprofiled Anthropic-protocol models still
    get a floor in ``build_model``). ``reasoning_effort`` resolves the same way
    minus the env step: per-call value -> TOML top-level -> unset.
    """
    cache: dict[tuple[str, int | None, str | None], str | BaseChatModel] = {}

    def resolve(
        model: str | None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str | BaseChatModel:
        spec = _model_spec(model, default_model)
        if max_tokens is None:
            max_tokens = default_max_tokens
        if max_tokens is None:
            env_raw = os.environ.get("MODEL_MAX_TOKENS")
            max_tokens = int(env_raw) if env_raw and env_raw.isdigit() else None
        if reasoning_effort is None:
            reasoning_effort = default_reasoning_effort
        key = (spec, max_tokens, reasoning_effort)
        if key not in cache:
            cache[key] = build_model(spec, max_tokens, reasoning_effort)
        return cache[key]

    return resolve


# --- Tool selection ------------------------------------------------------------


def filter_withheld_tools(all_tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Drop globally-withheld tools from the full tool set (see cli.py)."""
    return [tool for tool in all_tools if tool.name not in WITHHELD_TOOLS]


def read_only_tools(all_tools: Sequence[BaseTool]) -> list[BaseTool]:
    """``all_tools`` minus every mutating tool.

    Used for the plan-mode and ask-mode coordinators (and, with ``read_only``,
    the worker): drops ``save_knowledge``/``update_knowledge`` and every Ghidra
    write tool, keeping read-only navigation/search + knowledge-base reads.
    Filesystem ``write_file``/``read_file``/``edit_file`` come from middleware, so
    a plan can still be written to disk.
    """
    return [tool for tool in all_tools if tool.name not in MUTATION_TOOLS]


def build_worker(
    all_tools: Sequence[BaseTool],
    config: AgentConfig,
    resolve_model: ModelResolver,
    backend: Any,
    *,
    read_only: bool,
    skills: Sequence[str],
    cache_middleware: AgentMiddleware | None = None,
    async_middleware: AgentMiddleware | None = None,
    pruning_middleware: AgentMiddleware | None = None,
    auto_mode_middleware: AgentMiddleware | None = None,
    summary_model: str | BaseChatModel | None = None,
) -> SubAgent:
    """Build the single general-purpose ``SubAgent`` spec.

    It overrides deepagents' auto-added general-purpose sub-agent by name, and is
    given the same ``skills`` sources as the coordinator so it can read a skill's
    ``SKILL.md`` when its brief names one.

    The worker gets its own middleware (sub-agent middleware does not inherit from
    the main agent): model resilience (retry + optional provider fallback),
    argument validation, the shared Jev Auto Mode check on hard-to-undo tools
    (when enabled), the shared immutable-read cache (when enabled), async-task
    resolution, transient filesystem-tool retry, the shared Jev context pruner
    (when enabled), and a tuned auto-summarizer (aggressive sub-agent thresholds;
    replaces deepagents' stock instance by ``.name``). Plus its resolved model.

    ``backend`` is the shared filesystem backend the summarizer offloads evicted
    history to; ``summary_model`` (when given) routes summary calls to a cheaper
    model.

    ``read_only=True`` drops every ``MUTATION_TOOLS`` entry and appends the
    read-only write-scope section to the prompt, making the plan-mode and ask-mode
    workers read-only *by construction*.
    """
    tools = read_only_tools(all_tools) if read_only else list(all_tools)
    if config.worker_exclude:
        excluded = set(config.worker_exclude)
        tools = [tool for tool in tools if tool.name not in excluded]

    system_prompt = WORKER_SYSTEM_PROMPT
    if read_only:
        system_prompt = f"{system_prompt.rstrip()}\n\n{_READ_ONLY_SECTION}\n"

    model = resolve_model(
        config.worker_model, config.worker_max_tokens, config.worker_reasoning_effort
    )
    spec: SubAgent = {
        "name": WORKER_NAME,
        "description": (
            "General-purpose reverse-engineering worker. Delegate a self-contained "
            "task — a whole-program pass, a batch of functions, or an investigation "
            "— naming the skill it should follow and pasting in any known findings. "
            "It runs in its own context and returns a compact summary. Several may "
            "run in parallel over disjoint targets."
        ),
        "system_prompt": system_prompt,
        "tools": tools,
        "model": model,
        "skills": list(skills),
        "middleware": [
            *build_model_resilience_middleware(resolve_model),
            create_argument_validation_middleware(),
            # After validation, so a rejected call costs no Jev request.
            *([auto_mode_middleware] if auto_mode_middleware is not None else []),
            *([cache_middleware] if cache_middleware is not None else []),
            # Inside the cache so resolved (not stub) results are cached.
            *([async_middleware] if async_middleware is not None else []),
            build_tool_retry_middleware(),
            # Per-request relevance pruning of stale tool results; the
            # summarizer (outer) still sees the raw history.
            *([pruning_middleware] if pruning_middleware is not None else []),
            # Replaces deepagents' stock SummarizationMiddleware by name.
            build_tuned_summarization_middleware(
                model, backend, summary_model=summary_model, scope="subagent"
            ),
        ],
    }
    return spec
