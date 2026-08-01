"""Config-driven agent definitions (models + tools) loaded from TOML.

Agents are declared in ``subagents.toml`` (path overridable via ``AGENT_CONFIG``):
the main/coordinator agent's model + tool allowlist, and each sub-agent's
``name`` / ``description`` / ``system_prompt`` / ``model`` / ``tools`` /
``write_policy``. This module loads and validates that file and turns it into the
objects ``create_deep_agent`` expects.

How much of the program each sub-agent may change is a ``WritePolicy`` tier
rather than a read-only boolean — see ``WRITE_POLICIES`` below for why the middle
tier exists. A tier is enforced two ways: blocked tools are dropped from the
agent's tool set, and blocked ``action``s (plus the provisional-rename prefix) are
rejected per call by ``ArgumentValidationMiddleware``. The tier also *writes its
own prompt text*, so what the model is told always matches what it can do.

Why config-driven: models can be right-sized per agent (a cheap model for recon,
a capable one for analysis) without code edits, and the coordinator's tool set is
restricted to orchestration + navigation/search so heavy analysis stays in
sub-agents (context quarantine).

What stays in code (not expressible in TOML): each sub-agent's middleware — our
``ArgumentValidationMiddleware`` is a Python object, attached here — and the main
agent's ``SYSTEM_PROMPT`` (see prompt.py). Tool allowlists are name-based and
filtered against the live tool set, so a renamed/absent Ghidra MCP tool is
skipped with a startup warning rather than crashing.
"""

import os
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import SubAgent
from deepagents.middleware.subagents import DEFAULT_SUBAGENT_PROMPT
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.compaction import build_tuned_summarization_middleware
from ghidra_deep_agent.defaults import config_path
from ghidra_deep_agent.models import build_model
from ghidra_deep_agent.report_guard import SubagentReportGuardMiddleware
from ghidra_deep_agent.resilience import (
    build_model_resilience_middleware,
    build_tool_retry_middleware,
)
from ghidra_deep_agent.validation import create_argument_validation_middleware

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_CONFIG_FILENAME = "subagents.toml"
# `tools = "*"` in the config means "every available tool".
_ALL_TOOLS = "*"

# Every write-only tool: each mutates the program/project (or the knowledge base)
# with no read mode to preserve, so a restricted context drops it entirely.
# "Read-only" is then "everything else", meaning newly added *read* tools are
# auto-covered. Dual read/write tools (``variables``, ``comments``, ``types``,
# ``struct``, ``bookmarks``) are NOT listed here — they keep their read actions
# and have their write actions blocked per-call via ``ALL_WRITE_ACTIONS`` (see
# ``validation.py``).
MUTATION_TOOLS = frozenset(
    {
        "rename_symbol",
        "batch_rename",
        "create_function",
        "create_data_var",
        "assemble_code",
        "patch_bytes",
        "disassemble_at",  # converts undefined bytes to instructions / clears units
        "open_program",
        "close_program",
        "import_file",
        "export_program",
        "scripts",
        "recover_prototypes",  # local tool: commits recovered prototypes (mutates)
        "apply_switch_override",  # local tool: writes jump-table overrides (mutates)
        "deobfuscate_cff",  # local tool: rewrites flattened control flow (apply=True)
        "project_files",  # has a `delete` action
        "analysis_options",
        "analysis_control",
        "save_knowledge",  # KB writes blocked too
        "update_knowledge",
    }
)
# Write ``action`` values on GhidrAssistMCP's consolidated read/write tools. A
# restricted context rejects these via the argument-validation middleware while
# the tool's read actions (list/get/field_xrefs) still work.
# Action strings verified against the live server's tool schemas.
ALL_WRITE_ACTIONS: dict[str, frozenset[str]] = {
    "variables": frozenset({"rename", "retype", "set_prototype"}),
    "comments": frozenset({"set", "remove"}),
    "types": frozenset(
        {"set", "delete", "create_struct", "create_enum", "create_typedef"}
    ),
    "struct": frozenset(
        {
            "create",
            "modify",
            "merge",
            "set_field",
            "name_gap",
            "auto_create",
            "rename_field",
        }
    ),
    "bookmarks": frozenset({"set", "remove"}),
}

# --- Write policy tiers --------------------------------------------------------
#
# Sub-agents used to be either fully write-capable or fully read-only. That binary
# forced every analysis agent that shouldn't do type surgery to be read-only, which
# meant its conclusions could not be persisted at all: each follow-up delegation
# re-decompiled and re-reasoned about the same function from scratch, and any
# change it wanted was lost in free-text prose. The middle tier (``annotations``)
# exists so an investigator can persist *cheap, revisable* findings — renames,
# comments, bookmarks, knowledge-base entries — while heavyweight, hard-to-undo
# mutations stay with the specialists that own them.
#
# The restricted tiers are defined by SUBTRACTION from the full-lockdown sets
# above, so a write tool or write action added later is blocked by default in
# every restricted tier (fail closed) until it is explicitly allowed.

# Write-only tools the ``annotations`` tier keeps. Renames are metadata and
# trivially reversible; KB writes never touch the program at all.
ANNOTATION_TOOLS = frozenset(
    {
        "rename_symbol",
        "batch_rename",
        "save_knowledge",
        "update_knowledge",
    }
)
# Write ``action``s the ``annotations`` tier keeps on the dual read/write tools.
# Notably absent: ``variables:retype``/``set_prototype`` and everything on
# ``types``/``struct`` — type and signature surgery is program-global and stays
# with ``type-fixer`` / ``prototype-fixer``.
ANNOTATION_ACTIONS: dict[str, frozenset[str]] = {
    "variables": frozenset({"rename"}),
    "comments": frozenset({"set", "remove"}),
    "bookmarks": frozenset({"set", "remove"}),
}

# A rename applied by an ``annotations``-tier agent must carry this prefix. The
# tier is for *provisional* conclusions, so the uncertainty is encoded in the name
# itself rather than left to the agent's discretion; the rule is enforced by the
# argument-validation middleware, not by prompt text. Promotion (dropping the
# prefix once the evidence is settled) requires a ``full``-tier agent.
PROVISIONAL_RENAME_PREFIX = "maybe_"

_PENDING_PROTOCOL = f"""
## Your write scope in this session
You may apply CHEAP, REVISABLE annotations directly in Ghidra as you work, and you
SHOULD — what you persist is what the next sub-agent sees instead of re-deriving:
- rename functions, variables, and parameters (`rename_symbol`, `batch_rename`,
  `variables` with `action: rename`),
- add comments (`comments` with `action: set`),
- drop bookmarks (`bookmarks` with `action: set`),
- record findings in the knowledge base (`save_knowledge`, `update_knowledge`).

Every rename you apply MUST start with `{PROVISIONAL_RENAME_PREFIX}` — the tool
rejects the call otherwise. The prefix marks the name as provisional: your
evidence was good enough to be worth persisting, not good enough to be final. Also
`save_knowledge` each rename (category `rename`, confidence `provisional`) with the
evidence behind it, so a later agent can confirm or correct the guess instead of
starting over. Do NOT try to remove an existing `{PROVISIONAL_RENAME_PREFIX}`
prefix — confirming a provisional name is `function-analyst`'s job.

You may NOT retype variables, set prototypes, create or edit types/structs, patch
or assemble bytes, run scripts, or apply switch/CFF overrides.

## PENDING protocol — how a change you cannot apply gets made
When the evidence supports a change outside your scope, do BOTH of these:
1. File it in Ghidra: `bookmarks` with `action: set`, `category: pending-change`,
   at the relevant address, with the comment
   `pending <retype|prototype|struct|switch|patch|other>: <target> -> <detail>`.
   This is a durable queue — it survives this delegation, and the coordinator
   drains it to the specialist that owns the change.
2. List it in your summary under a `PENDING:` heading, one line per item.
A `read_only_error` result means the change is out of scope: file it as pending.
Never retry the blocked call.
""".strip()

# Appended to EVERY sub-agent prompt regardless of tier (full-tier specialists
# also end runs on writes). deepagents forwards only the text of the final
# non-empty AIMessage to the coordinator, so a run that ends on a tool call
# reports nothing but that call's preamble; SubagentReportGuardMiddleware
# backstops the runs where the model ignores this instruction anyway.
_REPORT_PROTOCOL = """
## Final report
The coordinator receives ONLY the text of your final message. Your tool calls
(knowledge-base writes, bookmarks, renames, comments) persist state but are
INVISIBLE to it. After your last tool call, ALWAYS end the turn with your
complete plain-text findings summary (including the `PENDING:` list where
applicable). Never end the turn on a tool call or with an empty message.
""".strip()

_READ_ONLY_SECTION = """
## Your write scope in this session
You are STRICTLY READ-ONLY. You cannot mutate the Ghidra program in any way and
cannot write the knowledge base: no renames, retypes, comments, bookmarks,
prototypes, types, structs, patches, or scripts. Do not ask for those tools.

State the concrete changes you would recommend (renames, retypes, comments,
prototypes, knowledge-base entries) in your summary instead of applying them, so
the coordinator can fold them into its plan or answer. A `read_only_error` result
means the tool is blocked here — do not retry it.
""".strip()


@dataclass(frozen=True)
class WritePolicy:
    """How much of the program a sub-agent may mutate.

    ``blocked_tools`` are dropped from the agent's tool set entirely;
    ``blocked_actions`` are rejected per-call by the argument-validation
    middleware. ``prompt_section`` and ``description_suffix`` are appended in
    ``build_subagents`` rather than written into ``subagents.toml``, so the text
    the model sees can never drift from what the middleware enforces, and one
    TOML entry can describe itself correctly under different policies (the same
    ``research`` agent annotates in the normal graph and is read-only in plan and
    ask mode).
    """

    name: str
    blocked_tools: frozenset[str]
    blocked_actions: Mapping[str, frozenset[str]]
    # Rename calls must use this prefix; None disables the check.
    rename_prefix: str | None
    prompt_section: str
    description_suffix: str


def _actions_minus(
    allowed: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """``ALL_WRITE_ACTIONS`` minus ``allowed``, dropping now-empty entries."""
    remaining: dict[str, frozenset[str]] = {}
    for tool, actions in ALL_WRITE_ACTIONS.items():
        blocked = actions - allowed.get(tool, frozenset())
        if blocked:
            remaining[tool] = blocked
    return remaining


WRITE_POLICIES: dict[str, WritePolicy] = {
    # Trusted specialists: everything their tool allowlist grants.
    "full": WritePolicy(
        name="full",
        blocked_tools=frozenset(),
        blocked_actions={},
        rename_prefix=None,
        prompt_section="",
        description_suffix="",
    ),
    # Investigators: provisional renames, comments, bookmarks, KB writes.
    "annotations": WritePolicy(
        name="annotations",
        blocked_tools=MUTATION_TOOLS - ANNOTATION_TOOLS,
        blocked_actions=_actions_minus(ANNOTATION_ACTIONS),
        rename_prefix=PROVISIONAL_RENAME_PREFIX,
        prompt_section=_PENDING_PROTOCOL,
        description_suffix=(
            " Applies provisional annotations only (renames prefixed "
            f"`{PROVISIONAL_RENAME_PREFIX}`, comments, bookmarks, knowledge-base "
            "entries); anything heavier it files as a `pending-change` bookmark "
            "and reports under PENDING: for you to route to a specialist."
        ),
    ),
    # Plan mode / ask mode: no writes at all.
    "none": WritePolicy(
        name="none",
        blocked_tools=MUTATION_TOOLS,
        blocked_actions=ALL_WRITE_ACTIONS,
        rename_prefix=None,
        prompt_section=_READ_ONLY_SECTION,
        description_suffix=" Read-only: applies no changes of any kind.",
    ),
}

DEFAULT_WRITE_POLICY = "full"
READ_ONLY_WRITE_POLICY = "none"
# Tools withheld from every agent. Filtered out of the full tool set once at
# startup (cli.py), before any per-agent selection — the only reliable block,
# since `tools = "*"` agents and the read-only research sub-agent would otherwise
# still receive them.
#   ``analyze_program`` runs full Ghidra Auto Analysis over the whole program; the
#     expected workflow is to analyze in the Ghidra GUI first, so the agent
#     triggering (or re-triggering) it is both slow and rarely wanted.
#   ``get_task_status`` is polled *internally* by ``AsyncTaskMiddleware`` to
#     resolve async submission stubs (see async_tasks.py); the model must never
#     receive it as a callable tool, or it starts manual polling — the very
#     context-bloating spin-loop the middleware exists to prevent. Withholding is
#     safe because the middleware and ``recover_prototypes`` look the tool up in
#     the *raw* MCP tool list (cli.py), upstream of this filter.
WITHHELD_TOOLS = frozenset(
    {
        "analyze_program",
        "get_task_status",
    }
)

# The read-only research sub-agent's name, referenced by both graphs.
RESEARCH_SUBAGENT_NAME = "research"

ModelResolver = Callable[[str | None], str | BaseChatModel]


@dataclass(frozen=True)
class SubAgentConfig:
    """A single sub-agent's declared configuration."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    all_tools: bool
    exclude: tuple[str, ...]
    model: str | None
    # A key of ``WRITE_POLICIES``; see that table for what each tier allows.
    write_policy: str


@dataclass(frozen=True)
class AgentConfig:
    """The full agent configuration parsed from ``subagents.toml``."""

    main_tools: tuple[str, ...]
    main_model: str | None
    default_model: str | None
    subagents: tuple[SubAgentConfig, ...]


# --- TOML loading / validation -------------------------------------------------


def _default_config_path() -> Path:
    """Resolve the config path: ``AGENT_CONFIG`` env, else repo-root TOML."""
    return config_path("AGENT_CONFIG", _CONFIG_FILENAME)


def _req_str(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' is required and must be a non-empty string")
    return value


def _opt_str(table: Mapping[str, Any], key: str, where: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string if set")
    return value


def _opt_bool(table: Mapping[str, Any], key: str, where: str) -> bool:
    """A boolean field that defaults to False when absent."""
    value = table.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{where}: '{key}' must be a boolean if set")
    return value


def _str_list(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where}: '{key}' must be a list of strings")
    return tuple(value)


def _opt_str_list(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    """A list-of-strings field that defaults to empty when absent."""
    if table.get(key) is None:
        return ()
    return _str_list(table, key, where)


def _parse_tools(table: Mapping[str, Any], where: str) -> tuple[tuple[str, ...], bool]:
    """Parse a ``tools`` field: a list of names, or ``"*"`` for all tools.

    Returns ``(names, all_tools)`` where ``all_tools`` is True for ``"*"``.
    """
    value = table.get("tools")
    if value == _ALL_TOOLS:
        return (), True
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where}: 'tools' must be a list of strings or \"*\"")
    return tuple(value), False


def _parse_write_policy(table: Mapping[str, Any], where: str) -> str:
    """Resolve an entry's write policy from ``write_policy`` / ``read_only``.

    ``read_only = true`` is kept as sugar for ``write_policy = "none"`` (it
    predates the tiers and reads well for the fully-locked-down case). Setting
    both is only an error when they disagree, so the redundant-but-consistent
    spelling doesn't break an existing config.
    """
    policy = _opt_str(table, "write_policy", where)
    if policy is not None and policy not in WRITE_POLICIES:
        raise ValueError(
            f"{where}: unknown write_policy {policy!r}; expected one of "
            f"{', '.join(sorted(WRITE_POLICIES))}"
        )
    read_only = _opt_bool(table, "read_only", where)
    if read_only:
        if policy is not None and policy != READ_ONLY_WRITE_POLICY:
            raise ValueError(
                f"{where}: read_only = true conflicts with "
                f"write_policy = {policy!r}; set only one"
            )
        return READ_ONLY_WRITE_POLICY
    return policy or DEFAULT_WRITE_POLICY


def _parse_subagent(raw: Any, path: Path) -> SubAgentConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: each [[subagents]] entry must be a table")
    name = _req_str(raw, "name", f"{path} [[subagents]]")
    where = f"{path} [[subagents]] '{name}'"
    tools, all_tools = _parse_tools(raw, where)
    # system_prompt is optional: omit it (e.g. for general-purpose) to fall back
    # to deepagents' stock sub-agent prompt.
    system_prompt = _opt_str(raw, "system_prompt", where) or DEFAULT_SUBAGENT_PROMPT
    return SubAgentConfig(
        name=name,
        description=_req_str(raw, "description", where),
        system_prompt=system_prompt,
        tools=tools,
        all_tools=all_tools,
        exclude=_opt_str_list(raw, "exclude", where),
        model=_opt_str(raw, "model", where),
        write_policy=_parse_write_policy(raw, where),
    )


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Load and validate the agent configuration from TOML.

    Raises:
        ValueError: if the file is missing, not valid TOML, or missing/ill-typed
            required keys.
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

    main_raw = raw.get("main", {})
    if not isinstance(main_raw, dict):
        raise ValueError(f"{path}: [main] must be a table")
    main_tools = _str_list(main_raw, "tools", f"{path} [main]")
    main_model = _opt_str(main_raw, "model", f"{path} [main]")

    subs_raw = raw.get("subagents", [])
    if not isinstance(subs_raw, list) or not subs_raw:
        raise ValueError(f"{path}: at least one [[subagents]] entry is required")
    subagents = tuple(_parse_subagent(entry, path) for entry in subs_raw)

    return AgentConfig(
        main_tools=main_tools,
        main_model=main_model,
        default_model=default_model,
        subagents=subagents,
    )


# --- Model resolution ----------------------------------------------------------


def _model_spec(model: str | None, default_model: str | None) -> str:
    """Resolve a model string: entry -> TOML default -> MODEL env -> built-in."""
    return model or default_model or os.environ.get("MODEL", _DEFAULT_MODEL)


def resolve_model_spec(model: str | None, config: AgentConfig) -> str:
    """The model string an agent will use (for display/logging)."""
    return _model_spec(model, config.default_model)


def make_model_resolver(default_model: str | None) -> ModelResolver:
    """Return a cached resolver building each distinct model string once."""
    cache: dict[str, str | BaseChatModel] = {}

    def resolve(model: str | None) -> str | BaseChatModel:
        spec = _model_spec(model, default_model)
        if spec not in cache:
            cache[spec] = build_model(spec)
        return cache[spec]

    return resolve


# --- Tool selection ------------------------------------------------------------


def _select(
    by_name: dict[str, BaseTool], names: Sequence[str], *, agent: str
) -> list[BaseTool]:
    """Return the tools whose names are in ``names``, skipping any not present.

    Tool names come from the Ghidra MCP server, which is the source of truth at
    runtime. A requested name that isn't available is reported and skipped so a
    renamed/removed tool can't crash agent startup.
    """
    selected: list[BaseTool] = []
    missing: list[str] = []
    for name in names:
        tool = by_name.get(name)
        if tool is None:
            missing.append(name)
        else:
            selected.append(tool)
    if missing:
        print(
            f"Warning: agent '{agent}' — {len(missing)} requested tool(s) not "
            f"available and skipped: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
    return selected


def filter_withheld_tools(all_tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Drop globally-withheld tools from the full tool set (see cli.py)."""
    return [tool for tool in all_tools if tool.name not in WITHHELD_TOOLS]


def build_main_tools(
    all_tools: Sequence[BaseTool], config: AgentConfig
) -> list[BaseTool]:
    """Select the coordinator's restricted tool set from the full tool list."""
    by_name = {tool.name: tool for tool in all_tools}
    return _select(by_name, config.main_tools, agent="main")


def _with_policy_section(system_prompt: str, policy: WritePolicy) -> str:
    """Append the policy's write-scope section and the report protocol."""
    sections = [system_prompt.rstrip()]
    if policy.prompt_section:
        sections.append(policy.prompt_section)
    sections.append(_REPORT_PROTOCOL)
    return "\n\n".join(sections) + "\n"


def build_subagents(
    all_tools: Sequence[BaseTool],
    config: AgentConfig,
    resolve_model: ModelResolver,
    backend: Any,
    *,
    cache_middleware: AgentMiddleware | None = None,
    async_middleware: AgentMiddleware | None = None,
    summary_model: str | BaseChatModel | None = None,
    policy_override: str | None = None,
) -> list[SubAgent]:
    """Build ``SubAgent`` specs from config, filtered against the live tools.

    Each sub-agent gets its own middleware (sub-agent middleware does not inherit
    from the main agent): model resilience (retry + optional provider fallback),
    argument validation, the shared immutable-read cache (when enabled),
    async-task resolution, transient filesystem-tool retry, and a tuned
    auto-summarizer (aggressive sub-agent thresholds; replaces deepagents'
    stock instance by ``.name``). Plus its resolved model.

    ``backend`` is the shared filesystem backend the summarizer offloads evicted
    history to; ``summary_model`` (when given) routes summary calls to a cheaper
    model.

    ``policy_override`` forces every sub-agent onto one ``WRITE_POLICIES`` tier,
    ignoring what the config asked for. Plan mode and ask mode build their
    delegates with ``"none"``, which is what makes those graphs read-only *by
    construction* rather than by trusting the config — the same TOML entry can
    then annotate in the normal graph and be locked down in the read-only ones.
    """
    if policy_override is not None and policy_override not in WRITE_POLICIES:
        raise ValueError(
            f"unknown policy_override {policy_override!r}; expected one of "
            f"{', '.join(sorted(WRITE_POLICIES))}"
        )
    by_name = {tool.name: tool for tool in all_tools}
    specs: list[SubAgent] = []
    for sub in config.subagents:
        if sub.all_tools:
            tools = list(all_tools)
        else:
            tools = _select(by_name, sub.tools, agent=sub.name)
        if sub.exclude:
            excluded = set(sub.exclude)
            tools = [tool for tool in tools if tool.name not in excluded]
        policy = WRITE_POLICIES[policy_override or sub.write_policy]
        # Drop the tier's blocked write-only tools outright; its blocked write
        # `action`s on the dual read/write tools are rejected per call by the
        # middleware, which also enforces the provisional-rename prefix.
        if policy.blocked_tools:
            tools = [tool for tool in tools if tool.name not in policy.blocked_tools]
        validation_mw = create_argument_validation_middleware(
            policy.blocked_actions or None,
            rename_prefix=policy.rename_prefix,
        )
        model = resolve_model(sub.model)
        spec: SubAgent = {
            "name": sub.name,
            "description": sub.description + policy.description_suffix,
            "system_prompt": _with_policy_section(sub.system_prompt, policy),
            "tools": tools,
            "model": model,
            "middleware": [
                *build_model_resilience_middleware(resolve_model),
                validation_mw,
                *([cache_middleware] if cache_middleware is not None else []),
                # Inside the cache so resolved (not stub) results are cached.
                *([async_middleware] if async_middleware is not None else []),
                build_tool_retry_middleware(),
                # Replaces deepagents' stock SummarizationMiddleware by name.
                build_tuned_summarization_middleware(
                    model, backend, summary_model=summary_model, scope="subagent"
                ),
                # Backstops deepagents' report extraction (subagents.py,
                # `_return_command_with_state_update`): it forwards the last
                # non-empty AIMessage's text even when that message is a
                # tool-call preamble. Runs after the loop, so order is moot.
                SubagentReportGuardMiddleware(),
            ],
        }
        specs.append(spec)
    return specs


# --- Plan mode (read-only) -----------------------------------------------------


def build_plan_mode_main_tools(
    all_tools: Sequence[BaseTool], config: AgentConfig
) -> list[BaseTool]:
    """The plan-mode coordinator's tools: ``build_main_tools`` minus mutations.

    Drops ``save_knowledge``/``update_knowledge`` (and any other blocked tool),
    keeping read-only navigation/search + knowledge-base reads. Filesystem
    ``write_file``/``read_file``/``edit_file`` come from middleware, so the plan
    can still be written to disk.
    """
    return [
        tool
        for tool in build_main_tools(all_tools, config)
        if tool.name not in MUTATION_TOOLS
    ]
