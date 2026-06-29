"""Config-driven agent definitions (models + tools) loaded from TOML.

Agents are declared in ``subagents.toml`` (path overridable via ``AGENT_CONFIG``):
the main/coordinator agent's model + tool allowlist, and each sub-agent's
``name`` / ``description`` / ``system_prompt`` / ``model`` / ``tools``. This module
loads and validates that file and turns it into the objects ``create_deep_agent``
expects.

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

from ghidra_deep_agent.models import build_model
from ghidra_deep_agent.resilience import (
    build_model_resilience_middleware,
    build_tool_retry_middleware,
)
from ghidra_deep_agent.validation import create_argument_validation_middleware

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_CONFIG_FILENAME = "subagents.toml"
# `tools = "*"` in the config means "every available tool".
_ALL_TOOLS = "*"

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
    env = os.environ.get("AGENT_CONFIG")
    if env:
        return Path(env).expanduser()
    # subagents.py -> ghidra_deep_agent -> src -> <repo root>
    return Path(__file__).resolve().parents[2] / _CONFIG_FILENAME


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


def build_main_tools(
    all_tools: Sequence[BaseTool], config: AgentConfig
) -> list[BaseTool]:
    """Select the coordinator's restricted tool set from the full tool list."""
    by_name = {tool.name: tool for tool in all_tools}
    return _select(by_name, config.main_tools, agent="main")


def build_subagents(
    all_tools: Sequence[BaseTool],
    config: AgentConfig,
    resolve_model: ModelResolver,
    *,
    cache_middleware: AgentMiddleware | None = None,
) -> list[SubAgent]:
    """Build ``SubAgent`` specs from config, filtered against the live tools.

    Each sub-agent gets its own middleware (sub-agent middleware does not inherit
    from the main agent): model resilience (retry + optional provider fallback),
    argument validation, the shared immutable-read cache (when enabled), and
    transient filesystem-tool retry. Plus its resolved model.
    """
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
        spec: SubAgent = {
            "name": sub.name,
            "description": sub.description,
            "system_prompt": sub.system_prompt,
            "tools": tools,
            "model": resolve_model(sub.model),
            "middleware": [
                *build_model_resilience_middleware(resolve_model),
                create_argument_validation_middleware(),
                *([cache_middleware] if cache_middleware is not None else []),
                build_tool_retry_middleware(),
            ],
        }
        specs.append(spec)
    return specs
