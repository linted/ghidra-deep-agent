"""Tests against the repo's own ``subagents.toml``, not a synthetic fixture.

The tier machinery is unit-tested in ``test_subagents.py``; what this file guards
is that the shipped config actually uses it correctly. Two classes of bug are only
visible here: a config that drifts out of sync with the code (an unknown policy, a
prompt telling the model to call an action the middleware blocks), and a change to
``_read_only_delegates`` that quietly hands plan or ask mode a write-capable agent.

Run:  uv run pytest tests/test_shipped_config.py -v
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.cli import _read_only_delegates
from ghidra_deep_agent.subagents import (
    ALL_WRITE_ACTIONS,
    MUTATION_TOOLS,
    READ_ONLY_WRITE_POLICY,
    WRITE_POLICIES,
    AgentConfig,
    build_subagents,
    load_agent_config,
)
from ghidra_deep_agent.validation import ArgumentValidationMiddleware

CONFIG_PATH = Path(__file__).resolve().parent.parent / "subagents.toml"

# Every tool any shipped agent asks for, so `_select` never warns about a missing
# one. Union of the config's allowlists plus the dual read/write tools.
_ALL_TOOL_NAMES = sorted(
    set(MUTATION_TOOLS)
    | set(ALL_WRITE_ACTIONS)
    | {
        "get_code",
        "xrefs",
        "analyze_function",
        "get_function_stack_layout",
        "get_basic_blocks",
        "get_data_at",
        "get_data_vars",
        "get_binary_info",
        "get_functions",
        "get_imports",
        "get_exports",
        "get_entry_points",
        "search_functions_by_name",
        "search_strings",
        "find_unrecovered_switches",
        "query_knowledge",
        "query_by_address",
        "query_by_category",
        "query_by_tags",
        "get_knowledge_summary",
        "list_all_knowledge",
        "list_analyzed_binaries",
    }
)


@pytest.fixture(scope="module")
def config() -> AgentConfig:
    return load_agent_config(CONFIG_PATH)


def _fake_tools() -> Sequence[BaseTool]:
    tools = [SimpleNamespace(name=name) for name in _ALL_TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(spec: str | None) -> FakeListChatModel:
    return FakeListChatModel(responses=["ok"])


def _build(config: AgentConfig, *, policy_override: str | None = None) -> list[Any]:
    return build_subagents(
        _fake_tools(),
        config,
        resolve_model=_resolver,
        backend=StateBackend(),
        policy_override=policy_override,
    )


def _tool_names(spec: Any) -> set[str]:
    return {getattr(tool, "name") for tool in spec["tools"]}  # noqa: B009


def _validation_mw(spec: Any) -> ArgumentValidationMiddleware:
    mws = [m for m in spec["middleware"] if isinstance(m, ArgumentValidationMiddleware)]
    assert len(mws) == 1
    return mws[0]


def test_shipped_config_parses_with_known_policies(config: AgentConfig) -> None:
    assert config.subagents, "config must declare at least one sub-agent"
    for sub in config.subagents:
        assert sub.write_policy in WRITE_POLICIES, (
            f"{sub.name}: unknown write_policy {sub.write_policy!r}"
        )


def test_shipped_config_builds_without_missing_tools(config: AgentConfig) -> None:
    specs = _build(config)
    assert {spec["name"] for spec in specs} == {sub.name for sub in config.subagents}


def test_read_only_delegates_cannot_mutate(config: AgentConfig) -> None:
    """Plan and ask mode are read-only by construction, not by config trust."""
    plan_subs, ask_subs = _read_only_delegates(
        _build(config, policy_override=READ_ONLY_WRITE_POLICY)
    )
    assert plan_subs and ask_subs

    for spec in [*plan_subs, *ask_subs]:
        leaked = _tool_names(spec) & MUTATION_TOOLS
        assert not leaked, f"{spec['name']} kept mutation tool(s) {sorted(leaked)}"
        assert _validation_mw(spec)._write_actions == ALL_WRITE_ACTIONS
        assert _validation_mw(spec)._rename_prefix is None


def test_delegates_built_from_the_normal_set_would_not_qualify(
    config: AgentConfig,
) -> None:
    """Guards the wiring: passing the *normal* build here must not be silently OK.

    If someone routes `_read_only_delegates` at the main sub-agent list again, the
    read-only guarantee is gone — this test fails loudly instead.
    """
    plan_subs, _ = _read_only_delegates(_build(config))
    research = plan_subs[0]
    assert _tool_names(research) & MUTATION_TOOLS, (
        "the annotating build of `research` should hold mutation tools; if it "
        "does not, this test no longer guards anything"
    )


def test_no_prompt_uses_a_blocked_or_nonexistent_bookmark_action() -> None:
    """`bookmarks` writes are `set`/`remove`; `add` silently fails on the server."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "action: add" not in text, (
        "a prompt tells the model to call `bookmarks` with action `add`, which "
        "the live server does not accept (it is `set`)"
    )


def test_annotating_agents_are_told_the_prefix_rule(config: AgentConfig) -> None:
    """The generated scope text must reach every annotations-tier agent."""
    annotating = [
        spec
        for spec, sub in zip(_build(config), config.subagents, strict=True)
        if sub.write_policy == "annotations"
    ]
    assert annotating, "expected at least one annotations-tier sub-agent"
    for spec in annotating:
        assert "maybe_" in spec["system_prompt"]
        assert "pending-change" in spec["system_prompt"]
