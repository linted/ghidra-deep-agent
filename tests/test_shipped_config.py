"""Tests against the repo's own ``subagents.toml``, not a synthetic fixture.

The tier machinery is unit-tested in ``test_subagents.py``; what this file guards
is that the shipped config actually uses it correctly. Two classes of bug are only
visible here: a config that drifts out of sync with the code (an unknown policy, a
tool name the server does not expose), and a change to
``_read_only_delegates`` that quietly hands plan or ask mode a write-capable agent.

Run:  uv run pytest tests/test_shipped_config.py -v
"""

from __future__ import annotations

import tomllib
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
    MUTATION_TOOLS,
    READ_ONLY_WRITE_POLICY,
    WRITE_POLICIES,
    AgentConfig,
    build_subagents,
    load_agent_config,
)
from ghidra_deep_agent.validation import ArgumentValidationMiddleware

CONFIG_PATH = Path(__file__).resolve().parent.parent / "subagents.toml"

# GhidrAssistMCP 3.0.0 read tools (every write tool is in MUTATION_TOOLS).
_GHIDRA_READ_TOOLS = {
    "get_program_info",
    "list_open_programs",
    "list_functions",
    "search_functions",
    "get_function_info",
    "get_function_byte_pattern",
    "get_function_hash",
    "get_function_prototype",
    "get_function_stack_layout",
    "get_program_function_statistics",
    "get_pseudocode",
    "get_disassembly",
    "get_pcode",
    "get_control_flow_graph",
    "get_xrefs_to",
    "get_xrefs_from",
    "get_callers",
    "get_callees",
    "get_call_graph",
    "get_hexdump",
    "list_memory_blocks",
    "list_imports",
    "list_exports",
    "list_strings",
    "search_strings",
    "search_bytes",
    "list_data_variables",
    "list_namespaces",
    "list_relocations",
    "list_entry_points",
    "get_current_address",
    "get_current_function",
    "get_comment",
    "get_function_comment",
    "list_function_comments",
    "list_function_variables",
    "list_data_types",
    "get_data_type",
    "get_structure_field_xrefs",
    "list_bookmarks",
    "list_classes",
    "get_class_info",
    "assemble_instructions",
    "list_project_files",
    "list_analysis_options",
    "list_analysis_presets",
    "get_analysis_status",
    "list_scripts",
    "get_script_source",
    "list_tasks",
}
# Local tools that are not mutations, plus the knowledge base's read tools.
_LOCAL_READ_TOOLS = {
    "find_unrecovered_switches",
    "query_knowledge",
    "query_by_address",
    "query_by_category",
    "query_by_tags",
    "get_knowledge_summary",
    "list_all_knowledge",
    "list_analyzed_binaries",
}
# Every tool any shipped agent could ask for, so `_select` never warns about a
# missing one.
_ALL_TOOL_NAMES = sorted(set(MUTATION_TOOLS) | _GHIDRA_READ_TOOLS | _LOCAL_READ_TOOLS)


@pytest.fixture(scope="module")
def config() -> AgentConfig:
    return load_agent_config(CONFIG_PATH)


def _fake_tools() -> Sequence[BaseTool]:
    tools = [SimpleNamespace(name=name) for name in _ALL_TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(model: str | None, max_tokens: int | None = None) -> FakeListChatModel:
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


def test_every_configured_tool_name_exists() -> None:
    """A misspelled or retired tool name is only a startup warning; fail here."""
    raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = [raw["main"], *raw["subagents"]]
    known = set(_ALL_TOOL_NAMES)
    for entry in entries:
        tools = entry.get("tools")
        if not isinstance(tools, list):
            continue
        unknown = set(tools) - known
        assert not unknown, f"{entry.get('name', 'main')}: unknown tools {unknown}"


def test_no_prompt_uses_the_retired_selector_api() -> None:
    """GhidrAssistMCP 3.x has no `action`/`format` selectors; unknown args error."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for selector in ("`action:", "`format:"):
        assert selector not in text, (
            f"a prompt still tells the model to pass {selector[1:]} — the "
            "consolidated tools it selected between no longer exist"
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
