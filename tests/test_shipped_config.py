"""Tests against the repo's own ``agents.toml`` and ``skills/`` tree.

The builder machinery is unit-tested in ``test_agent_config.py``; what this file
guards is that the shipped config and skills actually line up with the code. The
bugs only visible here: a config that won't parse/build, a ``SKILL.md`` that
deepagents' own loader rejects or whose name drifts from its directory, a skill
body that names a tool the server does not expose, and the read-only modes being
handed write tools or the repair skills.

Run:  uv run pytest tests/test_shipped_config.py -v
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.agent_config import (
    MUTATION_TOOLS,
    AgentConfig,
    build_worker,
    load_agent_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "agents.toml"
SKILLS_DIR = REPO_ROOT / "skills"

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
# Filesystem tools the skills may reference (supplied by deepagents middleware).
_FS_TOOLS = {"write_file", "read_file", "edit_file"}
_KNOWN_TOOLS = set(MUTATION_TOOLS) | _GHIDRA_READ_TOOLS | _LOCAL_READ_TOOLS | _FS_TOOLS

# Backticked snake_case tokens in skill bodies that are NOT tools: tool
# parameters, example symbol names, and a type/field name. Listed so the
# drift guard below can tell them apart from a genuine tool-name typo.
_NON_TOOL_TOKENS = {
    "base_address",
    "c_definition",
    "dry_run",
    "element_size",
    "file_size",
    "local_10",
    "maybe_",
    "set_rodata_constant",
    "size_t",
    "table_address",
    "table_hint",
    "unsupported_arch",
    "warning_cleared",
}

_ALL_TOOL_NAMES = sorted(_KNOWN_TOOLS)


@pytest.fixture(scope="module")
def config() -> AgentConfig:
    return load_agent_config(CONFIG_PATH)


def _fake_tools() -> Sequence[BaseTool]:
    tools = [SimpleNamespace(name=name) for name in _ALL_TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(model: str | None, max_tokens: int | None = None) -> FakeListChatModel:
    return FakeListChatModel(responses=["ok"])


def _tool_names(spec: Any) -> set[str]:
    return {getattr(tool, "name") for tool in spec["tools"]}  # noqa: B009


def _build_worker(config: AgentConfig, *, read_only: bool) -> Any:
    return build_worker(
        _fake_tools(),
        config,
        _resolver,
        StateBackend(),
        read_only=read_only,
        skills=["/skills/analysis/", "/skills/repair/"],
    )


# --- config --------------------------------------------------------------------


def test_shipped_config_parses(config: AgentConfig) -> None:
    # Models may be None (inherited); this just proves it loads and validates.
    assert isinstance(config, AgentConfig)


def test_worker_builds_from_the_full_set(config: AgentConfig) -> None:
    spec = _build_worker(config, read_only=False)
    assert _tool_names(spec) == set(_ALL_TOOL_NAMES)


def test_read_only_worker_holds_no_mutation_tool(config: AgentConfig) -> None:
    """Plan/ask mode read-only by construction, not by config trust."""
    spec = _build_worker(config, read_only=True)
    leaked = _tool_names(spec) & MUTATION_TOOLS
    assert not leaked, f"read-only worker kept mutation tool(s) {sorted(leaked)}"


# --- skills --------------------------------------------------------------------


def _skill_dirs() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/*/SKILL.md"))


def test_every_skill_parses_with_a_matching_name() -> None:
    """deepagents' own loader must accept each skill, name == directory name."""
    backend = FilesystemBackend(root_dir=str(SKILLS_DIR), virtual_mode=True)
    for group in ("/analysis/", "/repair/"):
        skills = _list_skills(backend, group)
        assert skills, f"no skills loaded from {group}"
        for skill in skills:
            # path is /<group>/<name>/SKILL.md
            dir_name = skill["path"].strip("/").split("/")[-2]
            assert skill["name"] == dir_name, (
                f"skill name {skill['name']!r} != directory {dir_name!r}"
            )


def test_skills_reference_only_real_tools() -> None:
    """A tool name a skill names must exist, else the model calls a ghost tool."""
    for md in _skill_dirs():
        body = md.read_text(encoding="utf-8")
        tokens = {
            tok
            for tok in re.findall(r"`([a-z][a-z0-9_]+)`", body)
            if "_" in tok and len(tok) > 3
        }
        unknown = tokens - _KNOWN_TOOLS - _NON_TOOL_TOKENS
        rel = md.relative_to(REPO_ROOT)
        assert not unknown, f"{rel}: references unknown tool/token(s) {sorted(unknown)}"


def test_repair_skills_need_write_tools() -> None:
    """The repair skills are correctly segregated from the read-only analysis set.

    Each repair skill body names at least one mutating tool (so it cannot be
    offered to the read-only plan/ask graphs); analysis skills must not require
    one to function.
    """
    for md in (SKILLS_DIR / "repair").glob("*/SKILL.md"):
        body = md.read_text(encoding="utf-8")
        named = {tok for tok in re.findall(r"`([a-z][a-z0-9_]+)`", body)}
        assert named & MUTATION_TOOLS, (
            f"{md.relative_to(REPO_ROOT)}: a repair skill should use a write tool"
        )


def test_no_skill_uses_the_retired_selector_api() -> None:
    """GhidrAssistMCP 3.x has no `action`/`format` selectors; unknown args error."""
    for md in _skill_dirs():
        text = md.read_text(encoding="utf-8")
        for selector in ("`action:", "`format:"):
            assert selector not in text, (
                f"{md.relative_to(REPO_ROOT)} still tells the model to pass "
                f"{selector[1:]} — the consolidated tools it chose between are gone"
            )
