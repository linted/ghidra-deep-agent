"""
Unit tests for the config-driven sub-agent builder, focused on the ``read_only``
flag: a read-only sub-agent must (a) drop the write-only tools from its tool set
and (b) get the write-action-blocking validation middleware, while a normal
sub-agent gets neither.

Run:  uv run pytest test_subagents.py -v
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from deepagents import SubAgent
from langchain_core.tools import BaseTool

from ghidra_deep_agent.subagents import (
    PLAN_MODE_BLOCKED_TOOLS,
    READ_ONLY_WRITE_ACTIONS,
    build_subagents,
    load_agent_config,
)
from ghidra_deep_agent.validation import ArgumentValidationMiddleware

# A mix of write-only tools (dropped for read-only agents) and read tools (kept).
_TOOL_NAMES = ["rename_symbol", "get_code", "xrefs", "save_knowledge", "variables"]


def _fake_tools() -> Sequence[BaseTool]:
    """Minimal stand-ins: build_subagents only ever reads ``.name``."""
    tools = [SimpleNamespace(name=name) for name in _TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(spec: str | None) -> str:
    """Stub model resolver: build_subagents only stores the returned value."""
    return spec or "stub-model"


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "subagents.toml"
    path.write_text(
        '[main]\ntools = ["get_code"]\n\n' + body,
        encoding="utf-8",
    )
    return path


def _tool_names(spec: SubAgent) -> set[str]:
    # spec["tools"] is a union type; our stubs expose ``.name`` via getattr.
    return {getattr(tool, "name") for tool in spec["tools"]}


def _validation_mw(spec: SubAgent) -> ArgumentValidationMiddleware:
    mws = [m for m in spec["middleware"] if isinstance(m, ArgumentValidationMiddleware)]
    assert len(mws) == 1, "each sub-agent gets exactly one validation middleware"
    return mws[0]


def _build(tmp_path: Path, body: str) -> dict[str, SubAgent]:
    config = load_agent_config(_write_config(tmp_path, body))
    specs = build_subagents(_fake_tools(), config, resolve_model=_resolver)
    return {spec["name"]: spec for spec in specs}


def test_read_only_subagent_drops_write_tools_and_blocks_write_actions(
    tmp_path: Path,
) -> None:
    specs = _build(
        tmp_path,
        "[[subagents]]\n"
        'name = "research"\n'
        "read_only = true\n"
        'description = "read-only"\n'
        'tools = "*"\n',
    )
    spec = specs["research"]

    tool_names = _tool_names(spec)
    # A blocked write-only tool is gone; read tools remain.
    assert "rename_symbol" not in tool_names
    assert "rename_symbol" in PLAN_MODE_BLOCKED_TOOLS  # guards against list drift
    assert {"get_code", "xrefs"} <= tool_names

    # The validation middleware rejects write actions on dual read/write tools.
    assert _validation_mw(spec)._write_actions == READ_ONLY_WRITE_ACTIONS


def test_normal_subagent_keeps_write_tools_and_allows_actions(tmp_path: Path) -> None:
    specs = _build(
        tmp_path,
        '[[subagents]]\nname = "analyst"\ndescription = "read-write"\ntools = "*"\n',
    )
    spec = specs["analyst"]

    assert "rename_symbol" in _tool_names(spec)
    # No read-only action blocking.
    assert _validation_mw(spec)._write_actions == {}


def test_read_only_defaults_to_false_and_rejects_non_bool(tmp_path: Path) -> None:
    config = load_agent_config(
        _write_config(
            tmp_path,
            '[[subagents]]\nname = "a"\ndescription = "d"\ntools = ["get_code"]\n',
        )
    )
    assert config.subagents[0].read_only is False

    with pytest.raises(ValueError, match="read_only"):
        load_agent_config(
            _write_config(
                tmp_path,
                "[[subagents]]\n"
                'name = "a"\n'
                'description = "d"\n'
                'tools = ["get_code"]\n'
                'read_only = "yes"\n',
            )
        )
