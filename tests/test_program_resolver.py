"""Unit tests for the async MCP-tool resolver helpers (no Ghidra, no network).

Drive each helper with a stub tool object exposing ``.name`` and an async
``ainvoke`` — the same surface langchain_mcp_adapters tools present — so the
web service's list/open orchestration can be exercised without a live engine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from ghidra_deep_agent import program_resolver as pr


class _StubTool:
    """Minimal stand-in for an MCP tool: a name and a canned ainvoke result."""

    def __init__(self, name: str, result: Any) -> None:
        self.name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


# ---- parse_project_files ---------------------------------------------------


def test_parse_project_files_filters_non_programs() -> None:
    payload = json.dumps(
        {
            "files": [
                {"name": "a.bin", "type": "Program"},
                {"name": "notes.txt", "type": "TextFile"},
                {"name": "b.bin", "type": "Program"},
            ],
            "folders": [{"name": "sub"}],
        }
    )
    assert pr.parse_project_files(payload) == ["a.bin", "b.bin"]


def test_parse_project_files_missing_type_kept() -> None:
    payload = json.dumps({"files": [{"name": "a.bin"}]})
    assert pr.parse_project_files(payload) == ["a.bin"]


def test_parse_project_files_falls_back_to_program_list() -> None:
    # No "files" key → fall back to the open-programs parser shape.
    payload = json.dumps([{"name": "a.bin"}, {"name": "b.bin"}])
    assert pr.parse_project_files(payload) == ["a.bin", "b.bin"]


# ---- list_project_programs -------------------------------------------------


def test_list_project_programs_returns_names() -> None:
    tool = _StubTool(
        "list_project_files",
        json.dumps({"files": [{"name": "a.bin", "type": "Program"}]}),
    )
    assert asyncio.run(pr.list_project_programs([tool])) == ["a.bin"]


def test_list_project_programs_raises_on_error_payload() -> None:
    tool = _StubTool(
        "list_project_files",
        json.dumps({"error": "Project listing requires GUI mode"}),
    )
    with pytest.raises(RuntimeError, match="GUI mode"):
        asyncio.run(pr.list_project_programs([tool]))


def test_list_project_programs_raises_when_tool_missing() -> None:
    with pytest.raises(RuntimeError, match="list_project_files"):
        asyncio.run(pr.list_project_programs([]))


# ---- open_program ----------------------------------------------------------


def test_open_program_invokes_with_path_and_no_analysis() -> None:
    tool = _StubTool("open_program", json.dumps({"status": "opened"}))
    asyncio.run(pr.open_program([tool], "/a.bin"))
    assert tool.calls == [{"path": "/a.bin", "auto_analyze": False}]


def test_open_program_raises_on_error_result() -> None:
    tool = _StubTool("open_program", json.dumps({"error": "not found"}))
    with pytest.raises(RuntimeError, match="not found"):
        asyncio.run(pr.open_program([tool], "/missing.bin"))


# ---- is_program_open -------------------------------------------------------


def test_is_program_open_true_and_false() -> None:
    tool = _StubTool(
        "list_open_programs",
        json.dumps({"programs": [{"name": "a.bin"}]}),
    )
    assert asyncio.run(pr.is_program_open([tool], "a.bin")) is True
    assert asyncio.run(pr.is_program_open([tool], "b.bin")) is False
