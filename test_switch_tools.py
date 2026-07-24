"""Unit tests for the local jump-table tools (``switch_tools.py``).

These cover the parts that run in *this* process — graceful degradation when the
server's ``scripts`` tool is absent, the JSON-manifest extraction/formatting, and
the ``apply_switch_override`` argument validation — without a live Ghidra. The
Java scripts themselves run inside Ghidra and are exercised by the end-to-end
check in the plan, not here.

Run:  uv run pytest test_switch_tools.py -v
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

from langchain_core.tools import BaseTool

from ghidra_deep_agent.apply_switch_override_script import (
    MARK_END as APPLY_END,
)
from ghidra_deep_agent.apply_switch_override_script import (
    MARK_START as APPLY_START,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_END as FIND_END,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_START as FIND_START,
)
from ghidra_deep_agent.switch_tools import (
    _APPLY_JSON_RE,
    _FIND_JSON_RE,
    _format_apply_summary,
    _format_find_summary,
    build_switch_tools,
)


def _tools(names: list[str]) -> Sequence[BaseTool]:
    """Minimal stand-ins: build_switch_tools only reads ``.name`` at build time."""
    return cast("Sequence[BaseTool]", [SimpleNamespace(name=n) for n in names])


def test_returns_empty_when_scripts_tool_absent() -> None:
    # No `scripts` tool from the server -> the feature disables itself, matching
    # build_prototype_tools' contract.
    assert build_switch_tools([]) == []
    assert build_switch_tools(list(_tools(["get_code", "xrefs"]))) == []


def test_builds_both_tools_when_scripts_present() -> None:
    tools = build_switch_tools(list(_tools(["scripts", "get_task_status"])))
    names = {t.name for t in tools}
    assert names == {"find_unrecovered_switches", "apply_switch_override"}


def test_find_manifest_regex_and_summary_roundtrip() -> None:
    manifest = {
        "counts": {
            "scanned": 1200,
            "unrecovered_funcs": 2,
            "unrecovered_jumps": 3,
            "review_known": 1,
            "decompile_failed": 0,
        },
        "switches": [
            {
                "func_addr": "0x401000",
                "name": "dispatch",
                "jump": "0x401080",
                "mnemonic": "JMP RAX",
                "table_hint": "0x4020a0",
            }
        ],
        "switches_truncated": False,
        "failed": [],
    }
    raw = f"console noise\n{FIND_START}\n{json.dumps(manifest)}\n{FIND_END}\ntrailing"
    match = _FIND_JSON_RE.search(raw)
    assert match is not None
    payload = json.loads(match.group(1))
    summary = _format_find_summary(payload)
    assert "unrecovered_jumps=3" in summary
    assert "0x401080 in 0x401000 dispatch" in summary
    assert "table@0x4020a0" in summary


def test_apply_manifest_regex_and_summary_roundtrip() -> None:
    manifest = {
        "applied": True,
        "jump": "0x401080",
        "func": "dispatch",
        "warning_cleared": True,
        "num_destinations": 7,
        "decompiled_c": "void dispatch(void) {\n  switch(x) { ... }\n}",
        "c_truncated": False,
        "notes": ["disassembled 3 new target(s)"],
    }
    raw = f"{APPLY_START}\n{json.dumps(manifest)}\n{APPLY_END}"
    match = _APPLY_JSON_RE.search(raw)
    assert match is not None
    summary = _format_apply_summary(json.loads(match.group(1)))
    assert "7 destination(s)" in summary
    assert "warning CLEARED" in summary
    assert "Fresh decompilation" in summary


def test_apply_summary_reports_uncleared_warning() -> None:
    summary = _format_apply_summary(
        {
            "applied": True,
            "jump": "0x1",
            "func": "f",
            "warning_cleared": False,
            "num_destinations": 4,
            "decompiled_c": "",
            "notes": [],
        }
    )
    assert "still present" in summary


def test_apply_summary_reports_failure() -> None:
    summary = _format_apply_summary(
        {"applied": False, "error": "writeOverride failed: bad addr", "notes": []}
    )
    assert "NOT applied" in summary
    assert "writeOverride failed" in summary


def _apply_tool() -> BaseTool:
    tools = build_switch_tools(list(_tools(["scripts", "get_task_status"])))
    tool = next(t for t in tools if t.name == "apply_switch_override")
    return tool


def test_apply_rejects_neither_contract() -> None:
    # Neither destinations nor table_address -> validated before any script runs.
    out = asyncio.run(_apply_tool().ainvoke({"jump_address": "0x401080"}))
    assert "EITHER `destinations` OR `table_address`" in out


def test_apply_rejects_both_contracts() -> None:
    out = asyncio.run(
        _apply_tool().ainvoke(
            {
                "jump_address": "0x401080",
                "destinations": ["0x401100"],
                "table_address": "0x4020a0",
                "element_size": 4,
                "count": 4,
            }
        )
    )
    assert "not both/neither" in out


def test_apply_rejects_incomplete_table_form() -> None:
    out = asyncio.run(
        _apply_tool().ainvoke({"jump_address": "0x401080", "table_address": "0x4020a0"})
    )
    assert "needs" in out and "element_size" in out
