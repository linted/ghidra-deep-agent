"""Unit tests for the local ``recover_prototypes`` tool (``prototype_tools.py``).

Mirrors ``test_switch_tools.py`` for the prototype-recovery sibling: graceful
degradation when the server's ``scripts`` tool is absent, and the manifest ->
summary formatting the model actually reads. The Java script itself runs inside
Ghidra and is only compile-checked (``test_switch_scripts_compile.py``).

The summary is the entire product of this tool — an escalation the formatter
drops is a function nobody reviews — so each branch is covered here.

Run:  uv run pytest tests/test_prototype_tools.py -v
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.tools import BaseTool

from ghidra_deep_agent.prototype_tools import (
    _JSON_RE,
    _format_summary,
    build_prototype_tools,
)
from ghidra_deep_agent.recover_prototypes_script import MARK_END, MARK_START


class FakeNamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


def _tools(names: list[str]) -> Sequence[BaseTool]:
    return [cast(BaseTool, cast(Any, FakeNamedTool(n))) for n in names]


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dry_run": False,
        "counts": {
            "scanned": 10,
            "already_correct": 4,
            "fixed": 3,
            "escalate": 2,
            "escalate_known": 1,
            "decompile_failed": 1,
        },
        "escalate": [],
        "failed": [],
    }
    base.update(overrides)
    return base


# ── tool construction ─────────────────────────────────────────────────────────


def test_no_scripts_tool_yields_no_tools() -> None:
    """The server ships `scripts` disabled by default; degrade, don't crash."""
    assert build_prototype_tools(list(_tools(["get_code"]))) == []


def test_scripts_tool_yields_the_recover_tool() -> None:
    tools = build_prototype_tools(list(_tools(["scripts", "get_task_status"])))
    assert [t.name for t in tools] == ["recover_prototypes"]


# ── manifest extraction ───────────────────────────────────────────────────────


def test_manifest_regex_matches_the_script_markers() -> None:
    raw = f"chatter\n{MARK_START}\n{json.dumps(_payload())}\n{MARK_END}\nmore"
    match = _JSON_RE.search(raw)
    assert match is not None
    assert json.loads(match.group(1))["counts"]["fixed"] == 3


# ── summary formatting ────────────────────────────────────────────────────────


def test_counts_line_reports_every_bucket() -> None:
    out = _format_summary(_payload())
    assert "scanned=10" in out
    assert "already_correct=4" in out
    assert "auto_fixed=3" in out
    assert "needs_review=2 new" in out
    assert "(1 already flagged)" in out
    assert "decompile_failed=1" in out


def test_dry_run_says_nothing_was_applied() -> None:
    """The counts read identically either way; only the header distinguishes them."""
    out = _format_summary(_payload(dry_run=True))
    assert "DRY RUN" in out
    assert "WOULD be fixed" in out


def test_clean_run_reports_nothing_to_review() -> None:
    out = _format_summary(_payload())
    assert "No new functions need manual review." in out


def test_escalations_are_listed_with_both_prototypes() -> None:
    """The point of an escalation is the committed-vs-recovered comparison."""
    out = _format_summary(
        _payload(
            escalate=[
                {
                    "addr": "0x401000",
                    "name": "FUN_00401000",
                    "committed": "undefined FUN_00401000(void)",
                    "recovered": "int FUN_00401000(char *, int)",
                    "reason": "variadic",
                }
            ]
        )
    )
    assert "0x401000" in out
    assert "undefined FUN_00401000(void)" in out
    assert "int FUN_00401000(char *, int)" in out
    assert "variadic" in out


def test_failed_decompiles_are_listed_with_their_error() -> None:
    out = _format_summary(
        _payload(
            failed=[{"addr": "0x402000", "name": "FUN_00402000", "error": "timeout"}]
        )
    )
    assert "Failed to decompile" in out
    assert "0x402000" in out
    assert "timeout" in out


def test_truncation_flags_are_surfaced() -> None:
    """Both detail lists are capped script-side; a silent cap under-reports."""
    out = _format_summary(
        _payload(
            escalate=[{"addr": "0x1", "name": "a"}],
            failed=[{"addr": "0x2", "name": "b"}],
            escalate_truncated=True,
            failed_truncated=True,
        )
    )
    assert out.count("list truncated") == 2
    assert "needs_review count above is the true total" in out
    assert "decompile_failed count above is the true total" in out


def test_missing_fields_do_not_crash_the_formatter() -> None:
    """A manifest from an older/partial run must degrade, not raise."""
    out = _format_summary({})
    assert "scanned=0" in out
