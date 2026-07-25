"""Tests for the shared Ghidra script runner (ghidra_script_tools.py).

Covers the deploy/run/parse path that prototype_tools and switch_tools now
share, including the failure messages the model sees when a script produces no
usable manifest.

Run:  uv run pytest test_ghidra_script_tools.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from ghidra_deep_agent.ghidra_script_tools import (
    GhidraScriptRunner,
    find_scripts_tools,
    manifest_pattern,
)

MARK_START = "<<<GDA_JSON"
MARK_END = "GDA_JSON>>>"
PATTERN = manifest_pattern(MARK_START, MARK_END)


class FakeScriptsTool:
    """Records the `scripts` calls and replays a canned run output."""

    name = "scripts"

    def __init__(self, run_output: str) -> None:
        self.run_output = run_output
        self.create_output = "created"
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        if args["action"] == "create":
            return self.create_output
        return self.run_output


class FakeNamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


def _runner(run_output: str) -> tuple[GhidraScriptRunner, FakeScriptsTool]:
    scripts = FakeScriptsTool(run_output)
    return GhidraScriptRunner(cast(Any, scripts), None), scripts


def _manifest(body: str) -> str:
    return f"noise before\n{MARK_START}\n{body}\n{MARK_END}\ntrailing noise"


# ── tool discovery ────────────────────────────────────────────────────────────


def test_missing_scripts_tool_returns_none() -> None:
    assert find_scripts_tools([], "widgets disabled.") is None
    others = [cast(Any, FakeNamedTool("get_code"))]
    assert find_scripts_tools(others, "widgets disabled.") is None


def test_finds_scripts_and_optional_status_tool() -> None:
    tools = [cast(Any, FakeNamedTool(n)) for n in ("scripts", "get_task_status")]
    found = find_scripts_tools(tools, "n/a")
    assert found is not None
    scripts_tool, status_tool = found
    assert scripts_tool.name == "scripts"
    assert status_tool is not None and status_tool.name == "get_task_status"


def test_status_tool_is_optional() -> None:
    found = find_scripts_tools([cast(Any, FakeNamedTool("scripts"))], "n/a")
    assert found is not None
    assert found[1] is None


# ── deploy + run ──────────────────────────────────────────────────────────────


def test_script_is_redeployed_with_overwrite_then_run() -> None:
    runner, scripts = _runner(_manifest('{"ok": true}'))

    asyncio.run(runner.run("x.java", "class X {}"))

    create, run = scripts.calls
    assert create == {
        "action": "create",
        "name": "x.java",
        "source": "class X {}",
        "overwrite": True,
    }
    assert run == {"action": "run", "name": "x.java"}


def test_run_args_are_forwarded() -> None:
    runner, scripts = _runner(_manifest("{}"))
    asyncio.run(runner.run("x.java", "src", ["dry_run"]))
    assert scripts.calls[1]["args"] == ["dry_run"]


def test_empty_run_args_are_omitted() -> None:
    runner, scripts = _runner(_manifest("{}"))
    asyncio.run(runner.run("x.java", "src", []))
    assert "args" not in scripts.calls[1]


# ── manifest parsing ──────────────────────────────────────────────────────────


def test_manifest_is_extracted_from_surrounding_output() -> None:
    runner, _ = _runner(_manifest('{"counts": {"scanned": 3}}'))

    payload, raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert error == ""
    assert payload == {"counts": {"scanned": 3}}
    # Raw output is returned too: deobfuscate_cff renders the human report from it.
    assert "trailing noise" in raw


def test_missing_manifest_reports_the_raw_tail() -> None:
    runner, _ = _runner("Exception in script: NullPointerException")

    payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert payload is None
    assert error.startswith("my_tool: no JSON manifest found")
    assert "NullPointerException" in error


def test_empty_output_is_reported_not_crashed() -> None:
    runner, _ = _runner("")

    payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert payload is None
    assert "(empty result)" in error


def test_malformed_json_is_reported() -> None:
    # Braces are present so the pattern matches; the body is still not JSON.
    runner, _ = _runner(_manifest("{'single': quoted}"))

    payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert payload is None
    assert error.startswith("my_tool: could not parse manifest JSON")


def test_two_marker_pairs_take_the_last_manifest() -> None:
    """A greedy pattern spans both pairs and yields unparseable JSON.

    Output can carry more than one manifest — a redeploy that echoes the previous
    run, or a script that emits twice. The current run's is the last one.
    """
    runner, _ = _runner(_manifest('{"run": 1}') + "\n" + _manifest('{"run": 2}'))

    payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert error == ""
    assert payload == {"run": 2}


def test_missing_manifest_surfaces_the_deploy_output() -> None:
    """A javac error lands in the deploy step and nowhere else.

    Ghidra compiles the script directory as one OSGi bundle, so a compile failure
    is by far the likeliest reason a run produced no manifest — the model can't
    act on it unless it is shown.
    """
    scripts = FakeScriptsTool("no manifest here")
    scripts.create_output = "error: cannot find symbol\n  symbol: class Addr"
    runner = GhidraScriptRunner(cast(Any, scripts), None)

    _payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    assert "Deploy (compile) output:" in error
    assert "cannot find symbol" in error


def test_raw_tail_is_bounded() -> None:
    runner, _ = _runner("z" * 5000)

    _payload, _raw, error = asyncio.run(
        runner.run_manifest("x.java", "src", PATTERN, "my_tool")
    )

    # The tail is capped so a runaway script can't flood the model's context.
    assert error.count("z") == 800
