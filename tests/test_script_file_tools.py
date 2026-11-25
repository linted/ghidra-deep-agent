"""Tests for run_script_with_file_output (script_file_tools.py).

A fake ``run_script`` plays the Ghidra side: it reads the output directory the
tool passes as the first script argument and writes files there, the way a real
GhidraScript would. The in-memory sandbox from the sync tests receives the
uploads, so each test can check exactly what landed where.

Run:  uv run pytest tests/test_script_file_tools.py -v
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from test_sandbox_sync import FakeSandboxBackend

from ghidra_deep_agent.agent_config import read_only_tools
from ghidra_deep_agent.sandbox import SANDBOX_WORKDIR
from ghidra_deep_agent.sandbox_sync import SEED_TAR_GLOB
from ghidra_deep_agent.script_file_tools import TOOL_NAME, build_script_file_tools

Writer = Callable[[Path, list[str]], object]


class FakeTool:
    def __init__(self, name: str, reply: Callable[[dict[str, Any]], str]) -> None:
        self.name = name
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return self._reply(args)


class Ghidra:
    """``create_script`` / ``run_script`` / ``get_task_status`` stand-ins."""

    def __init__(self, write: Writer, output: str = "done") -> None:
        self.seen_dirs: list[Path] = []

        def run(args: dict[str, Any]) -> str:
            out_dir = Path(args["args"][0])
            self.seen_dirs.append(out_dir)
            write(out_dir, list(args["args"][1:]))
            return output

        self.run = FakeTool("run_script", run)
        self.tools = [
            FakeTool("create_script", lambda _a: "created"),
            self.run,
            FakeTool("get_task_status", lambda _a: "Status: RUNNING"),
        ]


def _tool(ghidra: Ghidra, sandbox: FakeSandboxBackend, **kw: Any) -> Any:
    [built] = build_script_file_tools(cast(Any, ghidra.tools), sandbox, **kw)
    return built


def _call(tool: Any, **args: Any) -> str:
    return cast(str, asyncio.run(tool.ainvoke(args)))


def _sandbox() -> FakeSandboxBackend:
    return FakeSandboxBackend(remote_root=SANDBOX_WORKDIR)


def test_delivers_script_files_into_the_sandbox() -> None:
    def write(out: Path, args: list[str]) -> None:
        (out / "region.bin").write_bytes(b"\x00\x01\xff" * 10)
        (out / "tables").mkdir()
        (out / "tables" / "idx.txt").write_text("a,b\n")

    ghidra, sandbox = Ghidra(write, output="wrote 2 files"), _sandbox()
    result = _call(
        _tool(ghidra, sandbox), name="DumpRegion.java", script_args=["0x1000"]
    )

    dest = f"{SANDBOX_WORKDIR}/script_output/DumpRegion"
    assert sandbox.fs[f"{dest}/region.bin"] == b"\x00\x01\xff" * 10
    assert sandbox.fs[f"{dest}/tables/idx.txt"] == b"a,b\n"
    # The output dir comes first; the agent's own args follow it.
    assert ghidra.run.calls[0]["args"][1:] == ["0x1000"]
    assert f"{dest}/region.bin  30 B" in result
    assert "wrote 2 files" in result
    # Contents never reach the model, only the manifest.
    assert "\\x00" not in result and "a,b" not in result
    # The host temp dir is cleaned up.
    assert not ghidra.seen_dirs[0].exists()


def test_many_files_ship_as_one_tarball() -> None:
    def write(out: Path, _args: list[str]) -> None:
        for i in range(6):
            (out / f"page_{i}.bin").write_bytes(bytes([i]) * 64)

    sandbox = _sandbox()
    _call(_tool(Ghidra(write), sandbox), name="Pages.java", dest="pages")

    dest = f"{SANDBOX_WORKDIR}/pages"
    # One tarball upload (plus the small delivery list written afterwards).
    [(staged, _blob)] = sandbox.upload_calls[0]
    assert fnmatch.fnmatch(staged, f"{dest}/{SEED_TAR_GLOB}")
    for i in range(6):
        assert sandbox.fs[f"{dest}/page_{i}.bin"] == bytes([i]) * 64
    assert staged not in sandbox.fs


def test_no_files_written_explains_the_convention() -> None:
    sandbox = _sandbox()
    result = _call(
        _tool(Ghidra(lambda _o, _a: None, output="oops"), sandbox), name="X.java"
    )
    assert "wrote no files" in result and "getScriptArgs()[0]" in result
    assert "oops" in result
    assert sandbox.upload_calls == []


def test_dest_must_stay_inside_the_workdir() -> None:
    ghidra, sandbox = Ghidra(lambda o, _a: (o / "f").write_bytes(b"x")), _sandbox()
    tool = _tool(ghidra, sandbox)
    for bad in ("../escape", "/etc", "a/../../b", "/sandbox/outputx"):
        assert _call(tool, name="X.java", dest=bad).startswith("Error: dest")
    assert ghidra.run.calls == []  # rejected before the script ever runs

    _call(tool, name="X.java", dest=f"{SANDBOX_WORKDIR}/abs/ok")
    assert sandbox.fs[f"{SANDBOX_WORKDIR}/abs/ok/f"] == b"x"


def test_over_the_size_limit_delivers_nothing() -> None:
    sandbox = _sandbox()
    tool = _tool(
        Ghidra(lambda o, _a: (o / "big").write_bytes(b"x" * 100)),
        sandbox,
        max_bytes=50,
    )
    result = _call(tool, name="X.java")
    assert result.startswith("Error:") and "over the limit" in result
    assert sandbox.upload_calls == []


def test_symlinks_are_not_followed_out_of_the_output_dir(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("host-only")

    def write(out: Path, _args: list[str]) -> None:
        (out / "real.txt").write_text("ok")
        os.symlink(secret, out / "leak.txt")

    sandbox = _sandbox()
    _call(_tool(Ghidra(write), sandbox), name="X.java", dest="d")
    assert f"{SANDBOX_WORKDIR}/d/real.txt" in sandbox.fs
    assert f"{SANDBOX_WORKDIR}/d/leak.txt" not in sandbox.fs


def test_timed_out_script_delivers_nothing() -> None:
    def write(out: Path, _args: list[str]) -> None:
        (out / "partial.bin").write_bytes(b"half")

    ghidra = Ghidra(write, output="Script task submitted: abc123def running")
    sandbox = _sandbox()
    result = _call(_tool(ghidra, sandbox, timeout_s=0), name="Slow.java")
    assert "did not complete" in result and "No files were delivered" in result
    assert sandbox.upload_calls == []
    assert not ghidra.seen_dirs[0].exists()


def test_upload_failures_are_reported() -> None:
    sandbox = _sandbox()
    sandbox.upload_errors.add(f"{SANDBOX_WORKDIR}/d/bad.bin")

    def write(out: Path, _args: list[str]) -> None:
        (out / "good.bin").write_bytes(b"g")
        (out / "bad.bin").write_bytes(b"b")

    result = _call(_tool(Ghidra(write), sandbox), name="X.java", dest="d")
    assert "1 file(s), 1 bytes" in result
    assert "could NOT be uploaded" in result and "bad.bin" in result
    # Only what actually landed is recorded for the next call to clear.
    assert sandbox.fs[f"{SANDBOX_WORKDIR}/d/.script_output_files"] == b"good.bin\n"


def test_failures_become_error_results_not_exceptions() -> None:
    """A crash in Ghidra or the sandbox must not end the agent's turn."""

    def boom(_args: dict[str, Any]) -> str:
        raise ConnectionError("ghidra went away")

    ghidra = Ghidra(lambda _o, _a: None)
    ghidra.run._reply = boom
    result = _call(_tool(ghidra, _sandbox()), name="X.java")
    assert result.startswith("Error:") and "ghidra went away" in result
    assert not ghidra.seen_dirs  # never reached the fake's writer

    sandbox = _sandbox()
    sandbox.raise_on_execute = True
    good = Ghidra(lambda o, _a: (o / "f").write_bytes(b"x"))
    result = _call(_tool(good, sandbox), name="X.java")
    assert result.startswith("Error:") and "sandbox gone" in result
    assert not good.seen_dirs[0].exists()  # temp dir still cleaned up


def test_rerun_replaces_previous_delivery_but_keeps_agent_files() -> None:
    outputs = [["a.bin", "b.bin"], ["a.bin"]]

    def write(out: Path, _args: list[str]) -> None:
        for name in outputs.pop(0):
            (out / name).write_bytes(b"v")

    sandbox = _sandbox()
    tool = _tool(Ghidra(write), sandbox)
    dest = f"{SANDBOX_WORKDIR}/d"
    _call(tool, name="X.java", dest="d")
    sandbox.fs[f"{dest}/notes.md"] = b"agent's own"

    result = _call(tool, name="X.java", dest="d")
    assert f"{dest}/a.bin" in sandbox.fs
    assert f"{dest}/b.bin" not in sandbox.fs  # stale output from run 1
    assert sandbox.fs[f"{dest}/notes.md"] == b"agent's own"
    assert "2 old file(s) removed" in result


def test_tampered_delivery_list_cannot_escape_dest() -> None:
    sandbox = _sandbox()
    outside = f"{SANDBOX_WORKDIR}/keep.txt"
    sandbox.fs[outside] = b"keep"
    sandbox.fs[f"{SANDBOX_WORKDIR}/d/.script_output_files"] = (
        b"../keep.txt\n/sandbox/output/keep.txt\na/../../keep.txt\n"
    )
    _call(
        _tool(Ghidra(lambda o, _a: (o / "f").write_bytes(b"x")), sandbox),
        name="X.java",
        dest="d",
    )
    assert sandbox.fs[outside] == b"keep"


def test_files_over_the_sync_limit_are_flagged(monkeypatch: Any) -> None:
    monkeypatch.setenv("SANDBOX_SYNC_MAX_BYTES", "10")

    def write(out: Path, _args: list[str]) -> None:
        (out / "small").write_bytes(b"x")
        (out / "huge").write_bytes(b"x" * 11)

    result = _call(_tool(Ghidra(write), _sandbox()), name="X.java")
    huge = next(line for line in result.splitlines() if "/huge" in line)
    small = next(line for line in result.splitlines() if "/small" in line)
    assert "sandbox only" in huge and "sandbox only" not in small
    assert "will NOT be copied home" in result


def test_error_like_output_is_called_out() -> None:
    trace = (
        "java.lang.NullPointerException: boom\n"
        "    at DumpRegion.run(DumpRegion.java:12)"
    )
    ghidra = Ghidra(lambda o, _a: (o / "f").write_bytes(b"x"), output=trace)
    result = _call(_tool(ghidra, _sandbox()), name="X.java")
    assert result.startswith("Warning:") and "partial" in result
    assert "Script finished" not in result

    clean = Ghidra(lambda o, _a: (o / "f").write_bytes(b"x"), output="wrote 1 file")
    assert not _call(_tool(clean, _sandbox()), name="X.java").startswith("Warning")


def test_not_built_when_ghidra_is_remote() -> None:
    ghidra = Ghidra(lambda _o, _a: None)
    for url in ("http://10.0.0.5:8081/mcp", "http://ghidra.example.com/mcp"):
        assert (
            build_script_file_tools(cast(Any, ghidra.tools), _sandbox(), ghidra_url=url)
            == []
        )
    for url in (
        "http://localhost:8081/mcp",
        "http://127.0.0.1:8081/mcp",
        "http://[::1]:8081/mcp",
    ):
        assert build_script_file_tools(
            cast(Any, ghidra.tools), _sandbox(), ghidra_url=url
        )


def test_explicit_zero_limit_is_honored() -> None:
    sandbox = _sandbox()
    tool = _tool(
        Ghidra(lambda o, _a: (o / "f").write_bytes(b"x")), sandbox, max_bytes=0
    )
    assert "over the limit" in _call(tool, name="X.java")
    assert sandbox.upload_calls == []


def test_not_built_without_a_sandbox_or_script_tools() -> None:
    ghidra = Ghidra(lambda _o, _a: None)
    assert build_script_file_tools(cast(Any, ghidra.tools), None) == []
    no_scripts = [t for t in ghidra.tools if t.name != "run_script"]
    assert build_script_file_tools(cast(Any, no_scripts), _sandbox()) == []


def test_it_is_a_write_tool() -> None:
    """It runs arbitrary scripts, so read-only agents must never get it."""
    tool = _tool(Ghidra(lambda _o, _a: None), _sandbox())
    assert tool.name == TOOL_NAME
    assert read_only_tools([tool]) == []
