"""Run a Ghidra script and deliver the files it writes into the sandbox.

Ghidra runs on the host, the agent's shell runs in a remote OpenShell sandbox,
and nothing a Ghidra script writes is visible from the other side. Without a
bridge the agent has to pull bulk data (raw bytes, tables, listings) through
its own context — read it with a tool, then type it back out into a shell
command — which is slow, token-hungry, and lossy.

``run_script_with_file_output`` wraps GhidrAssistMCP's ``run_script``: it makes
a fresh host temp directory, passes its path to the script as the first script
argument, runs the script, then uploads every file the script wrote into
``SANDBOX_WORKDIR/<dest>/`` and returns a short manifest. The data never enters
the model's context, and because it lands in the synced work dir it also
syncs home with everything else.

Assumes Ghidra and this process share a filesystem (Ghidra runs locally). The
script is the agent's own, created beforehand with ``create_script``; this tool
only runs it.
"""

from __future__ import annotations

import asyncio
import hashlib
import posixpath
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.tools import BaseTool, tool

from ghidra_deep_agent.async_tasks import (
    _timeout_text,
    resolve_async_result,
    to_text,
)
from ghidra_deep_agent.defaults import env_int
from ghidra_deep_agent.ghidra_script_tools import find_scripts_tools, script_timeout_s
from ghidra_deep_agent.sandbox import SANDBOX_WORKDIR
from ghidra_deep_agent.sandbox_sync import upload_via_tar

TOOL_NAME = "run_script_with_file_output"

_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_FILES = 5000
# At or above this many files, ship them as one tarball instead of per-file.
_TAR_MIN_FILES = 4
# Manifest lines echoed back to the model; the rest are summarized as a count.
_MANIFEST_MAX_LINES = 40
# Tail of the script's own output (println) echoed back to the model.
_OUTPUT_TAIL_CHARS = 1500


@dataclass(frozen=True)
class _Written:
    rel: str  # path relative to the script's output dir (posix)
    size: int
    sha256: str


def _normalize_dest(dest: str, script_name: str) -> str | None:
    """Resolve ``dest`` to a path relative to ``SANDBOX_WORKDIR``, or ``None``.

    Accepts a relative path, or an absolute one already under the work dir.
    Rejects anything that would escape it, so the files always land somewhere
    the sync brings home.
    """
    raw = dest.strip() or f"script_output/{Path(script_name).stem}"
    if raw.startswith("/"):
        if raw != SANDBOX_WORKDIR and not raw.startswith(SANDBOX_WORKDIR + "/"):
            return None
        raw = raw[len(SANDBOX_WORKDIR) :].lstrip("/")
    norm = posixpath.normpath(raw) if raw else "."
    if norm == ".." or norm.startswith("../"):
        return None
    return "" if norm == "." else norm


def _collect(out_dir: Path) -> list[_Written]:
    """List regular files under ``out_dir``. Blocking — call via ``to_thread``.

    Symlinks are skipped: a script could otherwise point one at any host file
    and have this tool ship it into the sandbox.
    """
    found: list[_Written] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        found.append(
            _Written(
                rel=path.relative_to(out_dir).as_posix(),
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return found


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) <= _OUTPUT_TAIL_CHARS:
        return text
    return "…" + text[-_OUTPUT_TAIL_CHARS:]


class _Uploader:
    """Pushes a host directory's files into the sandbox."""

    def __init__(self, backend: SandboxBackendProtocol) -> None:
        self._backend = backend

    async def upload(
        self, out_dir: Path, files: list[_Written], remote_dir: str
    ) -> list[tuple[str, str]]:
        """Upload ``files`` under ``remote_dir``; returns ``(rel, error)`` failures."""
        mkdir = await self._backend.aexecute(f"mkdir -p {shlex.quote(remote_dir)}")
        if mkdir.exit_code not in (0, None):
            err = f"could not create {remote_dir}: {mkdir.output.strip()[:200]}"
            return [(f.rel, err) for f in files]

        def read_all() -> list[tuple[str, bytes]]:
            return [(f.rel, (out_dir / f.rel).read_bytes()) for f in files]

        items = await asyncio.to_thread(read_all)
        if len(items) >= _TAR_MIN_FILES:
            if await upload_via_tar(self._backend, remote_dir, items) is None:
                return []
            # Fall through to per-file uploads; the per-file errors are what
            # gets reported if those fail too.
        responses = await self._backend.aupload_files(
            [(f"{remote_dir}/{rel}", data) for rel, data in items]
        )
        by_path = {r.path: r for r in responses}
        failures: list[tuple[str, str]] = []
        for rel, _data in items:
            resp = by_path.get(f"{remote_dir}/{rel}")
            if resp is None:
                failures.append((rel, "no upload response"))
            elif resp.error is not None:
                failures.append((rel, str(resp.error)))
        return failures


def _format_result(
    remote_dir: str,
    files: list[_Written],
    failures: list[tuple[str, str]],
    script_output: str,
) -> str:
    failed = {rel for rel, _ in failures}
    landed = [f for f in files if f.rel not in failed]
    total = sum(f.size for f in landed)
    lines = [
        f"Script finished. {len(landed)} file(s), {total:,} bytes, written to "
        f"{remote_dir}/ in the sandbox (use the shell to work with them):"
    ]
    for f in landed[:_MANIFEST_MAX_LINES]:
        lines.append(f"- {remote_dir}/{f.rel}  {f.size:,} B  sha256:{f.sha256[:16]}")
    if len(landed) > _MANIFEST_MAX_LINES:
        lines.append(f"- … and {len(landed) - _MANIFEST_MAX_LINES} more")
    if failures:
        lines.append(f"\n{len(failures)} file(s) could NOT be uploaded:")
        for rel, err in failures[:10]:
            lines.append(f"- {rel}: {err[:200]}")
    if script_output.strip():
        lines.append("\nScript output (tail):\n" + _tail(script_output))
    return "\n".join(lines)


def build_script_file_tools(
    mcp_tools: list[BaseTool],
    sandbox: SandboxBackendProtocol | None,
    *,
    max_bytes: int | None = None,
    max_files: int | None = None,
    timeout_s: float | None = None,
) -> list[BaseTool]:
    """Build ``run_script_with_file_output``, or ``[]`` when it can't work.

    Needs a sandbox (without one the agent can already write host files
    directly) and GhidrAssistMCP's script tools (disabled by default).
    """
    if sandbox is None:
        return []
    found = find_scripts_tools(mcp_tools, f"{TOOL_NAME!r} disabled.")
    if found is None:
        return []
    _create, run_tool, status_tool = found
    uploader = _Uploader(sandbox)
    limit_bytes = max_bytes or env_int("SCRIPT_OUTPUT_MAX_BYTES", _DEFAULT_MAX_BYTES)
    limit_files = max_files or env_int("SCRIPT_OUTPUT_MAX_FILES", _DEFAULT_MAX_FILES)

    @tool(TOOL_NAME)
    async def run_script_with_file_output(
        name: str, script_args: list[str] | None = None, dest: str = ""
    ) -> str:
        """Run a Ghidra script and deliver the files it writes into the sandbox.

        This is the preferred way to get large blobs of data out of the program
        — raw memory/section bytes, tables, bulk listings, anything too big to
        read comfortably through tool results. The data goes straight to files
        in the sandbox, never through your context; process it there with the
        shell (xxd, python, your own tools).

        How it works: create the script first with ``create_script`` (Java
        GhidraScript), then call this instead of ``run_script``. The script
        receives a fresh, empty output directory as its FIRST script argument;
        your own ``script_args`` follow it. Write files there — relative paths and
        subdirectories are kept. Files written anywhere else are NOT delivered.

            String outDir = getScriptArgs()[0];
            String[] myArgs = Arrays.copyOfRange(getScriptArgs(), 1,
                                                 getScriptArgs().length);
            byte[] buf = new byte[len];
            currentProgram.getMemory().getBytes(toAddr(start), buf);
            Files.write(Paths.get(outDir, "region.bin"), buf);

        Keep the script bounded: read large ranges in chunks, call
        ``monitor.checkCancelled()`` in loops, and stop at a size you need.

        Args:
            name: The script's name as given to ``create_script``.
            script_args: Extra script arguments; they arrive after the output
                dir.
            dest: Sandbox directory for the files, relative to the working
                directory (default ``script_output/<script name>``). Must stay
                inside the working directory.

        Returns a manifest of delivered files (sandbox path, size, sha256) and
        the tail of the script's printed output — never the file contents.
        """
        rel_dest = _normalize_dest(dest, name)
        if rel_dest is None:
            return (
                f"Error: dest {dest!r} must be a path inside {SANDBOX_WORKDIR} "
                "(relative, without '..')."
            )
        remote_dir = f"{SANDBOX_WORKDIR}/{rel_dest}".rstrip("/")
        out_dir = Path(tempfile.mkdtemp(prefix="ghidra_script_out_"))
        try:
            run_call: dict[str, Any] = {
                "name": name,
                "args": [str(out_dir), *(script_args or [])],
            }
            limit = script_timeout_s() if timeout_s is None else timeout_s
            raw = to_text(await run_tool.ainvoke(run_call))
            output = await resolve_async_result(raw, status_tool, timeout_s=limit)
            if output == _timeout_text(limit):
                # The script may still be running and writing; don't ship a
                # half-written set as if it were the result.
                return output + "\nNo files were delivered."
            files = await asyncio.to_thread(_collect, out_dir)
            if not files:
                return (
                    "The script wrote no files to its output directory "
                    "(getScriptArgs()[0]), so nothing was delivered. Script "
                    "output (tail):\n" + (_tail(output) or "(empty)")
                )
            total = sum(f.size for f in files)
            if len(files) > limit_files or total > limit_bytes:
                return (
                    f"Error: the script wrote {len(files)} file(s), {total:,} "
                    f"bytes — over the limit of {limit_files} files / "
                    f"{limit_bytes:,} bytes per call. Nothing was delivered; "
                    "export a smaller range or split the work across calls."
                )
            failures = await uploader.upload(out_dir, files, remote_dir)
            return _format_result(remote_dir, files, failures, output)
        finally:
            await asyncio.to_thread(_remove, out_dir)

    return [run_script_with_file_output]


def _remove(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:  # leftover temp dir; not worth failing the call
        print(f"Warning: could not remove {path}: {exc}", file=sys.stderr)
