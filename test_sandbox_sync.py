"""
Unit tests for SandboxSyncMiddleware.

Focus: the sha256 manifest drives delta transfers in both directions — first
turn seeds everything, unchanged files transfer nothing, local edits re-upload,
sandbox edits download back — while oversized files are skipped and any backend
error is turned into a warning toast without killing the turn.

Uses a dict-backed FakeSandboxBackend (a real BaseSandbox subclass, so the
middleware's async calls exercise the genuine sync/async delegation). No network.

Run:  uv run pytest test_sandbox_sync.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from ghidra_deep_agent.sandbox_sync import SandboxSyncMiddleware
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink


@pytest.fixture(autouse=True)
def _clear_sinks() -> Generator[None, None, None]:
    """Toast sinks are module-global; reset between tests to avoid cross-talk."""
    import ghidra_deep_agent.toasts as toasts

    toasts._sinks.clear()
    yield
    toasts._sinks.clear()


class FakeSandboxBackend(BaseSandbox):
    """In-memory sandbox filesystem keyed by absolute path.

    Records upload batches and answers the middleware's ``find | sha256sum``
    probe from its own dict, so a test can mutate ``fs`` to simulate edits made
    inside the sandbox.
    """

    def __init__(self, remote_root: str = "/workspace") -> None:
        self.remote_root = remote_root
        self.fs: dict[str, bytes] = {}
        self.upload_calls: list[list[tuple[str, bytes]]] = []
        # Paths for which upload should report a failure (retry contract test).
        self.upload_errors: set[str] = set()
        # When set, execute() raises to exercise hook error handling.
        self.raise_on_execute = False

    @property
    def id(self) -> str:
        return "fake-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if self.raise_on_execute:
            raise RuntimeError("sandbox gone")
        if "sha256sum" not in command:
            return ExecuteResponse(output="", exit_code=0)
        prefix = f"{self.remote_root}/"
        lines = []
        for path, data in sorted(self.fs.items()):
            if not path.startswith(prefix):
                continue
            rel = path[len(prefix) :]
            digest = hashlib.sha256(data).hexdigest()
            lines.append(f"{len(data)} {digest}  ./{rel}")
        return ExecuteResponse(output="\n".join(lines), exit_code=0)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.upload_calls.append(list(files))
        responses = []
        for path, data in files:
            if path in self.upload_errors:
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
                continue
            self.fs[path] = data
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for path in paths:
            if path in self.fs:
                responses.append(FileDownloadResponse(path=path, content=self.fs[path]))
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="file_not_found"
                    )
                )
        return responses


def _mw(
    backend: FakeSandboxBackend, local_dir: Path, **kw: Any
) -> SandboxSyncMiddleware:
    return SandboxSyncMiddleware(
        cast(Any, backend), local_dir, remote_root=backend.remote_root, **kw
    )


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_first_seed_uploads_all_then_noop(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", b"alpha")
    _write(tmp_path, "sub/b.txt", b"bravo")
    backend = FakeSandboxBackend()
    mw = _mw(backend, tmp_path)

    asyncio.run(mw._seed())
    assert len(backend.upload_calls) == 1
    assert {p for p, _ in backend.upload_calls[0]} == {
        "/workspace/a.txt",
        "/workspace/sub/b.txt",
    }

    # Nothing changed: a second seed uploads nothing at all.
    asyncio.run(mw._seed())
    assert len(backend.upload_calls) == 1


def test_locally_changed_file_reuploads(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", b"alpha")
    backend = FakeSandboxBackend()
    mw = _mw(backend, tmp_path)
    asyncio.run(mw._seed())

    _write(tmp_path, "a.txt", b"alpha-v2")
    asyncio.run(mw._seed())

    assert len(backend.upload_calls) == 2
    assert backend.upload_calls[1] == [("/workspace/a.txt", b"alpha-v2")]


def test_remote_change_downloads_back(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", b"alpha")
    backend = FakeSandboxBackend()
    mw = _mw(backend, tmp_path)
    asyncio.run(mw._seed())

    # Agent edits the file and creates a new one inside the sandbox.
    backend.fs["/workspace/a.txt"] = b"edited-in-sandbox"
    backend.fs["/workspace/report.md"] = b"# findings"
    asyncio.run(mw._sync_back())

    assert (tmp_path / "a.txt").read_bytes() == b"edited-in-sandbox"
    assert (tmp_path / "report.md").read_bytes() == b"# findings"

    # Manifest now matches the sandbox: a follow-up sync-back downloads nothing new.
    (tmp_path / "a.txt").write_bytes(b"local-only-change-not-synced-back")
    asyncio.run(mw._sync_back())
    # unchanged remote content -> local file left as the test just wrote it
    assert (tmp_path / "a.txt").read_bytes() == b"local-only-change-not-synced-back"


def test_oversize_file_skipped_with_toast(tmp_path: Path) -> None:
    toasts: list[ToastRequest] = []
    register_toast_sink(toasts.append)
    _write(tmp_path, "big.bin", b"x" * 100)
    backend = FakeSandboxBackend()
    mw = _mw(backend, tmp_path, max_bytes=10)

    asyncio.run(mw._seed())

    assert backend.upload_calls == []  # nothing uploaded
    assert any("big.bin" in t.message for t in toasts)
    assert all(t.severity == "warning" for t in toasts)


def test_backend_error_in_hook_toasts_and_survives(tmp_path: Path) -> None:
    toasts: list[ToastRequest] = []
    register_toast_sink(toasts.append)
    _write(tmp_path, "a.txt", b"alpha")
    backend = FakeSandboxBackend()
    backend.raise_on_execute = True
    mw = _mw(backend, tmp_path)

    # aafter_agent must not raise even though execute() blows up.
    result = asyncio.run(mw.aafter_agent(cast(Any, {}), cast(Any, None)))
    assert result is None
    assert any("sync-back failed" in t.message.lower() for t in toasts)


def test_upload_error_keeps_file_out_of_manifest(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", b"alpha")
    backend = FakeSandboxBackend()
    backend.upload_errors.add("/workspace/a.txt")
    mw = _mw(backend, tmp_path)

    asyncio.run(mw._seed())
    # Errored upload is not recorded, so the next seed retries it.
    asyncio.run(mw._seed())

    assert len(backend.upload_calls) == 2
    assert backend.upload_calls[1] == [("/workspace/a.txt", b"alpha")]
