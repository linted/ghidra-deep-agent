"""Sync a local directory into an OpenShell sandbox and back, each turn.

A deep-agent sandbox gives the agent its own isolated filesystem, so files it
writes never appear on the host. This middleware bridges that gap: before each
turn it uploads ``AGENT_OUTPUT_DIR`` into the sandbox, and after each turn it
downloads changed files back — making the local directory the durable record
across runs (a fresh sandbox is created per app launch). It follows the
LangChain deep-agents "SandboxSyncMiddleware" guidance.

Change detection uses an in-memory sha256 manifest keyed by path relative to
``AGENT_OUTPUT_DIR``: only new or changed files are transferred in either
direction. Files deleted inside the sandbox are left in place locally. Any sync
failure is surfaced as a warning toast and swallowed — a sandbox hiccup must
never kill the turn.
"""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import tarfile
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from ghidra_deep_agent.sandbox import SANDBOX_WORKDIR
from ghidra_deep_agent.toasts import notify_toast

_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_MAX_BYTES_ENV = "SANDBOX_SYNC_MAX_BYTES"

_DEFAULT_TAR_MIN_FILES = 4
_TAR_MIN_FILES_ENV = "SANDBOX_SYNC_TAR_MIN_FILES"

# Staged inside the sync root — the only path guaranteed writable under the
# sandbox's Landlock policy (/tmp is not). The sync-back ``find`` excludes it
# by name, and the extract command removes it even when extraction fails.
_SEED_TAR_NAME = ".ghidra_seed.tar.gz"


def _default_max_bytes() -> int:
    """Per-file sync size cap from ``SANDBOX_SYNC_MAX_BYTES`` (bytes)."""
    raw = os.environ.get(_MAX_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BYTES
    return value if value > 0 else _DEFAULT_MAX_BYTES


def _default_tar_min_files() -> int:
    """Bulk-seed threshold from ``SANDBOX_SYNC_TAR_MIN_FILES`` (file count)."""
    raw = os.environ.get(_TAR_MIN_FILES_ENV, "").strip()
    if not raw:
        return _DEFAULT_TAR_MIN_FILES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_TAR_MIN_FILES
    return value if value > 0 else _DEFAULT_TAR_MIN_FILES


class SandboxSyncMiddleware(AgentMiddleware):
    """Keep ``local_dir`` and a sandbox directory in sync across turns."""

    def __init__(
        self,
        backend: SandboxBackendProtocol,
        local_dir: Path,
        *,
        remote_root: str = SANDBOX_WORKDIR,
        max_bytes: int | None = None,
        tar_min_files: int | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._local_dir = local_dir
        # A dedicated subdirectory, NOT the sandbox home (which holds the
        # language toolchain and dotfiles) — syncing the home would drag tens of
        # thousands of unrelated files into the local mirror.
        self._remote_root = remote_root.rstrip("/") or "/"
        self._max_bytes = max_bytes if max_bytes is not None else _default_max_bytes()
        self._tar_min_files = (
            tar_min_files if tar_min_files is not None else _default_tar_min_files()
        )
        # relpath -> sha256 hex of the last content synced in either direction.
        self._manifest: dict[str, str] = {}
        # relpaths already warned about for exceeding the size cap (warn once).
        self._warned_oversize: set[str] = set()
        # Whether the remote root has been created this session (created lazily).
        self._root_ready = False
        # Whether we already warned that bulk seeding fell back to per-file.
        self._warned_tar_fallback = False

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        try:
            await self._seed()
        except Exception as exc:  # noqa: BLE001 - a sync hiccup must not kill the turn
            notify_toast(
                f"Sandbox seed failed: {exc}", severity="warning", title="Sandbox"
            )
        return None

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        try:
            await self._sync_back()
        except Exception as exc:  # noqa: BLE001 - a sync hiccup must not kill the turn
            notify_toast(
                f"Sandbox sync-back failed: {exc}", severity="warning", title="Sandbox"
            )
        return None

    # -- internals -----------------------------------------------------------

    def _remote_path(self, rel: str) -> str:
        return f"{self._remote_root}/{rel}"

    async def _ensure_root(self) -> None:
        """Create the remote sync root once (uploads need its parent to exist).

        Raises on failure so the calling hook surfaces a warning toast; only
        marks the root ready on success, so a failed create is retried next turn
        rather than silently skipped.
        """
        if self._root_ready:
            return
        result = await self._backend.aexecute(
            f"mkdir -p {shlex.quote(self._remote_root)}"
        )
        if result.exit_code not in (0, None):
            raise RuntimeError(
                f"could not create sandbox dir {self._remote_root}: "
                f"{result.output.strip()[:200]}"
            )
        self._root_ready = True

    def _warn_oversize(self, rel: str) -> None:
        if rel in self._warned_oversize:
            return
        self._warned_oversize.add(rel)
        notify_toast(
            f"Sandbox sync skipped {rel}: larger than {self._max_bytes} bytes",
            severity="warning",
            title="Sandbox",
        )

    async def _seed(self) -> None:
        """Upload new/changed local files into the sandbox."""
        if not self._local_dir.is_dir():
            return
        uploads: list[tuple[str, bytes]] = []
        pending: dict[str, tuple[str, str]] = {}  # remote path -> (rel, digest)
        for path in sorted(self._local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self._local_dir).as_posix()
            data = path.read_bytes()
            if len(data) > self._max_bytes:
                self._warn_oversize(rel)
                continue
            digest = hashlib.sha256(data).hexdigest()
            if self._manifest.get(rel) == digest:
                continue
            remote = self._remote_path(rel)
            uploads.append((remote, data))
            pending[remote] = (rel, digest)
        if not uploads:
            return
        await self._ensure_root()
        if len(uploads) >= self._tar_min_files and await self._seed_via_tar(
            uploads, pending
        ):
            return
        for resp in await self._backend.aupload_files(uploads):
            meta = pending.get(resp.path)
            if meta is None or resp.error is not None:
                continue  # leave out of the manifest so it retries next turn
            rel, digest = meta
            self._manifest[rel] = digest

    async def _seed_via_tar(
        self,
        uploads: list[tuple[str, bytes]],
        pending: dict[str, tuple[str, str]],
    ) -> bool:
        """Upload all pending files as one gzipped tarball and extract it.

        Many small files cost one exec round-trip each when uploaded
        individually; a single tarball turns that into one upload plus one
        extract. Returns True when every pending file landed (manifest
        updated); False to make the caller fall back to per-file uploads —
        e.g. when the sandbox image lacks ``tar``.
        """
        buf = io.BytesIO()
        # compresslevel=1: the payload is uploaded base64-encoded, so light
        # compression already pays for itself; level 9 would just burn CPU.
        with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
            for remote, data in uploads:
                rel, _digest = pending[remote]
                info = tarfile.TarInfo(name=rel)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        seed_tar = f"{self._remote_root}/{_SEED_TAR_NAME}"
        try:
            responses = await self._backend.aupload_files([(seed_tar, buf.getvalue())])
            if not responses or responses[0].error is not None:
                error = responses[0].error if responses else "no response"
                self._warn_tar_fallback(f"tarball upload failed: {error}")
                return False
            # `;` not `&&`: the tarball must be removed even when extraction
            # fails, or the per-file fallback would leave it for the sync-back.
            result = await self._backend.aexecute(
                f"tar -xzf {shlex.quote(seed_tar)} "
                f"-C {shlex.quote(self._remote_root)}; "
                f"rc=$?; rm -f {shlex.quote(seed_tar)}; exit $rc"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to per-file uploads
            self._warn_tar_fallback(str(exc))
            return False
        if result.exit_code not in (0, None):
            self._warn_tar_fallback(
                f"extract exited {result.exit_code}: {result.output.strip()[:200]}"
            )
            return False
        for rel, digest in pending.values():
            self._manifest[rel] = digest
        return True

    def _warn_tar_fallback(self, detail: str) -> None:
        if self._warned_tar_fallback:
            return
        self._warned_tar_fallback = True
        notify_toast(
            f"Sandbox bulk seed unavailable ({detail}); using per-file uploads",
            severity="warning",
            title="Sandbox",
        )

    async def _sync_back(self) -> None:
        """Download files changed inside the sandbox back to the local dir."""
        # Ensure the root exists first, so a missing directory can't masquerade
        # as an empty listing; then a `cd` failure is a genuine error to raise.
        await self._ensure_root()
        listing = await self._backend.aexecute(
            f"cd {shlex.quote(self._remote_root)} && "
            f"find . -type f ! -name {shlex.quote(_SEED_TAR_NAME)} "
            "-printf '%s ' -exec sha256sum -- {} \\;"
        )
        if listing.exit_code not in (0, None):
            raise RuntimeError(
                f"sandbox listing failed (exit {listing.exit_code}): "
                f"{listing.output.strip()[:200]}"
            )
        downloads: list[str] = []
        meta: dict[str, tuple[str, str]] = {}  # remote path -> (rel, remote hash)
        for line in listing.output.splitlines():
            parsed = self._parse_find_line(line)
            if parsed is None:
                continue
            size, remote_hash, rel = parsed
            if size > self._max_bytes:
                self._warn_oversize(rel)
                continue
            if self._manifest.get(rel) == remote_hash:
                continue
            remote = self._remote_path(rel)
            downloads.append(remote)
            meta[remote] = (rel, remote_hash)
        if not downloads:
            return
        for resp in await self._backend.adownload_files(downloads):
            entry = meta.get(resp.path)
            if entry is None or resp.error is not None or resp.content is None:
                continue
            rel, remote_hash = entry
            dest = self._local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            self._manifest[rel] = remote_hash

    @staticmethod
    def _parse_find_line(line: str) -> tuple[int, str, str] | None:
        """Parse ``"<size> <sha256>  ./<relpath>"`` into ``(size, hash, rel)``."""
        line = line.strip()
        if not line:
            return None
        try:
            size_str, rest = line.split(" ", 1)
            size = int(size_str)
            hash_str, path_part = rest.split("  ", 1)
        except ValueError:
            return None
        path_part = path_part.strip()
        if path_part.startswith("./"):
            path_part = path_part[2:]
        if not path_part:
            return None
        return size, hash_str, path_part
