"""Sync a local directory into an OpenShell sandbox and back, each turn.

A deep-agent sandbox gives the agent its own isolated filesystem, so files it
writes never appear on the host. This middleware bridges that gap: before each
turn it uploads ``AGENT_OUTPUT_DIR`` into the sandbox, and after each turn it
downloads changed files back — making the local directory the durable record
across runs (a fresh sandbox is created per app launch). It follows the
LangChain deep-agents "SandboxSyncMiddleware" guidance.

Change detection uses an in-memory sha256 manifest keyed by path relative to
``AGENT_OUTPUT_DIR``: only new or changed files are transferred in either
direction. Directory symlinks inside ``AGENT_OUTPUT_DIR`` are followed on seed,
so an external directory can be symlinked into the sync root. Files deleted
inside the sandbox are left in place locally. Any sync failure is surfaced as a
warning toast and swallowed — a sandbox hiccup must never kill the turn.

The ``after_agent`` hook only runs when a turn completes. A turn that ends early
— paused on a usage limit, crashed, or cancelled — skips it, so the TUI calls
:meth:`SandboxSyncMiddleware.sync_back` itself on those paths and the CLI calls
it once more at shutdown, before the sandbox is torn down. Likewise a
``/continue`` resumes past the already-checkpointed ``before_agent`` node, so
the TUI calls :meth:`SandboxSyncMiddleware.seed` before resuming; that is what
repopulates a fresh sandbox when a paused session is resumed after a relaunch.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import shlex
import tarfile
import uuid
from collections.abc import Sequence
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
# sandbox's Landlock policy (/tmp is not). Each upload gets its own name so
# concurrent uploads into one directory can't overwrite or delete each other's
# tarball. The sync-back ``find`` excludes the whole pattern (which also covers
# the older fixed name, ``.ghidra_seed.tar.gz``), and the extract command
# removes the tarball even when extraction fails.
_SEED_TAR_PREFIX = ".ghidra_seed"
SEED_TAR_GLOB = f"{_SEED_TAR_PREFIX}*.tar.gz"


def staged_tar_name() -> str:
    """A fresh, unique name for a staged upload tarball."""
    return f"{_SEED_TAR_PREFIX}.{uuid.uuid4().hex}.tar.gz"


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


def build_tar(items: Sequence[tuple[str, bytes]]) -> bytes:
    """Gzip ``(relpath, data)`` pairs into a tarball. Blocking — use ``to_thread``."""
    buf = io.BytesIO()
    # compresslevel=1: the payload is uploaded base64-encoded, so light
    # compression already pays for itself; level 9 would just burn CPU.
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:
        for rel, data in items:
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def upload_via_tar(
    backend: SandboxBackendProtocol,
    remote_dir: str,
    items: Sequence[tuple[str, bytes]],
) -> str | None:
    """Upload ``(relpath, data)`` pairs under ``remote_dir`` as one tarball.

    Many small files cost one exec round-trip each when uploaded individually;
    a single tarball turns that into one upload plus one extract. The tarball
    is staged under a unique ``staged_tar_name()`` inside ``remote_dir`` — the
    sync-back listing skips that pattern at any depth — and removed even when
    extraction fails. Returns ``None`` on success, else a short error for the caller to
    report before falling back to per-file uploads. ``remote_dir`` must exist.
    """
    # gzip is CPU-bound; keep it off the event loop.
    payload = await asyncio.to_thread(build_tar, items)
    staged = f"{remote_dir.rstrip('/')}/{staged_tar_name()}"
    try:
        responses = await backend.aupload_files([(staged, payload)])
        if not responses or responses[0].error is not None:
            error = responses[0].error if responses else "no response"
            return f"tarball upload failed: {error}"
        # `;` not `&&`: the tarball must be removed even when extraction fails,
        # or a per-file fallback would leave it behind.
        result = await backend.aexecute(
            f"tar -xzf {shlex.quote(staged)} -C {shlex.quote(remote_dir)}; "
            f"rc=$?; rm -f {shlex.quote(staged)}; exit $rc"
        )
    except Exception as exc:  # the caller falls back to per-file uploads
        return str(exc)
    if result.exit_code not in (0, None):
        return f"extract exited {result.exit_code}: {result.output.strip()[:200]}"
    return None


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
        # relpath -> (size, mtime_ns) at the time its hash was last computed.
        # Lets the seed skip re-reading files that plainly have not changed.
        self._stats: dict[str, tuple[int, int]] = {}
        # relpaths already warned about for exceeding the size cap (warn once).
        self._warned_oversize: set[str] = set()
        # relpaths already warned about for a failed per-file upload (warn once).
        self._warned_upload_failures: set[str] = set()
        # Whether the remote root has been created this session (created lazily).
        self._root_ready = False
        # Whether we already warned that bulk seeding fell back to per-file.
        self._warned_tar_fallback = False

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        try:
            await self.seed()
        except Exception as exc:  # a sync hiccup must not kill the turn
            notify_toast(
                f"Sandbox seed failed: {exc}", severity="warning", title="Sandbox"
            )
        return None

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        try:
            await self.sync_back()
        except Exception as exc:  # a sync hiccup must not kill the turn
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

    def _scan_local(
        self,
    ) -> tuple[list[tuple[str, bytes]], dict[str, tuple[str, str]], list[str]]:
        """Find local files needing upload. Blocking — call via ``to_thread``.

        Returns ``(uploads, pending, oversize_relpaths)``. A file whose size and
        mtime are unchanged since it was last hashed is skipped without being
        read: hashing every file every turn meant re-reading the whole output
        directory each turn just to learn nothing changed.
        """
        uploads: list[tuple[str, bytes]] = []
        pending: dict[str, tuple[str, str]] = {}  # remote path -> (rel, digest)
        oversize: list[str] = []
        # recurse_symlinks: descend into symlinked directories so an external
        # directory symlinked into the sync root is seeded like a real one. A
        # symlink cycle would hang the walk; the sync root is user-controlled.
        for path in sorted(self._local_dir.rglob("*", recurse_symlinks=True)):
            if not path.is_file():
                continue
            rel = path.relative_to(self._local_dir).as_posix()
            try:
                stat = path.stat()
            except OSError:
                # Vanished mid-walk; it will be picked up next turn if it returns.
                continue
            if stat.st_size > self._max_bytes:
                oversize.append(rel)
                continue
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            if self._stats.get(rel) == fingerprint and rel in self._manifest:
                continue
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            self._stats[rel] = fingerprint
            if self._manifest.get(rel) == digest:
                continue
            remote = self._remote_path(rel)
            uploads.append((remote, data))
            pending[remote] = (rel, digest)
        return uploads, pending, oversize

    async def seed(self) -> None:
        """Upload new/changed local files into the sandbox.

        Raises on a backend failure; the ``before_agent`` hook turns that into a
        toast. Public so the TUI can seed before a ``/continue``, which resumes
        past the checkpointed ``before_agent`` node and so never re-runs it.
        """
        if not self._local_dir.is_dir():
            return
        # Off the event loop: the walk, the reads, and the hashing are all
        # blocking, and this runs before every turn.
        uploads, pending, oversize = await asyncio.to_thread(self._scan_local)
        for rel in oversize:
            self._warn_oversize(rel)
        if not uploads:
            return
        await self._ensure_root()
        if len(uploads) >= self._tar_min_files and await self._seed_via_tar(
            uploads, pending
        ):
            return
        failures: list[tuple[str, str]] = []
        for resp in await self._backend.aupload_files(uploads):
            meta = pending.get(resp.path)
            if meta is None:
                continue
            rel, digest = meta
            if resp.error is not None:
                # Left out of the manifest so it retries next turn.
                failures.append((rel, str(resp.error)))
                continue
            self._manifest[rel] = digest
        self._warn_upload_failures(failures)

    def _warn_upload_failures(self, failures: list[tuple[str, str]]) -> None:
        """Warn once per relpath when its per-file upload fails.

        The file stays out of the manifest and is retried every turn, so
        without a toast a persistently failing upload would be invisible.
        """
        fresh = [
            (rel, err)
            for rel, err in failures
            if rel not in self._warned_upload_failures
        ]
        if not fresh:
            return
        self._warned_upload_failures.update(rel for rel, _ in fresh)
        rel, err = fresh[0]
        more = f" (+{len(fresh) - 1} more)" if len(fresh) > 1 else ""
        notify_toast(
            f"Sandbox seed could not upload {rel}{more}: {err[:200]}",
            severity="warning",
            title="Sandbox",
        )

    async def _seed_via_tar(
        self,
        uploads: list[tuple[str, bytes]],
        pending: dict[str, tuple[str, str]],
    ) -> bool:
        """Upload all pending files as one gzipped tarball and extract it.

        Returns True when every pending file landed (manifest updated); False
        to make the caller fall back to per-file uploads — e.g. when the
        sandbox image lacks ``tar``.
        """
        items = [(pending[remote][0], data) for remote, data in uploads]
        error = await upload_via_tar(self._backend, self._remote_root, items)
        if error is not None:
            self._warn_tar_fallback(error)
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

    async def sync_back(self) -> None:
        """Download files changed inside the sandbox back to the local dir.

        Raises on a backend failure; the ``after_agent`` hook turns that into a
        toast. Public because that hook never runs for a turn that ends early
        (usage-limit pause, error, cancel), so the TUI and the CLI's shutdown
        path call this directly to keep that turn's work from being lost.
        """
        # Ensure the root exists first, so a missing directory can't masquerade
        # as an empty listing; then a `cd` failure is a genuine error to raise.
        await self._ensure_root()
        listing = await self._backend.aexecute(
            f"cd {shlex.quote(self._remote_root)} && "
            f"find . -type f ! -name {shlex.quote(SEED_TAR_GLOB)} "
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
        responses = await self._backend.adownload_files(downloads)
        # Writing the files is blocking; do it in one thread hop, not per file.
        await asyncio.to_thread(self._write_downloads, responses, meta)

    def _write_downloads(
        self,
        responses: list[Any],
        meta: dict[str, tuple[str, str]],
    ) -> None:
        """Write downloaded files locally. Blocking — call via ``to_thread``."""
        for resp in responses:
            entry = meta.get(resp.path)
            if entry is None or resp.error is not None or resp.content is None:
                continue
            rel, remote_hash = entry
            dest = self._local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            self._manifest[rel] = remote_hash
            # Record the stat of what we just wrote so the next seed doesn't
            # re-hash a file it already knows the content of.
            try:
                stat = dest.stat()
            except OSError:
                self._stats.pop(rel, None)
            else:
                self._stats[rel] = (stat.st_size, stat.st_mtime_ns)

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
