"""OpenShell sandbox lifecycle for the deep agent.

Deep-agent sandboxes are *backends*: when the backend implements
``SandboxBackendProtocol``, deepagents automatically adds an ``execute`` shell
tool (plus the standard filesystem tools) that run inside the sandbox instead of
on the host. This module creates an NVIDIA OpenShell sandbox, wraps it in the
``langchain-nvidia-openshell`` backend, and tears it down when the app exits.

Sandboxing is opt-in via the ``SANDBOX`` env var (unset/empty = disabled). The
gateway is resolved from the active OpenShell cluster config
(``~/.config/openshell/``); the workspace comes from ``OPENSHELL_WORKSPACE``
(default ``"default"``).
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

SANDBOX_ENV = "SANDBOX"
OPENSHELL_MODE = "openshell"
SUPPORTED_MODES = (OPENSHELL_MODE,)

# The agent's working directory inside the sandbox, and the directory synced to
# AGENT_OUTPUT_DIR. It is NOT the sandbox home (/sandbox), which holds the
# language toolchain — see SandboxSyncMiddleware. Commands and relative-path
# writes default here so the agent's output is captured; the sync middleware
# mirrors this exact path, so the two must stay equal.
SANDBOX_WORKDIR = "/sandbox/output"


class OpenShellSandboxError(RuntimeError):
    """A fresh OpenShell sandbox could not be created (fatal at startup)."""


def sandbox_mode() -> str:
    """Return the configured sandbox backend name; ``""`` means disabled."""
    return os.environ.get(SANDBOX_ENV, "").strip().lower()


@asynccontextmanager
async def open_sandbox_backend() -> AsyncIterator[SandboxBackendProtocol]:
    """Create a fresh OpenShell sandbox and yield a deepagents backend for it.

    The sandbox is created on entry (a blocking gRPC call, run in a worker
    thread) and destroyed on exit — including when the caller raises — so no
    sandbox is leaked on a crash. A failure to create the sandbox is raised as
    :class:`OpenShellSandboxError` for the caller to turn into a fatal startup
    error. Teardown failures are reported as warnings and never mask the
    original error.
    """
    # Imported lazily so the sandbox packages are only required when the
    # feature is actually enabled. A missing package is turned into the same
    # fatal error as any other creation failure, per this factory's contract.
    try:
        import openshell
        from langchain_nvidia_openshell import OpenShellSandbox
    except ImportError as exc:
        raise OpenShellSandboxError(
            f"OpenShell sandbox packages are not installed ({exc}); "
            "install with `uv sync` or unset SANDBOX."
        ) from exc

    class _RootedOpenShellSandbox(OpenShellSandbox):  # type: ignore[misc]
        """OpenShellSandbox that runs every command in ``SANDBOX_WORKDIR``.

        The base adapter runs commands in the sandbox home (``/sandbox``), which
        also holds the language toolchain — so shell commands and relative-path
        file writes would land there and be lost on teardown. Prepending a ``cd``
        into the synced working directory makes the agent's natural (relative)
        output persist. Absolute paths are unaffected (they can still escape).
        """

        def execute(
            self, command: str, *, timeout: int | None = None
        ) -> ExecuteResponse:
            return cast(
                ExecuteResponse, super().execute(_rooted(command), timeout=timeout)
            )

        async def aexecute(
            self, command: str, *, timeout: int | None = None
        ) -> ExecuteResponse:
            return cast(
                ExecuteResponse,
                await super().aexecute(_rooted(command), timeout=timeout),
            )

    def _rooted(command: str) -> str:
        # `|| true` keeps a missing dir from aborting the command; the dir is
        # created below at startup and by the sync middleware, so cd normally
        # succeeds.
        return f"cd {shlex.quote(SANDBOX_WORKDIR)} 2>/dev/null || true\n{command}"

    workspace = os.environ.get("OPENSHELL_WORKSPACE", "default")
    sandbox_cm = openshell.Sandbox(workspace=workspace)
    try:
        sandbox = await asyncio.to_thread(sandbox_cm.__enter__)
    except Exception as exc:  # noqa: BLE001 - surfaced as a fatal startup error
        raise OpenShellSandboxError(str(exc)) from exc

    try:
        sandbox_id = sandbox.id
    except Exception:  # noqa: BLE001 - id is best-effort for the status line
        sandbox_id = "unknown"
    print(f"Sandbox: openshell [id={sandbox_id}]")

    # Create the working/sync directory up front so the very first command and
    # the sync middleware's probe both find it.
    def _mkdir_workdir() -> Any:
        return sandbox.exec(["mkdir", "-p", SANDBOX_WORKDIR])

    try:
        await asyncio.to_thread(_mkdir_workdir)
    except Exception as exc:  # noqa: BLE001 - non-fatal; sync middleware retries
        print(
            f"Warning: could not pre-create {SANDBOX_WORKDIR}: {exc}",
            file=sys.stderr,
        )

    # 2 MiB upload chunks (default 512 KiB) cut large-upload round trips 4x
    # while the base64-inflated payload (~1.33x -> ~2.8 MB) stays under
    # gRPC's 4 MB message cap.
    backend = _RootedOpenShellSandbox(
        sandbox=sandbox, max_upload_chunk_bytes=2 * 1024 * 1024
    )
    try:
        yield backend
    finally:
        try:
            await asyncio.to_thread(sandbox_cm.__exit__, None, None, None)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask a crash
            print(
                f"Warning: OpenShell sandbox teardown failed: {exc}",
                file=sys.stderr,
            )
