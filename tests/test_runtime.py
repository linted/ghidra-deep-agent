"""The shared/per-program runtime split (pure parts; no Ghidra or Mongo)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend

from ghidra_deep_agent import runtime
from ghidra_deep_agent.program_resolver import ProgramRef
from ghidra_deep_agent.runtime import (
    Engine,
    Graphs,
    SharedRuntime,
    StartupError,
    open_storage,
    storage_config,
    validate_sandbox_mode,
)
from ghidra_deep_agent.sandbox import OpenShellSandboxError


def test_validate_sandbox_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SANDBOX", raising=False)
    validate_sandbox_mode()
    monkeypatch.setenv("SANDBOX", "openshell")
    validate_sandbox_mode()
    monkeypatch.setenv("SANDBOX", "docker")
    with pytest.raises(StartupError, match="unsupported SANDBOX='docker'"):
        validate_sandbox_mode()


def test_storage_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGODB_URI", "mongodb://x:1")
    monkeypatch.setenv("MONGODB_DB", "db")
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "legacy")
    assert storage_config() == ("mongodb://x:1", "db", "ollama:legacy")
    monkeypatch.setenv("EMBED_MODEL", "openai:text")
    assert storage_config().embed_model == "openai:text"


def test_open_storage_without_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("SANDBOX", raising=False)

    async def run() -> None:
        async with contextlib.AsyncExitStack() as stack:
            state = await open_storage(stack, output_dir="")
            assert isinstance(state.backend, StateBackend)
            assert state.sync_middleware is None and state.prompt_guidance == ""
            fs = await open_storage(stack, output_dir=str(tmp_path))
            assert isinstance(fs.backend, FilesystemBackend)

    asyncio.run(run())


def test_open_storage_enters_one_sandbox_per_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Each engine gets its own sandbox, torn down with its own stack."""
    monkeypatch.setenv("SANDBOX", "openshell")
    events: list[str] = []

    class FakeBackend:
        async def aexecute(self, cmd: str) -> Any:
            return None

    @contextlib.asynccontextmanager
    async def fake_sandbox() -> AsyncIterator[FakeBackend]:
        events.append("open")
        try:
            yield FakeBackend()
        finally:
            events.append("close")

    monkeypatch.setattr(runtime, "open_sandbox_backend", fake_sandbox)

    async def run() -> None:
        stack_a = contextlib.AsyncExitStack()
        stack_b = contextlib.AsyncExitStack()
        a = await open_storage(stack_a, output_dir=str(tmp_path / "a"))
        b = await open_storage(stack_b, output_dir="")
        assert a.sync_middleware is not None and b.sync_middleware is None
        assert "shell" in a.prompt_guidance.lower() or a.prompt_guidance
        assert events == ["open", "open"]
        await stack_a.aclose()
        assert events == ["open", "open", "close"]
        await stack_b.aclose()
        assert events == ["open", "open", "close", "close"]

    asyncio.run(run())


def test_open_storage_sandbox_failure_is_a_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX", "openshell")

    @contextlib.asynccontextmanager
    async def broken() -> AsyncIterator[Any]:
        raise OpenShellSandboxError("gateway down")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "open_sandbox_backend", broken)

    async def run() -> None:
        async with contextlib.AsyncExitStack() as stack:
            with pytest.raises(StartupError, match="gateway down"):
                await open_storage(stack, output_dir="")

    asyncio.run(run())


def _shared(**overrides: Any) -> SharedRuntime:
    fields: dict[str, Any] = dict(
        mcp_config={},
        agent_config=None,
        resolve_model=None,
        agents_md="",
        mongo=("mongodb://x", "db", "e"),
        checkpointer=None,
        session_store=None,
        built_model=None,
        main_model_spec="m",
        summary_override=None,
        summary_model=None,
        recursion_limit=42,
        app_name="app",
        max_context_tokens=1000,
        probe_tools=[],
        _stack=contextlib.ExitStack(),
    )
    fields.update(overrides)
    return SharedRuntime(**fields)


def test_config_for_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []
    stack = contextlib.ExitStack()
    stack.callback(lambda: closed.append("checkpointer"))
    monkeypatch.setattr(runtime, "close_mongo_clients", lambda: closed.append("mongo"))
    shared = _shared(_stack=stack)
    assert shared.config_for("t1") == {
        "configurable": {"thread_id": "t1"},
        "recursion_limit": 42,
    }
    shared.close()
    assert closed == ["checkpointer", "mongo"]


def test_engine_aclose_closes_its_stack() -> None:
    closed: list[bool] = []
    stack = contextlib.AsyncExitStack()
    stack.callback(lambda: closed.append(True))
    engine = Engine(
        program=ProgramRef("a", "/a"),
        graphs=Graphs(None, None, None),
        storage=runtime.Storage(None, None, ""),
        compaction_engine=None,
        knowledge_ok=True,
        tool_count=0,
        output_dir=None,
        _stack=stack,
    )
    asyncio.run(engine.aclose())
    assert closed == [True]


def test_build_engine_fails_closed_on_pin_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad pin is a StartupError, and nothing entered on the stack leaks."""

    class Probe:
        name = "get_binary_info"

        async def ainvoke(self, args: dict[str, Any]) -> str:
            return "[Context] Operating on: other.bin"

    async def fake_connect(mcp_config: Any, *, interceptors: Any) -> list[Any]:
        return [Probe()]

    monkeypatch.setattr(runtime, "connect_mcp", fake_connect)

    async def run() -> None:
        with pytest.raises(StartupError, match="cannot pin to /a.bin"):
            await runtime.build_engine(
                _shared(), ProgramRef("a.bin", "/a.bin"), output_dir=""
            )

    asyncio.run(run())


def test_runtime_modules_do_not_import_textual() -> None:
    """The server must be able to run without loading the TUI."""
    import subprocess
    import sys

    code = (
        "import sys, ghidra_deep_agent.runtime, ghidra_deep_agent.stream, "
        "ghidra_deep_agent.pinning, ghidra_deep_agent.program_resolver, "
        "ghidra_deep_agent.toasts; "
        "sys.exit(1 if 'textual' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
