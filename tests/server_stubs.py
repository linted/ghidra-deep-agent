"""Stub graphs, engines and runtimes for the server tests (no Ghidra, no Mongo)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

from ghidra_deep_agent.program_resolver import ProgramRef
from ghidra_deep_agent.resilience import UsageLimitError
from ghidra_deep_agent.runtime import Engine, Graphs, SharedRuntime, Storage
from ghidra_deep_agent.server.events import MemoryRunStore
from ghidra_deep_agent.server.registry import AgentRegistry
from ghidra_deep_agent.toasts import notify_toast

TWO_OPEN = (
    "[Context] Operating on: app.exe | Active window: app.exe\n\n"
    "Open Programs in Ghidra:\n\n"
    "1. app.exe [ACTIVE]\n   Project Path: /v1/app.exe\n"
    "   Format: PE\n   Language: x86:LE:64:default\n\n"
    "2. lib.so\n   Project Path: /libs/lib.so\n"
    "   Format: ELF\n   Language: ARM:LE:32:v8\n\n"
    "---\nTotal: 2 program(s) open\n"
)


class FakeTool:
    def __init__(self, name: str, reply: Any) -> None:
        self.name = name
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _Output:
    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        self.content = content
        self.usage_metadata = usage or {"input_tokens": 10, "output_tokens": 5}


class _State:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class StubGraph:
    """Streams one tool call and one reply; records what it was invoked with."""

    def __init__(self, reply: str = "hello from stub", *, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.inputs: list[Any] = []
        self.configs: list[Any] = []
        self.state_configs: list[Any] = []
        self.state_values: dict[str, Any] = {}

    async def astream_events(
        self, input_data: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.inputs.append(input_data)
        self.configs.append(config)
        yield {
            "event": "on_tool_start",
            "run_id": "t1",
            "name": "get_code",
            "metadata": {},
            "parent_ids": [],
            "data": {"input": {"address": "0x1000"}},
        }
        if self.delay:
            await asyncio.sleep(self.delay)
        yield {
            "event": "on_tool_end",
            "run_id": "t1",
            "metadata": {},
            "data": {"output": "mov x0, x1"},
        }
        yield {
            "event": "on_chat_model_stream",
            "run_id": "m1",
            "metadata": {},
            "data": {"chunk": _Output("hel")},
        }
        yield {
            "event": "on_chat_model_end",
            "run_id": "m1",
            "metadata": {},
            "data": {"output": _Output(self.reply)},
        }

    async def aget_state(self, config: Any) -> _State:
        self.state_configs.append(config)
        return _State(self.state_values)


class LimitGraph(StubGraph):
    async def astream_events(
        self, input_data: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.inputs.append(input_data)
        raise UsageLimitError(RuntimeError("429 rate limit"))
        yield  # pragma: no cover


class BrokenGraph(StubGraph):
    async def astream_events(
        self, input_data: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        raise RuntimeError("graph exploded")
        yield  # pragma: no cover


class HangingGraph(StubGraph):
    """Yields one event then blocks until cancelled."""

    async def astream_events(
        self, input_data: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "run_id": "m1", "metadata": {}}
        await asyncio.sleep(3600)


class ToastingGraph(StubGraph):
    async def astream_events(
        self, input_data: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        notify_toast("credits low", severity="warning", title="Provider")
        async for event in super().astream_events(input_data, config, version):
            yield event


def stub_engine(program: ProgramRef, graph: Any | None = None, **kw: Any) -> Engine:
    main = graph if graph is not None else StubGraph()
    return Engine(
        program=program,
        graphs=Graphs(main=main, plan=None, ask=kw.get("ask", StubGraph("ask reply"))),
        storage=Storage(None, None, kw.get("guidance", "")),
        compaction_engine=None,
        knowledge_ok=True,
        tool_count=3,
        output_dir=kw.get("output_dir"),
        _stack=contextlib.AsyncExitStack(),
    )


def stub_shared(
    *, programs: str = TWO_OPEN, session_store: Any | None = None
) -> SharedRuntime:
    return SharedRuntime(
        mcp_config={},
        agent_config=cast(Any, None),
        resolve_model=cast(Any, None),
        agents_md="",
        mongo=("mongodb://x", "db", "e"),  # type: ignore[arg-type]
        checkpointer=None,
        session_store=session_store,
        built_model=None,
        main_model_spec="stub:model",
        summary_override=None,
        summary_model=None,  # type: ignore[arg-type]
        recursion_limit=7,
        app_name="app",
        max_context_tokens=1000,
        probe_tools=[
            FakeTool("list_binaries", programs),
            FakeTool("open_program", "Opened 'x' (/x) in CodeBrowser."),
        ],
        _stack=contextlib.ExitStack(),
    )


class GraphPlan:
    """Which stub graph each created engine gets, by project path."""

    def __init__(self) -> None:
        self.by_path: dict[str, Any] = {}
        self.built: list[ProgramRef] = []
        self.closed: list[str] = []
        self.fail_for: set[str] = set()
        self.kwargs: dict[str, Any] = {}

    async def factory(self, shared: Any, program: ProgramRef, **kw: Any) -> Engine:
        self.built.append(program)
        self.kwargs = kw
        if program.project_path in self.fail_for:
            raise RuntimeError("engine build failed on purpose")
        engine = stub_engine(program, self.by_path.get(program.project_path))
        engine._stack.callback(lambda: self.closed.append(program.project_path))
        return engine


def make_registry(
    plan: GraphPlan | None = None, **shared_kw: Any
) -> tuple[AgentRegistry, GraphPlan, MemoryRunStore]:
    plan = plan or GraphPlan()
    store = MemoryRunStore()
    registry = AgentRegistry(
        stub_shared(**shared_kw), store, engine_factory=plan.factory
    )
    return registry, plan, store
