"""Agent instances, run lifecycle and event fan-out (stub graphs, memory store)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from server_stubs import (
    BrokenGraph,
    HangingGraph,
    LimitGraph,
    StubGraph,
    ToastingGraph,
    make_registry,
)

from ghidra_deep_agent.program_resolver import ProgramRef
from ghidra_deep_agent.prompt import ASK_MODE_TURN_PREFIX
from ghidra_deep_agent.server.registry import (
    AgentConflict,
    AgentNotReady,
    AgentRegistry,
    EngineBuildFailed,
    ProgramNotOpen,
    RunRecord,
    ThreadBusy,
    UnknownAgent,
    UnknownRun,
)
from ghidra_deep_agent.server.runner import turn_input


def _types(store: Any, run_id: str) -> list[str]:
    return [e["type"] for e in store.events_after(run_id, 0, 1000)]


def test_create_agent_is_idempotent_per_path() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        a, created = await registry.create_agent("/v1/app.exe")
        assert (
            created
            and a.status == "ready"
            and a.program == ProgramRef("app.exe", "/v1/app.exe")
        )
        again, created = await registry.create_agent("app.exe")  # by name
        assert not created and again is a
        assert plan.built == [a.program]
        b, _ = await registry.create_agent("/libs/lib.so")
        assert {x.id for x in registry.list_agents()} == {a.id, b.id}
        assert plan.kwargs["output_dir"] == ""  # no AGENT_OUTPUT_DIR root
        assert callable(plan.kwargs["on_mismatch"])

    asyncio.run(run())


def test_concurrent_creates_build_once() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        results = await asyncio.gather(
            *(registry.create_agent("/v1/app.exe") for _ in range(4))
        )
        assert len({inst.id for inst, _ in results}) == 1
        assert sum(created for _, created in results) == 1
        assert len(plan.built) == 1

    asyncio.run(run())


def test_create_agent_errors() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        with pytest.raises(ProgramNotOpen, match="not open"):
            await registry.create_agent("/nope")
        plan.fail_for.add("/v1/app.exe")
        with pytest.raises(EngineBuildFailed, match="on purpose"):
            await registry.create_agent("/v1/app.exe")
        assert registry.list_agents() == []  # a failed build leaves nothing behind
        with pytest.raises(UnknownAgent):
            registry.get_agent("zzz")

    asyncio.run(run())


def test_same_name_in_two_folders_is_refused() -> None:
    listing = (
        "1. app.exe [ACTIVE]\n   Project Path: /v1/app.exe\n\n"
        "2. app.exe\n   Project Path: /v2/app.exe\n"
    )

    async def run() -> None:
        registry, _, _ = make_registry(programs=listing)
        await registry.create_agent("/v1/app.exe")
        with pytest.raises(AgentConflict, match="keyed by name"):
            await registry.create_agent("/v2/app.exe")

    asyncio.run(run())


def test_output_root_gives_each_agent_a_subdir() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        registry._output_root = "/tmp/out/"
        await registry.create_agent("/libs/lib.so")
        assert plan.kwargs["output_dir"] == "/tmp/out/libs_lib.so"

    asyncio.run(run())


def test_run_lifecycle_done() -> None:
    async def run() -> None:
        registry, plan, store = make_registry()
        graph = StubGraph("the answer")
        plan.by_path["/v1/app.exe"] = graph
        inst, _ = await registry.create_agent("/v1/app.exe")
        run = await registry.start_run(inst.id, prompt="what is main?")
        assert not run.terminal
        assert inst.active_runs == {run.session_id: run.id}
        waited = await registry.wait(run.id, 5)
        assert waited is run and run.status == "done"
        assert run.reply == "the answer"
        assert (run.input_tokens, run.output_tokens) == (10, 5)
        assert run.thread_id == run.session_id
        assert inst.active_runs == {}
        assert graph.inputs == [
            {"messages": [{"role": "user", "content": "what is main?"}]}
        ]
        assert graph.configs[0]["configurable"]["thread_id"] == run.session_id
        assert graph.configs[0]["recursion_limit"] == 7
        assert _types(store, run.id) == [
            "started",
            "tool_start",
            "tool_end",
            "llm_end",
            "usage",
            "context",
            "reply",
            "final",
        ]
        final = store.events_after(run.id, 0, 100)[-1]["data"]
        assert final == {
            "reply": "the answer",
            "input_tokens": 10,
            "output_tokens": 5,
            "status": "done",
        }
        stored = store.load_run(run.id)
        assert stored is not None and stored["status"] == "done"
        assert registry.get_run(run.id) is run
        # Rehydrated from the store once the in-memory record is gone.
        registry._runs.clear()
        loaded = registry.get_run(run.id)
        assert isinstance(loaded, RunRecord) and loaded.reply == "the answer"
        assert loaded.done.is_set()
        with pytest.raises(UnknownRun):
            registry.get_run("nope")

    asyncio.run(run())


def test_ask_mode_uses_ask_graph_on_its_own_thread() -> None:
    async def run() -> None:
        registry, _, _ = make_registry()
        inst, _ = await registry.create_agent("/v1/app.exe")
        run = await registry.start_run(
            inst.id, prompt="why?", session_id="s1", mode="ask"
        )
        await registry.wait(run.id, 5)
        assert run.thread_id == "s1::ask" and run.reply == "ask reply"
        ask_graph = inst.engine.graphs.ask  # type: ignore[union-attr]
        content = ask_graph.inputs[0]["messages"][0]["content"]
        assert content.startswith(ASK_MODE_TURN_PREFIX) and content.endswith("why?")

    asyncio.run(run())


def test_continue_replays_with_no_input() -> None:
    run = RunRecord("r", "a", "s", "s", "normal", None, resume=True)
    assert turn_input(run) is None

    async def go() -> None:
        registry, plan, _ = make_registry()
        graph = StubGraph()
        plan.by_path["/v1/app.exe"] = graph
        inst, _ = await registry.create_agent("/v1/app.exe")
        run = await registry.start_run(
            inst.id, prompt=None, resume=True, session_id="s9"
        )
        await registry.wait(run.id, 5)
        assert graph.inputs == [None] and run.status == "done"

    asyncio.run(go())


def test_thread_busy_but_other_threads_and_agents_run_concurrently() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        plan.by_path["/v1/app.exe"] = StubGraph(delay=0.2)
        plan.by_path["/libs/lib.so"] = StubGraph(delay=0.2)
        a, _ = await registry.create_agent("/v1/app.exe")
        b, _ = await registry.create_agent("/libs/lib.so")
        r1 = await registry.start_run(a.id, prompt="x", session_id="s")
        with pytest.raises(ThreadBusy) as info:
            await registry.start_run(a.id, prompt="y", session_id="s")
        assert info.value.extra == {"active_run_id": r1.id}
        r2 = await registry.start_run(a.id, prompt="y", session_id="other")
        r3 = await registry.start_run(b.id, prompt="z", session_id="s")
        assert registry.active_run_count == 3
        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(registry.wait(r.id, 5) for r in (r1, r2, r3)))
        assert asyncio.get_running_loop().time() - started < 0.5  # ran in parallel
        assert {r.status for r in (r1, r2, r3)} == {"done"}
        r4 = await registry.start_run(a.id, prompt="again", session_id="s")
        await registry.wait(r4.id, 5)
        assert r4.status == "done"

    asyncio.run(run())


def test_paused_error_and_not_ready() -> None:
    async def run() -> None:
        registry, plan, store = make_registry()
        plan.by_path["/v1/app.exe"] = LimitGraph()
        plan.by_path["/libs/lib.so"] = BrokenGraph()
        a, _ = await registry.create_agent("/v1/app.exe")
        b, _ = await registry.create_agent("/libs/lib.so")
        paused = await registry.start_run(a.id, prompt="x")
        failed = await registry.start_run(b.id, prompt="x")
        await registry.wait(paused.id, 5)
        await registry.wait(failed.id, 5)
        assert paused.status == "paused" and "429" in (paused.error or "")
        assert _types(store, paused.id) == ["started", "paused"]
        assert failed.status == "error" and failed.error == "graph exploded"
        assert _types(store, failed.id)[-1] == "error"
        assert a.active_runs == {} and b.active_runs == {}

        # A degraded instance refuses runs until re-created.
        plan.kwargs["on_mismatch"]("rename_symbol", "other.bin")
        assert b.status == "degraded" and "other.bin" in (b.error or "")
        with pytest.raises(AgentNotReady, match="degraded"):
            await registry.start_run(b.id, prompt="x")

    asyncio.run(run())


def test_cancel_frees_the_thread_and_marks_cancelled() -> None:
    async def run() -> None:
        registry, plan, store = make_registry()
        plan.by_path["/v1/app.exe"] = HangingGraph()
        a, _ = await registry.create_agent("/v1/app.exe")
        r = await registry.start_run(a.id, prompt="x", session_id="s")
        await asyncio.sleep(0.05)
        assert await registry.cancel_run(r.id) is True
        assert r.status == "cancelled" and a.active_runs == {}
        assert _types(store, r.id) == ["started", "llm_start", "cancelled"]
        assert await registry.cancel_run(r.id) is False  # already terminal
        again = await registry.start_run(a.id, prompt=None, resume=True, session_id="s")
        assert again.thread_id == "s"
        await registry.cancel_run(again.id)

    asyncio.run(run())


def test_subscribers_get_live_events_and_a_terminal_sentinel() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        plan.by_path["/v1/app.exe"] = StubGraph(delay=0.05)
        a, _ = await registry.create_agent("/v1/app.exe")
        r = await registry.start_run(a.id, prompt="x")
        queue = registry.subscribe(r)
        seen = []
        while True:
            doc = await asyncio.wait_for(queue.get(), 5)
            if doc is None:
                break
            seen.append(doc["type"])
        assert seen[-1] == "final" and "reply" in seen
        registry.unsubscribe(r, queue)
        # Subscribing to a finished run yields the sentinel immediately.
        late = registry.subscribe(r)
        assert late.get_nowait() is None

    asyncio.run(run())


def test_toasts_inside_a_run_become_warning_events() -> None:
    async def run() -> None:
        registry, plan, store = make_registry()
        plan.by_path["/v1/app.exe"] = ToastingGraph()
        a, _ = await registry.create_agent("/v1/app.exe")
        r = await registry.start_run(a.id, prompt="x")
        await registry.wait(r.id, 5)
        warnings = [
            e for e in store.events_after(r.id, 0, 100) if e["type"] == "warning"
        ]
        assert warnings and warnings[0]["data"] == {
            "message": "credits low",
            "title": "Provider",
            "severity": "warning",
        }

    asyncio.run(run())


def test_session_store_is_recorded_and_touched() -> None:
    class FakeSessions:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        async def arecord_start(self, session_id: str, binary_name: str) -> None:
            self.calls.append(("start", session_id, binary_name))

        async def atouch(
            self, session_id: str, first_prompt: str | None = None
        ) -> None:
            self.calls.append(("touch", session_id, first_prompt))

    async def run() -> None:
        sessions = FakeSessions()
        registry, _, _ = make_registry(session_store=sessions)
        a, _ = await registry.create_agent("/v1/app.exe")
        r1 = await registry.start_run(a.id, prompt="first")
        await registry.wait(r1.id, 5)
        r2 = await registry.start_run(a.id, prompt="second", session_id=r1.session_id)
        await registry.wait(r2.id, 5)
        assert sessions.calls == [
            ("start", r1.session_id, "app.exe"),
            ("touch", r1.session_id, "first"),
            ("touch", r1.session_id, "second"),
        ]

    asyncio.run(run())


def test_delete_agent_cancels_runs_and_closes_engine() -> None:
    async def run() -> None:
        registry, plan, _ = make_registry()
        plan.by_path["/v1/app.exe"] = HangingGraph()
        a, _ = await registry.create_agent("/v1/app.exe")
        r = await registry.start_run(a.id, prompt="x")
        await asyncio.sleep(0.02)
        await registry.delete_agent(a.id)
        assert r.status == "cancelled"
        assert plan.closed == ["/v1/app.exe"]
        assert registry.list_agents() == []
        with pytest.raises(UnknownAgent):
            await registry.delete_agent(a.id)

    asyncio.run(run())


def test_startup_marks_interrupted_and_shutdown_cancels() -> None:
    async def run() -> None:
        registry, plan, store = make_registry()
        store.upsert_run(
            {
                "id": "old",
                "agent_id": "x",
                "session_id": "s",
                "status": "running",
                "created_at": "1",
            }
        )
        await registry.startup()
        assert store.load_run("old")["status"] == "error"  # type: ignore[index]
        plan.by_path["/v1/app.exe"] = HangingGraph()
        a, _ = await registry.create_agent("/v1/app.exe")
        r = await registry.start_run(a.id, prompt="x")
        await asyncio.sleep(0.02)
        await registry.shutdown(timeout=2)
        assert r.status == "cancelled" and registry.list_agents() == []
        assert plan.closed == ["/v1/app.exe"]

    asyncio.run(run())


def test_registry_is_plain_async() -> None:
    """No HTTP concepts leak into the registry (a future MCP wrapper relies on it)."""
    import ghidra_deep_agent.server.registry as mod

    assert "fastapi" not in mod.__dict__ and not hasattr(AgentRegistry, "app")
