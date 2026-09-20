"""Event serialization and the in-memory run store."""

from __future__ import annotations

import json

import pytest

from ghidra_deep_agent import stream
from ghidra_deep_agent.server import events
from ghidra_deep_agent.server.events import (
    TERMINAL_TYPES,
    WIRE_TYPE,
    AnyEvent,
    Cancelled,
    Failed,
    Final,
    MemoryRunStore,
    Paused,
    RunWarning,
    Started,
    event_type,
    serialize,
    sse_frame,
)


def test_every_event_serializes_with_its_wire_name() -> None:
    samples: list[AnyEvent] = [
        stream.ToolStart("c", "get_code", "0x1", False, ""),
        stream.ToolEnd("c"),
        stream.LLMStart("m", ""),
        stream.LLMEnd("m"),
        stream.SubagentReport("s", "desc", "text", False, 1.5),
        stream.Reply("hi"),
        stream.Usage(1, 2),
        stream.Context(3),
        stream.Compaction("start"),
        stream.Truncated(),
        stream.Token("t"),
        Started("a", "s", "t", "normal", False),
        RunWarning("m", "t", "warning"),
        Final("reply", 1, 2),
        Paused("limit"),
        Failed("boom"),
        Cancelled(),
    ]
    assert {type(s) for s in samples} == set(WIRE_TYPE)
    for i, sample in enumerate(samples, start=1):
        doc = serialize("run", i, sample)
        assert doc["run_id"] == "run" and doc["seq"] == i
        assert doc["type"] == event_type(sample)
        json.dumps(doc)  # JSON-clean
    assert serialize("r", 1, Final("x", 1, 2))["data"] == {
        "reply": "x",
        "input_tokens": 1,
        "output_tokens": 2,
        "status": "done",
    }
    assert serialize("r", 1, Failed("boom"))["type"] == "error"
    assert TERMINAL_TYPES == {"final", "paused", "error", "cancelled"}


def test_event_type_rejects_non_events() -> None:
    with pytest.raises(TypeError):
        event_type("nope")  # type: ignore[arg-type]


def test_sse_frame_uses_seq_as_id() -> None:
    doc = serialize("r", 7, stream.Reply("héllo"))
    frame = sse_frame(doc)
    assert frame == {"id": "7", "event": "reply", "data": '{"text": "héllo"}'}


def test_memory_store_cursor_and_runs() -> None:
    store = MemoryRunStore()
    for seq in range(1, 6):
        store.append_event(serialize("r1", seq, stream.Reply(str(seq))))
    store.append_event(serialize("r2", 1, stream.Reply("other")))
    assert [e["seq"] for e in store.events_after("r1", 0, 500)] == [1, 2, 3, 4, 5]
    assert [e["seq"] for e in store.events_after("r1", 3, 500)] == [4, 5]
    assert [e["seq"] for e in store.events_after("r1", 0, 2)] == [1, 2]
    assert store.events_after("nope", 0, 10) == []

    store.upsert_run(
        {
            "id": "r1",
            "agent_id": "a",
            "session_id": "s",
            "status": "running",
            "created_at": "2",
        }
    )
    store.upsert_run(
        {
            "id": "r2",
            "agent_id": "b",
            "session_id": "s",
            "status": "done",
            "created_at": "3",
        }
    )
    assert store.load_run("r1")["status"] == "running"  # type: ignore[index]
    assert store.load_run("zz") is None
    assert [r["id"] for r in store.list_runs()] == ["r2", "r1"]
    assert [r["id"] for r in store.list_runs(agent_id="a")] == ["r1"]
    assert [r["id"] for r in store.list_runs(session_id="s", limit=1)] == ["r2"]
    assert store.mark_interrupted("restart") == 1
    assert store.load_run("r1")["error"] == "restart"  # type: ignore[index]


def test_build_run_store_falls_back_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(uri: str) -> None:
        raise ConnectionError("no mongo")

    monkeypatch.setattr(events, "get_mongo_client", boom)
    assert isinstance(events.build_run_store("mongodb://x", "db"), MemoryRunStore)
