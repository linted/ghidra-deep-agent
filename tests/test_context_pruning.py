"""Unit tests for Jev context pruning.

The middleware must: pass small histories through untouched; blank only the
tool results Jev judges stale (never the recent, excluded, or small ones);
memoize verdicts so evictions are permanent and keeps are re-asked only when
the goal changes; batch under Jev's state budget; fail open on any Jev error;
leave graph state untouched; record one savings-log document per pass; and
aggregate that log into a report.

Run:  uv run pytest tests/test_context_pruning.py -v
The last test needs a live TypeSafe key: uv run pytest -m integration ...
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ghidra_deep_agent.context_pruning import (
    JevContextPruningMiddleware,
    Judgement,
    PruneLog,
    build_context_pruning_middleware,
    report,
)

BIG = "int x = 0;\n" * 400  # ~1.1k tokens per result under the approximate counter


class FakeJudge:
    """Scripted Jev: ``P(keep)`` per tool name, recording every request."""

    def __init__(self, keep: dict[str, float], *, fail: Exception | None = None):
        self.keep = keep
        self.fail = fail
        self.calls: list[tuple[dict[str, Any], list[str]]] = []

    def _answer(self, state: dict[str, Any], keys: list[str]) -> Judgement:
        self.calls.append((state, keys))
        if self.fail is not None:
            raise self.fail
        probs = {
            key: self.keep.get(state["exchanges"][key]["tool"], 1.0) for key in keys
        }
        return Judgement(probs, input_tokens=100)

    def judge(self, state: dict[str, Any], keys: Any) -> Judgement:
        return self._answer(state, list(keys))

    async def ajudge(self, state: dict[str, Any], keys: Any) -> Judgement:
        return self._answer(state, list(keys))


class FakeCollection:
    """The slice of pymongo the log and the report use."""

    def __init__(self, *, fail: bool = False) -> None:
        self.docs: list[dict[str, Any]] = []
        self.fail = fail

    def insert_one(self, doc: dict[str, Any]) -> None:
        if self.fail:
            raise RuntimeError("mongo down")
        self.docs.append(doc)

    def find(self, query: dict[str, Any]) -> FakeCollection:
        self._rows = [d for d in self.docs if self._match(d, query)]
        return self

    @staticmethod
    def _match(doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, cond in query.items():
            if isinstance(cond, dict) and "$gte" in cond:
                if doc[key] < cond["$gte"]:
                    return False
            elif doc.get(key) != cond:
                return False
        return True

    def sort(self, key: str, direction: int) -> list[dict[str, Any]]:
        return sorted(self._rows, key=lambda d: d[key], reverse=direction < 0)


def _exchange(i: int, tool: str, content: str = BIG) -> list[Any]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": tool, "args": {"addr": hex(i)}, "id": f"call-{i}"}],
        ),
        ToolMessage(content=content, tool_call_id=f"call-{i}", name=tool),
    ]


def _history(tools: list[str]) -> list[Any]:
    msgs: list[Any] = [HumanMessage(content="Analyze the crypto routine at 0x401000")]
    for i, tool in enumerate(tools):
        msgs += _exchange(i, tool)
    msgs.append(AIMessage(content="Looking at the results so far."))
    return msgs


def _request(messages: list[Any]) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]), messages=messages, state=None
    )


def _run(mw: JevContextPruningMiddleware, messages: list[Any]) -> list[Any]:
    """Drive ``awrap_model_call`` and return the messages the model was sent."""
    seen: list[list[Any]] = []

    async def handler(req: ModelRequest) -> ModelResponse:
        seen.append(list(req.messages))
        return ModelResponse(result=[AIMessage(content="ok")])

    asyncio.run(mw.awrap_model_call(_request(messages), handler))
    return seen[0]


def _mw(judge: FakeJudge, **kwargs: Any) -> JevContextPruningMiddleware:
    kwargs.setdefault("trigger_tokens", 1)
    kwargs.setdefault("keep_recent", 1)
    kwargs.setdefault("min_tokens", 100)
    return JevContextPruningMiddleware(judge=judge, **kwargs)


def _is_placeholder(msg: Any) -> bool:
    return isinstance(msg, ToolMessage) and "pruned from context" in msg.text


# --- pruning ------------------------------------------------------------------------


def test_below_trigger_passes_through_untouched() -> None:
    judge = FakeJudge({"get_code": 0.0})
    mw = _mw(judge, trigger_tokens=10**9)
    msgs = _history(["get_code", "get_code"])
    assert _run(mw, msgs) == msgs
    assert judge.calls == []


def test_stale_results_are_blanked_and_identity_kept() -> None:
    judge = FakeJudge({"get_code": 0.05, "xrefs": 0.9})
    mw = _mw(judge)
    msgs = _history(["get_code", "xrefs", "get_code", "task", "get_code"])
    # Index 2 is the get_code result; small results and excluded tools stay.
    msgs[2] = ToolMessage(content="tiny", tool_call_id="call-0", name="get_code")
    sent = _run(mw, msgs)

    assert len(sent) == len(msgs)
    assert sent[2].text == "tiny"  # under min_tokens: never judged
    assert sent[4].text == BIG  # xrefs judged relevant
    assert _is_placeholder(sent[6])  # stale get_code
    assert sent[6].tool_call_id == "call-2" and sent[6].name == "get_code"
    assert "`get_code`" in sent[6].text
    assert sent[8].text == BIG  # `task` excluded by default
    assert sent[10].text == BIG  # most recent exchange protected
    assert [m for m in sent if isinstance(m, AIMessage)] == [
        m for m in msgs if isinstance(m, AIMessage)
    ]
    assert mw.evicted == 1 and mw.judged == 2


def test_sync_hook_matches_async() -> None:
    judge = FakeJudge({"get_code": 0.0})
    mw = _mw(judge)
    msgs = _history(["get_code", "get_code", "xrefs"])
    seen: list[list[Any]] = []

    def handler(req: ModelRequest) -> ModelResponse:
        seen.append(list(req.messages))
        return ModelResponse(result=[AIMessage(content="ok")])

    mw.wrap_model_call(_request(msgs), handler)
    assert _is_placeholder(seen[0][2]) and _is_placeholder(seen[0][4])
    assert seen[0][6].text == BIG


def test_verdicts_are_memoized_and_evictions_permanent() -> None:
    judge = FakeJudge({"get_code": 0.0, "xrefs": 0.9})
    mw = _mw(judge)
    msgs = _history(["get_code", "xrefs", "strings"])
    _run(mw, msgs)
    assert len(judge.calls) == 1

    # Same goal, one more turn: only the exchange that just left the keep
    # window is asked about; the eviction sticks even though Jev would now
    # say "keep".
    judge.keep["get_code"] = 1.0
    msgs2 = msgs + _exchange(9, "strings")
    sent = _run(mw, msgs2)
    assert len(judge.calls) == 2
    state, keys = judge.calls[1]
    assert keys == ["x0"]
    assert state["exchanges"]["x0"]["arguments"] == '{"addr": "0x2"}'
    assert _is_placeholder(sent[2])
    assert sent[4].text == BIG

    # A new human message changes the goal: kept results are re-judged, the
    # evicted one is not.
    judge.keep["xrefs"] = 0.0
    msgs3 = msgs2 + [HumanMessage(content="Now find the key schedule")]
    sent = _run(mw, msgs3)
    assert len(judge.calls) == 3
    state, keys = judge.calls[2]
    asked = sorted(state["exchanges"][k]["arguments"] for k in keys)
    assert asked == ['{"addr": "0x1"}', '{"addr": "0x2"}']
    assert _is_placeholder(sent[2]) and _is_placeholder(sent[4])


def test_state_sent_to_jev_carries_goal_and_excerpts() -> None:
    judge = FakeJudge({})
    mw = _mw(judge, result_excerpt_tokens=50)
    msgs = _history(["get_code", "xrefs"])
    _run(mw, msgs)
    state, keys = judge.calls[0]
    assert state["task"].startswith("Analyze the crypto routine")
    assert state["latest_request"] == state["task"]
    assert state["recent_activity"] == "Looking at the results so far."
    entry = state["exchanges"][keys[0]]
    assert entry["tool"] == "get_code"
    assert "0x0" in entry["arguments"]
    assert "characters omitted" in entry["result"]
    assert len(entry["result"]) < 400


def test_batches_respect_the_state_budget() -> None:
    judge = FakeJudge({})
    mw = _mw(judge, batch_tokens=3_000, batch_questions=3)
    msgs = _history(["get_code"] * 9)
    _run(mw, msgs)
    # 8 candidates at ~1.1k tokens each: the token budget (~2 per batch) binds
    # before the question cap.
    assert len(judge.calls) >= 4
    assert all(len(keys) <= 3 for _, keys in judge.calls)
    assert sum(len(keys) for _, keys in judge.calls) == 8


def test_fails_open_on_jev_error(capsys: pytest.CaptureFixture[str]) -> None:
    from langchain_typesafe.client import TypeSafeAPIConnectionError

    judge = FakeJudge({}, fail=TypeSafeAPIConnectionError("no route"))
    log = FakeCollection()
    mw = _mw(judge, log=PruneLog(log))  # type: ignore[arg-type]
    msgs = _history(["get_code", "xrefs"])
    assert _run(mw, msgs) == msgs
    assert _run(mw, msgs) == msgs
    assert mw.failures == 2
    assert capsys.readouterr().err.count("Jev context pruning failed") == 1
    assert log.docs[0]["error"].startswith("TypeSafeAPIConnectionError")
    assert log.docs[0]["evicted"] == 0


def test_no_human_message_means_no_pruning() -> None:
    judge = FakeJudge({"get_code": 0.0})
    mw = _mw(judge)
    msgs = _exchange(0, "get_code") + _exchange(1, "get_code")
    assert _run(mw, msgs) == msgs
    assert judge.calls == []


# --- graph state ---------------------------------------------------------------------


class _Recorder(FakeListChatModel):
    """Fake model that records what it was sent and accepts tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        seen.append(list(messages))
        return super()._call(messages, *args, **kwargs)


seen: list[list[Any]] = []


def test_graph_state_keeps_raw_history() -> None:
    """Pruning is per request: the checkpoint never sees the placeholder."""
    judge = FakeJudge({"get_code": 0.0})
    mw = _mw(judge, keep_recent=0)
    seen.clear()

    async def run() -> None:
        agent = create_deep_agent(
            model=_Recorder(responses=["done"]),
            middleware=[mw],
            checkpointer=InMemorySaver(),
            backend=StateBackend(),
        )
        config: Any = {"configurable": {"thread_id": "t1"}}
        history = _history(["get_code"])
        await agent.ainvoke({"messages": history}, config=config)
        state = await agent.aget_state(config)
        stored = [m for m in state.values["messages"] if isinstance(m, ToolMessage)]
        assert stored and stored[0].text == BIG
        model_saw = [m for m in seen[-1] if isinstance(m, ToolMessage)]
        assert model_saw and _is_placeholder(model_saw[0])

    asyncio.run(run())


# --- factory --------------------------------------------------------------------------


def test_factory_off_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert build_context_pruning_middleware("u", "d", "s", "b") is None
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("JEV_PRUNE", "0")
    assert build_context_pruning_middleware("u", "d", "s", "b") is None


def test_factory_reads_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.delenv("JEV_PRUNE", raising=False)
    monkeypatch.setenv("JEV_PRUNE_LOG", "0")
    monkeypatch.setenv("JEV_PRUNE_THRESHOLD", "0.35")
    monkeypatch.setenv("JEV_PRUNE_TRIGGER_TOKENS", "12345")
    monkeypatch.setenv("JEV_PRUNE_KEEP_RECENT", "0")
    monkeypatch.setenv("JEV_PRUNE_EXCLUDE_TOOLS", "task, write_file")
    mw = build_context_pruning_middleware("u", "d", "sess", "bin")
    assert mw is not None
    assert mw._threshold == 0.35
    assert mw._trigger == 12345
    assert mw._keep_recent == 0
    assert mw._exclude == {"task", "write_file"}
    assert mw._log is None
    assert mw._session_id == "sess" and mw._binary == "bin"


# --- savings log and report ----------------------------------------------------------


def test_log_document_per_pass() -> None:
    judge = FakeJudge({"get_code": 0.1, "xrefs": 0.8})
    log = FakeCollection()
    mw = _mw(judge, log=PruneLog(log), session_id="s1", binary_name="a.out")  # type: ignore[arg-type]
    msgs = _history(["get_code", "xrefs", "strings"])
    _run(mw, msgs)
    _run(mw, msgs)

    assert len(log.docs) == 2
    first, second = log.docs
    assert first["session_id"] == "s1" and first["binary"] == "a.out"
    assert first["scope"] == "main"
    assert first["tokens_saved"] == first["tokens_before"] - first["tokens_after"]
    assert first["tokens_saved"] > 1000
    assert first["exchanges_total"] == 3
    assert first["candidates"] == 2 and first["judged"] == 2
    assert first["evicted"] == 1
    assert first["jev_requests"] == 1 and first["jev_input_tokens"] == 100
    by_tool = {d["tool"]: d for d in first["decisions"]}
    assert by_tool["get_code"]["evicted"] and by_tool["get_code"]["p_keep"] == 0.1
    assert not by_tool["xrefs"]["evicted"]
    assert not any(d["memoized"] for d in first["decisions"])
    # Second pass: same saving, no Jev traffic, verdicts marked memoized.
    assert second["tokens_saved"] == first["tokens_saved"]
    assert second["jev_requests"] == 0 and second["judged"] == 0
    assert all(d["memoized"] for d in second["decisions"])


def test_log_failure_never_reaches_the_model_call() -> None:
    judge = FakeJudge({"get_code": 0.0})
    mw = _mw(judge, log=PruneLog(FakeCollection(fail=True)))  # type: ignore[arg-type]
    sent = _run(mw, _history(["get_code", "xrefs"]))
    assert _is_placeholder(sent[2])


def _doc(session: str, ts: datetime, **overrides: Any) -> dict[str, Any]:
    doc = {
        "session_id": session,
        "binary": "a.out",
        "ts": ts,
        "scope": "subagent",
        "checkpoint_ns": "x|y",
        "tokens_before": 40_000,
        "tokens_after": 25_000,
        "tokens_saved": 15_000,
        "exchanges_total": 10,
        "candidates": 4,
        "judged": 4,
        "evicted": 3,
        "jev_requests": 1,
        "jev_input_tokens": 20_000,
        "jev_latency_ms": 300,
        "decisions": [
            {"tool": "get_code", "p_keep": 0.05, "evicted": True, "memoized": False},
            {"tool": "get_code", "p_keep": 0.12, "evicted": True, "memoized": False},
            {"tool": "xrefs", "p_keep": 0.15, "evicted": True, "memoized": False},
            {"tool": "xrefs", "p_keep": 0.95, "evicted": False, "memoized": False},
        ],
        "error": None,
    }
    doc.update(overrides)
    return doc


def test_report_aggregates_the_log() -> None:
    now = datetime.now(UTC)
    coll = FakeCollection()
    coll.docs = [
        _doc("s1", now - timedelta(hours=2)),
        _doc(
            "s1",
            now - timedelta(hours=1),
            judged=0,
            jev_requests=0,
            jev_input_tokens=0,
            decisions=[
                {"tool": "get_code", "p_keep": 0.05, "evicted": True, "memoized": True}
            ],
        ),
        _doc(
            "s2",
            now - timedelta(days=3),
            error="TypeSafeAPITimeoutError: 10s",
            judged=0,
            evicted=0,
            jev_requests=0,
            jev_input_tokens=0,
            decisions=[],
            tokens_after=40_000,
            tokens_saved=0,
        ),
    ]
    text = report(coll)  # type: ignore[arg-type]
    assert "passes: 3   sessions: 2   fail-open passes: 1" in text
    assert "verdicts made: 4" in text
    assert "evictions in effect: 6 summed over passes (20% of exchanges seen)" in text
    assert "saved ~30,000" in text
    assert "20,000 input tokens ≈ $0.0008" in text
    assert "s1: 2 passes, 6 evicted, saved ~30,000 of ~80,000 (38%)" in text
    assert "get_code: 2 judged, 2 evicted (100%)" in text  # memoized not re-counted
    assert "xrefs: 2 judged, 1 evicted (50%)" in text
    assert "0.0–0.1" in text and "0.9–1.0" in text

    assert "sessions: 1" in report(coll, session_id="s2")  # type: ignore[arg-type]
    assert "sessions: 1" in report(coll, since=timedelta(days=1))  # type: ignore[arg-type]
    assert "No pruning passes" in report(coll, session_id="nope")  # type: ignore[arg-type]


# --- live ---------------------------------------------------------------------


@pytest.mark.integration
def test_live_jev_separates_stale_from_needed() -> None:
    """Needs TYPESAFE_API_KEY. One tiny history, two obvious verdicts."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY not set")
    from ghidra_deep_agent.context_pruning import JevJudge

    judge = JevJudge()
    state = {
        "task": "Rename the function at 0x401000 based on what it does.",
        "latest_request": "Rename the function at 0x401000 based on what it does.",
        "recent_activity": "The decompilation at 0x401000 shows an AES key "
        "schedule; I will rename it aes_key_expand.",
        "exchanges": {
            "x0": {
                "tool": "get_code",
                "arguments": '{"address": "0x401000"}',
                "result": "void FUN_401000(uint *key, uint *w) { /* rotword, "
                "subword, rcon loop over 44 words */ ... }",
            },
            "x1": {
                "tool": "search_strings",
                "arguments": '{"query": "http"}',
                "result": "No strings matching 'http' were found in the program.",
            },
        },
    }
    result = asyncio.run(judge.ajudge(state, ["x0", "x1"]))
    assert result.keep_probabilities["x0"] > 0.5
    assert result.keep_probabilities["x1"] < 0.5
    assert result.input_tokens
