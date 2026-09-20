"""The TUI-free event translation shared by the TUI and the HTTP server."""

from __future__ import annotations

from typing import Any

from ghidra_deep_agent import stream
from ghidra_deep_agent.async_tasks import ASYNC_DONE_EVENT
from ghidra_deep_agent.stream import (
    EVENT_TYPE,
    Compaction,
    Context,
    LLMEnd,
    LLMStart,
    Reply,
    RunState,
    StreamEvent,
    SubagentReport,
    Token,
    ToolEnd,
    ToolStart,
    Truncated,
    Usage,
    parse_checkpoint_ns,
    translate,
)


class _Output:
    def __init__(
        self,
        content: Any = "",
        usage: dict[str, int] | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage
        self.response_metadata = {"stop_reason": stop_reason} if stop_reason else {}


class _ToolMessage:
    def __init__(self, text: str) -> None:
        self.content = text


class _Command:
    """Shape of a `task` tool's output: Command(update={"messages": [...]})."""

    def __init__(self, text: str) -> None:
        self.update = {"messages": [_ToolMessage(text)]}


def _tool_start(
    run_id: str,
    name: str,
    *,
    parents: list[str] | None = None,
    ns: str = "",
    inp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event": "on_tool_start",
        "run_id": run_id,
        "name": name,
        "metadata": {"langgraph_checkpoint_ns": ns},
        "parent_ids": parents or [],
        "data": {"input": inp if inp is not None else {"x": 1}},
    }


def _tool_end(run_id: str, output: Any, *, error: Any = None) -> dict[str, Any]:
    data: dict[str, Any] = {"output": output}
    if error is not None:
        data["error"] = error
    return {"event": "on_tool_end", "run_id": run_id, "metadata": {}, "data": data}


def _model_end(run_id: str, output: _Output, *, ns: str = "") -> dict[str, Any]:
    return {
        "event": "on_chat_model_end",
        "run_id": run_id,
        "metadata": {"langgraph_checkpoint_ns": ns},
        "data": {"output": output},
    }


def _drain(events: list[dict[str, Any]], run: RunState) -> list[StreamEvent]:
    out: list[StreamEvent] = []
    for event in events:
        out.extend(translate(event, run))
    return out


def test_parse_checkpoint_ns() -> None:
    assert parse_checkpoint_ns("") == ()
    assert parse_checkpoint_ns("tools:a") == ("tools:a",)
    assert parse_checkpoint_ns("tools:a|tools:b") == ("tools:a", "tools:b")


def test_every_event_type_has_a_wire_name() -> None:
    assert set(EVENT_TYPE) == set(StreamEvent.__args__)
    assert len(set(EVENT_TYPE.values())) == len(EVENT_TYPE)


def test_get_task_status_polls_are_hidden_and_balanced() -> None:
    run = RunState()
    out = _drain(
        [_tool_start("p1", "get_task_status"), _tool_end("p1", "still running")],
        run,
    )
    assert out == []
    assert run.hidden_tool_runs == set()
    assert run.active_tool_runs == set()


def test_nested_tool_calls_are_hidden() -> None:
    """A tool invoked from inside another tool's body (recover_prototypes →
    scripts) is suppressed entirely — no event, no deferred-async leak."""
    run = RunState()
    assert translate(_tool_start("outer", "recover_prototypes"), run) == [
        ToolStart("outer", "recover_prototypes", "{'x': 1}", False, "")
    ]
    assert (
        translate(_tool_start("inner", "scripts", parents=["chain", "outer"]), run)
        == []
    )
    assert "inner" in run.hidden_tool_runs
    # The hidden run ends with an async submission stub; it must not register
    # a deferred completion.
    assert translate(_tool_end("inner", "Script task submitted: abc123"), run) == []
    assert run.pending_async == {}
    assert run.hidden_tool_runs == set()
    assert translate(_tool_end("outer", "done"), run) == [ToolEnd("outer")]
    assert run.active_tool_runs == set()


def test_subagent_inner_tools_stay_visible() -> None:
    """Tool calls made by a sub-agent have the `task` run in their ancestry
    but must not be hidden — they are the sub-agent's real work."""
    run = RunState()
    translate(
        _tool_start("task1", "task", inp={"description": "dig"}, ns="tools:a"), run
    )
    out = translate(
        _tool_start("inner", "get_code", parents=["task1"], ns="tools:a|tools:b"),
        run,
    )
    assert out == [ToolStart("inner", "get_code", "{'x': 1}", False, "tools:a|tools:b")]


def test_async_stub_defers_completion_until_done_event() -> None:
    run = RunState()
    translate(_tool_start("t1", "get_code"), run)
    assert (
        translate(
            _tool_end("t1", "Task submitted for async execution. Task ID: deadbeef"),
            run,
        )
        == []
    )
    assert run.pending_async == {"deadbeef": "t1"}
    done = {
        "event": "on_custom_event",
        "name": ASYNC_DONE_EVENT,
        "run_id": "mw",
        "metadata": {},
        "data": {"task_id": "deadbeef"},
    }
    assert translate(done, run) == [ToolEnd("t1")]
    assert run.pending_async == {}
    # An unknown task id is ignored.
    assert translate(done, run) == []


def test_tool_error_carries_a_snippet() -> None:
    run = RunState()
    translate(_tool_start("t1", "get_code"), run)
    out = translate(_tool_end("t1", "Tool failed: nope", error=ValueError()), run)
    assert out == [ToolEnd("t1", True, "Tool failed: nope")]


def test_subagent_report_precedes_its_completion() -> None:
    run = RunState()
    description = "investigate the parser " * 5  # longer than the preview
    translate(_tool_start("task1", "task", inp={"description": description}), run)
    out = translate(_tool_end("task1", _Command("## Findings\n- entry")), run)
    assert len(out) == 2
    report, end = out
    assert isinstance(report, SubagentReport)
    assert report.call_id == "task1"
    assert report.description == description  # untruncated
    assert report.text == "## Findings\n- entry"
    assert report.error is False
    assert report.elapsed >= 0
    assert end == ToolEnd("task1")
    assert run.subagent_meta == {}


def test_subagent_report_skips_async_stub_detection() -> None:
    """A report that merely quotes an async submission stub must not defer the
    task — `task` is a local tool and never completes asynchronously."""
    run = RunState()
    translate(_tool_start("task1", "task", inp={"description": "export"}), run)
    text = "Task submitted for async execution. Task ID: deadbeef"
    out = translate(_tool_end("task1", _Command(text)), run)
    assert run.pending_async == {}
    assert isinstance(out[0], SubagentReport) and out[0].text == text
    assert out[1] == ToolEnd("task1")


def test_subagent_report_error_and_empty() -> None:
    run = RunState()
    translate(_tool_start("bad", "task", inp={"description": "doomed"}), run)
    errored, end = translate(_tool_end("bad", None, error=ValueError("boom")), run)
    assert isinstance(errored, SubagentReport) and errored.error is True
    assert end == ToolEnd("bad", True, "")
    translate(_tool_start("quiet", "task", inp={"description": "silent"}), run)
    empty, _ = translate(_tool_end("quiet", _Command("")), run)
    assert isinstance(empty, SubagentReport)
    assert empty.error is False and empty.text == ""


def test_top_level_model_end_is_the_reply() -> None:
    run = RunState()
    out = translate(
        _model_end("m1", _Output("answer", {"input_tokens": 10, "output_tokens": 5})),
        run,
    )
    assert out == [LLMEnd("m1"), Usage(10, 5), Context(10), Reply("answer")]
    assert run.last_reply_text == "answer"


def test_nested_model_end_is_not_a_reply() -> None:
    run = RunState()
    out = translate(
        _model_end(
            "m2",
            _Output("sub-agent narration", {"input_tokens": 3, "output_tokens": 1}),
            ns="tools:a|tools:b",
        ),
        run,
    )
    assert out == [LLMEnd("m2"), Usage(3, 1)]
    assert run.last_reply_text == ""


def test_last_top_level_reply_wins() -> None:
    run = RunState()
    translate(_model_end("m1", _Output("first")), run)
    translate(_model_end("m2", _Output("final")), run)
    assert run.last_reply_text == "final"


def test_compaction_events_are_flagged_and_never_replies() -> None:
    run = RunState()
    meta = {"lc_source": "summarization", "langgraph_checkpoint_ns": ""}
    start = {"event": "on_chat_model_start", "run_id": "c", "metadata": meta}
    chunk = {
        "event": "on_chat_model_stream",
        "run_id": "c",
        "metadata": meta,
        "data": {"chunk": _Output("summary…")},
    }
    end = {
        "event": "on_chat_model_end",
        "run_id": "c",
        "metadata": meta,
        "data": {"output": _Output("summary", {"input_tokens": 7, "output_tokens": 2})},
    }
    assert _drain([start, chunk, end], run) == [
        Compaction("start"),
        Compaction("end"),
        Usage(7, 2),
    ]
    assert run.last_reply_text == ""


def test_model_start_and_tokens() -> None:
    run = RunState()
    start = {
        "event": "on_chat_model_start",
        "run_id": "m",
        "metadata": {"langgraph_checkpoint_ns": "tools:a"},
    }
    chunk = {
        "event": "on_chat_model_stream",
        "run_id": "m",
        "metadata": {},
        "data": {"chunk": _Output("hel")},
    }
    empty = {
        "event": "on_chat_model_stream",
        "run_id": "m",
        "metadata": {},
        "data": {"chunk": _Output("")},
    }
    assert _drain([start, chunk, empty], run) == [
        LLMStart("m", "tools:a"),
        Token("hel"),
    ]


def test_truncation_is_surfaced() -> None:
    run = RunState()
    out = translate(_model_end("m", _Output("cut", stop_reason="max_tokens")), run)
    assert out == [LLMEnd("m"), Truncated(), Reply("cut")]


def test_unknown_events_are_ignored() -> None:
    run = RunState()
    assert translate({"event": "on_chain_start", "run_id": "x"}, run) == []
    assert stream.translate({}, run) == []
