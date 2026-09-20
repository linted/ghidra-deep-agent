"""TUI-free translation of LangGraph v2 stream events into a small typed stream.

Which events matter, which are hidden (``get_task_status`` polls, calls made from
inside another tool's body), how async submission stubs defer completion, and
which chat-model end carries the reply, are rules the TUI and the HTTP server
must agree on — so they live here once and both consume :func:`translate`.

``RunState`` is the per-turn bookkeeping :func:`translate` needs across events.
It used to live as four containers on ``GhidraAgentApp`` that the event handler
reached into directly, which leaked: nothing cleared them, so a turn cancelled
with Escape left its in-flight run ids behind forever. Making it an object the
caller *replaces* per turn fixes that structurally — a fresh run starts from a
fresh state, with no reset step to forget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from ghidra_deep_agent.async_tasks import ASYNC_DONE_EVENT, async_task_id
from ghidra_deep_agent.formatting import (
    extract_output_snippet,
    extract_preview,
    extract_stop_reason,
    extract_subagent_report,
    extract_text,
    extract_usage,
)


@dataclass
class RunState:
    """Tool bookkeeping for one agent turn."""

    # Runs suppressed from the activity view: `get_task_status` polls made by the
    # async middleware, and any call made from inside another tool's body. Tracked
    # rather than merely skipped so the paired on_tool_end stays balanced.
    hidden_tool_runs: set[str] = field(default_factory=set)
    # task_id -> run_id for async tool calls whose "completed" marker is deferred
    # until ASYNC_DONE_EVENT arrives (their own on_tool_end fires early, carrying
    # only the submission stub).
    pending_async: dict[str, str] = field(default_factory=dict)
    # run_id -> (description, start time) for sub-agent (`task`) runs in flight.
    subagent_meta: dict[str, tuple[str, float]] = field(default_factory=dict)
    # Plain (non-subagent) tool runs in flight. A call whose parent_ids chain
    # contains one of these was made from *inside* another tool and is hidden.
    active_tool_runs: set[str] = field(default_factory=set)
    # The main thread's latest assistant text, captured synchronously from the
    # stream loop so `/approve` never depends on reading the plan file back.
    last_reply_text: str = ""


# --- typed events -------------------------------------------------------------
# `call_id` is LangGraph's run id for the tool/model call. It is deliberately not
# named `run_id`: over the wire that name belongs to the server's run record.


@dataclass(frozen=True)
class ToolStart:
    call_id: str
    name: str
    preview: str
    is_subagent: bool
    checkpoint_ns: str


@dataclass(frozen=True)
class ToolEnd:
    call_id: str
    error: bool = False
    snippet: str = ""


@dataclass(frozen=True)
class LLMStart:
    call_id: str
    checkpoint_ns: str


@dataclass(frozen=True)
class LLMEnd:
    call_id: str


@dataclass(frozen=True)
class SubagentReport:
    """What one `task` run returned to the main agent."""

    call_id: str
    description: str
    text: str
    error: bool
    elapsed: float


@dataclass(frozen=True)
class Reply:
    """The main thread's assistant text; the last one of a turn is the answer."""

    text: str


@dataclass(frozen=True)
class Usage:
    """Token usage of one model call (a delta, not a running total)."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Context:
    """Snapshot of the main-thread prompt size from the latest model call."""

    input_tokens: int


@dataclass(frozen=True)
class Compaction:
    phase: Literal["start", "end"]


@dataclass(frozen=True)
class Truncated:
    """The model's reply stopped at the output-token limit."""


@dataclass(frozen=True)
class Token:
    """A streamed text chunk (the TUI's thinking panel; not served over HTTP)."""

    text: str


StreamEvent = (
    ToolStart
    | ToolEnd
    | LLMStart
    | LLMEnd
    | SubagentReport
    | Reply
    | Usage
    | Context
    | Compaction
    | Truncated
    | Token
)

# Wire names, shared by the server's JSON serializer and its docs.
EVENT_TYPE: dict[type[StreamEvent], str] = {
    ToolStart: "tool_start",
    ToolEnd: "tool_end",
    LLMStart: "llm_start",
    LLMEnd: "llm_end",
    SubagentReport: "subagent_report",
    Reply: "reply",
    Usage: "usage",
    Context: "context",
    Compaction: "compaction",
    Truncated: "truncated",
    Token: "token",
}


def parse_checkpoint_ns(checkpoint_ns: str) -> tuple[str, ...]:
    """Split a LangGraph checkpoint namespace into its segments.

    A namespace looks like "tools:<uuid>|tools:<inner_uuid>|…"; an empty
    string (the root) parses to an empty tuple.
    """
    if not checkpoint_ns:
        return ()
    return tuple(checkpoint_ns.split("|"))


def translate(event: dict[str, Any], run: RunState) -> list[StreamEvent]:
    """Translate one LangGraph v2 stream event into zero or more typed events.

    Returns them in the order a consumer should apply them (a sub-agent's end
    yields its report before its completion marker).
    """
    out: list[StreamEvent] = []
    kind = event.get("event", "")
    run_id: str = event.get("run_id", "")
    metadata: dict[str, Any] = event.get("metadata", {})
    checkpoint_ns: str = metadata.get("langgraph_checkpoint_ns", "")
    is_compaction = metadata.get("lc_source") == "summarization"
    is_top_level = "|" not in checkpoint_ns

    if kind == "on_tool_start":
        name = event.get("name", "")
        # The async-task middleware polls `get_task_status` internally; those
        # polls surface as tool runs but aren't the agent's work, so hide them
        # (tracking the run_id keeps the paired on_tool_end + counter balanced).
        if name == "get_task_status":
            run.hidden_tool_runs.add(run_id)
            return out
        # A tool call whose ancestry contains a plain tool run was made from
        # inside that tool's body (e.g. recover_prototypes invoking `scripts`
        # directly) — an implementation detail, so hide it. Sub-agent (`task`)
        # runs are deliberately not tracked as parents: their inner tool calls
        # are the sub-agent's real work and stay visible, nested via checkpoint
        # namespaces. Hidden runs count as parents too, so a hidden call's own
        # nested calls stay hidden.
        parent_ids = event.get("parent_ids") or []
        if any(
            pid in run.active_tool_runs or pid in run.hidden_tool_runs
            for pid in parent_ids
        ):
            run.hidden_tool_runs.add(run_id)
            return out
        raw_input = event.get("data", {}).get("input", {})
        preview = extract_preview(raw_input)
        is_subagent = name == "task"
        if not is_subagent:
            run.active_tool_runs.add(run_id)
        else:
            description = (
                raw_input.get("description") if isinstance(raw_input, dict) else None
            )
            run.subagent_meta[run_id] = (description or preview, time.monotonic())
        out.append(ToolStart(run_id, name, preview, is_subagent, checkpoint_ns))

    elif kind == "on_tool_end":
        run.active_tool_runs.discard(run_id)
        if run_id in run.hidden_tool_runs:
            run.hidden_tool_runs.discard(run_id)
            return out
        output = event.get("data", {}).get("output")
        error = bool(event.get("data", {}).get("error"))
        meta = run.subagent_meta.pop(run_id, None)
        if meta is not None:
            # `task` is a local tool that can never return an async submission
            # stub, so skip stub detection: its Command's str() contains the
            # report text, and a report merely quoting a stub would otherwise
            # defer this node forever. Keep the full report for the caller.
            description, started = meta
            out.append(
                SubagentReport(
                    run_id,
                    description,
                    extract_subagent_report(output),
                    error,
                    time.monotonic() - started,
                )
            )
            snippet = extract_output_snippet(output) if error else ""
            out.append(ToolEnd(run_id, error, snippet))
            return out
        # An async tool's own on_tool_end fires immediately with a submission
        # stub, before the real result is polled. Defer its "completed" marker:
        # remember the node by task_id and complete it on ASYNC_DONE_EVENT.
        task_id = async_task_id(output) if not error else None
        if task_id is not None:
            run.pending_async[task_id] = run_id
            return out
        snippet = extract_output_snippet(output) if error else ""
        out.append(ToolEnd(run_id, error, snippet))

    elif kind == "on_custom_event" and event.get("name") == ASYNC_DONE_EVENT:
        task_id = event.get("data", {}).get("task_id")
        done_run_id = run.pending_async.pop(task_id, None) if task_id else None
        if done_run_id is not None:
            out.append(ToolEnd(done_run_id))

    elif kind == "on_chat_model_start":
        if is_compaction:
            out.append(Compaction("start"))
        else:
            out.append(LLMStart(run_id, checkpoint_ns))

    elif kind == "on_chat_model_end":
        if is_compaction:
            out.append(Compaction("end"))
        else:
            out.append(LLMEnd(run_id))
        output = event.get("data", {}).get("output")
        # Truncation is otherwise invisible (an HTTP-success response that just
        # stops): surface it so a run that dead-ends on a cut-off tool call is
        # explainable.
        if not is_compaction and extract_stop_reason(output) in (
            "max_tokens",
            "length",
        ):
            out.append(Truncated())
        usage = extract_usage(output)
        if usage.input_tokens or usage.output_tokens:
            out.append(Usage(usage.input_tokens, usage.output_tokens))
        if not is_compaction and is_top_level and usage.input_tokens:
            out.append(Context(usage.input_tokens))
        # Capture the main thread's latest message; the final one (the turn that
        # ends the loop) wins, so a consumer renders only that — not the
        # intermediate narration accumulated mid-run.
        if not is_compaction and is_top_level:
            text = extract_text(output)
            # Stashed on the state synchronously so the stream loop can read it
            # right after the loop ends (the TUI uses it as the plan text for
            # `/approve`, independent of its async message flow).
            run.last_reply_text = text
            out.append(Reply(text))

    elif kind == "on_chat_model_stream":
        if is_compaction:
            return out  # suppress summary tokens from the normal output panels
        chunk = event.get("data", {}).get("chunk")
        if chunk is None:
            return out
        text = extract_text(chunk)
        if text:
            out.append(Token(text))

    return out
