"""Translation of LangGraph v2 stream events into Textual messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ghidra_deep_agent.async_tasks import ASYNC_DONE_EVENT, async_task_id
from ghidra_deep_agent.tui.formatting import (
    extract_output_snippet,
    extract_preview,
    extract_text,
    extract_usage,
)
from ghidra_deep_agent.tui.messages import (
    ContextUpdate,
    LLMDone,
    LLMThinking,
    ResponseFinal,
    StatusFlash,
    TextToken,
    TokenUpdate,
    ToolCountChanged,
    ToolEnded,
    ToolStarted,
)

if TYPE_CHECKING:
    from ghidra_deep_agent.tui.app import GhidraAgentApp
    from ghidra_deep_agent.tui.widgets import ActivityTree, ResponseLog, ThinkingPanel


def parse_checkpoint_ns(checkpoint_ns: str) -> tuple[str, ...]:
    """Split a LangGraph checkpoint namespace into its segments.

    A namespace looks like "tools:<uuid>|tools:<inner_uuid>|…"; an empty
    string (the root) parses to an empty tuple.
    """
    if not checkpoint_ns:
        return ()
    return tuple(checkpoint_ns.split("|"))


def handle_event(
    app: GhidraAgentApp,
    event: dict[str, Any],
    activity: ActivityTree,
    response: ResponseLog,
    thinking: ThinkingPanel,
) -> None:
    kind = event.get("event", "")
    run_id: str = event.get("run_id", "")
    metadata: dict[str, Any] = event.get("metadata", {})
    checkpoint_ns: str = metadata.get("langgraph_checkpoint_ns", "")
    is_compaction = metadata.get("lc_source") == "summarization"

    if kind == "on_tool_start":
        name = event.get("name", "")
        # The async-task middleware polls `get_task_status` internally; those
        # polls surface as tool runs but aren't the agent's work, so hide them
        # (tracking the run_id keeps the paired on_tool_end + counter balanced).
        if name == "get_task_status":
            app._hidden_tool_runs.add(run_id)
            return
        raw_input = event.get("data", {}).get("input", {})
        preview = extract_preview(raw_input)
        is_subagent = name == "task"
        activity.post_message(
            ToolStarted(run_id, name, preview, is_subagent, checkpoint_ns)
        )
        app.post_message(ToolCountChanged(1))

    elif kind == "on_tool_end":
        if run_id in app._hidden_tool_runs:
            app._hidden_tool_runs.discard(run_id)
            return
        output = event.get("data", {}).get("output")
        error = bool(event.get("data", {}).get("error"))
        # An async tool's own on_tool_end fires immediately with a submission
        # stub, before the real result is polled. Defer its "completed" marker:
        # remember the node by task_id and complete it on ASYNC_DONE_EVENT.
        task_id = async_task_id(output) if not error else None
        if task_id is not None:
            app._pending_async[task_id] = run_id
            return
        snippet = extract_output_snippet(output) if error else ""
        activity.post_message(ToolEnded(run_id, error, snippet))
        app.post_message(ToolCountChanged(-1))

    elif kind == "on_custom_event" and event.get("name") == ASYNC_DONE_EVENT:
        task_id = event.get("data", {}).get("task_id")
        run = app._pending_async.pop(task_id, None) if task_id else None
        if run is not None:
            activity.post_message(ToolEnded(run))
            app.post_message(ToolCountChanged(-1))

    elif kind == "on_chat_model_start":
        if is_compaction:
            app.post_message(StatusFlash("[yellow]⟳ Compacting context…[/yellow]"))
        else:
            activity.post_message(LLMThinking(run_id, checkpoint_ns))

    elif kind == "on_chat_model_end":
        if is_compaction:
            app.post_message(StatusFlash("[green]✓ Context compacted[/green]"))
        else:
            activity.post_message(LLMDone(run_id))
        output = event.get("data", {}).get("output")
        usage = extract_usage(output)
        if usage.input_tokens or usage.output_tokens:
            app.post_message(TokenUpdate(usage.input_tokens, usage.output_tokens))
        if not is_compaction and "|" not in checkpoint_ns and usage.input_tokens:
            app.post_message(ContextUpdate(usage.input_tokens))
        # Capture the main thread's latest message; the final one (the turn that
        # ends the loop) wins, so the main window renders only that — not the
        # intermediate narration accumulated mid-run.
        if not is_compaction and "|" not in checkpoint_ns:
            text = extract_text(output)
            # Stash on the app synchronously so `_run_agent` can read it right
            # after the stream loop (used as the plan text for `/approve`,
            # independent of the async ResponseFinal/AgentDone message flow).
            app._last_reply_text = text
            response.post_message(ResponseFinal(text))

    elif kind == "on_chat_model_stream":
        if is_compaction:
            return  # suppress summary tokens from the normal output panels
        chunk = event.get("data", {}).get("chunk")
        if chunk is None:
            return
        text = extract_text(chunk)
        if text:
            thinking.post_message(TextToken(text))
