"""Translation of LangGraph v2 stream events into Textual messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        raw_input = event.get("data", {}).get("input", {})
        preview = extract_preview(raw_input)
        is_subagent = name == "task"
        activity.post_message(
            ToolStarted(run_id, name, preview, is_subagent, checkpoint_ns)
        )
        app.post_message(ToolCountChanged(1))

    elif kind == "on_tool_end":
        output = event.get("data", {}).get("output")
        error = bool(event.get("data", {}).get("error"))
        snippet = extract_output_snippet(output) if error else ""
        activity.post_message(ToolEnded(run_id, error, snippet))
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

    elif kind == "on_chat_model_stream":
        if is_compaction:
            return  # suppress summary tokens from the normal output panels
        chunk = event.get("data", {}).get("chunk")
        if chunk is None:
            return
        text = extract_text(chunk)
        if text:
            response.post_message(TextToken(text))
            thinking.post_message(TextToken(text))
