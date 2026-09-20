"""Adapter from the typed event stream to Textual messages.

The rules for *which* LangGraph events matter live in
:mod:`ghidra_deep_agent.stream` (shared with the HTTP server); this module only
decides which widget each typed event is posted to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ghidra_deep_agent import stream
from ghidra_deep_agent.stream import RunState, parse_checkpoint_ns
from ghidra_deep_agent.tui.messages import (
    ContextUpdate,
    LLMDone,
    LLMThinking,
    ResponseFinal,
    StatusFlash,
    SubagentReportCaptured,
    TextToken,
    TokenUpdate,
    ToolCountChanged,
    ToolEnded,
    ToolStarted,
)

if TYPE_CHECKING:
    from ghidra_deep_agent.tui.app import GhidraAgentApp
    from ghidra_deep_agent.tui.widgets import ActivityTree, ResponseLog, ThinkingPanel

__all__ = ["handle_event", "parse_checkpoint_ns"]


def handle_event(
    app: GhidraAgentApp,
    event: dict[str, Any],
    activity: ActivityTree,
    response: ResponseLog,
    thinking: ThinkingPanel,
    run: RunState | None = None,
) -> None:
    """Translate one stream event into Textual messages.

    ``run`` carries the cross-event bookkeeping for the current turn; it defaults
    to the app's current state, which is what every caller outside tests wants.
    """
    if run is None:
        run = app.run_state
    for ev in stream.translate(event, run):
        match ev:
            case stream.ToolStart():
                activity.post_message(
                    ToolStarted(
                        ev.call_id,
                        ev.name,
                        ev.preview,
                        ev.is_subagent,
                        ev.checkpoint_ns,
                    )
                )
                app.post_message(ToolCountChanged(1))
            case stream.ToolEnd():
                activity.post_message(ToolEnded(ev.call_id, ev.error, ev.snippet))
                app.post_message(ToolCountChanged(-1))
            case stream.SubagentReport():
                # Keep the full report for ctrl+o.
                app.post_message(SubagentReportCaptured(ev))
            case stream.LLMStart():
                activity.post_message(LLMThinking(ev.call_id, ev.checkpoint_ns))
            case stream.LLMEnd():
                activity.post_message(LLMDone(ev.call_id))
            case stream.Compaction(phase="start"):
                app.post_message(StatusFlash("[yellow]⟳ Compacting context…[/yellow]"))
            case stream.Compaction():
                app.post_message(StatusFlash("[green]✓ Context compacted[/green]"))
            case stream.Truncated():
                app.post_message(
                    StatusFlash(
                        "[red]⚠ Model response truncated at the "
                        "output-token limit[/red]"
                    )
                )
            case stream.Usage():
                app.post_message(TokenUpdate(ev.input_tokens, ev.output_tokens))
            case stream.Context():
                app.post_message(ContextUpdate(ev.input_tokens))
            case stream.Reply():
                response.post_message(ResponseFinal(ev.text))
            case stream.Token():
                thinking.post_message(TextToken(ev.text))
