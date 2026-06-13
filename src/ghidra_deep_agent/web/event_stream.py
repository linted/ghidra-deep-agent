"""Translate LangGraph v2 stream events into JSON payloads for the web client.

This mirrors the TUI's [tui/events.py](../tui/events.py) but emits plain
JSON-serializable dicts (sent over the WebSocket) instead of Textual messages,
reusing the framework-agnostic extractors in
[tui/formatting.py](../tui/formatting.py).
"""

from __future__ import annotations

from typing import Any

from ghidra_deep_agent.tui.formatting import (
    extract_output_snippet,
    extract_preview,
    extract_text,
    extract_usage,
)

Payload = dict[str, Any]


def event_to_payloads(event: dict[str, Any]) -> list[Payload]:
    """Convert one LangGraph v2 event into zero or more client payloads.

    A single event can map to multiple client updates (e.g. an
    ``on_chat_model_end`` produces both an ``llm_done`` and a
    ``token_update``), so this returns a list.
    """
    kind = event.get("event", "")
    run_id: str = event.get("run_id", "")
    metadata: dict[str, Any] = event.get("metadata", {})
    checkpoint_ns: str = metadata.get("langgraph_checkpoint_ns", "")
    is_compaction = metadata.get("lc_source") == "summarization"

    payloads: list[Payload] = []

    if kind == "on_tool_start":
        name = event.get("name", "")
        raw_input = event.get("data", {}).get("input", {})
        payloads.append(
            {
                "type": "tool_start",
                "run_id": run_id,
                "name": name,
                "preview": extract_preview(raw_input),
                "is_subagent": name == "task",
                "checkpoint_ns": checkpoint_ns,
            }
        )
        payloads.append({"type": "tool_count", "delta": 1})

    elif kind == "on_tool_end":
        error = bool(event.get("data", {}).get("error"))
        output = event.get("data", {}).get("output")
        snippet = extract_output_snippet(output) if error else ""
        payloads.append(
            {"type": "tool_end", "run_id": run_id, "error": error, "snippet": snippet}
        )
        payloads.append({"type": "tool_count", "delta": -1})

    elif kind == "on_chat_model_start":
        if is_compaction:
            payloads.append({"type": "status_flash", "text": "⟳ Compacting context…"})
        else:
            payloads.append(
                {
                    "type": "llm_thinking",
                    "run_id": run_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            )

    elif kind == "on_chat_model_end":
        if is_compaction:
            payloads.append({"type": "status_flash", "text": "✓ Context compacted"})
        else:
            payloads.append({"type": "llm_done", "run_id": run_id})
        output = event.get("data", {}).get("output")
        usage = extract_usage(output)
        if usage.input_tokens or usage.output_tokens:
            payloads.append(
                {
                    "type": "token_update",
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                }
            )
        if not is_compaction and "|" not in checkpoint_ns and usage.input_tokens:
            payloads.append(
                {"type": "context_update", "current_input": usage.input_tokens}
            )

    elif kind == "on_chat_model_stream":
        if is_compaction:
            return payloads  # suppress summary tokens from the output panels
        chunk = event.get("data", {}).get("chunk")
        if chunk is not None:
            text = extract_text(chunk)
            if text:
                payloads.append({"type": "token", "text": text})

    return payloads
