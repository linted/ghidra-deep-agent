"""Pure helpers for extracting and formatting agent-event data."""

from __future__ import annotations

from typing import NamedTuple


def extract_preview(raw: object) -> str:
    if isinstance(raw, dict):
        text = (
            raw.get("description") or raw.get("task") or raw.get("prompt") or str(raw)
        )
    else:
        text = str(raw)
    return text[:60]


def extract_text(chunk: object) -> str:
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def extract_output_snippet(output: object) -> str:
    """Pull a short text snippet from a tool's output (typically a ToolMessage)."""
    if output is None:
        return ""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content[:80]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return str(block["text"])[:80]
            if isinstance(block, str):
                return block[:80]
    return str(content)[:80]


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int


def extract_usage(output: object) -> Usage:
    """Pull input/output tokens from a chat model's usage_metadata, if present."""
    if output is None:
        return Usage(0, 0)
    usage = getattr(output, "usage_metadata", None)
    if not isinstance(usage, dict):
        return Usage(0, 0)
    try:
        return Usage(
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return Usage(0, 0)


def fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m{secs:02d}s"


def fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"
