"""Seeding a read-only side thread with what the main session already knows.

Plan mode and ask mode run on their own threads, so the planner or answerer
would otherwise start blind. Before their first turn, the main thread's
history is summarized into a marked background block that is prepended as a
user message — background, not the model's own investigation.

Shared by the TUI (``/plan``, ``/ask``) and the HTTP server (``mode=ask``).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import get_buffer_string

from ghidra_deep_agent.formatting import extract_text
from ghidra_deep_agent.prompt import MARKED_BACKGROUND, PLAN_CONTEXT_SUMMARY_PROMPT

# Skip building a prior-context summary when the main thread has fewer than this
# many messages (nothing meaningful to hand over yet).
MIN_MESSAGES_FOR_SUMMARY = 3


class SeedError(RuntimeError):
    """The seed could not be built; ``key`` says which step failed."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


async def marked_prior_context(
    main_graph: Any, main_config: dict[str, Any], summary_model: BaseChatModel | None
) -> str | None:
    """Summarize the main thread into a marked background block.

    Returns ``None`` (skip seeding) when there is no summary model or the main
    thread is empty/tiny. Raises :class:`SeedError` when reading the state or
    calling the summary model fails, so the caller can warn and go on without
    a seed.
    """
    if summary_model is None:
        return None
    try:
        state = await main_graph.aget_state(main_config)
    except Exception as exc:
        raise SeedError("seed_state", f"started without prior context: {exc}") from exc
    messages = state.values.get("messages", [])
    if len(messages) < MIN_MESSAGES_FOR_SUMMARY:
        return None
    transcript = get_buffer_string(messages, format="xml")
    try:
        reply = await summary_model.ainvoke(
            PLAN_CONTEXT_SUMMARY_PROMPT.format(transcript=transcript)
        )
    except Exception as exc:
        raise SeedError(
            "seed_summary",
            f"started without prior context ({exc}); check SUMMARY_MODEL.",
        ) from exc
    summary = extract_text(reply).strip()
    return MARKED_BACKGROUND.format(summary=summary) if summary else None
