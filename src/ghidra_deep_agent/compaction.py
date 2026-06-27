"""On-demand conversation compaction middleware.

deepagents' built-in ``compact_conversation`` tool refuses to run until reported
usage reaches ~50% of the auto-summarization trigger. That makes a user-driven
``/compact`` a no-op while the conversation is still comfortably within budget.
This module swaps in a subclass that always treats manual compaction as eligible.
"""

from typing import Any

from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_middleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


class ForcedCompactionToolMiddleware(SummarizationToolMiddleware):
    """``compact_conversation`` that always compacts on demand.

    The upstream tool refuses to compact until usage reaches ~50% of the
    auto-summarization trigger. For an explicit user-driven ``/compact`` we
    always want it to run, so manual compaction is always eligible. The
    independent cutoff check still returns "nothing to compact" when there are
    too few messages to summarize.
    """

    def _is_eligible_for_compaction(self, messages: list[AnyMessage]) -> bool:
        return True


def create_forced_summarization_tool_middleware(
    model: str | BaseChatModel, backend: Any
) -> ForcedCompactionToolMiddleware:
    """Mirror of ``create_summarization_tool_middleware`` using the forced subclass.

    Resolves a model string to a ``BaseChatModel`` (as the upstream factory does)
    before building the summarization engine.
    """
    from deepagents._models import resolve_model

    if isinstance(model, str):
        model = resolve_model(model)
    summarization = create_summarization_middleware(model, backend)
    return ForcedCompactionToolMiddleware(summarization)
