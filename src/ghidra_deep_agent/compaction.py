"""On-demand and auto conversation compaction.

Two concerns live here:

1. **Manual ``/compact``.** deepagents' built-in ``compact_conversation`` tool
   refuses to run until reported usage reaches ~50% of the auto-summarization
   trigger, making a user-driven ``/compact`` a no-op while the conversation is
   still comfortably within budget. We swap in a subclass that always treats
   manual compaction as eligible.

2. **Tuning the *auto* summarizer.** ``create_deep_agent`` installs a stock
   ``SummarizationMiddleware(model, backend)`` for the main agent and every
   sub-agent, with no parameter to lower the trigger or route the (cheap,
   structured) summary call to a smaller model. Since deepagents 0.7, a
   ``middleware=`` (or sub-agent ``middleware``) entry whose ``.name`` matches a
   built-in **replaces** that default in place, so
   ``build_tuned_summarization_middleware`` builds a tuned
   ``SummarizationMiddleware`` to pass there. This keeps all of deepagents'
   summarization behavior (backend offload of evicted history,
   pre-summarization tool-arg truncation, ``ContextOverflowError`` fallback)
   while letting us compact earlier and summarize on a cheaper model.

   Tuning is **scope-aware**: sub-agents get aggressive built-in thresholds
   (they accumulate large decompiler dumps and, on models without a langchain
   context profile, deepagents' 170k-token fallback trigger effectively never
   fires), while the main agent keeps stock defaults — its baseline prompt
   (system prompt + tool schemas, which the trigger counts) sits far above the
   sub-agent trigger, so a shared low threshold would fire on every call and
   permanently squash its history. Each scope is an explicit argument at the
   construction site rather than inferred from the model.
"""

import os
import sys
from typing import Any, Literal

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


# --- Auto-summarization tuning -------------------------------------------------

# Sub-agent compaction thresholds. The trigger counts the *full* prompt
# (system message + tool schemas + history); a sub-agent baseline is ~11k, so
# 50k total ≈ 39k of accumulated history. Keep must be token-based: after a
# compaction the retained slice is guaranteed to sit well under the trigger,
# whereas a message-count keep can retain a few huge tool dumps and re-trigger
# immediately.
_SUBAGENT_DEFAULT_TRIGGER = ("tokens", 50000)
_SUBAGENT_DEFAULT_KEEP = ("tokens", 10000)


def _warn_no_profile(knob: str) -> None:
    print(
        f"Warning: {knob} ignored — the model exposes no context-window profile, "
        "so fractional thresholds can't be used. Use the *_TOKENS / *_MESSAGES "
        "form instead.",
        file=sys.stderr,
    )


def _trigger_from_env(default: Any, *, has_profile: bool, prefix: str) -> Any:
    """Override the auto-summarization trigger from env, else keep the default.

    ``{prefix}_TRIGGER_FRACTION`` (0-1 of the model's context) takes precedence
    over ``{prefix}_TRIGGER_TOKENS`` (absolute token count). Lowering it
    compacts earlier, trading a few extra summary calls for smaller per-call
    context. Fractional thresholds require a model profile; without one we warn
    and fall back to the token knob / default rather than crash at startup.
    """
    frac = os.environ.get(f"{prefix}_TRIGGER_FRACTION")
    if frac:
        if has_profile:
            return ("fraction", float(frac))
        _warn_no_profile(f"{prefix}_TRIGGER_FRACTION")
    tokens = os.environ.get(f"{prefix}_TRIGGER_TOKENS")
    if tokens:
        return ("tokens", int(tokens))
    return default


def _keep_from_env(default: Any, *, has_profile: bool, prefix: str) -> Any:
    """Override how much context to keep after compaction, else the default.

    Precedence: ``{prefix}_KEEP_TOKENS`` > ``{prefix}_KEEP_MESSAGES`` >
    ``{prefix}_KEEP_FRACTION``.
    """
    tokens = os.environ.get(f"{prefix}_KEEP_TOKENS")
    if tokens:
        return ("tokens", int(tokens))
    msgs = os.environ.get(f"{prefix}_KEEP_MESSAGES")
    if msgs:
        return ("messages", int(msgs))
    frac = os.environ.get(f"{prefix}_KEEP_FRACTION")
    if frac:
        if has_profile:
            return ("fraction", float(frac))
        _warn_no_profile(f"{prefix}_KEEP_FRACTION")
    return default


def build_tuned_summarization_middleware(
    model: str | BaseChatModel,
    backend: Any,
    *,
    summary_model: str | BaseChatModel | None = None,
    scope: Literal["main", "subagent"],
) -> Any:
    """Build a deepagents ``SummarizationMiddleware`` with tuned thresholds.

    The instance's ``.name`` is ``"SummarizationMiddleware"``, so passing it in
    ``create_deep_agent(middleware=...)`` (or a sub-agent's ``middleware``)
    replaces deepagents' stock instance in place (0.7 replace-by-name).

    ``scope="main"`` keeps deepagents' model-aware defaults, overridable via
    ``COMPACT_MAIN_*`` env knobs. ``scope="subagent"`` defaults to the
    aggressive ``_SUBAGENT_DEFAULT_*`` thresholds, overridable via ``COMPACT_*``
    knobs. ``summary_model`` (when given) routes the summary call to a cheaper
    model regardless of the agent's own model.
    """
    is_main = scope == "main"
    from deepagents._models import resolve_model
    from deepagents.middleware.summarization import (
        DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    resolved = resolve_model(model) if isinstance(model, str) else model
    defaults = compute_summarization_defaults(resolved)
    # Fraction thresholds are validated and evaluated against the model the
    # middleware itself holds — the summary model once one is routed — so the
    # profile check must look at that model, not the agent's.
    summary = (
        resolve_model(summary_model)
        if isinstance(summary_model, str)
        else summary_model
    )
    mw_model = summary if summary is not None else resolved
    profile = getattr(mw_model, "profile", None)
    has_profile = isinstance(profile, dict) and isinstance(
        profile.get("max_input_tokens"), int
    )
    if is_main:
        trigger = _trigger_from_env(
            defaults["trigger"], has_profile=has_profile, prefix="COMPACT_MAIN"
        )
        keep = _keep_from_env(
            defaults["keep"], has_profile=has_profile, prefix="COMPACT_MAIN"
        )
    else:
        trigger = _trigger_from_env(
            _SUBAGENT_DEFAULT_TRIGGER, has_profile=has_profile, prefix="COMPACT"
        )
        keep = _keep_from_env(
            _SUBAGENT_DEFAULT_KEEP, has_profile=has_profile, prefix="COMPACT"
        )
    return SummarizationMiddleware(
        mw_model,
        backend=backend,
        trigger=trigger,
        keep=keep,
        truncate_args_settings=defaults["truncate_args_settings"],
        summary_prompt=DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
    )
