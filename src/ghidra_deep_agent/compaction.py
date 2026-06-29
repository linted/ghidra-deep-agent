"""On-demand and auto conversation compaction.

Two concerns live here:

1. **Manual ``/compact``.** deepagents' built-in ``compact_conversation`` tool
   refuses to run until reported usage reaches ~50% of the auto-summarization
   trigger, making a user-driven ``/compact`` a no-op while the conversation is
   still comfortably within budget. We swap in a subclass that always treats
   manual compaction as eligible.

2. **Tuning the *auto* summarizer.** ``create_deep_agent`` hard-wires
   ``create_summarization_middleware(model, backend)`` for the main agent and
   every sub-agent, with no parameter to lower the trigger or route the (cheap,
   structured) summary call to a smaller model. deepagents' own
   ``SummarizationMiddleware`` *does* accept ``trigger``/``keep``/``model``, so
   ``install_tuned_summarization`` monkeypatches the factory symbol that
   ``create_deep_agent`` calls, returning a tuned instance instead. This keeps
   all of deepagents' summarization behavior (backend offload of evicted
   history, pre-summarization tool-arg truncation, ``ContextOverflowError``
   fallback) while letting us compact earlier and summarize on a cheaper model.
"""

import os
import sys
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


# --- Auto-summarization tuning -------------------------------------------------


def _warn_no_profile(knob: str) -> None:
    print(
        f"Warning: {knob} ignored — the model exposes no context-window profile, "
        "so fractional thresholds can't be used. Use the *_TOKENS / *_MESSAGES "
        "form instead.",
        file=sys.stderr,
    )


def _trigger_from_env(default: Any, *, has_profile: bool) -> Any:
    """Override the auto-summarization trigger from env, else keep the default.

    ``COMPACT_TRIGGER_FRACTION`` (0-1 of the model's context) takes precedence
    over ``COMPACT_TRIGGER_TOKENS`` (absolute token count). Lowering it compacts
    earlier, trading a few extra summary calls for smaller per-call context.
    Fractional thresholds require a model profile; without one we warn and fall
    back to the token knob / default rather than crash at startup.
    """
    frac = os.environ.get("COMPACT_TRIGGER_FRACTION")
    if frac:
        if has_profile:
            return ("fraction", float(frac))
        _warn_no_profile("COMPACT_TRIGGER_FRACTION")
    tokens = os.environ.get("COMPACT_TRIGGER_TOKENS")
    if tokens:
        return ("tokens", int(tokens))
    return default


def _keep_from_env(default: Any, *, has_profile: bool) -> Any:
    """Override how much context to keep after compaction, else the default."""
    msgs = os.environ.get("COMPACT_KEEP_MESSAGES")
    if msgs:
        return ("messages", int(msgs))
    frac = os.environ.get("COMPACT_KEEP_FRACTION")
    if frac:
        if has_profile:
            return ("fraction", float(frac))
        _warn_no_profile("COMPACT_KEEP_FRACTION")
    return default


def _tuned_auto_summarization(
    model: str | BaseChatModel,
    backend: Any,
    *,
    summary_model: str | BaseChatModel | None,
    **_: Any,
) -> Any:
    """Build a deepagents ``SummarizationMiddleware`` with env-tuned thresholds.

    Starts from deepagents' model-aware defaults (so behavior is unchanged when
    no env knobs are set) and overrides the trigger/keep thresholds and the
    summary model where configured. ``summary_model`` (when given) routes the
    summary call to a cheaper model regardless of the agent's own model.
    """
    from deepagents._models import resolve_model
    from deepagents.middleware.summarization import (
        DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    resolved = resolve_model(model) if isinstance(model, str) else model
    defaults = compute_summarization_defaults(resolved)
    profile = getattr(resolved, "profile", None)
    has_profile = isinstance(profile, dict) and isinstance(
        profile.get("max_input_tokens"), int
    )
    return SummarizationMiddleware(
        summary_model if summary_model is not None else model,
        backend=backend,
        trigger=_trigger_from_env(defaults["trigger"], has_profile=has_profile),
        keep=_keep_from_env(defaults["keep"], has_profile=has_profile),
        truncate_args_settings=defaults["truncate_args_settings"],
        summary_prompt=DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
    )


def auto_summarization_tuning_enabled() -> bool:
    """True when any auto-summarization knob (trigger/keep/model) is configured."""
    return any(
        os.environ.get(name)
        for name in (
            "COMPACT_TRIGGER_FRACTION",
            "COMPACT_TRIGGER_TOKENS",
            "COMPACT_KEEP_MESSAGES",
            "COMPACT_KEEP_FRACTION",
            "SUMMARY_MODEL",
        )
    )


def install_tuned_summarization(
    summary_model: str | BaseChatModel | None = None,
) -> None:
    """Patch the factory ``create_deep_agent`` uses for auto-summarization.

    ``create_deep_agent`` calls ``deepagents.graph.create_summarization_middleware``
    (a module-bound import) for the main agent and every sub-agent. Replacing that
    name with our tuned factory is the only seam to lower the trigger or change the
    summary model without forking deepagents — passing our own
    ``SummarizationMiddleware`` in ``middleware=`` would instead add a *second*
    instance and trip create_agent's duplicate-middleware assertion. Idempotent.
    """
    import deepagents.graph as graph

    # The factory is a module-bound import, not in deepagents.graph's public
    # exports, so reach it dynamically (keeps the type checker happy too).
    current = getattr(graph, "create_summarization_middleware")
    if getattr(current, "_ghidra_tuned", False):
        return

    def patched(model: Any, backend: Any, **kwargs: Any) -> Any:
        return _tuned_auto_summarization(
            model, backend, summary_model=summary_model, **kwargs
        )

    patched._ghidra_tuned = True  # type: ignore[attr-defined]
    setattr(graph, "create_summarization_middleware", patched)
