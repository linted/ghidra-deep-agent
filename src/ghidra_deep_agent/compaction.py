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

   Tuning is **scope-aware**: sub-agents get aggressive built-in thresholds
   (they accumulate large decompiler dumps and, on models without a langchain
   context profile, deepagents' 170k-token fallback trigger effectively never
   fires), while the main agent keeps stock defaults — its baseline prompt
   (system prompt + tool schemas, which the trigger counts) sits far above the
   sub-agent trigger, so a shared low threshold would fire on every call and
   permanently squash its history.
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


def _model_key(model: Any) -> str | None:
    """Best-effort identifier for a model spec or instance.

    Used only to tell the main agent's model apart from sub-agent models when
    object identity doesn't hold (e.g. the spec string was passed instead of
    the built instance). For ``provider:model`` spec strings the provider
    prefix is dropped so a spec compares equal to the instance built from it.
    """
    if isinstance(model, str):
        return model.split(":", 1)[-1]
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _tuned_auto_summarization(
    model: str | BaseChatModel,
    backend: Any,
    *,
    summary_model: str | BaseChatModel | None,
    is_main: bool,
    **_: Any,
) -> Any:
    """Build a deepagents ``SummarizationMiddleware`` with tuned thresholds.

    Main-agent scope keeps deepagents' model-aware defaults, overridable via
    ``COMPACT_MAIN_*`` env knobs. Sub-agent scope defaults to the aggressive
    ``_SUBAGENT_DEFAULT_*`` thresholds, overridable via ``COMPACT_*`` knobs.
    ``summary_model`` (when given) routes the summary call to a cheaper model
    regardless of the agent's own model.
    """
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


def install_tuned_summarization(
    summary_model: str | BaseChatModel | None = None,
    *,
    main_model: str | BaseChatModel | None = None,
) -> None:
    """Patch the factory ``create_deep_agent`` uses for auto-summarization.

    ``create_deep_agent`` calls ``deepagents.graph.create_summarization_middleware``
    (a module-bound import) for the main agent and every sub-agent. Replacing that
    name with our tuned factory is the only seam to lower the trigger or change the
    summary model without forking deepagents — passing our own
    ``SummarizationMiddleware`` in ``middleware=`` would instead add a *second*
    instance and trip create_agent's duplicate-middleware assertion. Idempotent.

    ``main_model`` identifies the coordinator's model so the patched factory can
    scope thresholds: agents built with it keep stock defaults, everything else
    gets the sub-agent thresholds. Matching is by object identity with a
    model-name fallback, so a sub-agent explicitly configured with the main
    agent's model inherits main-scope thresholds — acceptable, since its prompt
    baseline is the concern being scoped around, not its name.
    """
    import deepagents.graph as graph

    # The factory is a module-bound import, not in deepagents.graph's public
    # exports, so reach it dynamically (keeps the type checker happy too).
    current = getattr(graph, "create_summarization_middleware")
    if getattr(current, "_ghidra_tuned", False):
        return

    main_key = _model_key(main_model) if main_model is not None else None

    def patched(model: Any, backend: Any, **kwargs: Any) -> Any:
        is_main = main_model is not None and (
            model is main_model
            or (main_key is not None and _model_key(model) == main_key)
        )
        return _tuned_auto_summarization(
            model, backend, summary_model=summary_model, is_main=is_main, **kwargs
        )

    patched._ghidra_tuned = True  # type: ignore[attr-defined]
    setattr(graph, "create_summarization_middleware", patched)
