"""On-demand and auto conversation compaction.

Two concerns live here:

1. **Manual ``/compact``.** The TUI used to compact by asking the *agent* to
   call deepagents' ``compact_conversation`` tool — a full-context main-model
   turn spent at the exact moment context is at its largest, and one the model
   was free to ignore. ``compact_out_of_band`` instead drives the summarization
   engine directly: one summary-model call, no agent turn, always runs. It
   only computes the ``_summarization_event``; persisting it (via
   ``graph.aupdate_state``) is the caller's job, so a failure anywhere leaves
   the thread untouched.

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
from dataclasses import dataclass
from typing import Any, Literal

from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, get_buffer_string
from langchain_core.runnables.config import var_child_runnable_config


def create_manual_compaction_engine(model: str | BaseChatModel, backend: Any) -> Any:
    """Build the summarization engine `/compact` drives directly.

    A plain deepagents ``SummarizationMiddleware`` — never registered as
    middleware, only used as the engine behind ``compact_out_of_band`` and the
    agent's own ``compact_conversation`` tool (``SummarizationToolMiddleware``
    wraps this same instance). ``trim_tokens_to_summarize`` stays ``None``: the
    whole evicted slice goes to the (cheap, per ``SUMMARY_MODEL``) summary
    model, which beats summarizing a 4k-token tail. A slice exceeding that
    model's context window fails the summary call cleanly — nothing persisted.
    """
    from deepagents._models import resolve_model

    if isinstance(model, str):
        model = resolve_model(model)
    return create_summarization_middleware(model, backend)


@dataclass(frozen=True)
class ManualCompactionResult:
    """What an out-of-band compaction produced, ready to persist."""

    # The ``_summarization_event`` to write into thread state.
    event: dict[str, Any]
    summarized_count: int
    # Backend path the evicted history was offloaded to; None if that
    # (non-fatal) write failed.
    file_path: str | None


async def compact_out_of_band(
    engine: Any,
    messages: list[AnyMessage],
    prior_event: Any,
    *,
    thread_id: str,
) -> ManualCompactionResult | None:
    """Compact a thread's history without an agent turn.

    Runs the same steps as the ``compact_conversation`` tool body
    (``SummarizationToolMiddleware._arun_compact``), minus its eligibility gate
    (this path is explicitly user-driven) and minus the ``ToolMessage`` (there
    is no calling AI message to pair it with). Compaction never rewrites
    ``state["messages"]`` — the effective list is rebuilt from
    ``_summarization_event`` — so the returned event is the entire state
    change. Returns ``None`` when there is nothing to compact.

    Raises whatever the summary-model call raises; no side effect has happened
    by then, so the caller can report the error and leave the thread as-is.
    """
    effective = engine._apply_event_to_messages(list(messages), prior_event)
    cutoff = engine._determine_cutoff_index(effective)
    if cutoff == 0:
        return None
    to_summarize, _ = engine._partition_messages(effective, cutoff)
    summary = await _summarize_or_raise(engine, to_summarize)
    # The offload derives its per-thread history file from the thread_id in the
    # runnable-config contextvar. Outside a graph run that var is unset and the
    # engine falls back to a random `session_*` id, fragmenting the history
    # file — so pin the real thread id for the duration of the write.
    token = var_child_runnable_config.set({"configurable": {"thread_id": thread_id}})
    try:
        file_path = await engine._aoffload_to_backend(engine._backend, to_summarize)
    finally:
        var_child_runnable_config.reset(token)
    event = {
        "cutoff_index": engine._compute_state_cutoff(prior_event, cutoff),
        "summary_message": engine._build_new_messages_with_path(summary, file_path)[0],
        "file_path": file_path,
    }
    return ManualCompactionResult(event, len(to_summarize), file_path)


async def _summarize_or_raise(engine: Any, to_summarize: list[AnyMessage]) -> str:
    """The engine's summary call, except failures raise.

    ``_acreate_summary`` swallows model errors into an ``"Error generating
    summary: ..."`` string, which a persisting caller would then install as the
    summary — silently replacing real history with junk. Same prompt, same
    trim, but the exception propagates so nothing gets persisted.
    """
    lc = engine._lc_helper
    trimmed = lc._trim_messages_for_summary(to_summarize)
    formatted = get_buffer_string(trimmed, format="xml")
    response = await lc.model.ainvoke(
        lc.summary_prompt.format(messages=formatted).rstrip(),
        config={"metadata": {"lc_source": "summarization"}},
    )
    return str(response.text).strip()


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
