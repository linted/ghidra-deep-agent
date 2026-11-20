"""Resilience middleware: model retry, provider fallback, and tool retry.

The agent talks to OpenRouter / DeepSeek over the network, and (with a
``FilesystemBackend``) writes artifacts to disk. Without any retry layer a
transient 5xx / 429 / connection reset from the provider, or a transient I/O
error from a filesystem tool, bubbles straight up through the agent and crashes
the TUI. These factories wrap the model and tool calls with the stock LangChain
retry/fallback middleware so transient failures are retried with backoff (and,
optionally, fall back to a different provider/model) instead of aborting a run.

Configuration (env):
  MODEL_MAX_RETRIES   retry attempts per model call after the first (default 3)
  MODEL_FALLBACK      comma-separated ``provider:model`` fallbacks tried, in
                      order, after the primary model's retries are exhausted
                      (default: none — fallback disabled)
  TOOL_MAX_RETRIES    retry attempts for retryable filesystem tools (default 3)
  MODEL_CONCURRENCY_WAIT
                      seconds to keep retrying a "concurrency limit" 429
                      before pausing the run (default 300)
  MODEL_MAX_CONCURRENCY
                      cap on in-flight model calls across the main agent and
                      all sub-agents in this process (default: unlimited)
"""

import asyncio
import os
import random
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    hook_config,
)
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from ghidra_deep_agent.defaults import env_float, env_int
from ghidra_deep_agent.toasts import notify_toast

ModelResolver = Callable[[str | None], str | BaseChatModel]


class UsageLimitError(Exception):
    """Raised when a provider usage/rate limit outlasts the retry budget.

    A rate/quota limit that our short backoff retries can't clear (e.g. the
    multi-hour Anthropic "5-hour" limit) is not something we want to swallow into
    a synthetic error turn — that would poison the conversation and, inside a
    sub-agent, feed garbage back to the coordinator. Instead we raise this so the
    in-flight turn halts at a clean checkpoint boundary. The MongoDB checkpointer
    has already persisted every completed super-step (including finished
    sub-agent ``task`` results via pending writes), so the run can be resumed
    later on the same ``thread_id`` with a ``None`` input — see the TUI's
    ``/continue`` command.
    """

    def __init__(self, original: BaseException) -> None:
        self.original = original
        super().__init__(str(original))


# deepagents filesystem built-ins whose failures are transient I/O (and whose
# retries are safe — idempotent reads/writes of agent artifacts). We do NOT
# retry Ghidra MCP tools here: their transport already surfaces server errors as
# structured messages (see ``handle_mcp_errors`` in cli.py), and many are not
# idempotent.
_RETRYABLE_FS_TOOLS = ("write_file", "edit_file", "read_file", "delete")

# HTTP status codes worth retrying: request timeout, conflict, rate limit, and
# the 5xx family that providers return for transient overload.
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Substrings that mark a transient provider/network error when no status code is
# exposed on the exception.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "rate limit",
    "ratelimit",
    "too many requests",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)

# Subset of transient errors that mean "we've hit a usage/rate/quota limit" — the
# kind our short retries can't wait out. A 429 status, or any of these markers,
# routes the exhausted call to a clean halt (UsageLimitError) instead of a
# swallowed error turn, so the run stays cleanly resumable. Distinct from a plain
# network blip (timeout / connection reset / 5xx), which keeps the old behavior.
# NB: "overloaded" is deliberately absent. It means the *provider* is at capacity
# (a 529-style blip that clears in seconds), not that we've hit a quota — so it
# belongs in _TRANSIENT_MARKERS only. Listing it here too escalated a short
# capacity blip into a full halt once the retries were spent.
_USAGE_LIMIT_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota",
    "usage limit",
    "insufficient_quota",
)


def _is_transient(exc: BaseException) -> bool:
    """Would retrying this error plausibly succeed?

    True for rate limits, timeouts, connection resets, and 5xx responses; False
    for deterministic failures (bad request, auth, schema/validation), which
    would only waste time and money on retry.

    Since langchain-core 1.6.0, provider packages raise standard ``ModelError``
    subclasses (dual-inherited with the SDK's own types) carrying an
    authoritative ``is_retryable`` flag — trust it outright: the provider knows
    "overloaded" is a blip and "invalid api key" is not, no matter what the
    message text says. The status/text heuristics remain as the fallback for
    providers that haven't adopted the hierarchy (e.g. Ollama).
    """
    if isinstance(exc, ModelError):
        return exc.is_retryable
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _is_usage_limit(exc: BaseException) -> bool:
    """Would waiting hours (not seconds) be the only thing that clears this?

    True for provider rate/usage/quota limits: the standard
    ``ModelRateLimitError`` type (anthropic/openai-protocol providers since
    langchain-core 1.6.0), a 429 status, or a limit marker in the text — the
    latter two remain for providers without the standard hierarchy. Never
    relies on a provider-specific ``retry-after`` header (inconsistent across
    Anthropic/OpenRouter/DeepSeek, absent on Ollama).
    """
    if isinstance(exc, ModelRateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _USAGE_LIMIT_MARKERS)


# Markers of a *concurrency* limit: too many requests in flight at once on the
# workspace/key, not too many over a time window. It clears as soon as any
# in-flight call (ours or another client's) finishes — seconds to a minute or
# two, never hours — so it gets a long patient wait instead of the short retry
# budget and the usage-limit halt. Observed verbatim from the OpenCode gateway:
# "Error code: 429 - Your workspace has reached the concurrency limit".
_CONCURRENCY_MARKERS = (
    "concurrency limit",
    "concurrent request",
    "too many concurrent",
)


def _is_concurrency_limit(exc: BaseException) -> bool:
    """Is this a rate limit on requests in flight, rather than a quota?"""
    if not _is_usage_limit(exc):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _CONCURRENCY_MARKERS)


def _retry_on(exc: BaseException) -> bool:
    """``ModelRetryMiddleware``'s filter: transient, minus concurrency limits.

    A concurrency limit reaching the retry layer has already outlasted
    :class:`ConcurrencyLimitRetryMiddleware`'s whole wait budget; three more
    short retries would not clear it either, so hand it straight to the error
    boundary, which pauses the run.
    """
    return _is_transient(exc) and not _is_concurrency_limit(exc)


_CREDITS_MARKERS = ("requires more credits",)


def _is_out_of_credits(exc: BaseException) -> bool:
    """OpenRouter 402: prepaid credits / key daily limit can't cover the request.

    The openai SDK has no dedicated 402 exception class, so this arrives as a
    generic ``APIStatusError``; match on the status code, with the message text
    as a fallback for errors proxied through a fallback model.
    """
    if getattr(exc, "status_code", None) == 402:
        return True
    return any(marker in str(exc).lower() for marker in _CREDITS_MARKERS)


def _on_model_retries_exhausted(exc: BaseException) -> str:
    """Terminal model-error policy: halt on a limit, else continue.

    Called from two places that together see every terminal model error:
    ``ModelRetryMiddleware``'s ``on_failure`` when retries of a transient error
    are exhausted, and :class:`ModelErrorBoundaryMiddleware` for non-retryable
    errors (e.g. a 402), which langchain ≥1.3.16 re-raises past ``on_failure``.

    An out-of-credits error or a usage/rate limit is raised as
    :class:`UsageLimitError` so the turn stops at a clean, resumable checkpoint
    instead of committing a synthetic error message; the credits case also emits
    an error toast with provider-specific guidance, since the TUI's generic
    pause banner only mentions usage limits. Any other terminal error keeps the
    stock ``"continue"`` behavior — return a string that becomes the
    ``AIMessage`` content — so an unrelated blip still doesn't hard-crash a
    turn, plus an error toast so the failure isn't buried in the reply text.
    """
    if _is_out_of_credits(exc):
        notify_toast(
            "OpenRouter: not enough credits for this request — add credits or "
            "raise the key's daily limit, then /continue.",
            severity="error",
            title="Out of credits",
            timeout=10.0,
        )
        raise UsageLimitError(exc)
    if _is_usage_limit(exc):
        raise UsageLimitError(exc)
    notify_toast(
        f"Model call failed: {type(exc).__name__}. See reply for details.",
        severity="error",
        title="Model error",
        timeout=10.0,
    )
    return f"Model call failed after retries: {exc}"


# Deterministic configuration errors nothing downstream can route around: a bad
# API key, a nonexistent model id, or a permission denial fails every subsequent
# call identically, so "continuing" on a synthetic reply would only cascade
# fabricated messages through the rest of the run. These propagate raw (per
# upstream's rationale in langchain#38960) — loudly halting a turn that is still
# checkpointed and resumable. Deliberately absent: ModelInvalidRequestError and
# unclassified errors, which are often content-specific (one sub-agent's payload
# drawing a 400) — there the coordinator can route around the failure, and
# halting an hours-long autonomous run over it is the worse trade.
_CONFIG_ERRORS = (
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)


class ModelErrorBoundaryMiddleware(AgentMiddleware):
    """Route non-retryable model errors through the terminal-error policy.

    langchain 1.3.16 changed ``ModelRetryMiddleware``: exceptions not matched
    by ``retry_on`` are re-raised immediately instead of being handed to
    ``on_failure`` (see langchain#38960). Left alone, that raw raise loses the
    402 out-of-credits toast/halt and the synthetic-error-reply behavior this
    project relies on. This boundary wraps the retry layer — the position the
    ``on_failure`` hook used to occupy, inside the fallback layer — and sorts
    what it re-raises three ways:

    - credit/usage limits: :class:`UsageLimitError`, so the TUI pauses the run
      at a resumable checkpoint (and, configured, the fallback models are
      still tried first);
    - deterministic config errors (``_CONFIG_ERRORS``): re-raised untouched —
      nothing can route around a bad key, so fail loudly rather than feed the
      agent fabricated replies;
    - everything else: the ``on_failure`` "continue" policy
      (:func:`_on_model_retries_exhausted`) — toast plus synthetic reply, so a
      content-specific rejection in one sub-agent doesn't kill the whole run.

    ``UsageLimitError`` (already the product of this policy, raised by the
    retry layer's own ``on_failure``) and langgraph control-flow exceptions
    pass through untouched.
    """

    def _handle(self, exc: Exception) -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content=_on_model_retries_exhausted(exc))]
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            return handler(request)
        except (GraphBubbleUp, UsageLimitError, *_CONFIG_ERRORS):
            raise
        except Exception as exc:
            return self._handle(exc)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return await handler(request)
        except (GraphBubbleUp, UsageLimitError, *_CONFIG_ERRORS):
            raise
        except Exception as exc:
            return self._handle(exc)


class ConcurrencyLimitRetryMiddleware(AgentMiddleware):
    """Wait out a concurrency-limit 429 instead of pausing the run over it.

    The stock retry layer's budget (~7s over three retries) is far shorter than
    one in-flight model call (tens of seconds), so a workspace-wide concurrency
    limit routinely outlasted it and escalated to :class:`UsageLimitError` — a
    full pause waiting on ``/continue`` for a condition that clears on its own.
    This layer sits inside the retry layer and retries only concurrency limits,
    with capped exponential backoff and jitter (so parallel sub-agents don't
    retry in lockstep), until ``max_wait`` seconds have elapsed. Then it
    re-raises, and :func:`_retry_on` sends the error straight to the boundary,
    which pauses the run as before.
    """

    def __init__(
        self,
        max_wait: float,
        *,
        initial_delay: float = 2.0,
        max_delay: float = 30.0,
    ) -> None:
        super().__init__()
        self.max_wait = max_wait
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def _next_delay(self, exc: Exception, attempt: int, started: float) -> float:
        """Seconds to sleep before the next attempt; re-raise ``exc`` when spent."""
        if not _is_concurrency_limit(exc):
            raise exc
        delay = min(self.initial_delay * 2.0**attempt, self.max_delay)
        delay *= random.uniform(0.75, 1.25)
        remaining = self.max_wait - (time.monotonic() - started)
        if remaining <= 0:
            raise exc
        if attempt == 0:
            notify_toast(
                "Provider concurrency limit reached — waiting for a free slot "
                f"(up to {self.max_wait:.0f}s) before retrying.",
                severity="warning",
                title="Concurrency limit",
                timeout=8.0,
            )
        return min(delay, remaining)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                time.sleep(self._next_delay(exc, attempt, started))
                attempt += 1

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                await asyncio.sleep(self._next_delay(exc, attempt, started))
                attempt += 1


# How often a coroutine waiting on the model-call limiter re-checks for a slot.
# Polling (rather than ``to_thread(sem.acquire)``) keeps the wait cancellable
# without leaking a slot to an orphaned acquiring thread.
_LIMITER_POLL_SECONDS = 0.05


class ModelConcurrencyLimitMiddleware(AgentMiddleware):
    """Cap in-flight model calls across every agent sharing ``semaphore``.

    The main agent and its sub-agents each build their own middleware stack,
    so the cap only means something when they share one semaphore — see
    :func:`_shared_model_semaphore`. A thread semaphore (not an asyncio one)
    because sync and async callers, and possibly different event loops, draw
    on the same budget. Placed innermost, so a call waiting out a backoff in
    the retry layers does not hold a slot.
    """

    def __init__(self, semaphore: threading.Semaphore) -> None:
        super().__init__()
        self.semaphore = semaphore

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        with self.semaphore:
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        while not self.semaphore.acquire(blocking=False):
            await asyncio.sleep(_LIMITER_POLL_SECONDS)
        try:
            return await handler(request)
        finally:
            self.semaphore.release()


_model_semaphores: dict[int, threading.BoundedSemaphore] = {}
_model_semaphores_lock = threading.Lock()


def _shared_model_semaphore(limit: int) -> threading.BoundedSemaphore:
    """The process-wide semaphore for ``limit`` concurrent model calls."""
    with _model_semaphores_lock:
        if limit not in _model_semaphores:
            _model_semaphores[limit] = threading.BoundedSemaphore(limit)
        return _model_semaphores[limit]


def is_truncated_message(msg: BaseMessage) -> bool:
    """Was this model response cut off at the output-token limit?

    Anthropic-protocol providers set ``stop_reason: "max_tokens"``; OpenAI-shaped
    ones (OpenRouter, DeepSeek) set ``finish_reason: "length"``. Both land in
    ``response_metadata`` (langchain-anthropic sets it on streaming and
    non-streaming paths alike).
    """
    if not isinstance(msg, AIMessage):
        return False
    meta = msg.response_metadata or {}
    return meta.get("stop_reason") == "max_tokens" or meta.get("finish_reason") in (
        "length",
        "max_tokens",
    )


_TRUNCATION_NUDGE = (
    "[automated] Your previous response was cut off by the output-token limit "
    "before its tool calls could run. Nothing you announced was executed. "
    "Continue where you left off: re-issue the tool calls you intended (keep "
    "each payload concise; split large saves into several smaller ones), then "
    "finish with your final report."
)


def is_truncation_nudge(msg: BaseMessage) -> bool:
    """True for the automated ``HumanMessage`` this middleware appends."""
    return isinstance(msg, HumanMessage) and msg.content == _TRUNCATION_NUDGE


# Truncation-recovery attempts per turn. Two failed continuations mean the model
# keeps overrunning the cap; further retries would loop, so fall through and let
# the report/reply guards salvage what exists.
_MAX_TRUNCATION_RECOVERIES = 2


class TruncationRecoveryMiddleware(AgentMiddleware):
    """Resume the loop when a response is truncated instead of ending the run.

    A response cut off at ``max_tokens`` *before* a complete ``tool_use`` block
    parses leaves an ``AIMessage`` with no ``tool_calls`` — the agent loop then
    routes to END, silently dropping whatever the model was about to do (the
    observed failure: "Now let me save the key findings…" and the save never
    ran). This hook detects that dead end, appends a corrective ``HumanMessage``,
    and jumps back to the model node. Bounded per turn by counting its own nudge
    messages, so a model that keeps overrunning the cap cannot loop forever.
    """

    @hook_config(can_jump_to=["model"])
    def after_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if (
            not isinstance(last, AIMessage)
            or last.tool_calls
            or not is_truncated_message(last)
        ):
            return None
        if self._recoveries_this_turn(messages) >= _MAX_TRUNCATION_RECOVERIES:
            return None

        updates: list[BaseMessage] = []
        if last.invalid_tool_calls:
            # Resubmitting a half-parsed tool_use block can 400 on the provider;
            # replace the message (same id -> add_messages overwrites) with a
            # text-only copy before continuing.
            updates.append(AIMessage(content=last.text or "", id=last.id))
        updates.append(HumanMessage(content=_TRUNCATION_NUDGE))
        return {"jump_to": "model", "messages": updates}

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    @staticmethod
    def _recoveries_this_turn(messages: Sequence[BaseMessage]) -> int:
        """Count nudges since the last real (non-nudge) HumanMessage."""
        count = 0
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                if is_truncation_nudge(msg):
                    count += 1
                else:
                    break
        return count


def _fallback_specs() -> list[str]:
    raw = os.environ.get("MODEL_FALLBACK", "")
    return [spec.strip() for spec in raw.split(",") if spec.strip()]


def build_model_resilience_middleware(
    resolve_model: ModelResolver,
) -> list[AgentMiddleware]:
    """Model-call resilience, outermost first: fallback > error boundary >
    retry > concurrency-limit wait > concurrency cap.

    Fallback is listed first so it is the outermost wrapper: the primary model
    is retried on transient errors first, and only when the terminal-error
    policy raises (a usage/credit limit) does the call fall back to the next
    configured model (which is then itself retried). The error boundary between
    them applies that policy to non-retryable errors, which the retry layer
    re-raises rather than routing to ``on_failure`` since langchain 1.3.16.
    Concurrency-limit 429s are waited out below the retry layer, and the
    optional cap sits innermost so only calls actually in flight hold a slot.
    Omits the fallback layer when ``MODEL_FALLBACK`` is unset, and the cap when
    ``MODEL_MAX_CONCURRENCY`` is.
    """
    max_retries = env_int("MODEL_MAX_RETRIES", 3)
    concurrency_wait = env_float("MODEL_CONCURRENCY_WAIT", 300.0)
    max_concurrency = env_int("MODEL_MAX_CONCURRENCY", 0)
    middleware: list[AgentMiddleware] = []

    fallbacks = _fallback_specs()
    if fallbacks:
        resolved = [resolve_model(spec) for spec in fallbacks]
        middleware.append(ModelFallbackMiddleware(resolved[0], *resolved[1:]))

    # Between fallback and retry: catches the non-retryable errors the retry
    # layer re-raises (langchain ≥1.3.16) and applies the terminal-error policy.
    middleware.append(ModelErrorBoundaryMiddleware())
    middleware.append(
        ModelRetryMiddleware(
            max_retries=max_retries,
            retry_on=_retry_on,
            on_failure=_on_model_retries_exhausted,
        )
    )
    middleware.append(ConcurrencyLimitRetryMiddleware(concurrency_wait))
    if max_concurrency:
        middleware.append(
            ModelConcurrencyLimitMiddleware(_shared_model_semaphore(max_concurrency))
        )
    # Response-level (not exception-level) recovery: a max_tokens-truncated
    # response is a successful HTTP call the retry/fallback layers never see.
    middleware.append(TruncationRecoveryMiddleware())
    return middleware


def build_tool_retry_middleware() -> ToolRetryMiddleware:
    """Retry transient filesystem-tool I/O errors, scoped to idempotent tools."""
    max_retries = env_int("TOOL_MAX_RETRIES", 3)
    return ToolRetryMiddleware(
        max_retries=max_retries,
        tools=list(_RETRYABLE_FS_TOOLS),
        retry_on=(OSError,),
        on_failure="continue",
    )
