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
"""

import os
from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

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
# structured messages (see ``handle_mcp_errors`` in main.py), and many are not
# idempotent.
_RETRYABLE_FS_TOOLS = ("write_file", "edit_file", "read_file")

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
_USAGE_LIMIT_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "overloaded",
    "quota",
    "usage limit",
    "insufficient_quota",
)


def _is_transient(exc: BaseException) -> bool:
    """Heuristic: would retrying this error plausibly succeed?

    True for rate limits, timeouts, connection resets, and 5xx responses; False
    for deterministic failures (bad request, auth, schema/validation), which
    would only waste time and money on retry.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _is_usage_limit(exc: BaseException) -> bool:
    """Would waiting hours (not seconds) be the only thing that clears this?

    True for provider rate/usage/quota limits (429, or a limit marker in the
    text). Provider-agnostic: relies on status code + text, never on a
    provider-specific ``retry-after`` header (inconsistent across
    Anthropic/OpenRouter/DeepSeek, absent on Ollama).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _USAGE_LIMIT_MARKERS)


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
    """`on_failure` for ``ModelRetryMiddleware``: halt on a limit, else continue.

    Called both when retries are exhausted and immediately for non-retryable
    errors (``ModelRetryMiddleware`` skips the retry loop for those, e.g. a 402).

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


def _fallback_specs() -> list[str]:
    raw = os.environ.get("MODEL_FALLBACK", "")
    return [spec.strip() for spec in raw.split(",") if spec.strip()]


def build_model_resilience_middleware(
    resolve_model: ModelResolver,
) -> list[AgentMiddleware]:
    """Model-call resilience: optional provider fallback (outer) + retry (inner).

    Fallback is listed first so it is the outermost wrapper: the primary model
    is retried on transient errors first, and only if those retries are
    exhausted does the call fall back to the next configured model (which is then
    itself retried). Returns an empty-fallback list when ``MODEL_FALLBACK`` is
    unset.
    """
    max_retries = int(os.environ.get("MODEL_MAX_RETRIES", "3"))
    middleware: list[AgentMiddleware] = []

    fallbacks = _fallback_specs()
    if fallbacks:
        resolved = [resolve_model(spec) for spec in fallbacks]
        middleware.append(ModelFallbackMiddleware(resolved[0], *resolved[1:]))

    middleware.append(
        ModelRetryMiddleware(
            max_retries=max_retries,
            retry_on=_is_transient,
            on_failure=_on_model_retries_exhausted,
        )
    )
    return middleware


def build_tool_retry_middleware() -> ToolRetryMiddleware:
    """Retry transient filesystem-tool I/O errors, scoped to idempotent tools."""
    max_retries = int(os.environ.get("TOOL_MAX_RETRIES", "3"))
    return ToolRetryMiddleware(
        max_retries=max_retries,
        tools=list(_RETRYABLE_FS_TOOLS),
        retry_on=(OSError,),
        on_failure="continue",
    )
