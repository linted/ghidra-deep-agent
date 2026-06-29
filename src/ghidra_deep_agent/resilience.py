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

ModelResolver = Callable[[str | None], str | BaseChatModel]

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
            on_failure="continue",
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
