"""
Unit tests for model-retry exhaustion classification.

Focus: an exhausted usage/rate limit must halt cleanly (raise UsageLimitError)
so the run stays resumable, while any other exhausted error keeps the stock
"continue" behavior (return a string that becomes the AIMessage content).

Run:  uv run pytest test_resilience.py -v
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ghidra_deep_agent.resilience import (
    UsageLimitError,
    _is_out_of_credits,
    _is_usage_limit,
    _on_model_retries_exhausted,
)
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink


@pytest.fixture(autouse=True)
def _clear_sinks() -> Generator[None, None, None]:
    """Toast sinks are module-global; reset between tests to avoid cross-talk."""
    import ghidra_deep_agent.toasts as toasts

    toasts._sinks.clear()
    yield
    toasts._sinks.clear()


class _StatusError(Exception):
    """Mimic a provider SDK error exposing an HTTP ``status_code``."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_429_status_is_a_usage_limit() -> None:
    assert _is_usage_limit(_StatusError("slow down", 429)) is True


@pytest.mark.parametrize(
    "text",
    [
        "Rate limit exceeded",
        "429 too many requests",
        "model is overloaded, please retry",
        "you have exceeded your monthly quota",
        "usage limit reached for this window",
    ],
)
def test_limit_markers_are_usage_limits(text: str) -> None:
    assert _is_usage_limit(Exception(text)) is True


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError("bad gateway", 502),
        Exception("connection reset by peer"),
        Exception("request timed out"),
        Exception("invalid api key"),
    ],
)
def test_non_limit_errors_are_not_usage_limits(exc: Exception) -> None:
    assert _is_usage_limit(exc) is False


def test_on_failure_raises_on_usage_limit() -> None:
    original = _StatusError("rate limit exceeded", 429)
    with pytest.raises(UsageLimitError) as excinfo:
        _on_model_retries_exhausted(original)
    # The original exception is preserved for debugging.
    assert excinfo.value.original is original


def test_on_failure_continues_on_other_errors() -> None:
    # A non-limit exhausted error keeps the stock "continue" behavior: return a
    # string (the AIMessage content) rather than raising.
    result = _on_model_retries_exhausted(Exception("connection reset by peer"))
    assert isinstance(result, str)
    assert "connection reset by peer" in result


def test_402_status_is_out_of_credits() -> None:
    assert _is_out_of_credits(_StatusError("payment required", 402)) is True


def test_credits_marker_is_out_of_credits() -> None:
    exc = Exception(
        "This request requires more credits, or fewer max_tokens. "
        "You requested up to 65536 tokens, but can only afford 63176."
    )
    assert _is_out_of_credits(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError("bad gateway", 502),
        Exception("connection reset by peer"),
        Exception("invalid api key"),
    ],
)
def test_non_credit_errors_are_not_out_of_credits(exc: Exception) -> None:
    assert _is_out_of_credits(exc) is False


def test_on_failure_toasts_and_pauses_on_credits_error() -> None:
    received: list[ToastRequest] = []
    register_toast_sink(received.append)

    original = _StatusError(
        "This request requires more credits, or fewer max_tokens.", 402
    )
    with pytest.raises(UsageLimitError) as excinfo:
        _on_model_retries_exhausted(original)

    assert excinfo.value.original is original
    assert len(received) == 1
    assert received[0].severity == "error"
    assert "credits" in received[0].message


def test_on_failure_toasts_on_generic_terminal_error() -> None:
    received: list[ToastRequest] = []
    register_toast_sink(received.append)

    result = _on_model_retries_exhausted(Exception("connection reset by peer"))

    assert isinstance(result, str)
    assert len(received) == 1
    assert received[0].severity == "error"
    # The toast stays concise: exception type only, raw text lives in the reply.
    assert "connection reset by peer" not in received[0].message


def test_on_failure_does_not_toast_on_usage_limit() -> None:
    # The TUI already renders a dedicated pause banner for usage limits; a toast
    # there would be duplicate noise.
    received: list[ToastRequest] = []
    register_toast_sink(received.append)

    with pytest.raises(UsageLimitError):
        _on_model_retries_exhausted(_StatusError("rate limit exceeded", 429))

    assert received == []
