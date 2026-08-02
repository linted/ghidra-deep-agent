"""
Unit tests for model-retry exhaustion classification.

Focus: an exhausted usage/rate limit must halt cleanly (raise UsageLimitError)
so the run stays resumable, while any other exhausted error keeps the stock
"continue" behavior (return a string that becomes the AIMessage content).

Run:  uv run pytest tests/test_resilience.py -v
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ghidra_deep_agent.resilience import (
    _MAX_TRUNCATION_RECOVERIES,
    _TRUNCATION_NUDGE,
    TruncationRecoveryMiddleware,
    UsageLimitError,
    _is_out_of_credits,
    _is_transient,
    _is_usage_limit,
    _on_model_retries_exhausted,
    build_model_resilience_middleware,
    is_truncated_message,
)
from ghidra_deep_agent.toasts import ToastRequest


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
        "you have exceeded your monthly quota",
        "usage limit reached for this window",
    ],
)
def test_limit_markers_are_usage_limits(text: str) -> None:
    assert _is_usage_limit(Exception(text)) is True


def test_overloaded_is_transient_not_a_usage_limit() -> None:
    """Provider-at-capacity is a blip, not a quota block.

    It must stay retryable *and* must not escalate to UsageLimitError when the
    retries are spent — that would halt the turn for a condition that typically
    clears in seconds.
    """
    exc = Exception("model is overloaded, please retry")
    assert _is_transient(exc) is True
    assert _is_usage_limit(exc) is False


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


def test_on_failure_toasts_and_pauses_on_credits_error(
    captured_toasts: list[ToastRequest],
) -> None:
    received = captured_toasts

    original = _StatusError(
        "This request requires more credits, or fewer max_tokens.", 402
    )
    with pytest.raises(UsageLimitError) as excinfo:
        _on_model_retries_exhausted(original)

    assert excinfo.value.original is original
    assert len(received) == 1
    assert received[0].severity == "error"
    assert "credits" in received[0].message


def test_on_failure_toasts_on_generic_terminal_error(
    captured_toasts: list[ToastRequest],
) -> None:
    received = captured_toasts

    result = _on_model_retries_exhausted(Exception("connection reset by peer"))

    assert isinstance(result, str)
    assert len(received) == 1
    assert received[0].severity == "error"
    # The toast stays concise: exception type only, raw text lives in the reply.
    assert "connection reset by peer" not in received[0].message


def test_on_failure_does_not_toast_on_usage_limit(
    captured_toasts: list[ToastRequest],
) -> None:
    # The TUI already renders a dedicated pause banner for usage limits; a toast
    # there would be duplicate noise.
    received = captured_toasts

    with pytest.raises(UsageLimitError):
        _on_model_retries_exhausted(_StatusError("rate limit exceeded", 429))

    assert received == []


# --- truncation recovery --------------------------------------------------------


def _truncated_ai(
    text: str = "Now let me save the key findings to the knowledge base.",
    *,
    stop_reason: str | None = "max_tokens",
    **kwargs: object,
) -> AIMessage:
    meta = {"stop_reason": stop_reason} if stop_reason else {}
    return AIMessage(text, response_metadata=meta, **kwargs)


def test_max_tokens_stop_reason_is_truncated() -> None:
    assert is_truncated_message(_truncated_ai())
    assert is_truncated_message(
        AIMessage("x", response_metadata={"finish_reason": "length"})
    )


def test_normal_stop_reasons_are_not_truncated() -> None:
    assert not is_truncated_message(
        AIMessage("done", response_metadata={"stop_reason": "end_turn"})
    )
    assert not is_truncated_message(AIMessage("no metadata at all"))
    assert not is_truncated_message(HumanMessage("not an AIMessage"))


def _after_model(messages: list[object]) -> dict[str, Any] | None:
    return TruncationRecoveryMiddleware().after_model(
        {"messages": messages},  # type: ignore[typeddict-item]
        None,  # type: ignore[arg-type]  # runtime is unused
    )


def test_truncated_dead_end_jumps_back_to_the_model() -> None:
    # The observed failure: preamble text, tool_use cut off before it parsed,
    # no tool_calls -> the loop would route to END and the save never runs.
    update = _after_model([HumanMessage("analyze"), _truncated_ai()])
    assert update is not None
    assert update["jump_to"] == "model"
    (nudge,) = update["messages"]
    assert isinstance(nudge, HumanMessage)
    assert nudge.content == _TRUNCATION_NUDGE


def test_truncated_but_parsed_tool_calls_continue_normally() -> None:
    # A complete tool_use block parsed before the cut: the loop proceeds to the
    # tools node on its own; jumping would double-execute.
    msg = _truncated_ai(
        tool_calls=[
            {"name": "save_knowledge", "args": {}, "id": "c1", "type": "tool_call"}
        ]
    )
    assert _after_model([HumanMessage("analyze"), msg]) is None


def test_untruncated_response_is_untouched() -> None:
    assert _after_model([HumanMessage("analyze"), AIMessage("## Report")]) is None


def test_recovery_is_bounded_per_turn() -> None:
    messages: list[object] = [HumanMessage("analyze")]
    for _ in range(_MAX_TRUNCATION_RECOVERIES):
        messages += [_truncated_ai(), HumanMessage(_TRUNCATION_NUDGE)]
    messages.append(_truncated_ai())
    assert _after_model(messages) is None


def test_a_new_turn_resets_the_recovery_budget() -> None:
    messages: list[object] = [HumanMessage("first task")]
    for _ in range(_MAX_TRUNCATION_RECOVERIES):
        messages += [_truncated_ai(), HumanMessage(_TRUNCATION_NUDGE)]
    messages += [HumanMessage("second task"), _truncated_ai()]
    assert _after_model(messages) is not None


def test_invalid_tool_calls_are_replaced_not_resubmitted() -> None:
    msg = _truncated_ai(
        id="msg-1",
        invalid_tool_calls=[
            {
                "name": "save_knowledge",
                "args": '{"content": "half a',
                "id": "c1",
                "error": None,
                "type": "invalid_tool_call",
            }
        ],
    )
    update = _after_model([HumanMessage("analyze"), msg])
    assert update is not None
    replacement, nudge = update["messages"]
    assert isinstance(replacement, AIMessage)
    assert replacement.id == "msg-1"  # same id -> add_messages overwrites in place
    assert not replacement.invalid_tool_calls
    assert nudge.content == _TRUNCATION_NUDGE


def test_truncation_recovery_is_wired_into_the_resilience_stack() -> None:
    middleware = build_model_resilience_middleware(lambda spec: spec or "m")
    assert any(isinstance(m, TruncationRecoveryMiddleware) for m in middleware)
