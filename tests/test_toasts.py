"""
Unit tests for the toast notification pub/sub bus.

Run:  uv run pytest test_toasts.py -v
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ghidra_deep_agent.toasts import ToastRequest, notify_toast, register_toast_sink


@pytest.fixture(autouse=True)
def _clear_sinks() -> Generator[None, None, None]:
    """Toast sinks are module-global; reset between tests to avoid cross-talk."""
    import ghidra_deep_agent.toasts as toasts

    toasts._sinks.clear()
    yield
    toasts._sinks.clear()


def test_notify_toast_with_no_sinks_is_a_noop() -> None:
    notify_toast("hello")  # must not raise


def test_register_toast_sink_receives_request() -> None:
    received: list[ToastRequest] = []
    register_toast_sink(received.append)

    notify_toast("hello", severity="warning", title="Heads up", timeout=5.0)

    assert received == [
        ToastRequest("hello", severity="warning", title="Heads up", timeout=5.0)
    ]


def test_notify_toast_uses_defaults() -> None:
    received: list[ToastRequest] = []
    register_toast_sink(received.append)

    notify_toast("hello")

    assert received == [ToastRequest("hello")]


def test_multiple_sinks_all_receive_request() -> None:
    received_a: list[ToastRequest] = []
    received_b: list[ToastRequest] = []
    register_toast_sink(received_a.append)
    register_toast_sink(received_b.append)

    notify_toast("hello")

    assert received_a == [ToastRequest("hello")]
    assert received_b == [ToastRequest("hello")]


def test_unregister_stops_further_delivery() -> None:
    received: list[ToastRequest] = []
    unregister = register_toast_sink(received.append)

    notify_toast("first")
    unregister()
    notify_toast("second")

    assert received == [ToastRequest("first")]
