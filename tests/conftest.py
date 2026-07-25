"""Fixtures shared across the suite.

The toast sink registry is module-global, so tests that register one leak into
whatever runs next unless it is cleared. Three test modules each carried a
byte-identical copy of the reset fixture; it lives here instead.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

import ghidra_deep_agent.toasts as toasts
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink


@pytest.fixture(autouse=True)
def _clear_sinks() -> Generator[None, None, None]:
    """Toast sinks are module-global; reset between tests to avoid cross-talk."""
    toasts._sinks.clear()
    yield
    toasts._sinks.clear()


@pytest.fixture
def captured_toasts() -> Generator[list[ToastRequest], None, None]:
    """Collect every toast raised during the test.

    Saves repeating `register_toast_sink(list.append)` at the top of each test
    that asserts on a warning — it appeared six times in one module alone.
    """
    seen: list[ToastRequest] = []
    unregister = register_toast_sink(seen.append)
    try:
        yield seen
    finally:
        unregister()
