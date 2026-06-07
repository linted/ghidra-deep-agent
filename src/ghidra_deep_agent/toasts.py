"""Decoupled pub/sub bus for toast notifications.

Lets in-process code (MCP tool wrappers, middleware, knowledge-base helpers,
tool interceptors, ...) trigger a toast in a running TUI without holding a
reference to the `App` or any widget — analogous to how `logging` decouples
emitters from handlers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.notifications import SeverityLevel

_sinks: list[Callable[[ToastRequest], None]] = []


@dataclass(frozen=True)
class ToastRequest:
    message: str
    severity: SeverityLevel = "information"
    title: str = ""
    timeout: float | None = None


def register_toast_sink(sink: Callable[[ToastRequest], None]) -> Callable[[], None]:
    """Register a sink to receive toast requests; returns an unregister callable."""
    _sinks.append(sink)

    def unregister() -> None:
        _sinks.remove(sink)

    return unregister


def notify_toast(
    message: str,
    *,
    severity: SeverityLevel = "information",
    title: str = "",
    timeout: float | None = None,
) -> None:
    """Dispatch a toast request to every registered sink (a no-op if none)."""
    toast = ToastRequest(message, severity=severity, title=title, timeout=timeout)
    for sink in _sinks:
        sink(toast)
