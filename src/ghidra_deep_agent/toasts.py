"""Decoupled pub/sub bus for toast notifications.

Lets in-process code (MCP tool wrappers, middleware, knowledge-base helpers,
tool interceptors, ...) trigger a toast in a running TUI without holding a
reference to the `App` or any widget — analogous to how `logging` decouples
emitters from handlers.

Two delivery paths:

- **Global sinks** (:func:`register_toast_sink`): process-wide, what the TUI
  uses — one app, one run at a time.
- **A scoped sink** (:func:`toast_scope`): bound to the current task context.
  The HTTP server runs many agents concurrently in one process, so a warning
  raised inside one run must reach only that run's subscribers. The scope is a
  ``ContextVar``, which ``asyncio`` tasks, ``asyncio.to_thread`` and LangChain's
  executor hops all copy, so the emitters need no changes. A scoped sink takes
  precedence and *suppresses* global delivery: a server run must not spray the
  process-wide sinks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

# Mirrors textual.notifications.SeverityLevel without importing Textual, so the
# server (which never loads the TUI) can use the bus too.
SeverityLevel = Literal["information", "warning", "error"]

_sinks: list[Callable[[ToastRequest], None]] = []
_scoped_sink: ContextVar[Callable[[ToastRequest], None] | None] = ContextVar(
    "toast_sink", default=None
)


@dataclass(frozen=True)
class ToastRequest:
    message: str
    severity: SeverityLevel = "information"
    title: str = ""
    timeout: float | None = None


def register_toast_sink(sink: Callable[[ToastRequest], None]) -> Callable[[], None]:
    """Register a global sink for toast requests; returns an unregister callable."""
    _sinks.append(sink)

    def unregister() -> None:
        _sinks.remove(sink)

    return unregister


@contextmanager
def toast_scope(sink: Callable[[ToastRequest], None]) -> Iterator[None]:
    """Route toasts raised in this context (and tasks spawned from it) to ``sink``."""
    token = _scoped_sink.set(sink)
    try:
        yield
    finally:
        _scoped_sink.reset(token)


def notify_toast(
    message: str,
    *,
    severity: SeverityLevel = "information",
    title: str = "",
    timeout: float | None = None,
) -> None:
    """Dispatch a toast request to the scoped sink, else every global sink."""
    toast = ToastRequest(message, severity=severity, title=title, timeout=timeout)
    scoped = _scoped_sink.get()
    if scoped is not None:
        scoped(toast)
        return
    for sink in _sinks:
        sink(toast)
