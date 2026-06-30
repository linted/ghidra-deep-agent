from __future__ import annotations

from typing import Any

from textual.binding import Binding
from textual.suggester import SuggestFromList
from textual.widgets import Input

PLACEHOLDER_IDLE = "Enter task (Ctrl+C quit · /help for commands)…"
PLACEHOLDER_BUSY = "Agent is running… type ahead, Enter to queue"

SLASH_COMMANDS = ["/clear", "/compact", "/help", "/quit", "/resume", "/yank"]


class CommandInput(Input):
    """Single-line input with Up/Down history walking and slash autocomplete."""

    BINDINGS = [
        Binding("up", "history_up", show=False, priority=True),
        Binding("down", "history_down", show=False, priority=True),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "suggester", SuggestFromList(SLASH_COMMANDS, case_sensitive=False)
        )
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_idx: int | None = None
        self._draft: str = ""

    def add_to_history(self, query: str) -> None:
        if not query.strip():
            return
        if self._history and self._history[-1] == query:
            return
        self._history.append(query)
        self._history_idx = None
        self._draft = ""

    def action_history_up(self) -> None:
        if not self._history:
            return
        if self._history_idx is None:
            self._draft = self.value
            self._history_idx = len(self._history) - 1
        elif self._history_idx > 0:
            self._history_idx -= 1
        else:
            return
        self.value = self._history[self._history_idx]
        self.cursor_position = len(self.value)

    def action_history_down(self) -> None:
        if self._history_idx is None:
            return
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self.value = self._history[self._history_idx]
        else:
            self._history_idx = None
            self.value = self._draft
        self.cursor_position = len(self.value)
