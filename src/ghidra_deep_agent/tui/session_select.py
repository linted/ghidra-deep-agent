"""Modal session picker for the ``/resume`` command.

Pushed onto the running TUI (unlike ``program_select.py``, which is a standalone
app shown before the TUI starts). Lists previous sessions most-recent-first and
dismisses with the chosen ``session_id`` (or ``None`` on cancel).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

# Async callable: ``fetch(show_all)`` -> list of session records.
FetchSessions = Callable[[bool], Awaitable[list[dict[str, Any]]]]


def _format_when(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value or "")


class SessionSelectScreen(ModalScreen[str | None]):
    """Pick a previous session to resume."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("a", "toggle_all", "All binaries / current"),
    ]
    CSS = """
    SessionSelectScreen {
        align: center middle;
    }
    #session-box {
        width: 90;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: solid $accent;
        padding: 1 2;
    }
    #session-label {
        margin-bottom: 1;
        text-style: bold;
    }
    """

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        current_session_id: str,
        binary_name: str,
        fetch: FetchSessions,
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._current = current_session_id
        self._binary = binary_name
        self._fetch = fetch
        self._show_all = False

    def compose(self) -> ComposeResult:
        with Vertical(id="session-box"):
            yield Label(self._heading(), id="session-label")
            yield ListView(*self._rows())

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    # --- rendering -------------------------------------------------------------

    def _heading(self) -> str:
        scope = "all binaries" if self._show_all else f"binary: {self._binary}"
        return (
            f"Resume a session ({scope})  —  "
            "Enter to resume · 'a' toggle scope · Esc cancel"
        )

    def _rows(self) -> list[ListItem]:
        rows: list[ListItem] = []
        for doc in self._sessions:
            sid = str(doc.get("session_id") or doc.get("_id") or "")
            when = _format_when(doc.get("last_active_at"))
            title = doc.get("title") or "(no title)"
            marker = "[green]●[/green] " if sid == self._current else "  "
            label = f"{marker}[dim]{when}[/dim]  {title}"
            if self._show_all:
                label += f"  [cyan]\\[{doc.get('binary_name', '?')}][/cyan]"
            rows.append(ListItem(Label(label, markup=True), name=sid))
        return rows

    async def _reload(self) -> None:
        self._sessions = await self._fetch(self._show_all)
        self.query_one("#session-label", Label).update(self._heading())
        list_view = self.query_one(ListView)
        await list_view.clear()
        for row in self._rows():
            await list_view.append(row)
        list_view.focus()

    # --- actions ---------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name or None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def action_toggle_all(self) -> None:
        self._show_all = not self._show_all
        await self._reload()
