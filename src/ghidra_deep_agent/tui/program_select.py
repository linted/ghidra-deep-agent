"""Program selection screen (shown when multiple programs are open in Ghidra)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView


class ProgramSelectApp(App[str]):
    TITLE = "Ghidra Agent"
    BINDINGS = [Binding("ctrl+c", "quit", "Quit")]
    CSS = """
    Screen {
        align: center middle;
    }
    #select-container {
        width: 60;
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }
    #select-label {
        margin-bottom: 1;
        text-style: bold;
    }
    """

    def __init__(self, programs: list[str]) -> None:
        super().__init__()
        self._programs = programs

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="select-container"):
            yield Label(
                "Multiple programs are open. Select one to analyze:", id="select-label"
            )
            yield ListView(
                *[ListItem(Label(name), name=name) for name in self._programs]
            )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Program Selection"
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.exit(event.item.name or "")
