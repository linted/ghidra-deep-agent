from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from ghidra_deep_agent.tui.messages import TextToken


class ThinkingPanel(VerticalScroll):
    """Ephemeral strip that shows live-streaming LLM tokens while agent runs."""

    def compose(self) -> ComposeResult:
        yield Static("", id="thinking-text", markup=False)

    def on_mount(self) -> None:
        self.border_title = "thinking"
        self._buf = ""

    def reset(self) -> None:
        self._buf = ""
        self.query_one("#thinking-text", Static).update("")
        self.display = True

    def on_text_token(self, msg: TextToken) -> None:
        self._buf += msg.text
        self.query_one("#thinking-text", Static).update(self._buf[-3000:])
        self.scroll_end(animate=False)
