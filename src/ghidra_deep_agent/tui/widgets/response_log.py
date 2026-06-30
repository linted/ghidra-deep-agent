from __future__ import annotations

from rich.markdown import Markdown
from rich.rule import Rule
from textual.widgets import RichLog

from ghidra_deep_agent.tui.messages import AgentDone, ResponseFinal


class ResponseLog(RichLog):
    """Right pane: buffered markdown response."""

    def on_mount(self) -> None:
        self._response_buf = ""
        self.last_response = ""
        self.transcript: list[str] = []

    def clear(self) -> ResponseLog:
        self._response_buf = ""
        self.transcript = []
        return super().clear()

    def log_user(self, text: str) -> None:
        self.transcript.append(f"❯ {text}")
        self.write(Rule(style="dim cyan"))
        shown = text.replace("\n", "\n  ")
        self.write(f"[bold cyan]❯ {shown}[/bold cyan]")
        self.write(Rule(style="dim cyan"))

    def on_response_final(self, msg: ResponseFinal) -> None:
        self._response_buf = msg.text

    def on_agent_done(self, _msg: AgentDone) -> None:
        if self._response_buf:
            self.last_response = self._response_buf
            self.transcript.append(self._response_buf)
            self.write(Rule(style="dim green"))
            self.write("[bold green]✦ assistant[/bold green]")
            self.write(Rule(style="dim green"))
            self.write(Markdown(self._response_buf))
            self._response_buf = ""
