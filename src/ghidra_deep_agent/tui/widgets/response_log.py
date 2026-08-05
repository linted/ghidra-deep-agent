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

    def log_plan(self, text: str) -> None:
        """Render the authoritative current plan read back from disk/state."""
        self.transcript.append(text)
        self.write(Rule(style="dim magenta"))
        self.write("[bold magenta]📋 Current plan[/bold magenta]")
        self.write(Rule(style="dim magenta"))
        self.write(Markdown(text))

    def log_user(self, text: str) -> None:
        self.transcript.append(f"❯ {text}")
        self.write(Rule(style="dim cyan"))
        shown = text.replace("\n", "\n  ")
        self.write(f"[bold cyan]❯ {shown}[/bold cyan]")
        self.write(Rule(style="dim cyan"))

    def on_response_final(self, msg: ResponseFinal) -> None:
        self._response_buf = msg.text

    def log_assistant(self, text: str) -> None:
        """Render an assistant reply (live turn or replayed from a checkpoint)."""
        if not text:
            return
        self.last_response = text
        self.transcript.append(text)
        self.write(Rule(style="dim green"))
        self.write("[bold green]✦ assistant[/bold green]")
        self.write(Rule(style="dim green"))
        self.write(Markdown(text))

    def on_agent_done(self, _msg: AgentDone) -> None:
        if self._response_buf:
            self.log_assistant(self._response_buf)
            self._response_buf = ""
