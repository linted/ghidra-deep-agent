from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP_TEXT = """\
[bold]Slash commands[/bold]
  /clear    clear the response log and activity tree
  /yank     copy the last response to the clipboard
  /compact  compact the conversation history
  /resume   list & resume a previous session
  /plan [goal]  enter read-only plan mode (investigate & draft a plan)
  /approve  approve the current plan and execute it
  /plan-cancel  leave plan mode without executing
  /ask [question(s)]  enter read-only ask mode (answer questions, no changes)
  /ask-cancel  leave ask mode
  /help     show this help
  /quit     exit

[bold]Keys[/bold]
  ↑ / ↓          walk input history
  Escape         cancel a running agent · close this help
  Ctrl+T         toggle the activity pane
  Ctrl+Y         copy last response
  Ctrl+Shift+Y   copy full transcript
  Ctrl+L         clear log
  F1             show this help
  Ctrl+C         quit
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("f1", "close", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static("Ghidra Agent — help", id="help-title")
            yield Static(_HELP_TEXT)

    def action_close(self) -> None:
        self.dismiss()
