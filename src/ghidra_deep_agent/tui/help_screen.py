from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from ghidra_deep_agent.tui.commands import help_lines

_KEYS_TEXT = """\
[bold]Keys[/bold]
  ↑ / ↓          walk input history
  Escape         cancel a running agent · close this help
  Ctrl+T         toggle the activity pane
  Ctrl+O         view sub-agent reports (Esc close · Ctrl+Y copy)
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
        # Command help is generated from the command table, so a command can
        # never be dispatchable but undocumented (or vice versa).
        commands = "\n".join(["[bold]Slash commands[/bold]", *help_lines()])
        with Vertical(id="help-box"):
            yield Static("Ghidra Agent — help", id="help-title")
            yield Static(f"{commands}\n\n{_KEYS_TEXT}")

    def action_close(self) -> None:
        self.dismiss()
