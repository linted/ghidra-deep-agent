"""The TUI's slash commands, declared once.

Each command used to be written down three times — the ``if/elif`` chain in
``app._dispatch_slash``, the autocomplete list in ``widgets/command_input.py``,
and the help text in ``help_screen.py`` — with nothing keeping them in sync. A
command could be dispatchable but absent from autocomplete, or documented but
gone. This table is the one place a command is defined; the other three derive
from it.

The handler bodies stay on the app (they need its widgets and state); what lives
here is the name, what it takes, what it does, and whether it may start a run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    """One slash command's user-facing contract."""

    name: str
    help: str
    # Shown after the name in help/autocomplete, e.g. "[goal]". Empty for commands
    # that take no argument.
    arg_hint: str = ""
    # Whether the command starts (or resumes) an agent run, and so must be
    # refused while one is already in flight. Centralized because it was
    # previously copy-pasted into seven branches, and a new command that forgot
    # it would silently interleave two runs on one thread.
    needs_idle: bool = False

    @property
    def usage(self) -> str:
        return f"{self.name} {self.arg_hint}".strip()


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/clear", "clear the response log and activity tree"),
    SlashCommand("/yank", "copy the last response to the clipboard"),
    SlashCommand("/compact", "compact the conversation history", needs_idle=True),
    SlashCommand("/resume", "list & resume a previous session", needs_idle=True),
    SlashCommand(
        "/continue",
        "continue after a usage-limit pause (main, plan, or ask mode)",
        needs_idle=True,
    ),
    SlashCommand(
        "/plan",
        "enter read-only plan mode (investigate & draft a plan)",
        arg_hint="[goal]",
        needs_idle=True,
    ),
    SlashCommand(
        "/approve", "approve the current plan and execute it", needs_idle=True
    ),
    SlashCommand("/plan-cancel", "leave plan mode without executing"),
    SlashCommand(
        "/ask",
        "enter read-only ask mode (answer questions, no changes)",
        arg_hint="[question(s)]",
        needs_idle=True,
    ),
    SlashCommand("/ask-cancel", "leave ask mode"),
    SlashCommand("/help", "show this help"),
    SlashCommand("/quit", "exit"),
)

COMMANDS_BY_NAME: dict[str, SlashCommand] = {c.name: c for c in COMMANDS}

# Autocomplete source. Sorted so the suggestion order is stable and predictable.
SLASH_COMMAND_NAMES: list[str] = sorted(c.name for c in COMMANDS)


def help_lines() -> list[str]:
    """Rendered ``usage — help`` lines, column-aligned, for the help screen."""
    width = max(len(c.usage) for c in COMMANDS)
    return [f"  {c.usage.ljust(width)}  {c.help}" for c in COMMANDS]
