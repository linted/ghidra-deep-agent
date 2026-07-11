"""Textual messages passed between the agent event stream and the widgets."""

from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message


@dataclass(frozen=True)
class SubagentReport:
    """What one `task` run returned to the main agent."""

    run_id: str
    description: str
    text: str
    error: bool
    elapsed: float


class SubagentReportCaptured(Message):
    def __init__(self, report: SubagentReport) -> None:
        super().__init__()
        self.report = report


class ToolStarted(Message):
    def __init__(
        self,
        run_id: str,
        name: str,
        input_preview: str,
        is_subagent: bool,
        checkpoint_ns: str,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.name = name
        self.input_preview = input_preview
        self.is_subagent = is_subagent
        self.checkpoint_ns = checkpoint_ns


class ToolEnded(Message):
    def __init__(
        self, run_id: str, error: bool = False, output_snippet: str = ""
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.error = error
        self.output_snippet = output_snippet


class LLMThinking(Message):
    def __init__(self, run_id: str, checkpoint_ns: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.checkpoint_ns = checkpoint_ns


class LLMDone(Message):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


class TextToken(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ResponseFinal(Message):
    """The main agent's final message text, to replace the response buffer."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AgentDone(Message):
    pass


class StatusFlash(Message):
    """Transient text to surface in the status bar (auto-clears)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class TokenUpdate(Message):
    def __init__(self, delta_input: int, delta_output: int) -> None:
        super().__init__()
        self.delta_input = delta_input
        self.delta_output = delta_output


class ContextUpdate(Message):
    """Snapshot of the main-thread prompt size from the last LLM turn."""

    def __init__(self, current_input: int) -> None:
        super().__init__()
        self.current_input = current_input


class ToolCountChanged(Message):
    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta
