"""Request and response models for the HTTP API (also the OpenAPI contract)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ErrorOut(BaseModel):
    error: str
    detail: str
    extra: dict[str, Any] = Field(default_factory=dict)


class ProgramOut(BaseModel):
    name: str
    project_path: str
    active: bool = False
    format: str = ""
    language: str = ""


class ProgramsOut(BaseModel):
    programs: list[ProgramOut]


class ProjectFilesOut(BaseModel):
    files: list[str]


class OpenProgramIn(BaseModel):
    path: str = Field(description="Project path (or name) of a program in the project")
    auto_analyze: bool = False


class CreateAgentIn(BaseModel):
    program: str = Field(
        description="Project path (preferred) or name of a program open in Ghidra"
    )


class ProgramRefOut(BaseModel):
    name: str
    project_path: str


class AgentOut(BaseModel):
    id: str
    program: ProgramRefOut
    status: Literal["building", "ready", "degraded", "closed"]
    error: str | None = None
    created_at: str | None = None
    active_runs: dict[str, str] = Field(default_factory=dict)
    knowledge_ok: bool | None = None
    tool_count: int | None = None
    output_dir: str | None = None


class AgentsOut(BaseModel):
    agents: list[AgentOut]


class StartRunIn(BaseModel):
    prompt: str | None = Field(default=None, description="The turn's user message")
    continue_: bool = Field(
        default=False,
        alias="continue",
        description="Resume an interrupted turn on session_id (no prompt)",
    )
    session_id: str | None = Field(
        default=None, description="Thread to continue; omitted = new session"
    )
    mode: Literal["normal", "ask"] = "normal"

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _one_of_prompt_or_continue(self) -> StartRunIn:
        if self.continue_ == bool(self.prompt):
            raise ValueError("provide exactly one of 'prompt' or 'continue': true")
        if self.continue_ and not self.session_id:
            raise ValueError("'continue' needs a session_id")
        if self.continue_ and self.mode != "normal":
            raise ValueError("'continue' only applies to the normal thread")
        return self


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class RunOut(BaseModel):
    id: str
    agent_id: str
    session_id: str
    thread_id: str
    mode: str
    status: str
    prompt: str | None
    resume: bool
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    reply: str
    usage: UsageOut
    error: str | None
    last_seq: int


class RunLinks(BaseModel):
    self: str
    events: str
    wait: str
    cancel: str


class RunStartedOut(RunOut):
    links: RunLinks


class RunsOut(BaseModel):
    runs: list[RunOut]


class EventOut(BaseModel):
    seq: int
    ts: str
    type: str
    data: dict[str, Any]


class EventsPageOut(BaseModel):
    run_id: str
    status: str
    events: list[EventOut]
    next_after: int
    terminal: bool


class WaitIn(BaseModel):
    timeout: float = Field(default=300.0, ge=0, le=3600)


class CancelOut(BaseModel):
    cancelled: bool


class SessionOut(BaseModel):
    session_id: str
    binary_name: str | None = None
    title: str | None = None
    created_at: str | None = None
    last_active_at: str | None = None


class SessionsOut(BaseModel):
    sessions: list[SessionOut]


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class HistoryOut(BaseModel):
    session_id: str
    thread_id: str
    messages: list[MessageOut]


class FileEntryOut(BaseModel):
    path: str
    is_dir: bool
    size: int | None = None


class FilesOut(BaseModel):
    backend: Literal["filesystem", "state", "sandbox"]
    entries: list[FileEntryOut]


class HealthOut(BaseModel):
    status: str
    mcp_ok: bool
    db_ok: bool
    agents: int
    runs_active: int
    model: str
