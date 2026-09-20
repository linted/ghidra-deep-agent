"""The registry of agent instances and their runs.

One :class:`AgentInstance` per program (identified by project path), each
owning an :class:`~ghidra_deep_agent.runtime.Engine` pinned to it. Runs on
different instances execute concurrently as asyncio tasks; on one thread they
are serialized by refusing a second run while one is active (the caller waits
on the first and resubmits — an agent caller should see the conflict, not have
ordering hidden behind a queue).

The registry is plain async Python with dict-shaped results, so the HTTP layer
and a future MCP endpoint can both wrap it without duplicating any policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ghidra_deep_agent.program_resolver import (
    ProgramInfo,
    ProgramRef,
    find_program,
    list_open_programs,
    list_project_files,
    open_project_program,
)
from ghidra_deep_agent.runtime import (
    Engine,
    SharedRuntime,
    StartupError,
    build_engine,
)
from ghidra_deep_agent.server.events import (
    TERMINAL_TYPES,
    AnyEvent,
    Persisted,
    RunStore,
    serialize,
)
from ghidra_deep_agent.stream import RunState

RunStatus = Literal["queued", "running", "done", "paused", "error", "cancelled"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "paused", "error", "cancelled"})
AgentStatus = Literal["building", "ready", "degraded", "closed"]
Mode = Literal["normal", "ask"]

# How many live events a slow subscriber may lag before the oldest are dropped
# (they remain in the store; the client re-syncs from its cursor).
SUBSCRIBER_QUEUE_SIZE = 1000


# --- errors -------------------------------------------------------------------


class RegistryError(Exception):
    """Base for caller-facing failures; ``code`` is the wire error code."""

    code = "registry_error"

    def __init__(self, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra


class UnknownAgent(RegistryError):
    code = "unknown_agent"


class UnknownRun(RegistryError):
    code = "unknown_run"


class ProgramNotOpen(RegistryError):
    code = "program_not_open"


class AgentConflict(RegistryError):
    code = "name_conflict"


class AgentNotReady(RegistryError):
    code = "agent_not_ready"


class ThreadBusy(RegistryError):
    code = "thread_busy"


class EngineBuildFailed(RegistryError):
    code = "engine_build_failed"


class GhidraError(RegistryError):
    code = "ghidra_error"


# --- records ------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass
class RunRecord:
    id: str
    agent_id: str
    session_id: str
    thread_id: str
    mode: Mode
    prompt: str | None
    resume: bool
    status: RunStatus = "queued"
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reply: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    last_seq: int = 0
    # Per-turn bookkeeping for stream.translate; replaced per run.
    run_state: RunState = field(default_factory=RunState)
    task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    subscribers: list[asyncio.Queue[Persisted | None]] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "mode": self.mode,
            "status": self.status,
            "prompt": self.prompt,
            "resume": self.resume,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "reply": self.reply,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "error": self.error,
            "last_seq": self.last_seq,
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> RunRecord:
        """Rehydrate a stored run (no task; ``done`` is set if terminal)."""
        usage = doc.get("usage") or {}
        run = cls(
            id=doc["id"],
            agent_id=doc["agent_id"],
            session_id=doc["session_id"],
            thread_id=doc["thread_id"],
            mode=doc.get("mode", "normal"),
            prompt=doc.get("prompt"),
            resume=bool(doc.get("resume")),
            status=doc.get("status", "error"),
            created_at=_parse(doc.get("created_at")) or _now(),
            started_at=_parse(doc.get("started_at")),
            finished_at=_parse(doc.get("finished_at")),
            reply=doc.get("reply", ""),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            error=doc.get("error"),
            last_seq=int(doc.get("last_seq", 0)),
        )
        if run.terminal:
            run.done.set()
        return run


def _parse(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return None


@dataclass
class AgentInstance:
    id: str
    program: ProgramRef
    status: AgentStatus = "building"
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    engine: Engine | None = None
    # thread_id -> run_id for runs in flight; the per-thread serialization.
    active_runs: dict[str, str] = field(default_factory=dict)
    build_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def to_dict(self) -> dict[str, Any]:
        engine = self.engine
        return {
            "id": self.id,
            "program": {
                "name": self.program.name,
                "project_path": self.program.project_path,
            },
            "status": self.status,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "active_runs": dict(self.active_runs),
            "knowledge_ok": engine.knowledge_ok if engine else None,
            "tool_count": engine.tool_count if engine else None,
            "output_dir": engine.output_dir if engine else None,
        }


EngineFactory = Callable[..., Awaitable[Engine]]
Runner = Callable[
    ["RunRecord", "AgentInstance", "AgentRegistry"], Coroutine[Any, Any, None]
]


# --- registry -----------------------------------------------------------------


class AgentRegistry:
    def __init__(
        self,
        shared: SharedRuntime,
        store: RunStore,
        *,
        engine_factory: EngineFactory = build_engine,
        runner: Runner | None = None,
        output_root: str = "",
    ) -> None:
        self.shared = shared
        self.store = store
        self._engine_factory = engine_factory
        # Imported lazily to avoid a cycle (runner imports this module).
        self._runner = runner
        self._output_root = output_root
        self._agents: dict[str, AgentInstance] = {}
        self._runs: dict[str, RunRecord] = {}

    # --- programs ---------------------------------------------------------------

    async def list_programs(self) -> list[ProgramInfo]:
        try:
            return await list_open_programs(self.shared.probe_tools)
        except RuntimeError as exc:
            raise GhidraError(str(exc)) from exc

    async def list_project_files(self, folder: str = "/") -> list[str]:
        try:
            return await list_project_files(self.shared.probe_tools, folder)
        except RuntimeError as exc:
            raise GhidraError(str(exc)) from exc

    async def open_program(
        self, path: str, *, auto_analyze: bool = False
    ) -> ProgramInfo:
        try:
            return await open_project_program(
                self.shared.probe_tools, path, auto_analyze=auto_analyze
            )
        except RuntimeError as exc:
            if "not found" in str(exc).lower():
                raise ProgramNotOpen(str(exc)) from exc
            raise GhidraError(str(exc)) from exc

    # --- agents -----------------------------------------------------------------

    def list_agents(self) -> list[AgentInstance]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentInstance:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise UnknownAgent(f"no agent {agent_id!r}") from None

    def _find_by_path(self, project_path: str) -> AgentInstance | None:
        for inst in self._agents.values():
            if inst.program.project_path == project_path and inst.status != "closed":
                return inst
        return None

    async def create_agent(self, program_key: str) -> tuple[AgentInstance, bool]:
        """Create (or return) the instance for a program. Returns ``(inst, created)``.

        Idempotent per project path: a second request for the same program
        joins the existing instance, waiting for its build if one is under
        way. Two different programs with the same bare name are refused
        because knowledge, cache and sessions are keyed by that name.
        """
        entry = find_program(await self.list_programs(), program_key)
        if entry is None:
            raise ProgramNotOpen(
                f"{program_key!r} is not open in Ghidra; open it first "
                "(POST /programs/open) or check GET /programs."
            )
        program = ProgramRef(entry.name, entry.project_path)

        existing = self._find_by_path(program.project_path)
        if existing is not None:
            async with existing.build_lock:
                pass  # wait for an in-flight build
            return existing, False
        for inst in self._agents.values():
            if inst.program.name == program.name and inst.status != "closed":
                raise AgentConflict(
                    f"agent {inst.id} already covers a program named "
                    f"{program.name!r} ({inst.program.project_path}); knowledge "
                    "and cache are keyed by name, so a second one would collide."
                )

        inst = AgentInstance(
            id=f"{program.slug}-{uuid.uuid4().hex[:6]}", program=program
        )
        self._agents[inst.id] = inst
        async with inst.build_lock:
            try:
                inst.engine = await self._engine_factory(
                    self.shared,
                    program,
                    output_dir=self._output_dir_for(program),
                    on_mismatch=lambda tool, target: self._degrade(inst, tool, target),
                )
            except (StartupError, Exception) as exc:
                del self._agents[inst.id]
                raise EngineBuildFailed(str(exc)) from exc
            inst.status = "ready"
        return inst, True

    def _output_dir_for(self, program: ProgramRef) -> str:
        if not self._output_root:
            return ""
        return f"{self._output_root.rstrip('/')}/{program.slug}"

    def _degrade(self, inst: AgentInstance, tool: str, target: str) -> None:
        # The pin's last line of defense fired: Ghidra answered for another
        # program. Stop scheduling work here until the instance is re-created.
        inst.status = "degraded"
        inst.error = (
            f"tool {tool!r} operated on {target!r} instead of "
            f"{inst.program.project_path}; recreate the agent once the program "
            "is open again"
        )
        print(f"Agent {inst.id} degraded: {inst.error}", file=sys.stderr)

    async def delete_agent(self, agent_id: str) -> None:
        inst = self.get_agent(agent_id)
        for run_id in list(inst.active_runs.values()):
            await self.cancel_run(run_id)
        inst.status = "closed"
        del self._agents[agent_id]
        if inst.engine is not None:
            await inst.engine.aclose()
            inst.engine = None

    # --- runs -------------------------------------------------------------------

    async def start_run(
        self,
        agent_id: str,
        *,
        prompt: str | None,
        resume: bool = False,
        session_id: str | None = None,
        mode: Mode = "normal",
    ) -> RunRecord:
        inst = self.get_agent(agent_id)
        if inst.status != "ready" or inst.engine is None:
            raise AgentNotReady(
                f"agent {agent_id} is {inst.status}"
                + (f": {inst.error}" if inst.error else "")
            )
        new_session = session_id is None
        session_id = session_id or str(uuid.uuid4())
        thread_id = session_id if mode == "normal" else f"{session_id}::ask"
        if thread_id in inst.active_runs:
            raise ThreadBusy(
                f"a run is already active on thread {thread_id!r}",
                active_run_id=inst.active_runs[thread_id],
            )
        run = RunRecord(
            id=uuid.uuid4().hex[:12],
            agent_id=agent_id,
            session_id=session_id,
            thread_id=thread_id,
            mode=mode,
            prompt=prompt,
            resume=resume,
        )
        # Claimed before the first await so two concurrent requests for the
        # same thread cannot both pass the check above.
        inst.active_runs[thread_id] = run.id
        self._runs[run.id] = run
        await asyncio.to_thread(self.store.upsert_run, run.to_dict())
        store = self.shared.session_store
        if store is not None:
            with contextlib.suppress(Exception):  # best-effort bookkeeping
                if new_session:
                    await store.arecord_start(session_id, inst.program.name)
                await store.atouch(session_id, first_prompt=prompt)
        runner = self._runner or _default_runner()
        run.task = asyncio.create_task(runner(run, inst, self), name=f"run:{run.id}")
        return run

    def get_run(self, run_id: str) -> RunRecord:
        run = self._runs.get(run_id)
        if run is not None:
            return run
        doc = self.store.load_run(run_id)
        if doc is None:
            raise UnknownRun(f"no run {run_id!r}")
        return RunRecord.from_dict(doc)

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.store.list_runs(
            agent_id=agent_id, session_id=session_id, limit=limit
        )

    async def cancel_run(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if run.task is None or run.terminal:
            return False
        run.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run.task
        if not run.terminal:
            # Cancelled before the runner's body ever ran (the task was still
            # scheduled), so nothing else will finish the record.
            run.status = "cancelled"
            inst = self._agents.get(run.agent_id)
            if inst is not None:
                self.finish_run(run, inst)
            else:
                run.finished_at = _now()
            await asyncio.to_thread(self.store.upsert_run, run.to_dict())
            run.done.set()
        return True

    async def wait(self, run_id: str, timeout: float) -> RunRecord:
        run = self.get_run(run_id)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(run.done.wait(), timeout)
        return run

    # --- events -----------------------------------------------------------------

    def subscribe(self, run: RunRecord) -> asyncio.Queue[Persisted | None]:
        queue: asyncio.Queue[Persisted | None] = asyncio.Queue(SUBSCRIBER_QUEUE_SIZE)
        run.subscribers.append(queue)
        if run.terminal:
            queue.put_nowait(None)
        return queue

    def unsubscribe(
        self, run: RunRecord, queue: asyncio.Queue[Persisted | None]
    ) -> None:
        with contextlib.suppress(ValueError):
            run.subscribers.remove(queue)

    async def publish(self, run: RunRecord, event: AnyEvent) -> Persisted:
        run.last_seq += 1
        doc = serialize(run.id, run.last_seq, event)
        try:
            await asyncio.to_thread(self.store.append_event, doc)
        except Exception as exc:  # a persist failure must not kill the run
            print(
                f"Warning: could not persist event {doc['seq']}: {exc}", file=sys.stderr
            )
        for queue in list(run.subscribers):
            if queue.full():
                # Drop the oldest: the client re-syncs from the store by cursor.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(doc)
        if doc["type"] in TERMINAL_TYPES:
            for queue in list(run.subscribers):
                queue.put_nowait(None)
        return doc

    def finish_run(self, run: RunRecord, inst: AgentInstance) -> None:
        """Bookkeeping after a run ends: stamp it and free its thread.

        Deliberately does not set ``run.done``; the runner does that after the
        terminal record is persisted.
        """
        run.finished_at = _now()
        if inst.active_runs.get(run.thread_id) == run.id:
            del inst.active_runs[run.thread_id]

    # --- lifecycle --------------------------------------------------------------

    async def startup(self) -> None:
        """Fail any run the previous process left non-terminal."""
        count = await asyncio.to_thread(
            self.store.mark_interrupted, "server restarted while the run was active"
        )
        if count:
            print(f"Marked {count} interrupted run(s) from a previous server.")

    async def shutdown(self, timeout: float = 10.0) -> None:
        tasks = [r.task for r in self._runs.values() if r.task and not r.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        for inst in list(self._agents.values()):
            inst.status = "closed"
            if inst.engine is not None:
                with contextlib.suppress(Exception):
                    await inst.engine.aclose()
        self._agents.clear()

    @property
    def active_run_count(self) -> int:
        return sum(len(i.active_runs) for i in self._agents.values())


def _default_runner() -> Runner:
    from ghidra_deep_agent.server.runner import execute_run

    return execute_run
