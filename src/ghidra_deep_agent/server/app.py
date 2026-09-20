"""The FastAPI application over an :class:`AgentRegistry`.

Every route is a thin translation between HTTP and the registry: validation
by the pydantic schemas, registry errors mapped to status codes, and the two
event delivery shapes (SSE and JSON cursor pages) built from the same store.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ghidra_deep_agent.formatting import extract_text
from ghidra_deep_agent.server import schemas
from ghidra_deep_agent.server.events import TERMINAL_TYPES, Persisted, sse_frame
from ghidra_deep_agent.server.registry import (
    AgentConflict,
    AgentInstance,
    AgentNotReady,
    AgentRegistry,
    EngineBuildFailed,
    GhidraError,
    ProgramNotOpen,
    RegistryError,
    RunRecord,
    ThreadBusy,
    UnknownAgent,
    UnknownRun,
)

# Registry error -> HTTP status. Anything unlisted is a 500.
STATUS_FOR: dict[type[RegistryError], int] = {
    UnknownAgent: 404,
    UnknownRun: 404,
    ProgramNotOpen: 404,
    AgentConflict: 409,
    AgentNotReady: 409,
    ThreadBusy: 409,
    EngineBuildFailed: 502,
    GhidraError: 502,
}

# How long a JSON long-poll may block for one new event.
MAX_POLL_WAIT_S = 60.0


def _error(status: int, code: str, detail: str, **extra: Any) -> HTTPException:
    return HTTPException(status, {"error": code, "detail": detail, "extra": extra})


def _links(run_id: str) -> dict[str, str]:
    base = f"/runs/{run_id}"
    return {
        "self": base,
        "events": f"{base}/events",
        "wait": f"{base}/wait",
        "cancel": f"{base}/cancel",
    }


def _run_out(run: RunRecord) -> dict[str, Any]:
    return run.to_dict()


def create_app(registry: AgentRegistry, *, token: str | None = None) -> FastAPI:
    """Build the app.

    ``token`` (``WEB_TOKEN``) enables bearer auth on every route but /health.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await registry.startup()
        try:
            yield
        finally:
            await registry.shutdown()

    app = FastAPI(
        title="ghidra-deep-agent",
        description=(
            "Many reverse-engineering agents on one Ghidra. Create an agent per "
            "open program, start runs on it, and follow each run's events over "
            "SSE or by cursor."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RegistryError)
    async def _registry_error(_request: Request, exc: RegistryError) -> JSONResponse:
        status = next(
            (code for cls, code in STATUS_FOR.items() if isinstance(exc, cls)), 500
        )
        return JSONResponse(
            {"error": exc.code, "detail": exc.detail, "extra": exc.extra},
            status_code=status,
        )

    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        body: Any = exc.detail
        if not isinstance(body, dict):
            body = {"error": "http_error", "detail": str(body), "extra": {}}
        return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)

    async def require_token(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {token}":
            raise _error(401, "unauthorized", "missing or invalid bearer token")

    @app.get("/health", response_model=schemas.HealthOut)
    async def health() -> dict[str, Any]:
        agents = registry.list_agents()
        return {
            "status": "ok",
            "mcp_ok": bool(registry.shared.probe_tools),
            "db_ok": registry.shared.session_store is not None,
            "agents": len(agents),
            "runs_active": registry.active_run_count,
            "model": registry.shared.main_model_spec,
        }

    api = APIRouter(dependencies=[Depends(require_token)])

    # --- programs -----------------------------------------------------------

    @api.get(
        "/programs",
        response_model=schemas.ProgramsOut | schemas.ProjectFilesOut,
        summary="Programs open in Ghidra, or (project=true) files in the project",
    )
    async def programs(project: bool = False, folder: str = "/") -> dict[str, Any]:
        if project:
            return {"files": await registry.list_project_files(folder)}
        return {"programs": [p.__dict__ for p in await registry.list_programs()]}

    @api.post("/programs/open", response_model=schemas.ProgramOut)
    async def open_program(body: schemas.OpenProgramIn) -> dict[str, Any]:
        entry = await registry.open_program(body.path, auto_analyze=body.auto_analyze)
        return entry.__dict__

    # --- agents -------------------------------------------------------------

    @api.get("/agents", response_model=schemas.AgentsOut)
    async def list_agents() -> dict[str, Any]:
        return {"agents": [a.to_dict() for a in registry.list_agents()]}

    @api.post(
        "/agents",
        response_model=schemas.AgentOut,
        status_code=201,
        summary="Create (or return) the agent pinned to a program",
    )
    async def create_agent(
        body: schemas.CreateAgentIn, response: Response
    ) -> dict[str, Any]:
        inst, created = await registry.create_agent(body.program)
        response.status_code = 201 if created else 200
        return inst.to_dict()

    @api.get("/agents/{agent_id}", response_model=schemas.AgentOut)
    async def get_agent(agent_id: str) -> dict[str, Any]:
        return registry.get_agent(agent_id).to_dict()

    @api.delete("/agents/{agent_id}", status_code=204)
    async def delete_agent(agent_id: str) -> Response:
        await registry.delete_agent(agent_id)
        return Response(status_code=204)

    # --- runs ---------------------------------------------------------------

    @api.post(
        "/agents/{agent_id}/runs",
        response_model=schemas.RunStartedOut,
        status_code=202,
        summary="Start a turn (or resume an interrupted one) on an agent",
    )
    async def start_run(agent_id: str, body: schemas.StartRunIn) -> dict[str, Any]:
        run = await registry.start_run(
            agent_id,
            prompt=body.prompt,
            resume=body.continue_,
            session_id=body.session_id,
            mode=body.mode,
        )
        return {**_run_out(run), "links": _links(run.id)}

    @api.get("/runs", response_model=schemas.RunsOut)
    async def list_runs(
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = await asyncio.to_thread(
            registry.list_runs, agent_id=agent_id, session_id=session_id, limit=limit
        )
        return {"runs": [RunRecord.from_dict(r).to_dict() for r in rows]}

    @api.get("/runs/{run_id}", response_model=schemas.RunOut)
    async def get_run(run_id: str) -> dict[str, Any]:
        return _run_out(registry.get_run(run_id))

    @api.post("/runs/{run_id}/wait", response_model=schemas.RunOut)
    async def wait_run(run_id: str, body: schemas.WaitIn | None = None) -> JSONResponse:
        timeout = body.timeout if body is not None else 300.0
        run = await registry.wait(run_id, timeout)
        return JSONResponse(_run_out(run), status_code=200 if run.terminal else 202)

    @api.post("/runs/{run_id}/cancel", response_model=schemas.CancelOut)
    async def cancel_run(run_id: str) -> dict[str, Any]:
        return {"cancelled": await registry.cancel_run(run_id)}

    # --- events -------------------------------------------------------------

    async def _page(run: RunRecord, after: int, limit: int) -> list[Persisted]:
        return await asyncio.to_thread(
            registry.store.events_after, run.id, after, limit
        )

    @api.get(
        "/runs/{run_id}/events",
        summary="A run's events: SSE (Accept: text/event-stream) or a JSON page",
        response_model=schemas.EventsPageOut,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def events(
        request: Request,
        run_id: str,
        after: int = Query(0, ge=0, description="Cursor: last seq already seen"),
        limit: int = Query(500, ge=1, le=5000),
        wait: float = Query(
            0,
            ge=0,
            le=MAX_POLL_WAIT_S,
            description="JSON only: block up to N seconds for news",
        ),
    ) -> Any:
        run = registry.get_run(run_id)
        if "text/event-stream" in request.headers.get("accept", ""):
            last_id = request.headers.get("last-event-id")
            if last_id and last_id.isdigit():
                after = int(last_id)
            return EventSourceResponse(_sse(run, after), ping=15)

        page = await _page(run, after, limit)
        if not page and wait > 0 and not run.terminal:
            queue = registry.subscribe(run)
            try:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(queue.get(), wait)
            finally:
                registry.unsubscribe(run, queue)
            page = await _page(run, after, limit)
        next_after = page[-1]["seq"] if page else after
        terminal = any(e["type"] in TERMINAL_TYPES for e in page) or (
            run.terminal and next_after >= run.last_seq
        )
        return {
            "run_id": run.id,
            "status": run.status,
            "events": page,
            "next_after": next_after,
            "terminal": terminal,
        }

    async def _sse(run: RunRecord, after: int) -> AsyncIterator[dict[str, str]]:
        # Subscribe FIRST, then replay: an event published between the replay
        # read and the subscription would otherwise be lost. Duplicates from
        # the overlap are skipped by sequence number.
        queue = registry.subscribe(run)
        try:
            last = after
            replay_from = after
            while True:
                page = await _page(run, replay_from, 500)
                if not page:
                    break
                for doc in page:
                    last = doc["seq"]
                    yield sse_frame(doc)
                    if doc["type"] in TERMINAL_TYPES:
                        return
                replay_from = last
            if run.terminal:
                return
            while True:
                live = await queue.get()
                if live is None:
                    return
                if live["seq"] <= last:
                    continue
                last = live["seq"]
                yield sse_frame(live)
                if live["type"] in TERMINAL_TYPES:
                    return
        finally:
            registry.unsubscribe(run, queue)

    # --- sessions -----------------------------------------------------------

    @api.get("/sessions", response_model=schemas.SessionsOut)
    async def sessions(
        binary: str | None = None, limit: int = Query(50, ge=1, le=500)
    ) -> dict[str, Any]:
        store = registry.shared.session_store
        if store is None:
            raise _error(503, "sessions_unavailable", "session registry is disabled")
        rows = await store.alist_sessions(binary, limit)
        out = []
        for row in rows:
            out.append(
                {
                    "session_id": row.get("session_id") or row.get("_id"),
                    "binary_name": row.get("binary_name"),
                    "title": row.get("title"),
                    "created_at": _stamp(row.get("created_at")),
                    "last_active_at": _stamp(row.get("last_active_at")),
                }
            )
        return {"sessions": out}

    @api.get("/sessions/{session_id}/history", response_model=schemas.HistoryOut)
    async def history(
        session_id: str, agent_id: str | None = None, mode: str = "normal"
    ) -> dict[str, Any]:
        inst = _engine_for(agent_id)
        assert inst.engine is not None
        thread_id = session_id if mode == "normal" else f"{session_id}::ask"
        graph = inst.engine.graphs.main if mode == "normal" else inst.engine.graphs.ask
        state = await graph.aget_state(registry.shared.config_for(thread_id))
        messages = []
        for msg in state.values.get("messages", []):
            kind = getattr(msg, "type", "")
            role = {"human": "user", "ai": "assistant"}.get(kind)
            text = extract_text(msg)
            if role and text:
                messages.append({"role": role, "text": text})
        return {"session_id": session_id, "thread_id": thread_id, "messages": messages}

    def _engine_for(agent_id: str | None) -> AgentInstance:
        if agent_id is not None:
            inst = registry.get_agent(agent_id)
        else:
            # The main graphs are structurally identical across instances, so
            # any ready one can read a thread's state.
            inst = next(
                (a for a in registry.list_agents() if a.status == "ready"), None
            )  # type: ignore[assignment]
            if inst is None:
                raise _error(409, "no_agent", "no ready agent to read history with")
        if inst.engine is None:
            raise AgentNotReady(f"agent {inst.id} is {inst.status}")
        return inst

    # --- files --------------------------------------------------------------

    @api.get("/agents/{agent_id}/files", response_model=schemas.FilesOut)
    async def list_files(
        agent_id: str, path: str = "/", session_id: str | None = None
    ) -> dict[str, Any]:
        inst = registry.get_agent(agent_id)
        engine = inst.engine
        if engine is None:
            raise AgentNotReady(f"agent {agent_id} is {inst.status}")
        if engine.output_dir:
            root = Path(engine.output_dir).resolve()
            target = _safe_join(root, path)
            entries = []
            if target.is_dir():
                for child in sorted(target.iterdir()):
                    entries.append(
                        {
                            "path": "/" + child.relative_to(root).as_posix(),
                            "is_dir": child.is_dir(),
                            "size": child.stat().st_size if child.is_file() else None,
                        }
                    )
            return {"backend": "filesystem", "entries": entries}
        if engine.storage.prompt_guidance:
            # Sandboxed with no local mirror: files exist only in the sandbox.
            return {"backend": "sandbox", "entries": []}
        if session_id is None:
            raise _error(
                400, "session_required", "state-backed files need ?session_id="
            )
        files = await _state_files(inst, session_id)
        return {
            "backend": "state",
            "entries": [
                {"path": p, "is_dir": False, "size": len(_file_text(v))}
                for p, v in sorted(files.items())
                if p.startswith(path.rstrip("/") + "/") or path in ("", "/")
            ],
        }

    @api.get("/agents/{agent_id}/files/{path:path}")
    async def get_file(
        agent_id: str, path: str, session_id: str | None = None
    ) -> Response:
        inst = registry.get_agent(agent_id)
        engine = inst.engine
        if engine is None:
            raise AgentNotReady(f"agent {agent_id} is {inst.status}")
        if engine.output_dir:
            target = _safe_join(Path(engine.output_dir).resolve(), path)
            if not target.is_file():
                raise _error(404, "not_found", f"no file {path!r}")
            media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return Response(target.read_bytes(), media_type=media)
        if session_id is None:
            raise _error(
                400, "session_required", "state-backed files need ?session_id="
            )
        files = await _state_files(inst, session_id)
        key = "/" + path.lstrip("/")
        if key not in files:
            raise _error(404, "not_found", f"no file {path!r}")
        return Response(_file_text(files[key]), media_type="text/plain; charset=utf-8")

    async def _state_files(inst: AgentInstance, session_id: str) -> dict[str, Any]:
        assert inst.engine is not None
        state = await inst.engine.graphs.main.aget_state(
            registry.shared.config_for(session_id)
        )
        files = state.values.get("files") or {}
        return dict(files) if isinstance(files, dict) else {}

    app.include_router(api)
    return app


def _safe_join(root: Path, path: str) -> Path:
    """Resolve ``path`` under ``root``, refusing anything that escapes it."""
    target = (root / path.lstrip("/")).resolve()
    if target != root and root not in target.parents:
        raise _error(400, "bad_path", "path escapes the agent's output directory")
    return target


def _file_text(value: Any) -> str:
    """Best-effort text of a deepagents state ``files`` entry."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return "\n".join(str(line) for line in content)
        if content is not None:
            return str(content)
    return json.dumps(value, default=str)


def _stamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


# Re-exported for main.py and tests.
Handler = Callable[..., Awaitable[Any]]
__all__ = ["create_app", "os"]
