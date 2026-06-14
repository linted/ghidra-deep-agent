"""FastAPI app exposing the agent runtime to the browser.

REST for programs/sessions/history, a WebSocket for the live event stream, and
the vanilla-JS client served from ``static/``. Run with ``ghidra-deep-agent-web``
(reads ``WEB_HOST`` / ``WEB_PORT``).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ghidra_deep_agent.importer_client import import_binary
from ghidra_deep_agent.web.service import COMPACT_PROMPT, AgentService

STATIC_DIR = Path(__file__).parent / "static"

# Program names become analyzeHeadless args and repo paths in the importer; keep
# the web-side check identical to the importer's allowlist.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "256"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _safe_program_name(filename: str | None) -> str | None:
    """Reduce an upload filename to a safe program name, or None if unusable."""
    if not filename:
        return None
    name = os.path.basename(filename).strip()
    return name if _NAME_RE.match(name) else None


service = AgentService()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


app = FastAPI(title="Ghidra Deep Agent", lifespan=lifespan)


class NewSession(BaseModel):
    binary_name: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "model": service.settings.model,
        "max_context_tokens": service.max_context_tokens,
        "mcp_ok": service.mcp_ok,
        "db_ok": service.db_ok,
    }


@app.get("/api/programs")
async def get_programs() -> Response:
    try:
        return JSONResponse({"programs": await service.list_programs()})
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.post("/api/upload")
async def upload_binary(request: Request, name: str) -> Response:
    """Accept a binary and import it into the shared Ghidra Server repo.

    The file is sent as the raw request body (``application/octet-stream``) with
    ``?name=<filename>``; the bytes are forwarded to the importer in the
    ghidra-mcp container (the only place Ghidra is installed). Duplicate names are
    rejected (409) and no analysis runs at upload time — both per product decision.
    """
    safe = _safe_program_name(name)
    if safe is None:
        return JSONResponse(
            {"error": "invalid filename (allowed: letters, digits, . _ -)"},
            status_code=400,
        )

    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"file exceeds {MAX_UPLOAD_MB} MB"}, status_code=413
        )

    try:
        result = await import_binary(safe, data)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return JSONResponse(result.payload, status_code=result.status_code)


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": [s.to_dict() for s in service.sessions.list()]}


@app.post("/api/sessions")
async def create_session(body: NewSession) -> dict[str, Any]:
    return service.create_session(body.binary_name).to_dict()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    service.cancel(session_id)
    return {"deleted": service.sessions.delete(session_id)}


@app.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str) -> Response:
    try:
        return JSONResponse({"messages": await service.history(session_id)})
    except KeyError:
        return JSONResponse({"error": "unknown session"}, status_code=404)


@app.websocket("/api/sessions/{session_id}/stream")
async def stream_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    pump: asyncio.Task[None] | None = None

    async def run_pump(agent_input: str) -> None:
        try:
            async for payload in service.stream(session_id, agent_input):
                await websocket.send_json(payload)
        except asyncio.CancelledError:
            await websocket.send_json({"type": "cancelled"})
        except Exception as exc:  # surface agent errors to the client
            await websocket.send_json({"type": "error", "message": str(exc)})

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "cancel":
                service.cancel(session_id)
                continue

            if mtype != "query":
                continue

            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            if service.is_running(session_id):
                await websocket.send_json(
                    {
                        "type": "status_flash",
                        "text": "Agent still running — please wait.",
                    }
                )
                continue

            agent_input = COMPACT_PROMPT if text == "/compact" else text
            pump = asyncio.create_task(run_pump(agent_input))
    except WebSocketDisconnect:
        pass
    finally:
        if pump is not None and not pump.done():
            pump.cancel()
        service.cancel(session_id)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run() -> None:
    import uvicorn

    load_dotenv()
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
