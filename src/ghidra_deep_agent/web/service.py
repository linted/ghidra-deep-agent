"""Long-lived agent runtime shared across all web sessions.

Owns the single Ghidra MCP connection, the MongoDB checkpointer, the model, and
embeddings. Builds and caches one deep agent per binary (knowledge tools are
binary-scoped) and runs many ``thread_id``s against it concurrently.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.mongodb import MongoDBSaver

from ghidra_deep_agent.knowledge import build_knowledge_tools
from ghidra_deep_agent.models import build_embeddings, build_model
from ghidra_deep_agent.program_resolver import (
    is_program_open,
    list_project_programs,
    open_program,
)
from ghidra_deep_agent.runtime import (
    Settings,
    build_agent,
    connect_mcp_tools,
    context_window,
    make_backend,
    resolve_settings,
)
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink
from ghidra_deep_agent.web.event_stream import Payload, event_to_payloads
from ghidra_deep_agent.web.sessions import Session, SessionStore

# The canned prompt the TUI sends for /compact (see tui/app.py:_dispatch_slash).
COMPACT_PROMPT = (
    "Call the `compact_conversation` tool now to compact the conversation history."
)


class AgentService:
    """Orchestrates agent runs for the web UI."""

    def __init__(self) -> None:
        self.settings: Settings = resolve_settings()
        self.sessions = SessionStore(
            self.settings.mongodb_uri, self.settings.mongodb_db
        )
        self._tools: list[Any] = []
        self._agents: dict[str, Any] = {}
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._cm: Any = None
        self._checkpointer: Any = None
        self._model: Any = None
        self._embeddings: Any = None
        self._backend: Any = None
        self.mcp_ok = False
        self.db_ok = False
        self.max_context_tokens = 200_000

    # -- lifecycle -----------------------------------------------------------

    async def startup(self) -> None:
        """Connect to Ghidra and MongoDB. Call once at server start."""
        self._tools = await connect_mcp_tools()
        self.mcp_ok = bool(self._tools)

        self._model = build_model(self.settings.model)
        self.max_context_tokens = context_window(self._model)
        self._embeddings = build_embeddings(self.settings.embed_string)
        self._backend = make_backend(self.settings.output_dir)

        self._cm = MongoDBSaver.from_conn_string(
            self.settings.mongodb_uri, db_name=self.settings.mongodb_db
        )
        self._checkpointer = self._cm.__enter__()
        self.db_ok = True

    async def shutdown(self) -> None:
        """Cancel in-flight runs and release the checkpointer."""
        for task in list(self._running.values()):
            task.cancel()
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None

    # -- agents --------------------------------------------------------------

    def _agent_for(self, binary_name: str) -> Any:
        agent = self._agents.get(binary_name)
        if agent is None:
            knowledge_tools = build_knowledge_tools(
                self.settings.mongodb_uri,
                self.settings.mongodb_db,
                self._embeddings,
                binary_name,
            )
            agent = build_agent(
                self._model,
                knowledge_tools + self._tools,
                self._checkpointer,
                self._backend,
                self.settings.agents_md,
            )
            self._agents[binary_name] = agent
        return agent

    def _config(self, session_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.settings.recursion_limit,
        }

    # -- programs / sessions -------------------------------------------------

    async def list_programs(self) -> list[str]:
        return await list_project_programs(self._tools)

    def create_session(self, binary_name: str) -> Session:
        return self.sessions.create(binary_name)

    # -- running -------------------------------------------------------------

    def is_running(self, session_id: str) -> bool:
        task = self._running.get(session_id)
        return task is not None and not task.done()

    def cancel(self, session_id: str) -> bool:
        task = self._running.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    async def stream(self, session_id: str, query: str) -> AsyncIterator[Payload]:
        """Drive one agent run, yielding JSON payloads for the web client.

        Registers a toast sink for the duration of the run so backend toasts
        surface in this session's stream. Cancellable via :meth:`cancel`.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id}")

        # Lazily open the session's binary into the engine. Sessions are created
        # without touching Ghidra (so "New Session" is instant); the program is
        # opened on the first query against it. Opening is idempotent — skip it
        # when the binary is already open in the engine.
        try:
            if not await is_program_open(self._tools, session.binary_name):
                await open_program(self._tools, f"/{session.binary_name}")
        except RuntimeError as exc:
            yield {
                "type": "toast",
                "severity": "error",
                "title": "Open failed",
                "message": str(exc),
            }
            yield {"type": "agent_done"}
            return

        agent = self._agent_for(session.binary_name)
        config = self._config(session_id)

        current = asyncio.current_task()
        if current is not None:
            self._running[session_id] = current

        toast_queue: asyncio.Queue[Payload] = asyncio.Queue()

        def sink(toast: ToastRequest) -> None:
            toast_queue.put_nowait(
                {
                    "type": "toast",
                    "message": toast.message,
                    "severity": toast.severity,
                    "title": toast.title,
                }
            )

        unregister = register_toast_sink(sink)
        input_data = {"messages": [{"role": "user", "content": query}]}
        try:
            async for event in agent.astream_events(
                input_data, config=config, version="v2"
            ):
                while not toast_queue.empty():
                    yield toast_queue.get_nowait()
                for payload in event_to_payloads(event):
                    yield payload
            while not toast_queue.empty():
                yield toast_queue.get_nowait()
            yield {"type": "agent_done"}
        finally:
            unregister()
            self._running.pop(session_id, None)
            self.sessions.touch(session_id)

    async def history(self, session_id: str) -> list[Payload]:
        """Rebuild a session's user/assistant transcript from the checkpointer."""
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id}")
        agent = self._agent_for(session.binary_name)
        state = await agent.aget_state(self._config(session_id))
        values = getattr(state, "values", None) or {}
        messages = values.get("messages", []) if isinstance(values, dict) else []

        transcript: list[Payload] = []
        for msg in messages:
            role = getattr(msg, "type", "")
            text = _message_text(msg)
            if not text:
                continue
            if role == "human":
                transcript.append({"role": "user", "content": text})
            elif role == "ai":
                transcript.append({"role": "assistant", "content": text})
        return transcript


def _message_text(msg: Any) -> str:
    """Extract plain text from a LangChain message's possibly-blocky content."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return ""
