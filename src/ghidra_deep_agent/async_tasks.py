"""Transparently resolve GhidrAssistMCP async task results.

GhidrAssistMCP offloads long-running operations (e.g. ``get_code``) to async
tasks: the tool call returns a *stub* — "Task submitted for async execution.
Task ID: <uuid> ... Status: RUNNING" — instead of the result, and the caller is
expected to poll ``get_task_status`` with that id until the task finishes.

Making the model do that polling itself would add round-trips and a whole class
of "forgot to poll" failures. Instead this middleware intercepts every tool
result, detects the async stub, and polls ``get_task_status`` on the model's
behalf until the task completes — so agents see a synchronous result exactly as
they did with the previous MCP server. No prompt or allowlist changes needed, and
``get_task_status`` does not have to be granted to any agent.
"""

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

# The submission stub is very specific; require both signals to avoid ever
# mistaking real tool output (decompilation, disassembly) for a task stub.
_STUB_MARKER = "Task submitted for async execution"
_TASK_ID_RE = re.compile(r"Task ID:\s*(\S+)")
# A poll response still in flight.
_IN_FLIGHT = ("Status: RUNNING", "Status: PENDING")

# Custom event this middleware dispatches once an async task finishes (resolved
# or timed out), carrying ``{"task_id": ...}``. The TUI uses it to defer the
# originating tool's "completed" marker until the real result is in (the tool's
# own on_tool_end fires early, with the submission stub). See tui/events.py.
ASYNC_DONE_EVENT = "ghidra_async_done"


def _to_text(value: Any) -> str:
    """Flatten an MCP tool result (str, or a list of content blocks) to text.

    ``langchain_mcp_adapters`` results arrive either as a string or as a list of
    ``{"type": "text", "text": ...}`` blocks (dicts or objects); join the text
    blocks so downstream sees clean text rather than a Python repr.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(item, "text", None)
            )
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(value)


def _content_text(result: ToolMessage | Command[Any]) -> str | None:
    """Return a ToolMessage's content flattened to text, or None for a Command."""
    if isinstance(result, ToolMessage):
        return _to_text(result.content)
    return None


def _task_id(content: str) -> str | None:
    if _STUB_MARKER not in content:
        return None
    match = _TASK_ID_RE.search(content)
    return match.group(1) if match else None


def _is_in_flight(content: str) -> bool:
    return any(flag in content for flag in _IN_FLIGHT)


def async_task_id(output: Any) -> str | None:
    """Return the async task id if ``output`` is a submission stub, else ``None``.

    Public helper for the TUI: an async tool's own ``on_tool_end`` fires early
    with this stub, so the TUI keys the deferred completion on the returned id
    and matches it against the ``ASYNC_DONE_EVENT`` this middleware dispatches.
    """
    # A tool-end event's output may be a ToolMessage or the raw content.
    content = getattr(output, "content", output)
    return _task_id(_to_text(content))


class AsyncTaskMiddleware(AgentMiddleware):
    """Poll ``get_task_status`` so async tool results resolve transparently."""

    def __init__(
        self,
        status_tool: BaseTool,
        *,
        timeout_s: float = 180.0,
        initial_interval_s: float = 0.25,
        factor: float = 1.6,
        max_interval_s: float = 2.0,
    ) -> None:
        super().__init__()
        self._status = status_tool
        self._timeout = timeout_s
        # Exponential backoff between polls: quick at first (fast tasks resolve in
        # a poll or two) then backing off so long tasks don't hammer the server.
        self._initial_interval = initial_interval_s
        self._factor = factor
        self._max_interval = max_interval_s

    def _replace(self, result: ToolMessage, content: Any) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=result.name,
            tool_call_id=result.tool_call_id,
            status=result.status,
        )

    def _next_interval(self, interval: float) -> float:
        return min(interval * self._factor, self._max_interval)

    # --- sync ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)
        content = _content_text(result)
        if content is None:
            return result
        task_id = _task_id(content)
        if task_id is None:
            return result
        assert isinstance(result, ToolMessage)
        deadline = time.monotonic() + self._timeout
        interval = self._initial_interval
        while True:
            text = _to_text(self._status.invoke({"task_id": task_id}))
            if not _is_in_flight(text) or time.monotonic() >= deadline:
                self._signal_done(task_id)
                return self._replace(result, text)
            time.sleep(interval)
            interval = self._next_interval(interval)

    def _signal_done(self, task_id: str) -> None:
        """Tell the TUI the async task finished (see ASYNC_DONE_EVENT)."""
        try:
            dispatch_custom_event(ASYNC_DONE_EVENT, {"task_id": task_id})
        except Exception:  # noqa: BLE001 - purely a UI hint; never break the call
            pass

    # --- async -----------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        content = _content_text(result)
        if content is None:
            return result
        task_id = _task_id(content)
        if task_id is None:
            return result
        assert isinstance(result, ToolMessage)
        deadline = time.monotonic() + self._timeout
        interval = self._initial_interval
        while True:
            text = _to_text(await self._status.ainvoke({"task_id": task_id}))
            if not _is_in_flight(text) or time.monotonic() >= deadline:
                await self._asignal_done(task_id)
                return self._replace(result, text)
            await asyncio.sleep(interval)
            interval = self._next_interval(interval)

    async def _asignal_done(self, task_id: str) -> None:
        """Tell the TUI the async task finished (see ASYNC_DONE_EVENT)."""
        try:
            await adispatch_custom_event(ASYNC_DONE_EVENT, {"task_id": task_id})
        except Exception:  # noqa: BLE001 - purely a UI hint; never break the call
            pass


def build_async_task_middleware(
    tools: list[BaseTool],
) -> AsyncTaskMiddleware | None:
    """Build the middleware from the live tool set, or ``None`` if unsupported.

    Returns ``None`` when the server exposes no ``get_task_status`` tool (nothing
    to poll), so the agent behaves exactly as before.
    """
    status_tool = next((t for t in tools if t.name == "get_task_status"), None)
    if status_tool is None:
        return None
    timeout_s = float(os.environ.get("GHIDRA_ASYNC_TIMEOUT", "180"))
    initial_s = float(os.environ.get("GHIDRA_ASYNC_POLL_INTERVAL", "0.25"))
    factor = float(os.environ.get("GHIDRA_ASYNC_POLL_FACTOR", "1.6"))
    max_s = float(os.environ.get("GHIDRA_ASYNC_POLL_MAX", "2.0"))
    return AsyncTaskMiddleware(
        status_tool,
        timeout_s=timeout_s,
        initial_interval_s=initial_s,
        factor=factor,
        max_interval_s=max_s,
    )
