"""
Unit tests for transparent async-task resolution and its context hygiene.

Focus: the middleware must (1) hide GhidrAssistMCP's async submission stub by
polling to the resolved result, (2) on timeout hand back an explicit terminal
message rather than a dangling ``Status: RUNNING`` stub, and (3) never expose
``get_task_status`` to any agent (it is withheld, so the model can't reintroduce
manual polling).

Run:  uv run pytest test_async_tasks.py -v
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from ghidra_deep_agent.async_tasks import (
    AsyncTaskMiddleware,
    _timeout_text,
    build_async_task_middleware,
    resolve_async_result,
)
from ghidra_deep_agent.subagents import WITHHELD_TOOLS, filter_withheld_tools

STUB = "Task submitted for async execution. Task ID: abc123def. Status: RUNNING"
DONE = "Analysis complete. Decompiled body: return a + b;"


def _status_tool(responses: list[str]) -> BaseTool:
    """A fake ``get_task_status`` tool yielding ``responses`` in order.

    The last response repeats once exhausted, so a poll loop that outlives the
    scripted responses keeps seeing the final value.
    """
    calls = {"i": 0}

    @tool
    def get_task_status(task_id: str) -> str:
        """Return the (scripted) status for a task id."""
        i = calls["i"]
        calls["i"] = min(i + 1, len(responses) - 1)
        return responses[i]

    return get_task_status


def _fast_middleware(status_tool: BaseTool, *, timeout_s: float) -> AsyncTaskMiddleware:
    """A middleware with zero backoff so tests don't actually sleep."""
    return AsyncTaskMiddleware(
        status_tool,
        timeout_s=timeout_s,
        initial_interval_s=0.0,
        factor=1.0,
        max_interval_s=0.0,
    )


def _stub_handler(request: object) -> ToolMessage:
    return ToolMessage(content=STUB, name="get_code", tool_call_id="call-1")


async def _astub_handler(request: object) -> ToolMessage:
    return ToolMessage(content=STUB, name="get_code", tool_call_id="call-1")


# --- resolution (stub never surfaces) -----------------------------------------


def test_sync_middleware_resolves_stub_to_result() -> None:
    mw = _fast_middleware(_status_tool(["Status: RUNNING", DONE]), timeout_s=5.0)
    result = mw.wrap_tool_call(object(), _stub_handler)  # type: ignore[arg-type]
    assert isinstance(result, ToolMessage)
    assert result.content == DONE
    assert "Task submitted" not in result.content


def test_async_middleware_resolves_stub_to_result() -> None:
    mw = _fast_middleware(_status_tool(["Status: PENDING", DONE]), timeout_s=5.0)
    result = asyncio.run(mw.awrap_tool_call(object(), _astub_handler))  # type: ignore[arg-type]
    assert isinstance(result, ToolMessage)
    assert result.content == DONE


def test_non_stub_result_passes_through_untouched() -> None:
    mw = _fast_middleware(_status_tool([DONE]), timeout_s=5.0)

    def handler(request: object) -> ToolMessage:
        return ToolMessage(content="0x401000", name="get_address", tool_call_id="c")

    result = mw.wrap_tool_call(object(), handler)  # type: ignore[arg-type]
    assert isinstance(result, ToolMessage)
    assert result.content == "0x401000"


# --- timeout (explicit terminal message, no dangling stub) --------------------


def _assert_timeout_message(content: object) -> None:
    assert isinstance(content, str)
    assert content == _timeout_text(0.0)
    # Must not look like an in-flight stub the model might try to poll.
    assert "Status: RUNNING" not in content
    assert "Status: PENDING" not in content
    assert "Task ID:" not in content


def test_sync_middleware_timeout_returns_explicit_message() -> None:
    mw = _fast_middleware(_status_tool(["Status: RUNNING"]), timeout_s=0.0)
    result = mw.wrap_tool_call(object(), _stub_handler)  # type: ignore[arg-type]
    assert isinstance(result, ToolMessage)
    _assert_timeout_message(result.content)


def test_async_middleware_timeout_returns_explicit_message() -> None:
    mw = _fast_middleware(_status_tool(["Status: RUNNING"]), timeout_s=0.0)
    result = asyncio.run(mw.awrap_tool_call(object(), _astub_handler))  # type: ignore[arg-type]
    assert isinstance(result, ToolMessage)
    _assert_timeout_message(result.content)


def test_resolve_async_result_timeout_returns_explicit_message() -> None:
    status = _status_tool(["Status: RUNNING"])
    out = asyncio.run(
        resolve_async_result(
            STUB, status, timeout_s=0.0, initial_interval_s=0.0, factor=1.0
        )
    )
    _assert_timeout_message(out)


def test_resolve_async_result_resolves_stub() -> None:
    status = _status_tool(["Status: RUNNING", DONE])
    out = asyncio.run(
        resolve_async_result(
            STUB, status, timeout_s=5.0, initial_interval_s=0.0, factor=1.0
        )
    )
    assert out == DONE


# --- withholding (the model never receives get_task_status) -------------------


def test_get_task_status_is_withheld() -> None:
    assert "get_task_status" in WITHHELD_TOOLS


def test_filter_withheld_tools_drops_get_task_status() -> None:
    @tool
    def get_code(function: str) -> str:
        """Decompile a function."""
        return DONE

    status = _status_tool([DONE])
    kept = filter_withheld_tools([status, get_code])
    names = {t.name for t in kept}
    assert names == {"get_code"}


def test_middleware_still_built_from_raw_tool_list() -> None:
    # Withholding strips get_task_status from agent allowlists, but the middleware
    # is built from the raw MCP tool list (upstream of the filter), so it must
    # still find the tool there.
    status = _status_tool([DONE])
    assert build_async_task_middleware([status]) is not None
    assert build_async_task_middleware([]) is None
