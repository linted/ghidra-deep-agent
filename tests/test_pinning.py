"""Pinning every Ghidra tool call to one program, and catching a silent fallback."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from ghidra_deep_agent.pinning import (
    UNPINNED_TOOLS,
    PinMismatchError,
    matches_program,
    parse_operating_on,
    pin_program,
    result_text,
    verify_pinned,
)
from ghidra_deep_agent.program_resolver import ProgramRef

PIN = "/v1/app.exe"


def _request(name: str, args: dict[str, Any] | None = None) -> MCPToolCallRequest:
    return MCPToolCallRequest(name=name, args=args or {}, server_name="ghidra")


class _Block:
    """An mcp TextContent-like object."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _CallToolResult:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


def test_parse_operating_on() -> None:
    assert parse_operating_on("[Context] Operating on: app.exe\n\nbody") == "app.exe"
    assert (
        parse_operating_on("[Context] Operating on: app.exe | Active window: other\n")
        == "app.exe"
    )
    assert parse_operating_on("[Context] Active window: app.exe\n") is None
    assert parse_operating_on("no context here") is None


def test_matches_program_accepts_name_or_path() -> None:
    assert matches_program("app.exe", PIN)
    assert matches_program(PIN, PIN)
    assert not matches_program("app2.exe", PIN)
    assert not matches_program("/v2/app.exe", PIN)


def test_result_text_shapes() -> None:
    assert result_text("plain") == "plain"
    assert result_text([{"type": "text", "text": "block"}]) == "block"
    assert result_text(_CallToolResult("ctr")) == "ctr"


def _run(interceptor: Any, request: MCPToolCallRequest, reply: Any) -> Any:
    seen: list[MCPToolCallRequest] = []

    async def handler(req: MCPToolCallRequest) -> Any:
        seen.append(req)
        return reply

    result = asyncio.run(interceptor(request, handler))
    return result, seen


def test_pin_overrides_program_name_even_when_supplied() -> None:
    interceptor = pin_program(PIN)
    result, seen = _run(
        interceptor,
        _request("get_code", {"address": "0x1000", "program_name": "/v2/app.exe"}),
        "[Context] Operating on: app.exe\n\ncode",
    )
    assert result == "[Context] Operating on: app.exe\n\ncode"
    assert seen[0].args == {"address": "0x1000", "program_name": PIN}
    assert seen[0].name == "get_code"


def test_unpinned_tools_pass_through_untouched() -> None:
    interceptor = pin_program(PIN)
    for name in sorted(UNPINNED_TOOLS):
        _, seen = _run(
            interceptor, _request(name, {"x": 1}), "[Context] Operating on: zzz"
        )
        assert seen[0].args == {"x": 1}, name


def test_mismatch_raises_and_reports() -> None:
    reported: list[tuple[str, str]] = []
    interceptor = pin_program(PIN, on_mismatch=lambda n, t: reported.append((n, t)))
    with pytest.raises(PinMismatchError, match="pinned to /v1/app.exe.*'app2.exe'"):
        _run(
            interceptor,
            _request("rename_symbol"),
            _CallToolResult(
                "[Context] Operating on: app2.exe | Active window: app2.exe"
            ),
        )
    assert reported == [("rename_symbol", "app2.exe")]


def test_missing_context_line_passes() -> None:
    result, _ = _run(pin_program(PIN), _request("get_code"), "older server output")
    assert result == "older server output"


def test_verify_can_be_disabled() -> None:
    result, _ = _run(
        pin_program(PIN, verify=False),
        _request("get_code"),
        "[Context] Operating on: app2.exe",
    )
    assert result.startswith("[Context]")


def test_composes_under_an_error_wrapper() -> None:
    """The outer error interceptor turns the mismatch into a tool-failure string."""

    async def handle_errors(request: MCPToolCallRequest, handler: Any) -> Any:
        try:
            return await handler(request)
        except Exception as exc:
            return f"Tool '{request.name}' failed: {exc}"

    pin = pin_program(PIN)

    async def execute(request: MCPToolCallRequest) -> Any:
        return "[Context] Operating on: app2.exe"

    async def chain(request: MCPToolCallRequest) -> Any:
        return await handle_errors(request, lambda r: pin(r, execute))

    out = asyncio.run(chain(_request("get_code")))
    assert out.startswith("Tool 'get_code' failed:") and "pinned to" in out


class _Probe:
    name = "get_binary_info"

    def __init__(self, reply: Any) -> None:
        self.reply = reply

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        return self.reply


def test_verify_pinned() -> None:
    ref = ProgramRef("app.exe", PIN)
    asyncio.run(verify_pinned([_Probe("[Context] Operating on: app.exe\n\ninfo")], ref))
    asyncio.run(verify_pinned([_Probe("no context")], ref))
    asyncio.run(verify_pinned([], ref))  # no probe tool: nothing to check
    with pytest.raises(PinMismatchError):
        asyncio.run(verify_pinned([_Probe("[Context] Operating on: other.exe")], ref))
    with pytest.raises(PinMismatchError):
        asyncio.run(
            verify_pinned(
                [_Probe("Tool 'x' failed: is pinned to /v1 but operated on 'o'")], ref
            )
        )
