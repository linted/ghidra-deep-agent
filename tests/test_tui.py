"""Smoke tests for the TUI, driven with Textual's pilot (no backend services)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ghidra_deep_agent.tui import GhidraAgentApp
from ghidra_deep_agent.tui.events import parse_checkpoint_ns
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.widgets import (
    ActivityTree,
    CommandInput,
    ResponseLog,
    StatusBar,
)


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class _LLMOutput:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class StubAgent:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def astream_events(
        self, _input: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "run_id": "1", "metadata": {}}
        yield {
            "event": "on_chat_model_stream",
            "run_id": "1",
            "metadata": {},
            "data": {"chunk": _Chunk("hello from stub")},
        }
        if self.delay:
            await asyncio.sleep(self.delay)
        yield {
            "event": "on_chat_model_end",
            "run_id": "1",
            "metadata": {},
            "data": {"output": _LLMOutput()},
        }


def _make_app(agent: Any | None = None) -> GhidraAgentApp:
    return GhidraAgentApp(
        agent=agent if agent is not None else StubAgent(),
        config={},
        model="test-model",
        session_id="abc",
    )


def test_parse_checkpoint_ns() -> None:
    assert parse_checkpoint_ns("") == ()
    assert parse_checkpoint_ns("tools:a") == ("tools:a",)
    assert parse_checkpoint_ns("tools:a|tools:b") == ("tools:a", "tools:b")


def test_mount_widgets_and_tree_toggle() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            assert app.theme == "ghidra"
            app.query_one(ActivityTree)
            app.query_one(ResponseLog)
            app.query_one(StatusBar)
            assert app.query_one(CommandInput).has_focus
            await pilot.press("ctrl+t")
            assert app.query_one("#panes").has_class("hide-tree")
            await pilot.press("ctrl+t")
            assert not app.query_one("#panes").has_class("hide-tree")

    asyncio.run(run())


def test_help_screen_opens_and_closes() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app.query_one(CommandInput).value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)

    asyncio.run(run())


def test_input_history_walking() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            inp = app.query_one(CommandInput)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.press("escape")
            await pilot.pause()
            assert inp.value == ""
            await pilot.press("up")
            assert inp.value == "/help"
            await pilot.press("down")
            assert inp.value == ""

    asyncio.run(run())


def test_run_streams_response_into_transcript() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app.query_one(CommandInput).value = "analyze main"
            await pilot.press("enter")
            await pilot.pause(0.3)
            log = app.query_one(ResponseLog)
            assert log.transcript[0] == "❯ analyze main"
            assert log.transcript[1] == "hello from stub"
            assert log.last_response == "hello from stub"
            assert not app._agent_running

    asyncio.run(run())


def test_escape_cancels_running_agent() -> None:
    async def run() -> None:
        app = _make_app(StubAgent(delay=5.0))
        async with app.run_test() as pilot:
            app.query_one(CommandInput).value = "long task"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert app._agent_running
            await pilot.press("escape")
            await pilot.pause(0.3)
            assert not app._agent_running

    asyncio.run(run())
