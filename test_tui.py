"""Smoke tests for the TUI, driven with Textual's pilot (no backend services)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ghidra_deep_agent.tui import GhidraAgentApp
from ghidra_deep_agent.tui.events import handle_event, parse_checkpoint_ns
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.messages import SubagentReport
from ghidra_deep_agent.tui.report_screen import SubagentReportScreen
from ghidra_deep_agent.tui.widgets import (
    ActivityTree,
    CommandInput,
    ResponseLog,
    StatusBar,
    ThinkingPanel,
)


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class _LLMOutput:
    # Mirrors a real on_chat_model_end output: an AIMessage-like object carrying
    # both the final text (`content`, which the response pane renders) and token
    # usage. Streamed chunks feed the thinking panel; the final response text is
    # taken from this end-event output.
    def __init__(self, content: str = "") -> None:
        self.content = content
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
            "data": {"output": _LLMOutput("hello from stub")},
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


def test_nested_tool_calls_are_hidden() -> None:
    """A tool invoked from inside another tool's body (recover_prototypes →
    scripts) is suppressed entirely — no tree row, no deferred-async leak."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "outer",
                    "name": "recover_prototypes",
                    "metadata": {},
                    "parent_ids": [],
                    "data": {"input": {"dry_run": False}},
                }
            )
            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "inner",
                    "name": "scripts",
                    "metadata": {},
                    "parent_ids": ["chain", "outer"],
                    "data": {"input": {"action": "run"}},
                }
            )
            await pilot.pause()
            assert "outer" in activity._run_map
            assert "inner" not in activity._run_map
            assert "inner" in app._hidden_tool_runs
            assert len(activity.root.children) == 1

            # The hidden run ends with an async submission stub; it must not
            # register a deferred completion (there is no middleware to
            # dispatch ASYNC_DONE_EVENT for direct-invoke calls).
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "inner",
                    "metadata": {},
                    "data": {"output": "Script task submitted: abc123"},
                }
            )
            assert app._pending_async == {}
            assert "inner" not in app._hidden_tool_runs

            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "outer",
                    "metadata": {},
                    "data": {"output": "Prototype recovery pass complete."},
                }
            )
            await pilot.pause()
            assert app._active_tool_runs == set()
            assert "✓" in str(activity._run_map["outer"][0].label)

    asyncio.run(run())


def test_subagent_inner_tools_stay_visible() -> None:
    """Tool calls made by a sub-agent have the `task` run in their ancestry
    but must not be hidden — they are the sub-agent's real work."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "task1",
                    "name": "task",
                    "metadata": {"langgraph_checkpoint_ns": "tools:a"},
                    "parent_ids": [],
                    "data": {"input": {"description": "research"}},
                }
            )
            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "sub_tool",
                    "name": "get_code",
                    "metadata": {"langgraph_checkpoint_ns": "tools:a|tools:b"},
                    "parent_ids": ["task1"],
                    "data": {"input": {"address": "0x1000"}},
                }
            )
            await pilot.pause()
            assert "sub_tool" not in app._hidden_tool_runs
            assert "sub_tool" in activity._run_map

    asyncio.run(run())


class _FakeToolMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


class _FakeCommand:
    """Shape of a `task` tool's output: Command(update={"messages": [...]})."""

    def __init__(self, text: str) -> None:
        self.update = {"messages": [_FakeToolMessage(text)]}


def _task_start_event(run_id: str, description: str) -> dict[str, Any]:
    return {
        "event": "on_tool_start",
        "run_id": run_id,
        "name": "task",
        "metadata": {"langgraph_checkpoint_ns": "tools:a"},
        "parent_ids": [],
        "data": {"input": {"description": description, "subagent_type": "research"}},
    }


def test_subagent_report_captured() -> None:
    """A completed `task` run's full report is captured for the ctrl+o viewer."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            description = "map the crypto init path and report entry points"
            emit(_task_start_event("task1", description))
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "task1",
                    "metadata": {},
                    "data": {"output": _FakeCommand("## Findings\n- entry at 0x1000")},
                }
            )
            await pilot.pause()
            assert app._subagent_meta == {}
            [report] = app._subagent_reports
            assert report.run_id == "task1"
            assert report.description == description  # untruncated
            assert report.text == "## Findings\n- entry at 0x1000"
            assert report.error is False
            assert "✓" in str(activity._run_map["task1"][0].label)

    asyncio.run(run())


def test_subagent_report_skips_async_stub_detection() -> None:
    """A report that merely quotes an async submission stub must not defer the
    task node — `task` is a local tool and never completes asynchronously."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            emit(_task_start_event("task1", "run the export script"))
            text = "Task submitted for async execution. Task ID: deadbeef"
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "task1",
                    "metadata": {},
                    "data": {"output": _FakeCommand(text)},
                }
            )
            await pilot.pause()
            assert app._pending_async == {}
            [report] = app._subagent_reports
            assert report.text == text
            assert "✓" in str(activity._run_map["task1"][0].label)

    asyncio.run(run())


def test_subagent_report_error_and_empty() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            emit(_task_start_event("bad", "doomed run"))
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "bad",
                    "metadata": {},
                    "data": {"output": None, "error": ValueError("boom")},
                }
            )
            emit(_task_start_event("quiet", "silent run"))
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "quiet",
                    "metadata": {},
                    "data": {"output": _FakeCommand("")},
                }
            )
            await pilot.pause()
            errored, empty = app._subagent_reports
            assert errored.error is True
            assert empty.error is False
            assert empty.text == ""

    asyncio.run(run())


def test_report_screen_opens_and_closes() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            # No reports yet: ctrl+o only warns, no screen pushed.
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert not isinstance(app.screen, SubagentReportScreen)

            app._subagent_reports = [
                SubagentReport("r1", "first run", "old report", False, 1.0),
                SubagentReport("r2", "second run", "new report", False, 2.0),
            ]
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, SubagentReportScreen)
            # Most recent run is listed and selected first.
            screen = app.screen
            assert screen._reports[0].run_id == "r2"
            assert screen._selected() is screen._reports[0]
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, SubagentReportScreen)

    asyncio.run(run())


def test_report_screen_copy() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app._subagent_reports = [
                SubagentReport("r1", "the run", "the full report text", False, 1.0),
            ]
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, SubagentReportScreen)
            await pilot.press("ctrl+y")
            await pilot.pause()
            assert app.clipboard == "the full report text"

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
