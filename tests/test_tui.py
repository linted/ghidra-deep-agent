"""Smoke tests for the TUI, driven with Textual's pilot (no backend services)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from textual.widgets import Label

from ghidra_deep_agent.compaction import ManualCompactionResult
from ghidra_deep_agent.resilience import UsageLimitError
from ghidra_deep_agent.tui import GhidraAgentApp
from ghidra_deep_agent.tui.commands import COMMANDS, help_lines
from ghidra_deep_agent.tui.events import handle_event, parse_checkpoint_ns
from ghidra_deep_agent.tui.formatting import truncate_line
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.messages import AgentDone, SubagentReport
from ghidra_deep_agent.tui.report_screen import SubagentReportScreen
from ghidra_deep_agent.tui.side_mode import SideMode
from ghidra_deep_agent.tui.widgets import (
    ActivityTree,
    CommandInput,
    ResponseLog,
    StatusBar,
    ThinkingPanel,
)
from ghidra_deep_agent.tui.widgets.command_input import SLASH_COMMANDS


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


def _make_app(
    agent: Any | None = None, compaction_engine: Any | None = None
) -> GhidraAgentApp:
    return GhidraAgentApp(
        agent=agent if agent is not None else StubAgent(),
        config={},
        compaction_engine=compaction_engine,
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
            assert "inner" in app.run_state.hidden_tool_runs
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
            assert app.run_state.pending_async == {}
            assert "inner" not in app.run_state.hidden_tool_runs

            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "outer",
                    "metadata": {},
                    "data": {"output": "Prototype recovery pass complete."},
                }
            )
            await pilot.pause()
            assert app.run_state.active_tool_runs == set()
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
            assert "sub_tool" not in app.run_state.hidden_tool_runs
            assert "sub_tool" in activity._run_map

    asyncio.run(run())


def test_nested_subagents_nest_under_the_nearest_ancestor() -> None:
    """`_find_parent` must pick the longest *proper* namespace prefix.

    A tool two sub-agents deep belongs under the inner sub-agent, not the
    outer one and not the root.
    """

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            for run_id, ns in (("outer", "tools:a"), ("inner", "tools:a|tools:b")):
                emit(
                    {
                        "event": "on_tool_start",
                        "run_id": run_id,
                        "name": "task",
                        "metadata": {"langgraph_checkpoint_ns": ns},
                        "parent_ids": [],
                        "data": {"input": {"description": run_id}},
                    }
                )
            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "leaf",
                    "name": "get_code",
                    "metadata": {"langgraph_checkpoint_ns": "tools:a|tools:b|tools:c"},
                    "parent_ids": ["inner"],
                    "data": {"input": {"address": "0x1000"}},
                }
            )
            await pilot.pause()

            outer_node = activity._run_map["outer"][0]
            inner_node = activity._run_map["inner"][0]
            leaf_node = activity._run_map["leaf"][0]
            assert inner_node.parent is outer_node
            assert leaf_node.parent is inner_node

            # An unrelated namespace has no registered ancestor -> root.
            assert activity._find_parent("tools:zzz") is activity.root

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
            assert app.run_state.subagent_meta == {}
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
            assert app.run_state.pending_async == {}
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


def test_report_row_titles_stop_at_the_first_newline() -> None:
    # A multiline task description must not break the one-line list rows.
    assert truncate_line("analyze FUN_1400\nand also FUN_1500", 60) == (
        "analyze FUN_1400"
    )
    assert truncate_line("x" * 100, 60) == "x" * 60
    assert truncate_line("", 60) == ""
    assert truncate_line("\nleading newline", 60) == ""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app._subagent_reports = [
                SubagentReport("r1", "line one\nline two", "report", False, 1.0),
            ]
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, SubagentReportScreen)
            labels = app.screen.query_one("#report-list").query(Label)
            label = str(labels.first().content)
            assert "line one" in label
            assert "\n" not in label

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


def test_session_touch_is_not_cancelled_by_the_agent_run() -> None:
    """`_run_agent` is exclusive, so a shared worker group would kill the touch.

    Textual cancels every worker in an exclusive worker's group on the same node.
    Both used to default to "default", so `_start_run` killed the touch worker it
    had just launched — no session ever recorded a title or a fresh
    `last_active_at`, and `/resume` listed everything as "(no title)".
    """
    touched: list[tuple[str, str | None]] = []

    class RecordingStore:
        async def atouch(
            self, session_id: str, first_prompt: str | None = None
        ) -> None:
            touched.append((session_id, first_prompt))

        async def arecord_start(self, session_id: str, binary_name: str) -> None:
            pass

    async def run() -> None:
        app = _make_app(StubAgent())
        app._session_store = cast(Any, RecordingStore())
        async with app.run_test() as pilot:
            app.query_one(CommandInput).value = "analyze the entry point"
            await pilot.press("enter")
            await pilot.pause(0.3)

    asyncio.run(run())

    assert touched == [("abc", "analyze the entry point")]


def test_cancelled_run_leaves_no_active_tool_count() -> None:
    """A cancelled turn never delivers on_tool_end for what was in flight.

    Without a reset the app-side bookkeeping keeps those run_ids forever and the
    status bar's "⚙ N active" never returns to zero for the rest of the session.
    """

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            # Tools that started and will never report completion, as after an
            # Escape-cancel or an exception mid-stream.
            app.run_state.active_tool_runs.add("run-1")
            app.run_state.hidden_tool_runs.add("run-2")
            app.run_state.pending_async["task-1"] = "run-3"
            app.run_state.subagent_meta["run-4"] = ("recon", 0.0)
            app.query_one(StatusBar).active_tools = 3
            await pilot.pause()

            app._reset_run_bookkeeping()

            assert app.run_state.active_tool_runs == set()
            assert app.run_state.hidden_tool_runs == set()
            assert app.run_state.pending_async == {}
            assert app.run_state.subagent_meta == {}
            assert app.query_one(StatusBar).active_tools == 0

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


class _LimitAgent:
    """A stub whose stream halts on a usage limit before yielding anything."""

    async def astream_events(
        self, _input: Any, config: Any, version: str
    ) -> AsyncIterator[dict[str, Any]]:
        raise UsageLimitError(RuntimeError("429 rate limit"))
        yield  # pragma: no cover - marks this an async generator


def test_continue_resumes_active_side_mode() -> None:
    """`/continue` resumes the active plan/ask thread, not just the main one."""

    async def run() -> None:
        for flag, cfg_attr in (
            ("_plan_mode", "_plan_config"),
            ("_ask_mode", "_ask_config"),
        ):
            app = _make_app()
            async with app.run_test() as pilot:
                called: list[bool] = []
                # Bind `called` per iteration; it is rebound by the enclosing loop.
                setattr(app, "_resume_run", lambda c=called: c.append(True))
                setattr(app, flag, True)
                setattr(app, cfg_attr, {"configurable": {"thread_id": "abc::x"}})
                app._dispatch_slash("/continue")
                await pilot.pause()
                assert called == [True], flag

    asyncio.run(run())


def test_continue_keeps_the_activity_tree() -> None:
    """`/continue` resumes the same turn, so its tree must survive.

    It used to call `ActivityTree.reset()`, wiping the pane — and since the
    sub-agents LangGraph restores from pending writes emit no events, that
    history never came back. What was in flight when the pause hit is frozen
    instead, because its `on_tool_end` will never arrive.
    """

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)
            # Decorated with @work, so its bound type isn't a plain callable.
            setattr(app, "_run_agent", lambda _q: None)

            def emit(event: dict[str, Any]) -> None:
                handle_event(app, event, activity, response, thinking)

            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "done",
                    "name": "get_code",
                    "metadata": {},
                    "parent_ids": [],
                    "data": {"input": {"address": "0x1000"}},
                }
            )
            emit(
                {
                    "event": "on_tool_end",
                    "run_id": "done",
                    "metadata": {},
                    "data": {"output": "int main(void) {}"},
                }
            )
            emit(
                {
                    "event": "on_tool_start",
                    "run_id": "interrupted",
                    "name": "decompile",
                    "metadata": {},
                    "parent_ids": [],
                    "data": {"input": {"address": "0x2000"}},
                }
            )
            await pilot.pause()
            finished = activity._run_map["done"][0]
            paused = activity._run_map["interrupted"][0]

            app._resume_run()
            await pilot.pause()

            labels = [str(node.label) for node in activity.root.children]
            assert "✓" in str(finished.label), "completed work was wiped"
            assert "⏸ paused" in str(paused.label), "in-flight node still reads as live"
            assert "interrupted" not in activity._run_map, "a run that can never end"
            assert activity._pending_runs == set()
            assert any("↻ continued" in label for label in labels)

    asyncio.run(run())


def test_resumed_subagent_reuses_its_node() -> None:
    """A replayed task keeps its (deterministic) id, so it must not grow a twin."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            activity = app.query_one(ActivityTree)
            response = app.query_one(ResponseLog)
            thinking = app.query_one(ThinkingPanel)
            # Decorated with @work, so its bound type isn't a plain callable.
            setattr(app, "_run_agent", lambda _q: None)

            def start_subagent(run_id: str) -> None:
                handle_event(
                    app,
                    {
                        "event": "on_tool_start",
                        "run_id": run_id,
                        "name": "task",
                        "metadata": {"langgraph_checkpoint_ns": "tools:abc"},
                        "parent_ids": [],
                        "data": {"input": {"description": "research"}},
                    },
                    activity,
                    response,
                    thinking,
                )

            start_subagent("task1")
            await pilot.pause()
            original = activity._ns_to_node[("tools:abc",)]
            before = len(activity.root.children)

            app._resume_run()
            await pilot.pause()
            start_subagent("task1-replayed")
            await pilot.pause()

            # Only the "↻ continued" marker was added.
            assert len(activity.root.children) == before + 1
            assert activity._ns_to_node[("tools:abc",)] is original
            assert activity._run_map["task1-replayed"][0] is original
            assert "●" in str(original.label), "the relit node still reads as paused"

    asyncio.run(run())


def test_entering_a_side_mode_always_mints_a_thread() -> None:
    """ "Mode on, no thread" used to need a runtime guard on every path.

    A side-mode turn must never fall back to the main session's thread, and the
    flags previously allowed a state where it could. SideMode carries its config,
    so the invalid state is unrepresentable and `/continue` is always safe.
    """

    async def run() -> None:
        for kind in ("plan", "ask"):
            app = _make_app()
            async with app.run_test() as pilot:
                app._start_run = lambda *a: None  # type: ignore[method-assign]
                app._enter_side_mode(kind, "something")
                assert app._side is not None, kind
                thread = app._side.config["configurable"]["thread_id"]
                assert thread.startswith(f"{app._session_id}::{kind}::"), kind
                await pilot.pause()

    asyncio.run(run())


def test_every_command_is_dispatchable_documented_and_autocompleted() -> None:
    """The three views of a command are generated from one table.

    They used to be written out separately, so a command could be dispatchable
    but missing from autocomplete, or documented but no longer handled.
    """
    app = _make_app()
    handlers = set(app._slash_handlers())
    declared = {c.name for c in COMMANDS}

    assert handlers == declared, "handler map and command table disagree"
    assert set(SLASH_COMMANDS) == declared, "autocomplete list is out of sync"
    documented = "\n".join(help_lines())
    for name in declared:
        assert name in documented, f"{name} is undocumented"


def test_run_starting_commands_are_marked_needs_idle() -> None:
    """The busy guard is now a table flag, not seven copy-pasted branches."""
    by_name = {c.name: c for c in COMMANDS}
    for name in ("/compact", "/resume", "/continue", "/plan", "/ask", "/approve"):
        assert by_name[name].needs_idle, f"{name} could interleave two runs"
    for name in ("/clear", "/yank", "/help", "/quit"):
        assert not by_name[name].needs_idle


def test_commands_that_start_a_run_are_refused_while_busy() -> None:
    """Every run-starting command shares one busy guard; none may slip through."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            started: list[Any] = []
            app._start_run = lambda *a: started.append(a)  # type: ignore[method-assign]
            app._resume_run = lambda: started.append(("resume",))  # type: ignore[method-assign]
            # Decorated with @work, so its bound type isn't a plain callable.
            setattr(app, "_open_resume_picker", lambda: started.append(("picker",)))
            app._agent_running = True

            for command in ("/compact", "/resume", "/continue", "/plan x", "/ask y"):
                app._dispatch_slash(command)
            await pilot.pause()

            assert started == []

    asyncio.run(run())


class _CompactStubAgent(StubAgent):
    """StubAgent whose thread state can be read and written — never streamed.

    ``/compact`` must run out-of-band: one state read, one state write, and no
    agent turn at all. Streaming through this stub is therefore an error.
    """

    def __init__(self, messages: list[Any] | None = None, event: Any = None) -> None:
        super().__init__()
        self.messages = messages if messages is not None else []
        self.event = event
        self.updates: list[tuple[dict[str, Any], str | None]] = []

    def astream_events(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("/compact must not start an agent stream")

    async def aget_state(self, config: Any) -> Any:
        return SimpleNamespace(
            values={"messages": self.messages, "_summarization_event": self.event}
        )

    async def aupdate_state(
        self, config: Any, values: dict[str, Any], as_node: str | None = None
    ) -> None:
        self.updates.append((values, as_node))


def test_compact_persists_the_event_without_an_agent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/compact drives the summarization engine directly: the only state change
    is a `_summarization_event` written as the tools node."""

    async def run() -> None:
        agent = _CompactStubAgent(messages=["m"] * 20, event={"cutoff_index": 3})
        app = _make_app(agent=agent, compaction_engine=object())
        seen: list[tuple[Any, ...]] = []
        result = ManualCompactionResult(
            event={"cutoff_index": 14, "summary_message": "s", "file_path": "h.md"},
            summarized_count=12,
            file_path="h.md",
        )

        async def fake_compact(
            engine: Any, messages: Any, prior: Any, *, thread_id: str
        ) -> ManualCompactionResult:
            seen.append((engine, list(messages), prior, thread_id))
            return result

        monkeypatch.setattr(
            "ghidra_deep_agent.tui.app.compact_out_of_band", fake_compact
        )
        async with app.run_test() as pilot:
            app._dispatch_slash("/compact")
            await pilot.pause(0.2)
            # The driver got the thread's real state, keyed by the session id.
            assert seen == [
                (app._compaction_engine, ["m"] * 20, {"cutoff_index": 3}, "abc")
            ]
            assert agent.updates == [({"_summarization_event": result.event}, "tools")]
            assert app.query_one(ResponseLog).transcript == ["❯ /compact"]
            assert not app._agent_running

    asyncio.run(run())


def test_compact_with_nothing_to_do_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        agent = _CompactStubAgent()
        app = _make_app(agent=agent, compaction_engine=object())

        async def fake_compact(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            "ghidra_deep_agent.tui.app.compact_out_of_band", fake_compact
        )
        async with app.run_test() as pilot:
            app._dispatch_slash("/compact")
            await pilot.pause(0.2)
            assert agent.updates == []
            assert not app._agent_running

    asyncio.run(run())


def test_compact_failure_leaves_the_thread_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summary-model failure must be reported, not persisted — and must not
    leave the app stuck busy."""

    async def run() -> None:
        agent = _CompactStubAgent(messages=["m"] * 20)
        app = _make_app(agent=agent, compaction_engine=object())

        async def fake_compact(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("summary model down")

        monkeypatch.setattr(
            "ghidra_deep_agent.tui.app.compact_out_of_band", fake_compact
        )
        async with app.run_test() as pilot:
            app._dispatch_slash("/compact")
            await pilot.pause(0.2)
            assert agent.updates == []
            assert not app._agent_running

    asyncio.run(run())


def test_compact_is_refused_in_a_side_mode() -> None:
    """Side-mode threads are ephemeral; /compact must neither touch them nor
    silently reach around to the main thread."""

    async def run() -> None:
        agent = _CompactStubAgent(messages=["m"] * 20)
        app = _make_app(agent=agent, compaction_engine=object())
        async with app.run_test() as pilot:
            app._side = SideMode(kind="plan", config={}, plan_path=None)
            app._dispatch_slash("/compact")
            await pilot.pause(0.2)
            assert agent.updates == []
            assert not app._agent_running

    asyncio.run(run())


def test_compact_without_an_engine_flashes_and_does_nothing() -> None:
    async def run() -> None:
        agent = _CompactStubAgent(messages=["m"] * 20)
        app = _make_app(agent=agent)  # no compaction engine
        async with app.run_test() as pilot:
            app._dispatch_slash("/compact")
            await pilot.pause(0.2)
            assert agent.updates == []
            assert not app._agent_running

    asyncio.run(run())


def test_side_modes_mint_one_thread_and_reuse_it() -> None:
    """Entering a side mode mints an ephemeral thread; staying in it reuses it."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app._start_run = lambda *a: None  # type: ignore[method-assign]

            app._enter_side_mode("plan", "first goal")
            assert app._side is not None
            first_config = app._side.config
            first_path = app._side.plan_path
            assert app._side.needs_seed is True

            # Already in plan mode: the same plan file and thread keep being revised.
            app._side.needs_seed = False
            app._enter_side_mode("plan", "second goal")
            assert app._side is not None
            assert app._side.config == first_config
            assert app._side.plan_path == first_path
            assert app._side.needs_seed is False

            # The thread is this session's, namespaced by mode.
            thread_id = first_config["configurable"]["thread_id"]
            assert thread_id.startswith(f"{app._session_id}::plan::")
            assert "recursion_limit" in first_config

            await pilot.pause()

    asyncio.run(run())


def test_plan_and_ask_are_mutually_exclusive() -> None:
    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app._start_run = lambda *a: None  # type: ignore[method-assign]

            app._enter_side_mode("plan", "goal")
            assert app._side is not None and app._side.is_plan

            app._enter_side_mode("ask", "question")
            assert app._side is not None
            assert app._side.is_ask and not app._side.is_plan
            assert app._side.plan_path is None
            assert "::ask::" in app._side.config["configurable"]["thread_id"]

            app._enter_side_mode("plan", "goal again")
            assert app._side is not None
            assert app._side.is_plan and not app._side.is_ask
            assert "::plan::" in app._side.config["configurable"]["thread_id"]

            await pilot.pause()

    asyncio.run(run())


def test_usage_limit_banner_in_plan_mode_omits_relaunch() -> None:
    """The pause banner for a side-mode run must not advise `--session-id`
    relaunch — the ephemeral thread isn't restorable across launches."""

    async def run() -> None:
        app = _make_app()
        app._plan_agent = _LimitAgent()
        async with app.run_test() as pilot:
            log = app.query_one(ResponseLog)
            writes: list[str] = []
            orig_write = log.write

            def capture(content: Any, *args: Any, **kwargs: Any) -> Any:
                if isinstance(content, str):
                    writes.append(content)
                return orig_write(content, *args, **kwargs)

            log.write = capture  # type: ignore[method-assign]
            app._side = SideMode(
                kind="plan",
                config={"configurable": {"thread_id": "abc::plan::x"}},
                plan_path="plans/x.md",
            )
            app._start_run("plan turn", "hi")
            await pilot.pause(0.3)
            joined = "\n".join(writes)
            assert "plan-mode" in joined
            assert "--session-id" not in joined

    asyncio.run(run())


def test_empty_turn_renders_placeholder_not_nothing() -> None:
    """A turn that ends with no reply must say so, not leave a blank pane."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test():
            log = app.query_one(ResponseLog)
            before = len(log.lines)
            log.on_agent_done(AgentDone())
            assert len(log.lines) > before  # placeholder rendered

            before = len(log.lines)
            log.on_agent_done(AgentDone(errored=True))
            assert len(log.lines) == before  # the error box already told the story

    asyncio.run(run())


def test_guard_appended_reply_is_surfaced_after_stream() -> None:
    """MainReplyGuardMiddleware appends its reply from a graph node, invisible
    to on_chat_model_end capture — the app must read it back from final state."""

    salvaged = "[Reply recovered: the agent ended its turn without a final reply]"

    class _GuardStub(StubAgent):
        async def aget_state(self, _config: Any) -> Any:
            return SimpleNamespace(values={"messages": [AIMessage(salvaged)]})

    async def run() -> None:
        app = _make_app(agent=_GuardStub())
        async with app.run_test() as pilot:
            app.query_one(CommandInput).value = "answer the questions"
            await pilot.press("enter")
            await pilot.pause(0.3)
            log = app.query_one(ResponseLog)
            assert salvaged in log.transcript

    asyncio.run(run())
