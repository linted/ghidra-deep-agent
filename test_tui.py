"""Smoke tests for the TUI, driven with Textual's pilot (no backend services)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ghidra_deep_agent.resilience import UsageLimitError
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


def test_continue_in_side_mode_without_thread_is_a_noop() -> None:
    """A side-mode flag set with no minted thread flashes instead of resuming."""

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
                setattr(app, cfg_attr, None)
                app._dispatch_slash("/continue")
                await pilot.pause()
                assert called == [], flag

    asyncio.run(run())


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


def test_side_modes_mint_one_thread_and_reuse_it() -> None:
    """Entering a side mode mints an ephemeral thread; staying in it reuses it."""

    async def run() -> None:
        app = _make_app()
        async with app.run_test() as pilot:
            app._start_run = lambda *a: None  # type: ignore[method-assign]

            app._enter_plan_mode("first goal")
            first_config = app._plan_config
            first_path = app._plan_path
            assert first_config is not None
            assert app._plan_needs_seed is True

            # Already in plan mode: the same plan file and thread keep being revised.
            app._plan_needs_seed = False
            app._enter_plan_mode("second goal")
            assert app._plan_config == first_config
            assert app._plan_path == first_path
            assert app._plan_needs_seed is False

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

            app._enter_plan_mode("goal")
            assert app._plan_mode is True

            app._enter_ask_mode("question")
            assert app._ask_mode is True
            assert app._plan_mode is False
            assert app._plan_config is None
            assert app._ask_config is not None
            assert "::ask::" in app._ask_config["configurable"]["thread_id"]

            app._enter_plan_mode("goal again")
            assert app._plan_mode is True
            assert app._ask_mode is False
            assert app._ask_config is None

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
            app._plan_mode = True
            app._plan_config = {"configurable": {"thread_id": "abc::plan::x"}}
            app._start_run("plan turn", "hi")
            await pilot.pause(0.3)
            joined = "\n".join(writes)
            assert "plan-mode" in joined
            assert "--session-id" not in joined

    asyncio.run(run())
