from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.rule import Rule
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import Footer, Header, Input
from textual.worker import Worker, WorkerState

from ghidra_deep_agent.prompt import (
    APPROVED_PLAN_INSTRUCTION,
    MARKED_BACKGROUND,
    PLAN_CONTEXT_SUMMARY_PROMPT,
)
from ghidra_deep_agent.resilience import UsageLimitError
from ghidra_deep_agent.sessions import SessionStore
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink
from ghidra_deep_agent.tui.events import handle_event
from ghidra_deep_agent.tui.formatting import extract_text
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.messages import (
    AgentDone,
    ContextUpdate,
    StatusFlash,
    SubagentReport,
    SubagentReportCaptured,
    TokenUpdate,
    ToolCountChanged,
)
from ghidra_deep_agent.tui.report_screen import SubagentReportScreen
from ghidra_deep_agent.tui.session_select import SessionSelectScreen
from ghidra_deep_agent.tui.widgets import (
    PLACEHOLDER_BUSY,
    PLACEHOLDER_IDLE,
    ActivityTree,
    CommandInput,
    ResponseLog,
    StatusBar,
    ThinkingPanel,
)

GHIDRA_THEME = Theme(
    name="ghidra",
    primary="#4ebf71",
    secondary="#22d3ee",
    accent="#2dd4bf",
    warning="#fbbf24",
    error="#f87171",
    success="#4ebf71",
    foreground="#d6e2e8",
    background="#0f1419",
    surface="#151b21",
    panel="#1d252e",
    dark=True,
    variables={"footer-key-foreground": "#4ebf71"},
)


# Skip building a prior-context summary when the main thread has fewer than this
# many messages (nothing meaningful to hand the planner yet).
MIN_MESSAGES_FOR_SUMMARY = 3


def _slug(text: str, max_len: int = 40) -> str:
    """A filesystem-safe slug for a plan goal; 'plan' when empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "plan"


def _file_content(value: Any) -> str | None:
    """Best-effort extract text from a deepagents state ``files`` entry."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return "\n".join(str(line) for line in content)
        if content is not None:
            return str(content)
    if isinstance(value, list):
        return "\n".join(str(line) for line in value)
    return None


class GhidraAgentApp(App[None]):
    TITLE = "Ghidra Agent"
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "cancel_agent", "Cancel"),
        Binding("ctrl+y", "yank", "Copy response"),
        Binding("ctrl+shift+y", "yank_all", "Copy transcript", show=False),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("ctrl+t", "toggle_tree", "Tree"),
        Binding("ctrl+o", "reports", "Reports"),
        Binding("f1", "help", "Help", show=False),
    ]

    def __init__(
        self,
        agent: Any,
        config: dict[str, Any],
        plan_agent: Any = None,
        ask_agent: Any = None,
        summary_model: Any = None,
        model: str = "",
        session_id: str = "",
        mcp_ok: bool = True,
        db_ok: bool = True,
        max_context_tokens: int = 200_000,
        session_store: SessionStore | None = None,
        binary_name: str = "",
    ) -> None:
        super().__init__()
        self._agent = agent
        self._plan_agent = plan_agent
        self._ask_agent = ask_agent
        self._summary_model = summary_model
        self._plan_mode = False
        self._plan_path: str | None = None
        # Ephemeral planning-thread config, minted per `/plan`, cleared on
        # approve/cancel. Keeps planning off the main thread.
        self._plan_config: dict[str, Any] | None = None
        # Set on fresh plan-mode entry; the first planning turn seeds the marked
        # background summary and clears it (revisions don't re-seed).
        self._plan_needs_seed = False
        # Ask mode mirrors plan mode's ephemeral-thread machinery (minus the
        # plan-file/approve step): a read-only Q&A coordinator on its own thread.
        # `/plan` and `/ask` are mutually exclusive.
        self._ask_mode = False
        self._ask_config: dict[str, Any] | None = None
        self._ask_needs_seed = False
        # The last top-level assistant reply text, captured synchronously from
        # the stream loop (see events.py). During plan mode this is the full plan
        # markdown the planner echoed; `_last_plan_text` snapshots it per turn so
        # `/approve` never depends on reading the plan file back from disk/state.
        self._last_reply_text = ""
        self._last_plan_text: str | None = None
        # Async-task middleware bookkeeping (see tui/events.py): run_ids of hidden
        # `get_task_status` polls, and task_id -> run_id for async tool calls whose
        # "completed" marker is deferred until ASYNC_DONE_EVENT arrives.
        self._hidden_tool_runs: set[str] = set()
        self._pending_async: dict[str, str] = {}
        # Sub-agent (`task`) run bookkeeping: run_id -> (description, start time)
        # while in flight, and the completed runs' final reports for the ctrl+o
        # viewer (what each sub-agent returned to the main agent).
        self._subagent_meta: dict[str, tuple[str, float]] = {}
        self._subagent_reports: list[SubagentReport] = []
        # Plain (non-subagent) tool runs currently in flight. A tool call whose
        # parent_ids chain contains one of these was made from *inside* another
        # tool (e.g. recover_prototypes invoking `scripts` directly) and is
        # hidden — it's the composite tool's implementation detail, not the
        # agent's work.
        self._active_tool_runs: set[str] = set()
        self._output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
        self._config = config
        self._model = model
        self._session_id = session_id
        self._session_store = session_store
        self._binary_name = binary_name
        self._agent_running = False
        self._agent_worker: Worker[None] | None = None
        self._unregister_toast_sink: Callable[[], None] | None = None
        self._mcp_ok = mcp_ok
        self._db_ok = db_ok
        self._max_context_tokens = max_context_tokens
        self._elapsed_timer: Timer | None = None
        self._run_start: float | None = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="panes"):
            yield ActivityTree("root agent")
            with Vertical(id="right-pane"):
                yield ResponseLog(highlight=True, markup=True)
                yield ThinkingPanel()
        yield StatusBar()
        yield CommandInput(placeholder=PLACEHOLDER_IDLE, id="query")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(GHIDRA_THEME)
        self.theme = "ghidra"
        self.sub_title = f"{self._model}  ·  session: {self._session_id}"
        self.query_one("#query", Input).focus()
        bar = self.query_one(StatusBar)
        bar.mcp_ok = self._mcp_ok
        bar.db_ok = self._db_ok
        bar.max_context = self._max_context_tokens
        self._unregister_toast_sink = register_toast_sink(self._on_toast_request)

    def on_unmount(self) -> None:
        if self._unregister_toast_sink is not None:
            self._unregister_toast_sink()
            self._unregister_toast_sink = None

    def _on_toast_request(self, toast: ToastRequest) -> None:
        self.notify(
            toast.message,
            title=toast.title,
            severity=toast.severity,
            timeout=toast.timeout,
        )

    # -- status-bar plumbing -------------------------------------------------

    def on_status_flash(self, msg: StatusFlash) -> None:
        self.query_one(StatusBar).flash(msg.text)

    def on_token_update(self, msg: TokenUpdate) -> None:
        self._total_input_tokens += msg.delta_input
        self._total_output_tokens += msg.delta_output
        bar = self.query_one(StatusBar)
        bar.input_tokens = self._total_input_tokens
        bar.output_tokens = self._total_output_tokens

    def on_context_update(self, msg: ContextUpdate) -> None:
        self.query_one(StatusBar).current_context = msg.current_input

    def on_tool_count_changed(self, msg: ToolCountChanged) -> None:
        bar = self.query_one(StatusBar)
        bar.active_tools = max(0, bar.active_tools + msg.delta)

    # -- input ---------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        inp = event.input
        event.input.clear()
        if isinstance(inp, CommandInput):
            inp.add_to_history(query)

        if query.startswith("/"):
            self._dispatch_slash(query)
            return

        if self._agent_running:
            self.query_one(StatusBar).flash(
                "[yellow]Agent still running — please wait.[/yellow]"
            )
            return

        self._start_run(query, query)

    def _start_run(self, display: str, agent_input: str) -> None:
        self._set_busy(True)
        self.query_one(ResponseLog).log_user(display)
        self.query_one(ActivityTree).reset()
        self._touch_session(display)
        self._agent_worker = self._run_agent(agent_input)

    def _resume_run(self) -> None:
        """Continue an interrupted turn on whichever thread is active.

        Re-invokes the graph with a ``None`` input so LangGraph replays from the
        last checkpoint: only the failed task re-runs, while completed sub-agents
        are restored from pending writes. Used after a run pauses on a usage
        limit (see ``UsageLimitError`` handling in ``_run_agent``). ``_run_agent``
        resolves the agent+config from the current mode flags, so this resumes
        the main, plan, or ask thread transparently.
        """
        self._set_busy(True)
        response = self.query_one(ResponseLog)
        response.write("[dim]↻ Continuing from the last checkpoint…[/dim]")
        self.query_one(ActivityTree).reset()
        self._touch_session("/continue")
        self._agent_worker = self._run_agent(None)

    @work(exclusive=False)
    async def _touch_session(self, prompt: str) -> None:
        """Bump the session's activity time (fire-and-forget, best-effort)."""
        if self._session_store is None:
            return
        try:
            await self._session_store.atouch(self._session_id, first_prompt=prompt)
        except Exception:
            # Session bookkeeping must never disrupt the run.
            pass

    def _dispatch_slash(self, command: str) -> None:
        cmd = command.split()[0].lower()
        bar = self.query_one(StatusBar)
        if cmd == "/clear":
            self.action_clear_log()
            bar.flash("[green]Cleared.[/green]")
        elif cmd == "/yank":
            self.action_yank()
        elif cmd == "/quit":
            self.exit()
        elif cmd == "/compact":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            self._start_run(
                "/compact",
                "Call the `compact_conversation` tool now to compact the "
                "conversation history.",
            )
        elif cmd == "/resume":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            self._open_resume_picker()
        elif cmd == "/continue":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            # In a side-mode, resume that mode's ephemeral thread. Guard against
            # a mode flag set with no thread minted (nothing to replay).
            if self._plan_mode and self._plan_config is None:
                bar.flash("[yellow]Nothing to resume in plan mode.[/yellow]")
                return
            if self._ask_mode and self._ask_config is None:
                bar.flash("[yellow]Nothing to resume in ask mode.[/yellow]")
                return
            self._resume_run()
        elif cmd == "/plan":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            goal = command[len(cmd) :].strip()
            self._enter_plan_mode(goal)
        elif cmd == "/approve":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            self._approve_plan()
        elif cmd == "/plan-cancel":
            if not self._plan_mode:
                bar.flash("[yellow]Not in plan mode.[/yellow]")
                return
            self._exit_plan_mode()
            bar.flash("[magenta]Plan mode cancelled.[/magenta]")
        elif cmd == "/ask":
            if self._agent_running:
                bar.flash("[yellow]Agent still running — please wait.[/yellow]")
                return
            question = command[len(cmd) :].strip()
            self._enter_ask_mode(question)
        elif cmd == "/ask-cancel":
            if not self._ask_mode:
                bar.flash("[yellow]Not in ask mode.[/yellow]")
                return
            self._exit_ask_mode()
            bar.flash("[cyan]Ask mode cancelled.[/cyan]")
        elif cmd == "/help":
            self.action_help()
        else:
            bar.flash(f"[red]Unknown command: {cmd}[/red]")

    @work(exclusive=False)
    async def _open_resume_picker(self) -> None:
        bar = self.query_one(StatusBar)
        if self._session_store is None:
            bar.flash("[yellow]Session registry unavailable.[/yellow]")
            return
        store = self._session_store
        sessions = await store.alist_sessions(self._binary_name)
        if not sessions:
            bar.flash("[yellow]No previous sessions for this binary.[/yellow]")
            return

        async def fetch(show_all: bool) -> list[dict[str, Any]]:
            return await store.alist_sessions(None if show_all else self._binary_name)

        chosen = await self.push_screen_wait(
            SessionSelectScreen(sessions, self._session_id, self._binary_name, fetch)
        )
        if chosen and chosen != self._session_id:
            await self._switch_session(chosen)

    async def _switch_session(self, session_id: str) -> None:
        plan_was_active = self._plan_mode
        ask_was_active = self._ask_mode
        if plan_was_active:
            # The planning thread was seeded from the old main session; drop it
            # rather than carry it across a session switch.
            self._exit_plan_mode()
        if ask_was_active:
            # Same rationale for the ephemeral ask thread.
            self._exit_ask_mode()
        self._session_id = session_id
        self._config["configurable"]["thread_id"] = session_id
        self._subagent_meta.clear()
        self._subagent_reports.clear()
        self.action_clear_log()
        await self._replay_last_reply()
        self.sub_title = f"{self._model}  ·  session: {session_id}"
        if self._session_store is not None:
            await self._session_store.arecord_start(session_id, self._binary_name)
        msg = f"[green]Resumed session {session_id[:8]}.[/green]"
        if plan_was_active:
            msg += " [magenta]Plan mode cancelled.[/magenta]"
        if ask_was_active:
            msg += " [cyan]Ask mode cancelled.[/cyan]"
        self.query_one(StatusBar).flash(msg)

    async def _replay_last_reply(self) -> None:
        """After a resume, paint the session's last assistant reply back into
        the main window so the user sees the session loaded and what happened
        last."""
        try:
            state = await self._agent.aget_state(self._config)
        except Exception:
            return
        for msg in reversed(state.values.get("messages", [])):
            if getattr(msg, "type", None) == "ai":
                text = extract_text(msg).strip()
                if text:
                    self.query_one(ResponseLog).log_assistant(text)
                    return

    # -- plan mode -----------------------------------------------------------

    def _set_plan_mode(self, on: bool) -> None:
        """Flip plan mode and mirror it onto the status bar indicator."""
        self._plan_mode = on
        self.query_one(StatusBar).plan_mode = on

    def _exit_plan_mode(self) -> None:
        """Leave plan mode and drop the ephemeral planning thread + file."""
        self._set_plan_mode(False)
        self._plan_path = None
        self._plan_config = None
        self._plan_needs_seed = False
        self._last_plan_text = None

    def _enter_plan_mode(self, goal: str) -> None:
        """Enter plan mode, minting a fresh timestamped plan file + thread.

        A new plan file and planning thread are minted only when entering from
        the normal state; while already in plan mode the current plan file and
        thread keep being revised (and no background summary is re-seeded).
        """
        bar = self.query_one(StatusBar)
        # `/plan` and `/ask` are mutually exclusive — leave ask mode first.
        if self._ask_mode:
            self._exit_ask_mode()
        if not self._plan_mode or self._plan_path is None:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            self._plan_path = f"plans/{stamp}-{_slug(goal)}.md"
            self._plan_config = {
                "configurable": {"thread_id": f"{self._session_id}::plan::{stamp}"},
                "recursion_limit": self._config.get("recursion_limit", 10000),
            }
            self._plan_needs_seed = True
        self._set_plan_mode(True)
        if goal:
            self._start_run("/plan " + goal, goal)
        else:
            bar.flash("[magenta]Plan mode ON — describe what to plan.[/magenta]")

    # -- ask mode ------------------------------------------------------------

    def _set_ask_mode(self, on: bool) -> None:
        """Flip ask mode and mirror it onto the status bar indicator."""
        self._ask_mode = on
        self.query_one(StatusBar).ask_mode = on

    def _exit_ask_mode(self) -> None:
        """Leave ask mode and drop the ephemeral ask thread."""
        self._set_ask_mode(False)
        self._ask_config = None
        self._ask_needs_seed = False

    def _enter_ask_mode(self, question: str) -> None:
        """Enter ask mode, minting a fresh ephemeral Q&A thread.

        A new thread is minted only when entering from the normal state; while
        already in ask mode the current thread keeps handling follow-ups (and no
        background summary is re-seeded). `/plan` is left first — the two
        side-modes are mutually exclusive.
        """
        bar = self.query_one(StatusBar)
        if self._plan_mode:
            self._exit_plan_mode()
        if not self._ask_mode or self._ask_config is None:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            self._ask_config = {
                "configurable": {"thread_id": f"{self._session_id}::ask::{stamp}"},
                "recursion_limit": self._config.get("recursion_limit", 10000),
            }
            self._ask_needs_seed = True
        self._set_ask_mode(True)
        if question:
            self._start_run("/ask " + question, question)
        else:
            bar.flash("[cyan]Ask mode ON — ask a question.[/cyan]")

    async def _build_marked_prior_context(self) -> str | None:
        """Summarize the main session so far into a marked background block.

        Returns None (skip seeding) when there's no summary model, the main
        thread is empty/tiny, or the summary call fails — the planner then just
        starts from the goal.
        """
        if self._summary_model is None:
            return None
        try:
            state = await self._agent.aget_state(self._config)
        except Exception:
            return None
        messages = state.values.get("messages", [])
        if len(messages) < MIN_MESSAGES_FOR_SUMMARY:
            return None
        from langchain_core.messages import get_buffer_string

        transcript = get_buffer_string(messages, format="xml")
        try:
            reply = await self._summary_model.ainvoke(
                PLAN_CONTEXT_SUMMARY_PROMPT.format(transcript=transcript)
            )
        except Exception:
            return None
        summary = extract_text(reply).strip()
        return MARKED_BACKGROUND.format(summary=summary) if summary else None

    def _approve_plan(self) -> None:
        """Leave plan mode and tell the normal agent to execute the plan.

        The plan text is taken from the planner's streamed reply (captured per
        turn as `_last_plan_text`), which the plan prompt guarantees contains the
        full plan markdown. This is backend-agnostic and does not depend on where
        the planner persisted the plan file — the disk/state read is only a
        fallback. The execution agent runs on the MAIN thread, so it never
        inherits any planner-authored messages.
        """
        bar = self.query_one(StatusBar)
        if not self._plan_mode:
            bar.flash("[yellow]Not in plan mode — nothing to approve.[/yellow]")
            return
        plan_path = self._plan_path
        plan_text = self._last_plan_text or (
            self._read_plan_text(plan_path, self._plan_config)
            if plan_path and self._plan_config is not None
            else None
        )
        if not plan_text:
            bar.flash("[yellow]No plan to approve yet — write a plan first.[/yellow]")
            return
        self._exit_plan_mode()
        self._start_run(
            "/approve",
            APPROVED_PLAN_INSTRUCTION.format(plan_path=plan_path, plan_text=plan_text),
        )

    def _read_plan_text(
        self, plan_path: str, config: dict[str, Any] | None
    ) -> str | None:
        """Read the current plan back from disk (FilesystemBackend) or state.

        ``config`` selects which thread's state to read (the planning thread);
        it is only needed for the StateBackend branch. Returns None if it can't
        be found, so the caller can fall back to the streamed reply (the prompt
        also makes the model echo the full plan).
        """
        if self._output_dir:
            try:
                return (Path(self._output_dir) / plan_path).read_text(encoding="utf-8")
            except OSError:
                return None
        if config is None:
            return None
        try:
            files = self._plan_agent.get_state(config).values.get("files", {})
        except Exception:
            return None
        return _file_content(files.get(plan_path))

    # -- bindings ------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "cancel_agent" and not self._agent_running:
            return None
        return True

    def action_yank(self) -> None:
        text = self.query_one(ResponseLog).last_response
        if not text:
            self.notify("Nothing to copy yet.", severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify("Response copied to clipboard.")

    def action_yank_all(self) -> None:
        transcript = self.query_one(ResponseLog).transcript
        if not transcript:
            self.notify("Nothing to copy yet.", severity="warning")
            return
        self.copy_to_clipboard("\n\n".join(transcript))
        self.notify("Transcript copied to clipboard.")

    def action_clear_log(self) -> None:
        self.query_one(ResponseLog).clear()
        self.query_one(ActivityTree).reset()

    def action_toggle_tree(self) -> None:
        self.query_one("#panes").toggle_class("hide-tree")

    def on_subagent_report_captured(self, msg: SubagentReportCaptured) -> None:
        self._subagent_reports.append(msg.report)

    def action_reports(self) -> None:
        """Open the sub-agent report viewer.

        Reports survive `/clear` (they're the session's audit trail) and are
        dropped only on a session switch.
        """
        if not self._subagent_reports:
            self.notify("No sub-agent reports yet.", severity="warning")
            return
        self.push_screen(SubagentReportScreen(list(reversed(self._subagent_reports))))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_cancel_agent(self) -> None:
        if self._agent_running and self._agent_worker is not None:
            self._agent_worker.cancel()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker is self._agent_worker and event.state == WorkerState.CANCELLED:
            response = self.query_one(ResponseLog)
            response.write(Rule(style="dim yellow"))
            response.write("[bold yellow]■ Run cancelled[/bold yellow]")
            self.query_one(StatusBar).flash("[yellow]Run cancelled.[/yellow]")

    # -- run state -----------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._agent_running = busy
        inp = self.query_one("#query", Input)
        bar = self.query_one(StatusBar)
        if busy:
            inp.placeholder = PLACEHOLDER_BUSY
            inp.add_class("busy")
            bar.add_class("busy")
            self._run_start = time.monotonic()
            bar.elapsed_seconds = 0
            if self._elapsed_timer is not None:
                self._elapsed_timer.stop()
            self._elapsed_timer = self.set_interval(1.0, self._tick_elapsed)
        else:
            inp.placeholder = PLACEHOLDER_IDLE
            inp.remove_class("busy")
            bar.remove_class("busy")
            if self._elapsed_timer is not None:
                self._elapsed_timer.stop()
                self._elapsed_timer = None
            self._run_start = None
        self.refresh_bindings()

    def _tick_elapsed(self) -> None:
        if self._run_start is None:
            return
        self.query_one(StatusBar).elapsed_seconds = int(
            time.monotonic() - self._run_start
        )

    @work(exclusive=True)
    async def _run_agent(self, query: str | None) -> None:
        # Pick the graph AND its thread config together, captured for the lifetime
        # of this turn so a later mode flip can't change which thread we stream to.
        # `/plan` and `/ask` are mutually exclusive side-modes, each on its own
        # ephemeral thread; otherwise the normal agent on the main thread.
        plan_run = self._plan_mode
        ask_run = self._ask_mode
        plan_path = self._plan_path
        if plan_run:
            agent, config = self._plan_agent, self._plan_config
        elif ask_run:
            agent, config = self._ask_agent, self._ask_config
        else:
            agent, config = self._agent, self._config
        response = self.query_one(ResponseLog)
        if (plan_run or ask_run) and config is None:
            # Programming error: a side-mode is on but no thread was minted. Never
            # fall back to the main config (that reintroduces the bug we fixed).
            mode = "Plan" if plan_run else "Ask"
            self.query_one(StatusBar).flash(
                f"[red]{mode} mode has no thread — aborting run.[/red]"
            )
            self._set_busy(False)
            self.query_one("#query", Input).focus()
            return

        input_data: dict[str, Any] | None
        if query is None:
            # Resume (`/continue`): re-invoke with no input so LangGraph replays
            # from the last checkpoint on this thread — only the failed task
            # re-runs, completed sub-agents are restored from pending writes.
            input_data = None
        else:
            messages: list[dict[str, str]] = []
            # On the first turn of a side-mode, seed its fresh thread with a
            # marked summary of the main session so far (background, not work).
            if (plan_run and self._plan_needs_seed) or (
                ask_run and self._ask_needs_seed
            ):
                self._plan_needs_seed = False
                self._ask_needs_seed = False
                background = await self._build_marked_prior_context()
                if background:
                    messages.append({"role": "user", "content": background})
            if plan_run and plan_path:
                query = (
                    f"[Plan mode — write/maintain the complete plan at "
                    f"`{plan_path}`]\n\n{query}"
                )
            elif ask_run:
                query = (
                    "[Ask mode — decompose the question(s), delegate investigation "
                    "to the research sub-agent, and synthesize a grounded, cited "
                    f"answer]\n\n{query}"
                )
            messages.append({"role": "user", "content": query})
            input_data = {"messages": messages}

        activity = self.query_one(ActivityTree)
        thinking = self.query_one(ThinkingPanel)
        thinking.reset()
        # Reset before streaming so a turn that produces no top-level reply can't
        # reuse a stale capture (see events.py for where this gets set).
        self._last_reply_text = ""
        try:
            async for event in agent.astream_events(
                input_data, config=config, version="v2"
            ):
                handle_event(self, event, activity, response, thinking)
            if plan_run and plan_path:
                # The streamed reply is the source of truth for the plan; the
                # disk/state read is only a fallback. Snapshot it for `/approve`.
                plan_text = self._last_reply_text or self._read_plan_text(
                    plan_path, config
                )
                self._last_plan_text = plan_text
                if plan_text:
                    response.log_plan(plan_text)
        except UsageLimitError:
            # Not a crash: the provider usage/rate limit outlasted our retries.
            # Everything committed so far (history + finished sub-agents) is
            # durable in the checkpointer, so tell the user how to resume rather
            # than showing a scary error.
            sid = self._session_id
            response.write(Rule(style="yellow"))
            response.write(
                "[bold yellow]⏸ Usage limit reached — run paused and safely "
                "checkpointed.[/bold yellow]"
            )
            if plan_run or ask_run:
                # A side-mode run lives on an ephemeral thread that is not
                # restorable across launches, so resume must happen in-session.
                mode = "plan" if plan_run else "ask"
                response.write(
                    f"This {mode}-mode run is checkpointed. When your limit "
                    "resets, type [b]/continue[/b] to pick up where it left off "
                    "(finished sub-agents won't re-run). Leaving "
                    f"{mode} mode or closing the app abandons the paused run."
                )
            else:
                response.write(
                    f"Completed work is saved to session [b]{sid}[/b]. When your "
                    "limit resets, type [b]/continue[/b] to pick up where this "
                    "turn left off (finished sub-agents won't re-run). If you've "
                    f"since closed the app, relaunch with [b]--session-id {sid}[/b] "
                    "and then run [b]/continue[/b]."
                )
            response.write(Rule(style="yellow"))
        except Exception as exc:
            response.write(Rule(style="red"))
            response.write(f"[bold red]✗ Error: {exc}[/bold red]")
            response.write(Rule(style="red"))
        finally:
            # Runs on cancellation too (CancelledError is a BaseException and
            # is not swallowed above), so the UI always returns to idle.
            thinking.display = False
            response.post_message(AgentDone())
            self._set_busy(False)
            self.query_one("#query", Input).focus()
