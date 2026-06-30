from __future__ import annotations

import time
from collections.abc import Callable
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

from ghidra_deep_agent.sessions import SessionStore
from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink
from ghidra_deep_agent.tui.events import handle_event
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.messages import (
    AgentDone,
    ContextUpdate,
    StatusFlash,
    TokenUpdate,
    ToolCountChanged,
)
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
        Binding("f1", "help", "Help", show=False),
    ]

    def __init__(
        self,
        agent: Any,
        config: dict[str, Any],
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
        self._session_id = session_id
        self._config["configurable"]["thread_id"] = session_id
        self.action_clear_log()
        self.sub_title = f"{self._model}  ·  session: {session_id}"
        if self._session_store is not None:
            await self._session_store.arecord_start(session_id, self._binary_name)
        self.query_one(StatusBar).flash(
            f"[green]Resumed session {session_id[:8]}.[/green]"
        )

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
    async def _run_agent(self, query: str) -> None:
        input_data = {"messages": [{"role": "user", "content": query}]}
        activity = self.query_one(ActivityTree)
        response = self.query_one(ResponseLog)
        thinking = self.query_one(ThinkingPanel)
        thinking.reset()
        try:
            async for event in self._agent.astream_events(
                input_data, config=self._config, version="v2"
            ):
                handle_event(self, event, activity, response, thinking)
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
