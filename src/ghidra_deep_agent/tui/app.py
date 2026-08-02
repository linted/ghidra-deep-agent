from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from rich.rule import Rule
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import Footer, Header, Input
from textual.worker import Worker, WorkerState

from ghidra_deep_agent.compaction import compact_out_of_band
from ghidra_deep_agent.defaults import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RECURSION_LIMIT,
)
from ghidra_deep_agent.prompt import (
    APPROVED_PLAN_INSTRUCTION,
    MARKED_BACKGROUND,
    PLAN_CONTEXT_SUMMARY_PROMPT,
)
from ghidra_deep_agent.resilience import UsageLimitError
from ghidra_deep_agent.sessions import SessionStore
from ghidra_deep_agent.toasts import ToastRequest, notify_toast, register_toast_sink
from ghidra_deep_agent.tui.commands import COMMANDS_BY_NAME
from ghidra_deep_agent.tui.events import handle_event
from ghidra_deep_agent.tui.formatting import extract_text
from ghidra_deep_agent.tui.help_screen import HelpScreen
from ghidra_deep_agent.tui.messages import (
    AgentDone,
    ContextUpdate,
    ResponseFinal,
    StatusFlash,
    SubagentReport,
    SubagentReportCaptured,
    TokenUpdate,
    ToolCountChanged,
)
from ghidra_deep_agent.tui.report_screen import SubagentReportScreen
from ghidra_deep_agent.tui.run_state import RunState
from ghidra_deep_agent.tui.session_select import SessionSelectScreen
from ghidra_deep_agent.tui.side_mode import Kind, SideMode
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
        summary_model: BaseChatModel | None = None,
        compaction_engine: Any = None,
        model: str = "",
        session_id: str = "",
        mcp_ok: bool = True,
        db_ok: bool = True,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        session_store: SessionStore | None = None,
        binary_name: str = "",
    ) -> None:
        super().__init__()
        self._agent = agent
        self._plan_agent = plan_agent
        self._ask_agent = ask_agent
        self._summary_model = summary_model
        # Summarization engine driven directly by /compact (no agent turn).
        self._compaction_engine = compaction_engine
        # The active read-only side mode (`/plan` or `/ask`), or None for normal
        # operation. One object rather than two parallel sets of flags, so "both
        # modes on" and "mode on with no thread" are unrepresentable.
        self._side: SideMode | None = None
        # Per-turn tool bookkeeping, owned by events.py. Replaced wholesale at the
        # start of each run so a cancelled turn can't leak its in-flight run_ids
        # into the next one — see RunState's docstring.
        self.run_state = RunState()
        # During plan mode the reply is the full plan markdown the planner echoed;
        # this snapshots it per turn so `/approve` never depends on reading the
        # plan file back from disk/state.
        self._last_plan_text: str | None = None
        # Completed sub-agent reports for the ctrl+o viewer (what each sub-agent
        # returned to the main agent). Spans turns, so not part of RunState.
        self._subagent_reports: list[SubagentReport] = []
        self._output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
        self._config = config
        self._model = model
        self._session_id = session_id
        self._session_store = session_store
        self._binary_name = binary_name
        self._agent_running = False
        self._agent_worker: Worker[None] | None = None
        self._unregister_toast_sink: Callable[[], None] | None = None
        # Keys of best-effort failures already toasted (see `_warn_once`).
        self._warned_failures: set[str] = set()
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

        if not self._require_idle(self.query_one(StatusBar)):
            return

        self._start_run(query, query)

    def _warn_once(self, key: str, message: str) -> None:
        """Toast a best-effort failure the first time it happens.

        These paths deliberately swallow errors so a hiccup can't kill a turn,
        but silence makes a *persistent* failure invisible — a session that
        never records, or a plan that never gets its background context. One
        toast per kind per session: enough to notice, not enough to nag.
        """
        if key in self._warned_failures:
            return
        self._warned_failures.add(key)
        notify_toast(message, severity="warning", title="Agent")

    def _require_idle(self, bar: StatusBar) -> bool:
        """True when no run is in flight; otherwise flash and return False.

        Every command that would start or replace a run goes through this.
        """
        if self._agent_running:
            bar.flash("[yellow]Agent still running — please wait.[/yellow]")
            return False
        return True

    def _reset_run_bookkeeping(self) -> None:
        """Start the next turn from a clean slate.

        A run cancelled with Escape (or aborted by an exception) never delivers
        ``on_tool_end`` for whatever was in flight, so its run_ids would otherwise
        persist and the status bar's active-tool count would never return to zero.
        The tree forgets its own in-flight runs in ``ActivityTree.reset()`` (a
        fresh turn) or ``mark_resumed()`` (``/continue``); this is the app-side
        half of the same bookkeeping.
        """
        self.run_state = RunState()
        self.query_one(StatusBar).active_tools = 0

    def _start_run(self, display: str, agent_input: str) -> None:
        self._set_busy(True)
        self.query_one(ResponseLog).log_user(display)
        self.query_one(ActivityTree).reset()
        self._reset_run_bookkeeping()
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

        The activity tree is carried across rather than cleared: this is the same
        turn continuing, and the restored sub-agents emit no events, so wiping it
        would lose that history for good.
        """
        self._set_busy(True)
        response = self.query_one(ResponseLog)
        response.write("[dim]↻ Continuing from the last checkpoint…[/dim]")
        self.query_one(ActivityTree).mark_resumed()
        self._reset_run_bookkeeping()
        self._touch_session("/continue")
        self._agent_worker = self._run_agent(None)

    # Own worker group: `_run_agent` is `exclusive=True`, and Textual cancels every
    # worker sharing the *group* of an exclusive worker on the same node. Both used
    # to default to "default", so starting a run killed the touch worker launched one
    # line earlier — before its first await — and no session ever got a title or a
    # refreshed `last_active_at`.
    @work(exclusive=False, group="session")
    async def _touch_session(self, prompt: str) -> None:
        """Bump the session's activity time (fire-and-forget, best-effort)."""
        if self._session_store is None:
            return
        try:
            await self._session_store.atouch(self._session_id, first_prompt=prompt)
        except Exception as exc:
            # Session bookkeeping must never disrupt the run — but if it keeps
            # failing, `/resume` will quietly not list this session.
            self._warn_once("touch_session", f"Session not recorded for /resume: {exc}")

    def _slash_handlers(self) -> dict[str, Callable[[str], None]]:
        """Command name -> handler, taking whatever followed the command."""
        return {
            "/clear": self._cmd_clear,
            "/yank": lambda _arg: self.action_yank(),
            "/quit": lambda _arg: self.exit(),
            "/compact": self._cmd_compact,
            "/resume": self._cmd_resume,
            # A side mode always carries its own thread, so `/continue` inside one
            # resumes that thread rather than the main session's.
            "/continue": lambda _arg: self._resume_run(),
            "/plan": lambda arg: self._enter_side_mode("plan", arg),
            "/approve": lambda _arg: self._approve_plan(),
            "/plan-cancel": lambda _arg: self._cmd_cancel_side("plan"),
            "/ask": lambda arg: self._enter_side_mode("ask", arg),
            "/ask-cancel": lambda _arg: self._cmd_cancel_side("ask"),
            "/help": lambda _arg: self.action_help(),
        }

    def _cmd_resume(self, _arg: str) -> None:
        # `_open_resume_picker` is a @work method; discard the Worker it returns.
        self._open_resume_picker()

    def _cmd_clear(self, _arg: str) -> None:
        self.action_clear_log()
        self.query_one(StatusBar).flash("[green]Cleared.[/green]")

    def _cmd_compact(self, _arg: str) -> None:
        """Compact the main thread's history without an agent turn.

        This used to start a normal run asking the model to call the
        ``compact_conversation`` tool — a full-context main-model call spent at
        the exact moment context is at its largest, and one the model was free
        to ignore. Now the summarization engine runs directly: one
        summary-model call, deterministic.
        """
        bar = self.query_one(StatusBar)
        if self._side is not None:
            # A side mode's ephemeral thread is dropped on exit — compacting it
            # is pointless, and /compact must not silently touch the main
            # thread from inside one.
            bar.flash(
                "[yellow]/compact works on the main session — "
                "leave plan/ask mode first.[/yellow]"
            )
            return
        if self._compaction_engine is None:
            bar.flash("[yellow]Compaction unavailable in this session.[/yellow]")
            return
        self._agent_worker = self._run_compaction()

    @work(exclusive=True)
    async def _run_compaction(self) -> None:
        """Read state → summarize → persist, atomically from the UI's view.

        Runs busy and exclusive (and is refused while a run is in flight via
        ``needs_idle``), so no turn can grow ``messages`` between the state
        read and the ``cutoff_index`` write. Nothing is persisted until the
        final ``aupdate_state`` — a failure or an Escape-cancel anywhere before
        that leaves the thread untouched.
        """
        self._set_busy(True)
        response = self.query_one(ResponseLog)
        response.log_user("/compact")
        try:
            state = await self._agent.aget_state(self._config)
            result = await compact_out_of_band(
                self._compaction_engine,
                state.values.get("messages", []),
                state.values.get("_summarization_event"),
                thread_id=self._session_id,
            )
            if result is None:
                self.query_one(StatusBar).flash(
                    "[yellow]Nothing to compact yet.[/yellow]"
                )
                return
            # `as_node="tools"` writes the event as the node that owns it on
            # the in-graph path (the compact tool); pinned by a real-graph test.
            await self._agent.aupdate_state(
                self._config, {"_summarization_event": result.event}, as_node="tools"
            )
            saved = (
                f" Full history saved to {result.file_path}."
                if result.file_path
                else ""
            )
            response.write(
                f"[green]✦ Compacted {result.summarized_count} messages into a "
                f"summary.{saved}[/green]"
            )
        except Exception as exc:
            response.write(
                f"[bold red]✗ Compaction failed: {exc} — "
                "the conversation was not changed.[/bold red]"
            )
        finally:
            self._set_busy(False)
            self.query_one("#query", Input).focus()

    def _cmd_cancel_side(self, kind: Kind) -> None:
        bar = self.query_one(StatusBar)
        if self._side is None or self._side.kind != kind:
            bar.flash(f"[yellow]Not in {kind} mode.[/yellow]")
            return
        color = "magenta" if kind == "plan" else "cyan"
        self._exit_side_mode()
        bar.flash(f"[{color}]{self._side_label(kind)} mode cancelled.[/{color}]")

    @staticmethod
    def _side_label(kind: Kind) -> str:
        return "Plan" if kind == "plan" else "Ask"

    def _dispatch_slash(self, command: str) -> None:
        cmd = command.split()[0].lower()
        bar = self.query_one(StatusBar)
        spec = COMMANDS_BY_NAME.get(cmd)
        if spec is None:
            bar.flash(f"[red]Unknown command: {cmd}[/red]")
            return
        # One busy guard for every run-starting command, driven by the table —
        # previously copy-pasted into each branch, where a new command could
        # silently omit it.
        if spec.needs_idle and not self._require_idle(bar):
            return
        self._slash_handlers()[cmd](command[len(cmd) :].strip())

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
        # A side mode's thread was seeded from the *old* main session, so drop it
        # rather than carry it across the switch.
        was_active = self._side
        if was_active is not None:
            self._exit_side_mode()
        self._session_id = session_id
        self._config["configurable"]["thread_id"] = session_id
        self._reset_run_bookkeeping()
        self._subagent_reports.clear()
        self.action_clear_log()
        await self._replay_last_reply()
        self.sub_title = f"{self._model}  ·  session: {session_id}"
        if self._session_store is not None:
            await self._session_store.arecord_start(session_id, self._binary_name)
        msg = f"[green]Resumed session {session_id[:8]}.[/green]"
        if was_active is not None:
            color = "magenta" if was_active.is_plan else "cyan"
            msg += f" [{color}]{was_active.label} mode cancelled.[/{color}]"
        self.query_one(StatusBar).flash(msg)

    async def _replay_last_reply(self) -> None:
        """After a resume, paint the session's last assistant reply back into
        the main window so the user sees the session loaded and what happened
        last."""
        try:
            state = await self._agent.aget_state(self._config)
        except Exception as exc:
            # Painting nothing looks exactly like "the session was empty".
            self._warn_once("replay", f"Could not load the session's history: {exc}")
            return
        for msg in reversed(state.values.get("messages", [])):
            if getattr(msg, "type", None) == "ai":
                text = extract_text(msg).strip()
                if text:
                    self.query_one(ResponseLog).log_assistant(text)
                    return

    # -- side modes (plan / ask) ---------------------------------------------
    # Both are read-only modes that run on their own ephemeral checkpointer
    # thread, minted on entry and dropped on exit. The helpers below hold what
    # they share; the pairs after them hold what actually differs (plan also
    # mints a plan file).

    def _new_side_stamp(self, *, active: bool, has_thread: bool) -> str | None:
        """Timestamp for a fresh side-mode thread, or ``None`` to keep the current.

        A new thread is minted only when entering from the normal state — while
        already in the mode, the existing thread keeps handling follow-ups (and
        no background summary is re-seeded).
        """
        if active and has_thread:
            return None
        return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    def _side_thread_config(self, kind: str, stamp: str) -> dict[str, Any]:
        """Graph config for a side mode's ephemeral thread."""
        return {
            "configurable": {"thread_id": f"{self._session_id}::{kind}::{stamp}"},
            "recursion_limit": self._config.get(
                "recursion_limit", DEFAULT_RECURSION_LIMIT
            ),
        }

    def _start_or_hint(self, bar: StatusBar, text: str, prefix: str, hint: str) -> None:
        """Run the mode's first turn, or hint at what to type when it's empty."""
        if text:
            self._start_run(prefix + text, text)
        else:
            bar.flash(hint)

    def _sync_mode_indicators(self) -> None:
        """Mirror the active side mode onto the status bar's two indicators."""
        bar = self.query_one(StatusBar)
        bar.plan_mode = self._side is not None and self._side.is_plan
        bar.ask_mode = self._side is not None and self._side.is_ask

    def _exit_side_mode(self) -> None:
        """Leave whichever side mode is active, dropping its ephemeral thread."""
        self._side = None
        self._last_plan_text = None
        self._sync_mode_indicators()

    def _enter_side_mode(self, kind: Kind, text: str) -> None:
        """Enter (or continue) a side mode, minting its thread on fresh entry.

        Entering the mode already active keeps its thread — follow-ups continue
        the same conversation and don't re-seed. Entering the *other* mode
        replaces it: the two are mutually exclusive.
        """
        bar = self.query_one(StatusBar)
        current = self._side if self._side and self._side.kind == kind else None
        stamp = self._new_side_stamp(
            active=current is not None, has_thread=current is not None
        )
        if stamp is not None:
            self._side = SideMode(
                kind=kind,
                config=self._side_thread_config(kind, stamp),
                plan_path=(
                    f"plans/{stamp}-{_slug(text)}.md" if kind == "plan" else None
                ),
            )
        self._sync_mode_indicators()
        if kind == "plan":
            self._start_or_hint(
                bar,
                text,
                "/plan ",
                "[magenta]Plan mode ON — describe what to plan.[/magenta]",
            )
        else:
            self._start_or_hint(
                bar, text, "/ask ", "[cyan]Ask mode ON — ask a question.[/cyan]"
            )

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
        except Exception as exc:
            self._warn_once(
                "seed_state", f"Plan/ask mode started without prior context: {exc}"
            )
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
        except Exception as exc:
            self._warn_once(
                "seed_summary",
                f"Plan/ask mode started without prior context ({exc}); "
                "check SUMMARY_MODEL.",
            )
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
        side = self._side
        if side is None or not side.is_plan:
            bar.flash("[yellow]Not in plan mode — nothing to approve.[/yellow]")
            return
        plan_path = side.plan_path
        plan_text = self._last_plan_text or (
            self._read_plan_text(plan_path, side.config) if plan_path else None
        )
        if not plan_text:
            bar.flash("[yellow]No plan to approve yet — write a plan first.[/yellow]")
            return
        self._exit_side_mode()
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

    async def _surface_salvaged_reply(
        self, agent: Any, config: dict[str, Any], response: ResponseLog
    ) -> None:
        """Render a reply appended after the model loop ended.

        ``MainReplyGuardMiddleware`` appends its salvaged reply from a graph
        node, not a model call, so events.py's ``on_chat_model_end`` capture
        never sees it — read the final state instead. On a normal turn the last
        message's text equals the captured reply and this is a no-op.
        """
        try:
            state = await agent.aget_state(config)
        except Exception:
            return  # display-only nicety; never fail the turn over it
        for msg in reversed(state.values.get("messages") or []):
            text = extract_text(msg) if isinstance(msg, AIMessage) else ""
            if text:
                if text != self.run_state.last_reply_text:
                    self.run_state.last_reply_text = text
                    response.post_message(ResponseFinal(text))
                return

    @work(exclusive=True)
    async def _run_agent(self, query: str | None) -> None:
        # Pick the graph AND its thread config together, captured for the lifetime
        # of this turn so a later mode flip can't change which thread we stream to.
        # `/plan` and `/ask` are mutually exclusive side-modes, each on its own
        # ephemeral thread; otherwise the normal agent on the main thread.
        side = self._side
        if side is None:
            agent, config = self._agent, self._config
        elif side.is_plan:
            agent, config = self._plan_agent, side.config
        else:
            agent, config = self._ask_agent, side.config
        response = self.query_one(ResponseLog)

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
            if side is not None and side.needs_seed:
                side.needs_seed = False
                background = await self._build_marked_prior_context()
                if background:
                    messages.append({"role": "user", "content": background})
            if side is not None:
                query = side.decorate(query)
            messages.append({"role": "user", "content": query})
            input_data = {"messages": messages}

        activity = self.query_one(ActivityTree)
        thinking = self.query_one(ThinkingPanel)
        thinking.reset()
        # Reset before streaming so a turn that produces no top-level reply can't
        # reuse a stale capture (see events.py for where this gets set).
        self.run_state.last_reply_text = ""
        errored = False
        try:
            async for event in agent.astream_events(
                input_data, config=config, version="v2"
            ):
                handle_event(self, event, activity, response, thinking)
            await self._surface_salvaged_reply(agent, config, response)
            if side is not None and side.is_plan and side.plan_path:
                # The streamed reply is the source of truth for the plan; the
                # disk/state read is only a fallback. Snapshot it for `/approve`.
                plan_text = self.run_state.last_reply_text or self._read_plan_text(
                    side.plan_path, config
                )
                self._last_plan_text = plan_text
                if plan_text:
                    response.log_plan(plan_text)
        except UsageLimitError:
            errored = True  # pause banner below; no no-reply placeholder on top
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
            if side is not None:
                # A side-mode run lives on an ephemeral thread that is not
                # restorable across launches, so resume must happen in-session.
                mode = side.kind
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
            errored = True
            response.write(Rule(style="red"))
            response.write(f"[bold red]✗ Error: {exc}[/bold red]")
            response.write(Rule(style="red"))
        finally:
            # Runs on cancellation too (CancelledError is a BaseException and
            # is not swallowed above), so the UI always returns to idle.
            thinking.display = False
            response.post_message(AgentDone(errored))
            self._set_busy(False)
            self.query_one("#query", Input).focus()
