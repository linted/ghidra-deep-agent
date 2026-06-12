from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rich.markdown import Markdown
from rich.rule import Rule
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

from ghidra_deep_agent.toasts import ToastRequest, register_toast_sink

_PLACEHOLDER_IDLE = "Enter task (Ctrl+C quit · /help for commands)…"
_PLACEHOLDER_BUSY = "Agent is running… type ahead, Enter to queue"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class ToolStarted(Message):
    def __init__(
        self,
        run_id: str,
        name: str,
        input_preview: str,
        is_subagent: bool,
        checkpoint_ns: str,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.name = name
        self.input_preview = input_preview
        self.is_subagent = is_subagent
        self.checkpoint_ns = checkpoint_ns


class ToolEnded(Message):
    def __init__(
        self, run_id: str, error: bool = False, output_snippet: str = ""
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.error = error
        self.output_snippet = output_snippet


class LLMThinking(Message):
    def __init__(self, run_id: str, checkpoint_ns: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.checkpoint_ns = checkpoint_ns


class LLMDone(Message):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


class TextToken(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AgentDone(Message):
    pass


class StatusFlash(Message):
    """Transient text to surface in the status bar (auto-clears)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class TokenUpdate(Message):
    def __init__(self, delta_tokens: int) -> None:
        super().__init__()
        self.delta_tokens = delta_tokens


class ToolCountChanged(Message):
    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class ActivityTree(Tree[None]):
    """Left pane: live agent/tool call hierarchy."""

    DEFAULT_CSS = """
    ActivityTree {
        width: 35%;
        border-right: solid $accent-darken-2;
        scrollbar-size: 1 1;
    }
    """

    def on_mount(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.clear()
        self.root.expand()
        # run_id -> (node, start_time, base_label, is_subagent)
        self._run_map: dict[str, tuple[TreeNode[None], float, str, bool]] = {}
        # checkpoint_ns -> sub-agent node, used to nest events under sub-agents.
        self._ns_to_node: dict[str, TreeNode[None]] = {}
        self._thinking_node: TreeNode[None] | None = None
        self._thinking_run_id: str | None = None

    def reset(self) -> None:  # type: ignore[override]
        self._reset()

    # -- tool tracking -------------------------------------------------------

    def on_tool_started(self, msg: ToolStarted) -> None:
        self._clear_thinking()
        parent_node = self._find_parent(msg.checkpoint_ns)
        preview = msg.input_preview[:40].replace("\n", " ")
        if msg.is_subagent:
            base = "[bold]▶ sub-agent[/bold]"
            if preview:
                base += f": [dim]{preview}[/dim]"
            label = f"{base}  [yellow]●[/yellow]"
            node = parent_node.add(label, expand=True)
            self._ns_to_node[msg.checkpoint_ns] = node
        else:
            base = f"⚙ {msg.name}"
            if preview:
                base += f": [dim]{preview}[/dim]"
            label = f"{base}  [yellow]●[/yellow]"
            node = parent_node.add_leaf(label)
        self._run_map[msg.run_id] = (
            node,
            time.monotonic(),
            base,
            msg.is_subagent,
        )

    def on_tool_ended(self, msg: ToolEnded) -> None:
        entry = self._run_map.get(msg.run_id)
        if entry is None:
            return
        node, start_time, base, is_subagent = entry
        elapsed = time.monotonic() - start_time
        marker = "[red]✗[/red]" if msg.error else "[green]✓[/green]"
        duration = f"[dim]({_fmt_duration(elapsed)})[/dim]"
        node.set_label(f"{base}  {marker} {duration}")
        if msg.error and msg.output_snippet:
            snippet = msg.output_snippet[:80].replace("\n", " ")
            node.add_leaf(f"[red]└ {snippet}[/red]")
        if is_subagent:
            node.collapse()

    # -- LLM thinking indicator ----------------------------------------------

    def on_llm_thinking(self, msg: LLMThinking) -> None:
        self._clear_thinking()
        parent_node = self._find_parent(msg.checkpoint_ns)
        self._thinking_node = parent_node.add_leaf("[italic]⋯ thinking…[/italic]")
        self._thinking_run_id = msg.run_id

    def on_llm_done(self, msg: LLMDone) -> None:
        if self._thinking_run_id == msg.run_id:
            self._clear_thinking()

    def _clear_thinking(self) -> None:
        if self._thinking_node is not None:
            self._thinking_node.remove()
            self._thinking_node = None
            self._thinking_run_id = None

    # -- helpers -------------------------------------------------------------

    def _find_parent(self, checkpoint_ns: str) -> TreeNode[None]:
        """Return the sub-agent node whose checkpoint_ns is the longest prefix of
        checkpoint_ns.

        A task tool's ns looks like "tools:<uuid>".  Events belonging to the
        sub-agent it spawned have ns "tools:<uuid>|tools:<inner_uuid>…".
        The longest matching prefix wins, which handles nested sub-agents correctly.
        """
        best_ns = ""
        best_node = self.root
        for ns, node in self._ns_to_node.items():
            if checkpoint_ns.startswith(ns + "|") and len(ns) > len(best_ns):
                best_ns = ns
                best_node = node
        return best_node


class ThinkingPanel(VerticalScroll):
    """Ephemeral strip that shows live-streaming LLM tokens while agent runs."""

    DEFAULT_CSS = """
    ThinkingPanel {
        height: 10;
        border-top: dashed $warning-darken-2;
        padding: 0 1;
        scrollbar-size: 1 1;
        display: none;
    }
    ThinkingPanel Static {
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="thinking-text", markup=False)

    def on_mount(self) -> None:
        self._buf = ""

    def reset(self) -> None:
        self._buf = ""
        self.query_one("#thinking-text", Static).update("")
        self.display = True

    def on_text_token(self, msg: TextToken) -> None:
        self._buf += msg.text
        self.query_one("#thinking-text", Static).update(self._buf[-3000:])
        self.scroll_end(animate=False)


class ResponseLog(RichLog):
    """Right pane: buffered markdown response."""

    DEFAULT_CSS = """
    ResponseLog {
        width: 100%;
        height: 1fr;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    def on_mount(self) -> None:
        self._response_buf = ""
        self.last_response = ""

    def clear(self) -> ResponseLog:
        self._response_buf = ""
        return super().clear()

    def on_text_token(self, msg: TextToken) -> None:
        self._response_buf += msg.text

    def on_agent_done(self, _msg: AgentDone) -> None:
        if self._response_buf:
            self.last_response = self._response_buf
            self.write(Rule(style="dim green"))
            self.write("[bold green]✦ assistant[/bold green]")
            self.write(Rule(style="dim green"))
            self.write(Markdown(self._response_buf))
            self._response_buf = ""


class StatusBar(Static):
    """Single-line status bar above the input: connections, timer, tokens, tools."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    """

    mcp_ok: reactive[bool] = reactive(True)
    db_ok: reactive[bool] = reactive(True)
    elapsed_seconds: reactive[int] = reactive(0)
    tokens: reactive[int] = reactive(0)
    active_tools: reactive[int] = reactive(0)
    flash_text: reactive[str] = reactive("")

    def on_mount(self) -> None:
        self._flash_timer: Timer | None = None
        self._refresh_status()

    def watch_mcp_ok(self, _old: bool, _new: bool) -> None:
        self._refresh_status()

    def watch_db_ok(self, _old: bool, _new: bool) -> None:
        self._refresh_status()

    def watch_elapsed_seconds(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_tokens(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_active_tools(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_flash_text(self, _old: str, _new: str) -> None:
        self._refresh_status()

    def flash(self, text: str, duration: float = 3.0) -> None:
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self.flash_text = text
        self._flash_timer = self.set_timer(duration, self._clear_flash)

    def _clear_flash(self) -> None:
        self.flash_text = ""

    def _refresh_status(self) -> None:
        if self.flash_text:
            self.update(self.flash_text)
            return
        mcp = "[green]✓[/green]" if self.mcp_ok else "[red]✗[/red]"
        db = "[green]✓[/green]" if self.db_ok else "[red]✗[/red]"
        mins, secs = divmod(int(self.elapsed_seconds), 60)
        elapsed = f"{mins:02d}:{secs:02d}"
        toks = _fmt_tokens(self.tokens)
        self.update(
            f" mcp {mcp}  db {db}   ⏱ {elapsed}   🧠 {toks} tok   "
            f"⚙ {self.active_tools} active"
        )


class CommandInput(Input):
    """Single-line input with Up/Down history walking."""

    BINDINGS = [
        Binding("up", "history_up", show=False, priority=True),
        Binding("down", "history_down", show=False, priority=True),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_idx: int | None = None
        self._draft: str = ""

    def add_to_history(self, query: str) -> None:
        if not query.strip():
            return
        if self._history and self._history[-1] == query:
            return
        self._history.append(query)
        self._history_idx = None
        self._draft = ""

    def action_history_up(self) -> None:
        if not self._history:
            return
        if self._history_idx is None:
            self._draft = self.value
            self._history_idx = len(self._history) - 1
        elif self._history_idx > 0:
            self._history_idx -= 1
        else:
            return
        self.value = self._history[self._history_idx]
        self.cursor_position = len(self.value)

    def action_history_down(self) -> None:
        if self._history_idx is None:
            return
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self.value = self._history[self._history_idx]
        else:
            self._history_idx = None
            self.value = self._draft
        self.cursor_position = len(self.value)


# ---------------------------------------------------------------------------
# Program selection screen (shown when multiple programs are open in Ghidra)
# ---------------------------------------------------------------------------


class ProgramSelectApp(App[str]):
    TITLE = "Ghidra Agent"
    BINDINGS = [Binding("ctrl+c", "quit", "Quit")]
    CSS = """
    Screen {
        align: center middle;
    }
    #select-container {
        width: 60;
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }
    #select-label {
        margin-bottom: 1;
        text-style: bold;
    }
    """

    def __init__(self, programs: list[str]) -> None:
        super().__init__()
        self._programs = programs

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="select-container"):
            yield Label(
                "Multiple programs are open. Select one to analyze:", id="select-label"
            )
            yield ListView(
                *[ListItem(Label(name), name=name) for name in self._programs]
            )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Program Selection"
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.exit(event.item.name or "")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class GhidraAgentApp(App[None]):
    TITLE = "Ghidra Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+y", "yank", "Copy response"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }
    #panes {
        height: 1fr;
    }
    #right-pane {
        width: 65%;
    }
    #query {
        height: 3;
        border-top: solid $accent-darken-2;
    }
    #query.busy {
        border-top: solid $warning-darken-2;
    }
    """

    def __init__(
        self,
        agent: Any,
        config: dict[str, Any],
        model: str = "",
        session_id: str = "",
        mcp_ok: bool = True,
        db_ok: bool = True,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._config = config
        self._model = model
        self._session_id = session_id
        self._agent_running = False
        self._unregister_toast_sink: Callable[[], None] | None = None
        self._mcp_ok = mcp_ok
        self._db_ok = db_ok
        self._elapsed_timer: Timer | None = None
        self._run_start: float | None = None
        self._total_tokens = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="panes"):
            yield ActivityTree("root agent")
            with Vertical(id="right-pane"):
                yield ResponseLog(highlight=True, markup=True)
                yield ThinkingPanel()
        yield StatusBar()
        yield CommandInput(placeholder=_PLACEHOLDER_IDLE, id="query")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self._model}  ·  session: {self._session_id}"
        self.query_one("#query", Input).focus()
        bar = self.query_one(StatusBar)
        bar.mcp_ok = self._mcp_ok
        bar.db_ok = self._db_ok
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
        self._total_tokens += msg.delta_tokens
        self.query_one(StatusBar).tokens = self._total_tokens

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
        response = self.query_one(ResponseLog)
        response.write(Rule(style="dim cyan"))
        shown = display.replace("\n", "\n  ")
        response.write(f"[bold cyan]❯ {shown}[/bold cyan]")
        response.write(Rule(style="dim cyan"))
        self.query_one(ActivityTree).reset()
        self._run_agent(agent_input)

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
        elif cmd == "/help":
            bar.flash(
                "/clear  /yank  /compact  /quit  /help  ·  "
                "↑/↓ history · Ctrl+Y yank · Ctrl+L clear · Ctrl+C quit"
            )
        else:
            bar.flash(f"[red]Unknown command: {cmd}[/red]")

    # -- bindings ------------------------------------------------------------

    def action_yank(self) -> None:
        text = self.query_one(ResponseLog).last_response
        if not text:
            self.notify("Nothing to copy yet.", severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify("Response copied to clipboard.")

    def action_clear_log(self) -> None:
        self.query_one(ResponseLog).clear()
        self.query_one(ActivityTree).reset()

    # -- run state -----------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._agent_running = busy
        inp = self.query_one("#query", Input)
        bar = self.query_one(StatusBar)
        if busy:
            inp.placeholder = _PLACEHOLDER_BUSY
            inp.add_class("busy")
            self._run_start = time.monotonic()
            bar.elapsed_seconds = 0
            if self._elapsed_timer is not None:
                self._elapsed_timer.stop()
            self._elapsed_timer = self.set_interval(1.0, self._tick_elapsed)
        else:
            inp.placeholder = _PLACEHOLDER_IDLE
            inp.remove_class("busy")
            if self._elapsed_timer is not None:
                self._elapsed_timer.stop()
                self._elapsed_timer = None
            self._run_start = None

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
                self._handle_event(event, activity, response, thinking)
        except Exception as exc:
            response.write(Rule(style="red"))
            response.write(f"[bold red]✗ Error: {exc}[/bold red]")
            response.write(Rule(style="red"))
        finally:
            thinking.display = False
            response.post_message(AgentDone())
            self._set_busy(False)
            self.query_one("#query", Input).focus()

    def _handle_event(
        self,
        event: dict[str, Any],
        activity: ActivityTree,
        response: ResponseLog,
        thinking: ThinkingPanel,
    ) -> None:
        kind = event.get("event", "")
        run_id: str = event.get("run_id", "")
        metadata: dict[str, Any] = event.get("metadata", {})
        checkpoint_ns: str = metadata.get("langgraph_checkpoint_ns", "")
        is_compaction = metadata.get("lc_source") == "summarization"

        if kind == "on_tool_start":
            name = event.get("name", "")
            raw_input = event.get("data", {}).get("input", {})
            preview = _extract_preview(raw_input)
            is_subagent = name == "task"
            activity.post_message(
                ToolStarted(run_id, name, preview, is_subagent, checkpoint_ns)
            )
            self.post_message(ToolCountChanged(1))

        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output")
            error = bool(event.get("data", {}).get("error"))
            snippet = _extract_output_snippet(output) if error else ""
            activity.post_message(ToolEnded(run_id, error, snippet))
            self.post_message(ToolCountChanged(-1))

        elif kind == "on_chat_model_start":
            if is_compaction:
                self.post_message(StatusFlash("[yellow]⟳ Compacting context…[/yellow]"))
            else:
                activity.post_message(LLMThinking(run_id, checkpoint_ns))

        elif kind == "on_chat_model_end":
            if is_compaction:
                self.post_message(StatusFlash("[green]✓ Context compacted[/green]"))
            else:
                activity.post_message(LLMDone(run_id))
            output = event.get("data", {}).get("output")
            tokens = _extract_usage(output)
            if tokens:
                self.post_message(TokenUpdate(tokens))

        elif kind == "on_chat_model_stream":
            if is_compaction:
                return  # suppress summary tokens from the normal output panels
            chunk = event.get("data", {}).get("chunk")
            if chunk is None:
                return
            text = _extract_text(chunk)
            if text:
                response.post_message(TextToken(text))
                thinking.post_message(TextToken(text))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_preview(raw: object) -> str:
    if isinstance(raw, dict):
        text = (
            raw.get("description") or raw.get("task") or raw.get("prompt") or str(raw)
        )
    else:
        text = str(raw)
    return text[:60]


def _extract_text(chunk: object) -> str:
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _extract_output_snippet(output: object) -> str:
    """Pull a short text snippet from a tool's output (typically a ToolMessage)."""
    if output is None:
        return ""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content[:80]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return str(block["text"])[:80]
            if isinstance(block, str):
                return block[:80]
    return str(content)[:80]


def _extract_usage(output: object) -> int:
    """Pull total_tokens from a chat model's usage_metadata, if present."""
    if output is None:
        return 0
    usage = getattr(output, "usage_metadata", None)
    if isinstance(usage, dict):
        try:
            return int(usage.get("total_tokens", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m{secs:02d}s"


def _fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"
