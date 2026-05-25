from __future__ import annotations

from rich.markdown import Markdown
from rich.rule import Rule
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Footer, Header, Input, RichLog, Static, Tree
from textual.widgets.tree import TreeNode
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import work


_PLACEHOLDER_IDLE = "Enter analysis task (Ctrl+C to quit)…"
_PLACEHOLDER_BUSY = "Agent is running… type ahead, Enter to queue"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class ToolStarted(Message):
    def __init__(self, run_id: str, parent_run_id: str | None, name: str,
                 input_preview: str, is_subagent: bool) -> None:
        super().__init__()
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.name = name
        self.input_preview = input_preview
        self.is_subagent = is_subagent


class ToolEnded(Message):
    def __init__(self, run_id: str, error: bool = False) -> None:
        super().__init__()
        self.run_id = run_id
        self.error = error


class LLMThinking(Message):
    def __init__(self, run_id: str, parent_run_id: str | None) -> None:
        super().__init__()
        self.run_id = run_id
        self.parent_run_id = parent_run_id


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


class ChainLinked(Message):
    """Invisible intermediary: records a chain run_id → parent_run_id link."""
    def __init__(self, run_id: str, parent_run_id: str | None) -> None:
        super().__init__()
        self.run_id = run_id
        self.parent_run_id = parent_run_id


class StatusUpdate(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ActivityTree(Tree):
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
        self._run_map: dict[str, TreeNode] = {}
        self._parent_map: dict[str, str | None] = {}
        self._thinking_node: TreeNode | None = None
        self._thinking_run_id: str | None = None

    def reset(self) -> None:
        self._reset()

    # -- tool tracking -------------------------------------------------------

    def on_tool_started(self, msg: ToolStarted) -> None:
        self._clear_thinking()
        parent_node = self._resolve_parent(msg.parent_run_id)
        if msg.is_subagent:
            label = f"▶ sub-agent: {msg.input_preview}" if msg.input_preview else "▶ sub-agent"
            node = parent_node.add(label, expand=True)
        else:
            node = parent_node.add_leaf(f"⚙ {msg.name}  ●")
        self._run_map[msg.run_id] = node
        self._parent_map[msg.run_id] = msg.parent_run_id

    def on_tool_ended(self, msg: ToolEnded) -> None:
        node = self._run_map.get(msg.run_id)
        if node is None:
            return
        current = str(node.label)
        marker = "  ✗" if msg.error else "  ✓"
        node.set_label(current.replace("  ●", marker))

    # -- LLM thinking indicator ----------------------------------------------

    def on_llm_thinking(self, msg: LLMThinking) -> None:
        self._clear_thinking()
        parent_node = self._resolve_parent(msg.parent_run_id)
        self._thinking_node = parent_node.add_leaf("⋯ thinking…")
        self._thinking_run_id = msg.run_id

    def on_llm_done(self, msg: LLMDone) -> None:
        if self._thinking_run_id == msg.run_id:
            self._clear_thinking()

    def _clear_thinking(self) -> None:
        if self._thinking_node is not None:
            self._thinking_node.remove()
            self._thinking_node = None
            self._thinking_run_id = None

    def on_chain_linked(self, msg: ChainLinked) -> None:
        self._parent_map[msg.run_id] = msg.parent_run_id

    # -- helpers -------------------------------------------------------------

    def _resolve_parent(self, parent_run_id: str | None) -> TreeNode:
        """Walk the parent chain to find the nearest tracked node."""
        rid = parent_run_id
        while rid is not None:
            if rid in self._run_map:
                return self._run_map[rid]
            rid = self._parent_map.get(rid)
        return self.root


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

    def clear(self) -> "ResponseLog":
        self._response_buf = ""
        return super().clear()

    def on_text_token(self, msg: TextToken) -> None:
        self._response_buf += msg.text

    def on_status_update(self, msg: StatusUpdate) -> None:
        self.write(f"[dim]{msg.text}[/dim]")

    def on_agent_done(self, _msg: AgentDone) -> None:
        if self._response_buf:
            self.last_response = self._response_buf
            self.write(Markdown(self._response_buf))
            self._response_buf = ""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class GhidraAgentApp(App):
    TITLE = "Ghidra Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("y", "yank", "Copy response"),
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
        dock: bottom;
        height: 3;
        border-top: solid $accent-darken-2;
    }
    #query.busy {
        border-top: solid $warning-darken-2;
    }
    """

    def __init__(self, agent, config: dict, model: str = "", session_id: str = "") -> None:
        super().__init__()
        self._agent = agent
        self._config = config
        self._model = model
        self._session_id = session_id
        self._agent_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="panes"):
            yield ActivityTree("root agent")
            with Vertical(id="right-pane"):
                yield ResponseLog(highlight=True, markup=True)
                yield ThinkingPanel()
        yield Input(placeholder=_PLACEHOLDER_IDLE, id="query")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self._model}  ·  session: {self._session_id}"
        self.query_one("#query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._agent_running:
            self.query_one(ResponseLog).post_message(
                StatusUpdate("[yellow]Agent still running — please wait.[/yellow]")
            )
            return
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self._set_busy(True)
        response = self.query_one(ResponseLog)
        response.write(Rule(style="dim cyan"))
        response.write(f"[bold cyan]❯ {query}[/bold cyan]")
        response.write(Rule(style="dim cyan"))
        self.query_one(ActivityTree).reset()
        self._run_agent(query)

    def action_yank(self) -> None:
        text = self.query_one(ResponseLog).last_response
        if not text:
            self.notify("Nothing to copy yet.", severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify("Response copied to clipboard.")

    def _set_busy(self, busy: bool) -> None:
        self._agent_running = busy
        inp = self.query_one("#query", Input)
        if busy:
            inp.placeholder = _PLACEHOLDER_BUSY
            inp.add_class("busy")
        else:
            inp.placeholder = _PLACEHOLDER_IDLE
            inp.remove_class("busy")

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
            response.post_message(StatusUpdate(f"[red]Error: {exc}[/red]"))
        finally:
            thinking.display = False
            response.post_message(AgentDone())
            self._set_busy(False)
            self.query_one("#query", Input).focus()

    def _handle_event(self, event: dict, activity: ActivityTree, response: ResponseLog, thinking: ThinkingPanel) -> None:
        kind = event.get("event", "")
        run_id: str = event.get("run_id", "")
        parent_run_id: str | None = event.get("parent_run_id")

        if kind == "on_chain_start":
            activity.post_message(ChainLinked(run_id, parent_run_id))

        elif kind == "on_tool_start":
            name = event.get("name", "")
            raw_input = event.get("data", {}).get("input", {})
            preview = _extract_preview(raw_input)
            is_subagent = name == "task"
            activity.post_message(ToolStarted(run_id, parent_run_id, name, preview, is_subagent))

        elif kind == "on_tool_end":
            error = bool(event.get("data", {}).get("error"))
            activity.post_message(ToolEnded(run_id, error))

        elif kind == "on_chat_model_start":
            activity.post_message(LLMThinking(run_id, parent_run_id))

        elif kind == "on_chat_model_end":
            activity.post_message(LLMDone(run_id))

        elif kind == "on_chat_model_stream":
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
        text = raw.get("description") or raw.get("task") or raw.get("prompt") or str(raw)
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
