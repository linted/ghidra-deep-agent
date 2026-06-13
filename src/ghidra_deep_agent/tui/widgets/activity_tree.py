from __future__ import annotations

import time

from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from ghidra_deep_agent.tui.events import parse_checkpoint_ns
from ghidra_deep_agent.tui.formatting import fmt_duration
from ghidra_deep_agent.tui.messages import LLMDone, LLMThinking, ToolEnded, ToolStarted


class ActivityTree(Tree[None]):
    """Left pane: live agent/tool call hierarchy."""

    def on_mount(self) -> None:
        self.border_title = "activity"
        self.guide_depth = 2
        self._reset()

    def _reset(self) -> None:
        self.clear()
        self.root.expand()
        # run_id -> (node, start_time, base_label, is_subagent)
        self._run_map: dict[str, tuple[TreeNode[None], float, str, bool]] = {}
        # checkpoint_ns segments -> sub-agent node, used to nest events
        # under sub-agents.
        self._ns_to_node: dict[tuple[str, ...], TreeNode[None]] = {}
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
            base = "[bold cyan]▶ sub-agent[/bold cyan]"
            if preview:
                base += f": [dim]{preview}[/dim]"
            label = f"{base}  [yellow]●[/yellow]"
            node = parent_node.add(label, expand=True)
            self._ns_to_node[parse_checkpoint_ns(msg.checkpoint_ns)] = node
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
        duration = f"[dim]({fmt_duration(elapsed)})[/dim]"
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
        """Return the sub-agent node whose namespace is the longest proper
        prefix of checkpoint_ns.

        A task tool's ns looks like "tools:<uuid>".  Events belonging to the
        sub-agent it spawned have ns "tools:<uuid>|tools:<inner_uuid>…".
        The longest matching prefix wins, which handles nested sub-agents
        correctly.
        """
        segments = parse_checkpoint_ns(checkpoint_ns)
        best_node = self.root
        best_len = 0
        for ns_segments, node in self._ns_to_node.items():
            depth = len(ns_segments)
            if (
                depth < len(segments)
                and segments[:depth] == ns_segments
                and depth > best_len
            ):
                best_len = depth
                best_node = node
        return best_node
