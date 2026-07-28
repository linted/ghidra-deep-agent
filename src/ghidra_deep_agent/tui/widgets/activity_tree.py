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
        # run_id -> (node, start_time, base_label, is_subagent). Entries outlive
        # the run: `on_tool_ended` relabels the node but keeps the mapping.
        self._run_map: dict[str, tuple[TreeNode[None], float, str, bool]] = {}
        # The subset of `_run_map` still showing the in-progress marker, so
        # `mark_resumed` can freeze those without touching finished nodes.
        self._pending_runs: set[str] = set()
        # checkpoint_ns segments -> sub-agent node, used to nest events
        # under sub-agents.
        self._ns_to_node: dict[tuple[str, ...], TreeNode[None]] = {}
        self._thinking_node: TreeNode[None] | None = None
        self._thinking_run_id: str | None = None

    def reset(self) -> None:  # type: ignore[override]
        self._reset()

    def mark_resumed(self) -> None:
        """Freeze the interrupted turn's in-flight nodes and mark the boundary.

        `/continue` keeps the tree rather than clearing it: the work already done
        is still this run's history, and the sub-agents LangGraph restores from
        pending writes never re-emit the events that would redraw them. What
        *was* in flight when the pause hit will never deliver its `on_tool_end`
        — the resumed stream carries fresh run_ids — so relabel those nodes
        instead of leaving a `●` that reads as "still running".

        `_ns_to_node` deliberately survives: it is what lets the resumed events
        nest back under their original sub-agent nodes.
        """
        self._clear_thinking()
        for run_id in self._pending_runs:
            node, _start, base, _is_subagent = self._run_map.pop(run_id)
            node.set_label(f"{base}  [dim]⏸ paused[/dim]")
        self._pending_runs.clear()
        self.root.add_leaf("[dim]↻ continued[/dim]")

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
            ns = parse_checkpoint_ns(msg.checkpoint_ns)
            node = self._ns_to_node.get(ns)
            if node is None:
                node = parent_node.add(label, expand=True)
                self._ns_to_node[ns] = node
            else:
                # After `/continue` the re-run task replays with the same
                # (deterministic) task id, so this sub-agent already has a node
                # holding the work it did before the pause. Relight it rather
                # than growing a twin alongside it.
                node.set_label(label)
                node.expand()
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
        self._pending_runs.add(msg.run_id)

    def on_tool_ended(self, msg: ToolEnded) -> None:
        self._pending_runs.discard(msg.run_id)
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

        Probes candidate prefixes longest-first rather than scanning every
        registered namespace: this runs on every tool-start and every thinking
        event, while ``_ns_to_node`` grows for the whole turn.
        """
        segments = parse_checkpoint_ns(checkpoint_ns)
        for depth in range(len(segments) - 1, 0, -1):
            node = self._ns_to_node.get(segments[:depth])
            if node is not None:
                return node
        return self.root
