"""Per-turn bookkeeping for the stream-event handler.

``handle_event`` has to remember things across events within a single turn: which
tool runs are hidden, which are in flight, which async task each deferred node is
waiting on, and what the main thread last said. That state used to live as four
containers on ``GhidraAgentApp`` that ``events.py`` reached into directly — which
both coupled the two modules (hence the ``TYPE_CHECKING``-only import to break the
resulting import cycle) and leaked: nothing cleared them, so a turn cancelled with
Escape left its in-flight run_ids behind forever and the status bar's active-tool
count never returned to zero.

Making it an object the app *replaces* per turn fixes that structurally — a fresh
run starts from a fresh state, with no reset step to forget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunState:
    """Tool bookkeeping for one agent turn."""

    # Runs suppressed from the activity tree: `get_task_status` polls made by the
    # async middleware, and any call made from inside another tool's body. Tracked
    # rather than merely skipped so the paired on_tool_end stays balanced.
    hidden_tool_runs: set[str] = field(default_factory=set)
    # task_id -> run_id for async tool calls whose "completed" marker is deferred
    # until ASYNC_DONE_EVENT arrives (their own on_tool_end fires early, carrying
    # only the submission stub).
    pending_async: dict[str, str] = field(default_factory=dict)
    # run_id -> (description, start time) for sub-agent (`task`) runs in flight.
    subagent_meta: dict[str, tuple[str, float]] = field(default_factory=dict)
    # Plain (non-subagent) tool runs in flight. A call whose parent_ids chain
    # contains one of these was made from *inside* another tool and is hidden.
    active_tool_runs: set[str] = field(default_factory=set)
    # The main thread's latest assistant text, captured synchronously from the
    # stream loop so `/approve` never depends on reading the plan file back.
    last_reply_text: str = ""
