"""Backstops so an agent's final output is never just a tool-call preamble.

Two guards share one salvage core:

``SubagentReportGuardMiddleware`` — deepagents extracts a sub-agent's report as
the text of the last non-empty ``AIMessage`` in its transcript, with no check
for ``tool_calls`` (``deepagents/middleware/subagents.py``,
``_return_command_with_state_update``). That extraction lives in a closure
inside deepagents' ``task`` tool and strips ``messages`` from the returned
state, so it cannot be patched or repaired from outside the sub-agent run; this
middleware runs *inside* it instead. Three observed shapes forward a preamble
("Now let me save the findings…") instead of a report:
- the run ends on a tool call followed by an empty ``end_turn`` message;
- the response is truncated at the output-token limit before the ``tool_use``
  block parses (no ``tool_calls`` at all, or ``invalid_tool_calls``);
- the model announces an action and emits ``end_turn`` without calling anything.
Rather than blacklisting those shapes, the guard checks the *positive* contract
from ``_REPORT_PROTOCOL`` (subagents.py): a real report carries a line starting
with ``REPORT_SENTINEL``. Anything else is salvaged. A genuine report that
merely forgot the sentinel is embedded in full inside the recovery wrapper, so
the failure mode is noise, not loss.

``MainReplyGuardMiddleware`` — the coordinator has the same disease with the
human in the coordinator's role: the TUI renders only the turn's final message,
so a turn ending on saves (or truncated) shows a bare preamble or nothing.
Chat replies stay conversational (no sentinel), so the reply guard relies on
structural signals plus a preamble heuristic. Unlike the sub-agent guard, its
appended message persists in the checkpointed conversation — intentionally: the
salvaged text IS the turn's reply, on resume too.

``after_agent`` appends the salvaged text as a plain ``AIMessage`` so both
consumers (deepagents' backward walk; the TUI's final-state read) land on it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from ghidra_deep_agent.resilience import _TRUNCATION_NUDGE, is_truncated_message

# The line a sub-agent's final report must start with (see ``_REPORT_PROTOCOL``
# in subagents.py, which quotes this constant so prompt and guard cannot drift).
REPORT_SENTINEL = "## Final report"

# Tool calls whose arguments carry the actual findings text; rendered in full.
_KNOWLEDGE_TOOLS = frozenset({"save_knowledge", "update_knowledge"})

_REPORT_MARKER = (
    "[Report recovered: the sub-agent did not end its run with a final report; "
    "its last message and the findings it persisted this turn follow.]"
)

_REPLY_MARKER = (
    "[Reply recovered: the agent ended its turn without a final reply; its "
    "last message and what it persisted this turn follow.]"
)

_NOTHING_PERSISTED_NOTE = (
    "Nothing was persisted this turn: the run appears to have been cut short "
    "(truncation or an unexecuted tool call). Treat the findings as lost and "
    "consider re-dispatching the task."
)


def _message_text(msg: AIMessage) -> str:
    # Mirror deepagents' pick exactly (`msg.text` property + `.rstrip()`), so
    # our "would extraction misfire" predicate matches its real behavior.
    return msg.text.rstrip() if msg.text else ""


def _clean_final_message(msg: AIMessage) -> bool:
    """Text-bearing, no pending/broken tool calls, not truncated."""
    return bool(
        _message_text(msg)
        and not msg.tool_calls
        and not msg.invalid_tool_calls
        and not is_truncated_message(msg)
    )


def _looks_like_report(msg: AIMessage) -> bool:
    """Does this message satisfy the sub-agent report contract?"""
    if not _clean_final_message(msg):
        return False
    # Line-prefix match (not startswith on the whole text) so a one-sentence
    # lead-in above the header doesn't trigger a false salvage.
    return any(
        line.lstrip().startswith(REPORT_SENTINEL)
        for line in _message_text(msg).splitlines()
    )


def _looks_like_reply(msg: AIMessage) -> bool:
    """Is this a plausible user-facing reply (no sentinel required)?

    A trailing colon is the announced-action shape ("Now let me save …:") —
    prose replies don't end mid-sentence pointing at work that never happened.
    """
    return _clean_final_message(msg) and not _message_text(msg).endswith(":")


def _turn_start(messages: Sequence[BaseMessage]) -> int:
    """Index just after the turn's opening HumanMessage.

    Truncation-recovery nudges are HumanMessages injected mid-turn by
    ``TruncationRecoveryMiddleware``; they do not start a new turn.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and msg.content != _TRUNCATION_NUDGE:
            return i + 1
    return 0


def _render_tool_call(name: str, args: Any) -> str:
    if name in _KNOWLEDGE_TOOLS and isinstance(args, dict):
        lines = [f"### {name}"]
        for key, value in args.items():
            if isinstance(value, str):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {json.dumps(value, default=str)}")
        return "\n".join(lines)
    try:
        rendered = args if isinstance(args, str) else json.dumps(args, default=str)
    except (TypeError, ValueError):
        rendered = repr(args)
    return f"- `{name}` {rendered}"


def _persisted_findings(
    messages: Sequence[BaseMessage], start: int, tail_start: int
) -> list[str]:
    """Render what the turn's tool calls persisted.

    Knowledge-tool calls are rendered in full across the whole turn — since #56
    investigators save provisionally as they go, so findings usually exist even
    when the final save never ran. Other tool calls (annotations) are listed
    compactly only from ``tail_start`` (the misextracted message) onward, to
    bound the noise. Half-parsed knowledge calls from the tail are included raw:
    a truncated save still carries findings text.
    """
    persisted: list[str] = []
    for i in range(start, len(messages)):
        msg = messages[i]
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls:
            if call["name"] in _KNOWLEDGE_TOOLS or i >= tail_start:
                persisted.append(_render_tool_call(call["name"], call["args"]))
        if i >= tail_start:
            for invalid in msg.invalid_tool_calls:
                if invalid.get("name") in _KNOWLEDGE_TOOLS:
                    rendered = _render_tool_call(
                        invalid.get("name") or "?", invalid.get("args")
                    )
                    persisted.append(f"{rendered} (truncated/unparsed)")
    return persisted


def _salvage(
    messages: Sequence[BaseMessage],
    *,
    looks_like: Callable[[AIMessage], bool],
    marker: str,
    scope_pick_to_turn: bool,
) -> str | None:
    start = _turn_start(messages)
    pick: AIMessage | None = None
    pick_index = -1
    pick_stop = start if scope_pick_to_turn else 0
    for i in range(len(messages) - 1, pick_stop - 1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and _message_text(msg):
            pick = msg
            pick_index = i
            break
    if pick is not None and looks_like(pick):
        return None

    # If the model wrote a real report/reply earlier in this same turn and then
    # kept saving, prefer its own words over a reconstruction.
    for i in range(len(messages) - 1, start - 1, -1):
        msg = messages[i]
        if i != pick_index and isinstance(msg, AIMessage) and looks_like(msg):
            return _message_text(msg)

    tail_start = pick_index if pick is not None else len(messages)
    persisted = _persisted_findings(messages, start, tail_start)
    if pick is None and not persisted:
        return None  # nothing to say and nothing saved: appending would be noise

    parts = [marker]
    if pick is not None and _message_text(pick):
        parts.append(_message_text(pick))
    if persisted:
        parts.append("## Persisted findings\n" + "\n".join(persisted))
    else:
        parts.append(_NOTHING_PERSISTED_NOTE)
    return "\n\n".join(parts)


def salvage_report(messages: Sequence[BaseMessage]) -> str | None:
    """Report text to append when deepagents' extraction would misfire.

    Returns ``None`` when the transcript already extracts correctly: the last
    text-bearing ``AIMessage`` (searched globally, mirroring the upstream walk)
    is a clean message carrying the ``REPORT_SENTINEL`` line.
    """
    return _salvage(
        messages,
        looks_like=_looks_like_report,
        marker=_REPORT_MARKER,
        scope_pick_to_turn=False,
    )


def salvage_reply(messages: Sequence[BaseMessage]) -> str | None:
    """Reply text to append when the coordinator's turn ends without one.

    Scoped to the current turn (earlier turns have their own replies): returns
    ``None`` when the turn's last text-bearing ``AIMessage`` is clean prose.
    """
    return _salvage(
        messages,
        looks_like=_looks_like_reply,
        marker=_REPLY_MARKER,
        scope_pick_to_turn=True,
    )


class SubagentReportGuardMiddleware(AgentMiddleware):
    """Append a salvaged final report when a sub-agent run ends without one.

    Runs once, after the model loop has finished, so it cannot perturb the run
    itself; and `messages` never propagates into the coordinator's state, so
    the appended message only affects report extraction.
    """

    def after_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        salvaged = salvage_report(state.get("messages") or [])
        if salvaged is None:
            return None
        return {"messages": [AIMessage(content=salvaged)]}

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


class MainReplyGuardMiddleware(AgentMiddleware):
    """Append a salvaged reply when a coordinator turn ends without one.

    Attached to the main/plan/ask graphs (see cli.py). The appended message
    persists in the checkpointed conversation deliberately: it is the turn's
    reply, including after a resume.
    """

    def after_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        salvaged = salvage_reply(state.get("messages") or [])
        if salvaged is None:
            return None
        return {"messages": [AIMessage(content=salvaged)]}

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
