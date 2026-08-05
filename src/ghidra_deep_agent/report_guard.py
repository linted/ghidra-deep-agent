"""Backstop so a sub-agent's report is never just a tool-call preamble.

deepagents extracts a sub-agent's report as the text of the last non-empty
``AIMessage`` in its transcript, with no check for ``tool_calls``
(``deepagents/middleware/subagents.py``, ``_return_command_with_state_update``).
Since the write-policy tiers told investigators to persist findings via tool
calls (#56), runs frequently end on a ``save_knowledge``/bookmark call followed
by Anthropic's empty ``end_turn`` message — and the extraction then forwards the
tool call's one-line preamble ("Now let me save the findings...") as the entire
report. The extraction lives in a closure inside deepagents' ``task`` tool and
strips ``messages`` from the returned state, so it cannot be patched or repaired
from outside the sub-agent run; this middleware runs *inside* it instead.

``after_agent`` appends a salvaged report as a plain ``AIMessage`` whenever the
transcript would otherwise misextract, so deepagents' backward walk lands on it.
The prompt-side fix (``_REPORT_PROTOCOL`` in ``subagents.py``) makes this a
no-op in the common case; the middleware covers the runs where the model ends on
a tool call anyway.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

# Tool calls whose arguments carry the actual findings text; rendered in full.
_KNOWLEDGE_TOOLS = frozenset({"save_knowledge", "update_knowledge"})

_RECOVERY_MARKER = (
    "[Report recovered: the sub-agent ended its run on a tool call without a "
    "final summary; its last message and the findings it persisted follow.]"
)


def _message_text(msg: AIMessage) -> str:
    # Mirror deepagents' pick exactly (`msg.text` property + `.rstrip()`), so
    # our "would extraction misfire" predicate matches its real behavior.
    return msg.text.rstrip() if msg.text else ""


def _render_tool_call(name: str, args: dict[str, Any]) -> str:
    if name in _KNOWLEDGE_TOOLS:
        lines = [f"### {name}"]
        for key, value in args.items():
            if isinstance(value, str):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {json.dumps(value, default=str)}")
        return "\n".join(lines)
    try:
        rendered = json.dumps(args, default=str)
    except (TypeError, ValueError):
        rendered = repr(args)
    return f"- `{name}` {rendered}"


def salvage_report(messages: Sequence[BaseMessage]) -> str | None:
    """Report text to append when deepagents' extraction would misfire.

    Returns ``None`` when the transcript already extracts correctly (the last
    text-bearing ``AIMessage`` has no tool calls) or when there is nothing to
    salvage from (no AI text at all).
    """
    pick: AIMessage | None = None
    pick_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and _message_text(msg):
            pick = msg
            pick_index = i
            break
    if pick is None or not pick.tool_calls:
        return None

    # The extraction would forward `pick`'s tool-call preamble. If the model
    # wrote a real report earlier in this same turn (a text-only AIMessage
    # since the last HumanMessage) and then kept saving, resurface that.
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = _message_text(msg)
            if text:
                return text

    # Otherwise reconstruct deterministically: the preamble plus what the
    # trailing tool calls persisted (knowledge-base payloads carry the actual
    # findings; annotations are listed by name and args).
    persisted: list[str] = []
    for msg in messages[pick_index:]:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                persisted.append(_render_tool_call(call["name"], call["args"]))
    parts = [_RECOVERY_MARKER, _message_text(pick)]
    if persisted:
        parts.append("## Persisted findings\n" + "\n".join(persisted))
    return "\n\n".join(parts)


class SubagentReportGuardMiddleware(AgentMiddleware):
    """Append a salvaged final report when a sub-agent run ends on a tool call.

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
