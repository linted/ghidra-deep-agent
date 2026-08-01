"""
Unit tests for the sub-agent report guard: ``salvage_report`` must fire exactly
when deepagents' report extraction would forward a tool-call preamble instead of
a real findings summary, and the middleware's appended message must be what that
extraction then picks.

Run:  uv run pytest tests/test_report_guard.py -v
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from ghidra_deep_agent.report_guard import (
    SubagentReportGuardMiddleware,
    salvage_report,
)

_PREAMBLE = "Now let me save the findings to the knowledge base."
_REPORT = "## Findings\nThe function parses TLV frames.\n\nPENDING: retype arg1"


def _save_call(content: str = "parses TLV frames") -> dict[str, Any]:
    return {
        "name": "save_knowledge",
        "args": {"category": "finding", "content": content},
        "id": "call_1",
        "type": "tool_call",
    }


def _tool_result(call_id: str = "call_1") -> ToolMessage:
    return ToolMessage("saved", tool_call_id=call_id)


def test_clean_final_report_needs_no_salvage() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
        AIMessage(_REPORT),
    ]
    assert salvage_report(messages) is None


def test_no_ai_text_at_all_needs_no_salvage() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage("", tool_calls=[_save_call()]),
        _tool_result(),
    ]
    assert salvage_report(messages) is None
    assert salvage_report([]) is None


def test_run_ending_on_tool_call_reconstructs_from_persisted_findings() -> None:
    # The #56 failure mode: final act is a save, then Anthropic's empty
    # end_turn AIMessage — deepagents' walk would forward the bare preamble.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call("dispatch table at 0x1400")]),
        _tool_result(),
        AIMessage(""),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")
    assert _PREAMBLE in salvaged
    assert "dispatch table at 0x1400" in salvaged
    assert "save_knowledge" in salvaged


def test_earlier_report_in_same_turn_is_resurfaced_verbatim() -> None:
    # The model wrote a real summary, then kept saving: prefer its own words
    # over a reconstruction.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_REPORT),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
    ]
    assert salvage_report(messages) == _REPORT


def test_report_from_a_previous_turn_is_not_resurfaced() -> None:
    # A text-only AIMessage BEFORE the last HumanMessage answers an earlier
    # request; salvage must reconstruct from this turn instead.
    messages: list[BaseMessage] = [
        HumanMessage("first task"),
        AIMessage("old report about something else"),
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")
    assert "old report about something else" not in salvaged


def test_middleware_appends_message_that_deepagents_extraction_picks() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
        AIMessage(""),
    ]
    update = SubagentReportGuardMiddleware().after_agent(
        {"messages": messages},  # type: ignore[typeddict-item]
        None,  # type: ignore[arg-type]  # runtime is unused
    )
    assert update is not None
    appended = update["messages"][0]
    assert isinstance(appended, AIMessage)
    assert not appended.tool_calls

    # Pin the deepagents contract this guard targets: the report is the last
    # AIMessage with non-empty text (deepagents/middleware/subagents.py,
    # `_return_command_with_state_update`). If a deepagents upgrade changes
    # this selection, revisit report_guard.py.
    content = ""
    for msg in reversed([*messages, appended]):
        if isinstance(msg, AIMessage):
            text = msg.text.rstrip() if msg.text else ""
            if text:
                content = text
                break
    assert content == appended.text
    assert content.startswith("[Report recovered:")


def test_middleware_is_a_no_op_on_clean_transcripts() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_REPORT),
    ]
    update = SubagentReportGuardMiddleware().after_agent(
        {"messages": messages},  # type: ignore[typeddict-item]
        None,  # type: ignore[arg-type]
    )
    assert update is None
