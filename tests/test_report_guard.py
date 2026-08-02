"""
Unit tests for the report/reply guards: ``salvage_report`` must fire exactly
when deepagents' report extraction would forward something other than a
sentinel-bearing final report, ``salvage_reply`` when a coordinator turn ends
without a plausible prose reply, and the middleware's appended message must be
what the downstream consumer then picks.

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
    REPORT_SENTINEL,
    MainReplyGuardMiddleware,
    SubagentReportGuardMiddleware,
    salvage_reply,
    salvage_report,
)
from ghidra_deep_agent.resilience import _TRUNCATION_NUDGE

_PREAMBLE = "Now let me save the findings to the knowledge base."
# The shape that escaped PR #57: announced action, colon, and then nothing.
_BARE_PREAMBLE = "Now let me save all the key findings and apply annotations:"
_REPORT = f"{REPORT_SENTINEL}\nThe function parses TLV frames.\n\nPENDING: retype arg1"


def _save_call(
    content: str = "parses TLV frames", call_id: str = "call_1"
) -> dict[str, Any]:
    return {
        "name": "save_knowledge",
        "args": {"category": "finding", "content": content},
        "id": call_id,
        "type": "tool_call",
    }


def _tool_result(call_id: str = "call_1") -> ToolMessage:
    return ToolMessage("saved", tool_call_id=call_id)


# --- salvage_report: no-salvage cases -------------------------------------------


def test_clean_final_report_needs_no_salvage() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
        AIMessage(_REPORT),
    ]
    assert salvage_report(messages) is None


def test_report_after_a_lead_in_line_still_counts() -> None:
    # The sentinel must match as a line prefix, not via startswith on the text.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage("Analysis complete.\n\n" + _REPORT),
    ]
    assert salvage_report(messages) is None


def test_empty_transcript_needs_no_salvage() -> None:
    assert salvage_report([]) is None
    assert salvage_report([HumanMessage("analyze FUN_1400")]) is None


# --- salvage_report: misfire shapes ---------------------------------------------


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


def test_preamble_without_tool_calls_is_salvaged() -> None:
    # The shape that escaped PR #57: the model announced the saves and stopped
    # (truncation before the tool_use parsed, or a plain end_turn). No parsed
    # tool_calls on the final message — the old guard called this clean.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call("dispatch table at 0x1400")]),
        _tool_result(),
        AIMessage(_BARE_PREAMBLE),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")
    assert _BARE_PREAMBLE in salvaged
    # Provisional saves from earlier in the turn are recovered.
    assert "dispatch table at 0x1400" in salvaged


def test_sentinel_report_truncated_at_max_tokens_is_salvaged() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(
            f"{REPORT_SENTINEL}\nThe function pa",
            response_metadata={"stop_reason": "max_tokens"},
        ),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")
    assert "The function pa" in salvaged


def test_openai_style_length_finish_reason_is_salvaged() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_REPORT, response_metadata={"finish_reason": "length"}),
    ]
    assert salvage_report(messages) is not None


def test_invalid_tool_calls_trigger_salvage_and_surface_partial_args() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(
            _PREAMBLE,
            invalid_tool_calls=[
                {
                    "name": "save_knowledge",
                    "args": '{"category": "finding", "content": "half a find',
                    "id": "c1",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
        ),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert "half a find" in salvaged
    assert "(truncated/unparsed)" in salvaged


def test_real_report_missing_sentinel_is_wrapped_not_lost() -> None:
    # The benign failure mode of the sentinel contract: a genuine report that
    # forgot the header is embedded in full, only wrapped.
    text = "Findings:\n- parses TLV frames\n- dispatch table at 0x1400"
    messages: list[BaseMessage] = [HumanMessage("analyze FUN_1400"), AIMessage(text)]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert text in salvaged


def test_nothing_persisted_marker_recommends_redispatch() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_BARE_PREAMBLE),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert "re-dispatching" in salvaged


def test_no_ai_text_but_persisted_saves_are_still_reported() -> None:
    # Zero AI text used to mean "nothing to salvage"; the saves themselves are
    # the findings.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage("", tool_calls=[_save_call("dispatch table at 0x1400")]),
        _tool_result(),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert "dispatch table at 0x1400" in salvaged


# --- salvage_report: earlier-summary resurfacing --------------------------------


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


def test_preamble_shaped_earlier_summary_is_not_resurfaced() -> None:
    # An earlier text-only message that is itself just narration must not be
    # promoted to "the report"; reconstruct instead.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage("Let me check the callers first."),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")


def test_report_from_a_previous_turn_is_not_resurfaced() -> None:
    # A text-only AIMessage BEFORE the last HumanMessage answers an earlier
    # request; salvage must reconstruct from this turn instead.
    messages: list[BaseMessage] = [
        HumanMessage("first task"),
        AIMessage(f"{REPORT_SENTINEL}\nold report about something else"),
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Report recovered:")
    assert "old report about something else" not in salvaged


def test_truncation_nudge_does_not_start_a_new_turn() -> None:
    # TruncationRecoveryMiddleware injects HumanMessages mid-turn; saves made
    # before the nudge still belong to this turn's findings.
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call("dispatch table at 0x1400")]),
        _tool_result(),
        AIMessage(_BARE_PREAMBLE, response_metadata={"stop_reason": "max_tokens"}),
        HumanMessage(_TRUNCATION_NUDGE),
        AIMessage(""),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert "dispatch table at 0x1400" in salvaged


def test_early_saves_are_included_but_early_annotations_are_not() -> None:
    # Pins the split: knowledge tools render across the whole turn, other tool
    # calls only from the misextracted tail.
    early_bookmark = {
        "name": "bookmarks",
        "args": {"action": "set", "address": "0x1400"},
        "id": "b1",
        "type": "tool_call",
    }
    late_comment = {
        "name": "comments",
        "args": {"action": "set", "comment": "dispatcher"},
        "id": "c2",
        "type": "tool_call",
    }
    messages: list[BaseMessage] = [
        HumanMessage("analyze FUN_1400"),
        AIMessage(
            "saving as I go", tool_calls=[_save_call("early finding"), early_bookmark]
        ),
        _tool_result(),
        ToolMessage("ok", tool_call_id="b1"),
        AIMessage(_PREAMBLE, tool_calls=[late_comment]),
        ToolMessage("ok", tool_call_id="c2"),
    ]
    salvaged = salvage_report(messages)
    assert salvaged is not None
    assert "early finding" in salvaged
    assert "bookmarks" not in salvaged
    assert "comments" in salvaged


# --- salvage_reply (coordinator) ------------------------------------------------


def test_normal_prose_reply_needs_no_salvage() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("what does FUN_1400 do?"),
        AIMessage("It parses TLV frames from the UART ring buffer."),
    ]
    assert salvage_reply(messages) is None


def test_turn_ending_on_saves_salvages_a_reply() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("answer the questions in 14-q.md"),
        AIMessage(_PREAMBLE, tool_calls=[_save_call("device powers off via PMIC")]),
        _tool_result(),
        AIMessage(""),
    ]
    salvaged = salvage_reply(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Reply recovered:")
    assert "device powers off via PMIC" in salvaged


def test_colon_preamble_reply_is_salvaged() -> None:
    # No sentinel contract for chat replies; the announced-action shape is the
    # structural tell.
    messages: list[BaseMessage] = [
        HumanMessage("answer the questions in 14-q.md"),
        AIMessage(_BARE_PREAMBLE),
    ]
    salvaged = salvage_reply(messages)
    assert salvaged is not None
    assert salvaged.startswith("[Reply recovered:")
    assert _BARE_PREAMBLE in salvaged


def test_reply_from_a_previous_turn_is_never_picked() -> None:
    # The pick is scoped to the current turn: a turn with no AI text and no
    # saves stays a no-op rather than resurfacing last turn's reply.
    messages: list[BaseMessage] = [
        HumanMessage("first question"),
        AIMessage("first answer"),
        HumanMessage("second question"),
    ]
    assert salvage_reply(messages) is None


def test_earlier_reply_in_same_turn_is_resurfaced() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("answer the questions"),
        AIMessage("The device powers off via the PMIC, details saved."),
        AIMessage(_PREAMBLE, tool_calls=[_save_call()]),
        _tool_result(),
    ]
    assert (
        salvage_reply(messages) == "The device powers off via the PMIC, details saved."
    )


# --- middleware plumbing --------------------------------------------------------


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


def test_reply_middleware_appends_a_plain_reply_message() -> None:
    messages: list[BaseMessage] = [
        HumanMessage("answer the questions"),
        AIMessage(_BARE_PREAMBLE),
    ]
    update = MainReplyGuardMiddleware().after_agent(
        {"messages": messages},  # type: ignore[typeddict-item]
        None,  # type: ignore[arg-type]
    )
    assert update is not None
    appended = update["messages"][0]
    assert isinstance(appended, AIMessage)
    assert appended.text.startswith("[Reply recovered:")
