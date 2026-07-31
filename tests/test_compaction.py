"""
Unit tests for the two halves of compaction:

- the out-of-band ``/compact`` driver (``compact_out_of_band``), which must
  summarize on exactly one model call, offload history under the *passed*
  thread id, handle chained compaction, and persist nothing on failure; and
- scope-aware auto-summarization tuning: the builder must hand sub-agent scope
  the aggressive built-in thresholds (trigger 50k tokens / keep 10k tokens),
  leave main scope on deepagents' stock defaults, honor the per-scope
  ``COMPACT_*`` / ``COMPACT_MAIN_*`` env knobs, route summaries to
  ``summary_model`` when one is given, and produce an instance deepagents
  0.7's replace-by-name merge swaps in for the stock
  ``SummarizationMiddleware``.

Run:  uv run pytest tests/test_compaction.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.graph import _apply_custom_middleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ghidra_deep_agent.compaction import (
    build_tuned_summarization_middleware,
    compact_out_of_band,
    create_manual_compaction_engine,
)


def _model() -> FakeListChatModel:
    """Fake chat model with no context profile (the common proxied case)."""
    return FakeListChatModel(responses=["ok"])


def _build(scope: Any, summary_model: Any = None) -> Any:
    return build_tuned_summarization_middleware(
        _model(), StateBackend(), summary_model=summary_model, scope=scope
    )


def test_subagent_scope_gets_builtin_defaults() -> None:
    mw = _build("subagent")
    assert mw._lc_helper.trigger == ("tokens", 50000)
    assert mw._lc_helper.keep == ("tokens", 10000)


def test_main_scope_gets_stock_defaults() -> None:
    mw = _build("main")
    # deepagents' no-profile fallbacks.
    assert mw._lc_helper.trigger == ("tokens", 170000)
    assert mw._lc_helper.keep == ("messages", 6)


def test_env_overrides_land_in_their_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPACT_TRIGGER_TOKENS", "30000")
    monkeypatch.setenv("COMPACT_MAIN_TRIGGER_TOKENS", "100000")
    monkeypatch.setenv("COMPACT_MAIN_KEEP_MESSAGES", "12")
    sub_mw = _build("subagent")
    assert sub_mw._lc_helper.trigger == ("tokens", 30000)
    assert sub_mw._lc_helper.keep == ("tokens", 10000)
    main_mw = _build("main")
    assert main_mw._lc_helper.trigger == ("tokens", 100000)
    assert main_mw._lc_helper.keep == ("messages", 12)


def test_keep_tokens_beats_keep_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPACT_KEEP_TOKENS", "8000")
    monkeypatch.setenv("COMPACT_KEEP_MESSAGES", "10")
    mw = _build("subagent")
    assert mw._lc_helper.keep == ("tokens", 8000)


def test_fraction_without_profile_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COMPACT_TRIGGER_FRACTION", "0.5")
    mw = _build("subagent")
    assert mw._lc_helper.trigger == ("tokens", 50000)
    assert "COMPACT_TRIGGER_FRACTION ignored" in capsys.readouterr().err


def test_summary_model_routes_the_summary_call() -> None:
    cheap = _model()
    assert _build("subagent", summary_model=cheap)._lc_helper.model is cheap
    assert _build("main", summary_model=cheap)._lc_helper.model is cheap


def test_no_summary_model_uses_the_agents_own() -> None:
    agent_model = _model()
    mw = build_tuned_summarization_middleware(
        agent_model, StateBackend(), scope="subagent"
    )
    assert mw._lc_helper.model is agent_model


# --- Out-of-band /compact driver ----------------------------------------------


class _ExplodingModel(FakeListChatModel):
    """A summary model whose every call fails."""

    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("summary model down")


def _history(n: int = 30) -> list[Any]:
    """An alternating human/ai transcript long enough to have a cutoff."""
    out: list[Any] = []
    for i in range(n // 2):
        out.append(HumanMessage(content=f"question {i}"))
        out.append(AIMessage(content=f"answer {i}"))
    return out


def _engine(tmp_path: Path, model: Any | None = None) -> Any:
    return create_manual_compaction_engine(
        model if model is not None else FakeListChatModel(responses=["the summary"]),
        FilesystemBackend(root_dir=tmp_path),
    )


def test_compact_out_of_band_summarizes_and_offloads(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    result = asyncio.run(
        compact_out_of_band(engine, _history(), None, thread_id="thread-42")
    )
    assert result is not None
    event = result.event
    # The summary message is what later turns see in place of the evicted slice.
    assert event["summary_message"].additional_kwargs["lc_source"] == "summarization"
    assert "the summary" in event["summary_message"].content
    assert 0 < event["cutoff_index"] < 30
    assert result.summarized_count == event["cutoff_index"]
    # The history file is named by the *passed* thread id — the engine resolves
    # it from the runnable-config contextvar, which the driver must pin (unset,
    # it falls back to a random `session_*` name).
    history = tmp_path / "conversation_history" / "thread-42.md"
    assert history.exists()
    assert "question 0" in history.read_text(encoding="utf-8")


def test_compact_out_of_band_with_tiny_history_is_a_noop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    messages: list[Any] = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = asyncio.run(
        compact_out_of_band(engine, messages, None, thread_id="thread-42")
    )
    assert result is None
    assert not (tmp_path / "conversation_history").exists()


def test_compact_out_of_band_chains_on_a_prior_event(tmp_path: Path) -> None:
    """A second /compact must extend the first, not restart from index zero."""
    engine = _engine(tmp_path)
    messages = _history(40)
    prior = {
        "cutoff_index": 10,
        "summary_message": HumanMessage(
            content="PRIOR-SUMMARY",
            additional_kwargs={"lc_source": "summarization"},
        ),
        "file_path": None,
    }
    effective = engine._apply_event_to_messages(list(messages), prior)
    effective_cutoff = engine._determine_cutoff_index(effective)
    result = asyncio.run(compact_out_of_band(engine, messages, prior, thread_id="t"))
    assert result is not None
    # -1: the prior summary sits at effective index 0 but is not a state message.
    assert result.event["cutoff_index"] == 10 + effective_cutoff - 1
    # The prior summary is filtered from the offload — its originals are
    # already stored.
    history = (tmp_path / "conversation_history" / "t.md").read_text(encoding="utf-8")
    assert "PRIOR-SUMMARY" not in history


def test_compact_out_of_band_summary_failure_raises_before_any_write(
    tmp_path: Path,
) -> None:
    """Upstream ``_acreate_summary`` swallows model errors into an "Error
    generating summary" *string*, which would get persisted as the summary.
    The driver must instead raise — before the history offload, so nothing
    is written anywhere."""
    engine = _engine(tmp_path, model=_ExplodingModel(responses=["unused"]))
    with pytest.raises(RuntimeError, match="summary model down"):
        asyncio.run(compact_out_of_band(engine, _history(), None, thread_id="t"))
    assert not (tmp_path / "conversation_history").exists()


def test_engine_exposes_every_private_attr_the_driver_uses(tmp_path: Path) -> None:
    """``compact_out_of_band`` drives deepagents/langchain private methods (there
    is no public programmatic-compaction API). If an upgrade renames any of
    them, fail here at test time — not inside a live ``/compact``."""
    engine = _engine(tmp_path)
    for attr in (
        "_apply_event_to_messages",
        "_determine_cutoff_index",
        "_partition_messages",
        "_aoffload_to_backend",
        "_build_new_messages_with_path",
        "_compute_state_cutoff",
        "_backend",
        "_lc_helper",
    ):
        assert hasattr(engine, attr), attr
    lc = engine._lc_helper
    for attr in ("_trim_messages_for_summary", "summary_prompt", "model"):
        assert hasattr(lc, attr), attr


class _ToolFake(FakeListChatModel):
    """FakeListChatModel that a tool-calling agent graph will accept."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_summarization_event_written_as_tools_node_round_trips() -> None:
    """Pins the TUI's ``aupdate_state(..., as_node="tools")`` coupling.

    Out-of-band compaction persists only ``_summarization_event`` (no
    ``ToolMessage`` — there is no calling AI message to pair one with), written
    as the ``tools`` node, the node that owns this key on the in-graph tool
    path. If a deepagents/langchain upgrade renames the node or stops
    accepting the write, fail here."""

    async def run() -> None:
        agent = create_deep_agent(
            model=_ToolFake(responses=["first reply", "second reply"]),
            checkpointer=InMemorySaver(),
            backend=StateBackend(),
        )
        config: Any = {"configurable": {"thread_id": "t1"}}
        turn_input: Any = {"messages": [{"role": "user", "content": "hello"}]}
        await agent.ainvoke(turn_input, config=config)
        n_messages = len((await agent.aget_state(config)).values["messages"])
        event = {
            "cutoff_index": n_messages,
            "summary_message": HumanMessage(
                content="THE-SUMMARY",
                additional_kwargs={"lc_source": "summarization"},
            ),
            "file_path": None,
        }
        await agent.aupdate_state(
            config, {"_summarization_event": event}, as_node="tools"
        )
        state = await agent.aget_state(config)
        assert state.values["_summarization_event"]["cutoff_index"] == n_messages
        # Raw history is untouched — compaction only records the event.
        assert len(state.values["messages"]) == n_messages
        # And the next turn still runs on the compacted thread.
        second_input: Any = {"messages": [{"role": "user", "content": "again"}]}
        out = await agent.ainvoke(second_input, config=config)
        assert out["messages"][-1].content == "second reply"

    asyncio.run(run())


def test_replaces_stock_summarization_by_name() -> None:
    """The whole design rides on deepagents 0.7's replace-by-name merge.

    Canary: a tuned instance must carry the stock middleware's ``.name`` and
    replace it in place (no duplicate) when merged the way ``create_deep_agent``
    merges ``middleware=`` — if a future deepagents changes either half of that
    contract, fail here rather than silently double-summarizing.
    """
    backend = StateBackend()
    stock = create_summarization_middleware(_model(), backend)
    tuned = _build("subagent")
    assert tuned.name == stock.name == "SummarizationMiddleware"
    merged = _apply_custom_middleware([stock], [tuned])
    assert merged == [tuned]
