"""
Unit tests for scope-aware auto-summarization tuning: the builder must hand
sub-agent scope the aggressive built-in thresholds (trigger 50k tokens / keep
10k tokens), leave main scope on deepagents' stock defaults, honor the
per-scope ``COMPACT_*`` / ``COMPACT_MAIN_*`` env knobs, route summaries to
``summary_model`` when one is given, and produce an instance deepagents 0.7's
replace-by-name merge swaps in for the stock ``SummarizationMiddleware``.

Run:  uv run pytest tests/test_compaction.py -v
"""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends import StateBackend
from deepagents.graph import _apply_custom_middleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ghidra_deep_agent.compaction import build_tuned_summarization_middleware


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
