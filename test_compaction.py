"""
Unit tests for scope-aware auto-summarization tuning: the monkeypatched
deepagents factory must hand sub-agents the aggressive built-in thresholds
(trigger 50k tokens / keep 10k tokens), leave the main agent on deepagents'
stock defaults, honor the per-scope ``COMPACT_*`` / ``COMPACT_MAIN_*`` env
knobs, and route summaries to ``summary_model`` when one is given.

Run:  uv run pytest test_compaction.py -v
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import deepagents.graph as graph
import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ghidra_deep_agent.compaction import install_tuned_summarization


class _NamedFakeModel(FakeListChatModel):
    """Fake chat model exposing ``model_name`` (no context profile)."""

    model_name: str = ""


def _model(name: str = "") -> FakeListChatModel:
    if name:
        return _NamedFakeModel(responses=["ok"], model_name=name)
    return FakeListChatModel(responses=["ok"])


@pytest.fixture(autouse=True)
def _restore_factory() -> Iterator[None]:
    """Undo the monkeypatch so each test installs from a clean slate."""
    original = getattr(graph, "create_summarization_middleware")
    try:
        yield
    finally:
        setattr(graph, "create_summarization_middleware", original)


def _build(model: Any) -> Any:
    """Invoke the (patched) factory the way deepagents' graph does."""
    factory = getattr(graph, "create_summarization_middleware")
    return factory(model, StateBackend())


def test_patch_applied_and_idempotent() -> None:
    assert not getattr(create_summarization_middleware, "_ghidra_tuned", False)
    install_tuned_summarization()
    patched = getattr(graph, "create_summarization_middleware")
    assert getattr(patched, "_ghidra_tuned", False)
    install_tuned_summarization()
    assert getattr(graph, "create_summarization_middleware") is patched


def test_subagent_gets_builtin_defaults() -> None:
    install_tuned_summarization(main_model=_model())
    mw = _build(_model())
    assert mw._lc_helper.trigger == ("tokens", 50000)
    assert mw._lc_helper.keep == ("tokens", 10000)


def test_main_model_gets_stock_defaults() -> None:
    main = _model()
    install_tuned_summarization(main_model=main)
    mw = _build(main)
    # deepagents' no-profile fallbacks.
    assert mw._lc_helper.trigger == ("tokens", 170000)
    assert mw._lc_helper.keep == ("messages", 6)


def test_main_matched_by_name_from_spec_string() -> None:
    install_tuned_summarization(main_model="openrouter:acme/big-model")
    mw = _build(_model("acme/big-model"))
    assert mw._lc_helper.trigger == ("tokens", 170000)
    mw = _build(_model("acme/other-model"))
    assert mw._lc_helper.trigger == ("tokens", 50000)


def test_no_main_model_treats_everything_as_subagent() -> None:
    install_tuned_summarization()
    mw = _build(_model())
    assert mw._lc_helper.trigger == ("tokens", 50000)


def test_env_overrides_land_in_their_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPACT_TRIGGER_TOKENS", "30000")
    monkeypatch.setenv("COMPACT_MAIN_TRIGGER_TOKENS", "100000")
    monkeypatch.setenv("COMPACT_MAIN_KEEP_MESSAGES", "12")
    main = _model()
    install_tuned_summarization(main_model=main)
    sub_mw = _build(_model())
    assert sub_mw._lc_helper.trigger == ("tokens", 30000)
    assert sub_mw._lc_helper.keep == ("tokens", 10000)
    main_mw = _build(main)
    assert main_mw._lc_helper.trigger == ("tokens", 100000)
    assert main_mw._lc_helper.keep == ("messages", 12)


def test_keep_tokens_beats_keep_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPACT_KEEP_TOKENS", "8000")
    monkeypatch.setenv("COMPACT_KEEP_MESSAGES", "10")
    install_tuned_summarization()
    mw = _build(_model())
    assert mw._lc_helper.keep == ("tokens", 8000)


def test_fraction_without_profile_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COMPACT_TRIGGER_FRACTION", "0.5")
    install_tuned_summarization()
    mw = _build(_model())
    assert mw._lc_helper.trigger == ("tokens", 50000)
    assert "COMPACT_TRIGGER_FRACTION ignored" in capsys.readouterr().err


def test_summary_model_routes_the_summary_call() -> None:
    cheap = _model("cheap-summarizer")
    main = _model("main-model")
    install_tuned_summarization(cheap, main_model=main)
    assert _build(_model())._lc_helper.model is cheap
    assert _build(main)._lc_helper.model is cheap
