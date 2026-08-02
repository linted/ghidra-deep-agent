"""
Unit tests for ``build_model``'s max_tokens handling.

The regression these pin down: an Anthropic-protocol endpoint serving a model
langchain-anthropic has no profile for (e.g. ``anthropic:glm-5.2`` on z.ai)
silently got the library's 4096-token output fallback, truncating long
responses mid-tool_use so agent runs ended on a bare preamble with nothing
persisted. ``build_model`` must therefore cap unprofiled anthropic models
explicitly and honor an explicit ``max_tokens`` for the providers it builds.

Run:  uv run pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from ghidra_deep_agent.models import (
    _UNPROFILED_ANTHROPIC_MAX_TOKENS,
    build_model,
)


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider clients want a key at construction; none is ever called."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.delenv("ZAI_API_URL", raising=False)


def test_unprofiled_anthropic_model_gets_the_floor_cap() -> None:
    # glm-5.2 has no langchain-anthropic profile; without our floor it would
    # inherit the library's 4096 fallback.
    model = build_model("anthropic:glm-5.2")
    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens == _UNPROFILED_ANTHROPIC_MAX_TOKENS


def test_profiled_anthropic_model_stays_a_bare_string() -> None:
    # Real Claude models have good profile defaults (32k-128k); leave them to
    # init_chat_model.
    assert build_model("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"


def test_explicit_max_tokens_overrides_the_floor() -> None:
    model = build_model("anthropic:glm-5.2", max_tokens=50_000)
    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens == 50_000


def test_explicit_max_tokens_applies_to_profiled_anthropic_models() -> None:
    model = build_model("anthropic:claude-sonnet-4-6", max_tokens=8192)
    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens == 8192


def test_deepseek_passes_max_tokens_through() -> None:
    model = build_model("deepseek:deepseek-v4-pro", max_tokens=16_384)
    assert not isinstance(model, str)
    assert getattr(model, "max_tokens") == 16_384  # noqa: B009


def test_unsupported_provider_warns_and_ignores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert build_model("ollama:llama3", max_tokens=1000) == "ollama:llama3"
    assert "max_tokens is not supported" in capsys.readouterr().err


# --- zai: provider (z.ai OpenAI-compatible endpoint) ----------------------------


def test_zai_builds_chatopenai_against_the_documented_endpoint() -> None:
    model = build_model("zai:glm-5.2")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "glm-5.2"
    assert "api.z.ai/api/paas/v4" in str(model.openai_api_base)
    # No silent cap on this path: unset means the server default applies.
    assert model.max_tokens is None


def test_zai_threads_max_tokens_through() -> None:
    model = build_model("zai:glm-5.2", max_tokens=32_768)
    assert isinstance(model, ChatOpenAI)
    assert model.max_tokens == 32_768


def test_zai_base_url_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_URL", "https://example.test/v4/")
    model = build_model("zai:glm-5.2")
    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base) == "https://example.test/v4/"


def test_zai_without_key_fails_with_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        build_model("zai:glm-5.2")
