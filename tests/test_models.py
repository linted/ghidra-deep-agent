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

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from ghidra_deep_agent.models import (
    _UNPROFILED_ANTHROPIC_MAX_TOKENS,
    _ChatDeepSeekFixed,
    _usage_to_delta,
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
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    monkeypatch.delenv("OPENCODE_API_URL", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_URL", raising=False)


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


# --- opencode: provider (OpenCode Console gateway) -------------------------------


def test_opencode_claude_uses_the_anthropic_endpoint() -> None:
    model = build_model("opencode:claude-sonnet-4-6")
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.anthropic_api_url == "https://opencode.ai/inference/anthropic"
    # The OpenCode key, not ANTHROPIC_API_KEY.
    assert model.anthropic_api_key.get_secret_value() == "oc-key"


def test_opencode_gpt_uses_the_responses_api() -> None:
    model = build_model("opencode:gpt-5.4")
    assert isinstance(model, ChatOpenAI)
    assert model.use_responses_api is True
    assert model.openai_api_base == "https://opencode.ai/inference/openai/v1"


def test_opencode_other_models_use_chat_completions_with_reasoning_round_trip() -> None:
    model = build_model("opencode:deepseek-v4-pro")
    assert isinstance(model, _ChatDeepSeekFixed)
    assert model.api_base == "https://opencode.ai/inference/openai/v1"
    assert model.api_key is not None
    assert model.api_key.get_secret_value() == "oc-key"


def test_opencode_threads_max_tokens_through() -> None:
    for name in ("claude-sonnet-4-6", "gpt-5.4", "glm-5.3"):
        model = build_model(f"opencode:{name}", max_tokens=12_345)
        assert getattr(model, "max_tokens") == 12_345  # noqa: B009


def test_opencode_base_url_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_URL", "https://example.test/inference/")
    model = build_model("opencode:kimi-k3")
    assert isinstance(model, _ChatDeepSeekFixed)
    assert model.api_base == "https://example.test/inference/openai/v1"


def test_opencode_without_key_fails_with_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        build_model("opencode:glm-5.3")


# --- opencode-go: provider (OpenCode Go subscription) ---------------------------


def _sdk_headers(model: Any) -> dict[str, str]:
    """Headers the underlying SDK client sends on every request."""
    client = getattr(model, "root_async_client", None) or model._async_client
    return dict(client.default_headers)


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("claude-haiku-5-5", "/anthropic"),
        ("gpt-6-luna", "/openai/v1"),
        ("glm-5.3", "/openai/v1"),
    ],
)
def test_opencode_go_uses_the_go_base_url(name: str, path: str) -> None:
    model = build_model(f"opencode-go:{name}")
    base = getattr(model, "anthropic_api_url", None) or getattr(
        model, "openai_api_base", None
    )
    if base is None:
        base = getattr(model, "api_base")  # noqa: B009
    assert base == f"https://opencode.ai/inference/go{path}"


def test_opencode_go_base_url_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_URL", "https://example.test/go/")
    model = build_model("opencode-go:glm-5.3")
    assert isinstance(model, _ChatDeepSeekFixed)
    assert model.api_base == "https://example.test/go/openai/v1"


def test_opencode_sends_one_stable_session_id_from_every_model() -> None:
    """Go requires x-opencode-session; it must be the same across all agents."""
    sessions = set()
    for spec in (
        "opencode-go:claude-haiku-5-5",
        "opencode-go:gpt-6-luna",
        "opencode-go:glm-5.3",
        "opencode:glm-5.3",
    ):
        headers = _sdk_headers(build_model(spec))
        sessions.add(headers["x-opencode-session"])
        assert headers["User-Agent"].startswith("ghidra-deep-agent/")
    assert len(sessions) == 1
    assert next(iter(sessions)).startswith("ses_")


def test_opencode_go_without_key_names_the_go_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="opencode-go:glm-5.3 requires"):
        build_model("opencode-go:glm-5.3")


# --- reasoning_effort ------------------------------------------------------------


def test_opencode_sends_reasoning_effort_on_chat_completions() -> None:
    model = build_model("opencode-go:glm-5.3", reasoning_effort="high")
    assert isinstance(model, _ChatDeepSeekFixed)
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["reasoning_effort"] == "high"


def test_opencode_omits_reasoning_effort_when_unset() -> None:
    model = build_model("opencode-go:glm-5.3")
    assert isinstance(model, _ChatDeepSeekFixed)
    assert "reasoning_effort" not in model._get_request_payload([HumanMessage("hi")])


def test_opencode_responses_models_take_reasoning_effort() -> None:
    model = build_model("opencode:gpt-5.4", reasoning_effort="low")
    assert getattr(model, "reasoning_effort") == "low"  # noqa: B009


@pytest.mark.parametrize(
    "spec", ["opencode:claude-sonnet-4-6", "deepseek:deepseek-v4-pro", "zai:glm-5.2"]
)
def test_reasoning_effort_is_ignored_with_a_warning_elsewhere(
    spec: str, capsys: pytest.CaptureFixture[str]
) -> None:
    model = build_model(spec, reasoning_effort="high")
    assert getattr(model, "reasoning_effort", None) is None
    assert "reasoning_effort is not supported" in capsys.readouterr().err


# --- streamed usage --------------------------------------------------------------


def _merged_usage(usages: list[tuple[int, int] | None]) -> tuple[int, int]:
    """Stream chunks carrying ``usages`` through the delta rewrite and merge."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="x",
                usage_metadata=None
                if u is None
                else {
                    "input_tokens": u[0],
                    "output_tokens": u[1],
                    "total_tokens": u[0] + u[1],
                },
            )
        )
        for u in usages
    ]
    seen = None
    for chunk in chunks:
        seen = _usage_to_delta(chunk, seen)
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert isinstance(merged.message, AIMessageChunk)
    usage = merged.message.usage_metadata
    assert usage is not None
    return usage["input_tokens"], usage["output_tokens"]


def test_cumulative_streamed_usage_is_counted_once() -> None:
    """A backend sending the running total on every chunk must not be summed."""
    assert _merged_usage([(100, 1), (100, 2), None, (100, 5)]) == (100, 5)


def test_single_final_usage_chunk_is_unchanged() -> None:
    assert _merged_usage([None, None, (100, 5)]) == (100, 5)
