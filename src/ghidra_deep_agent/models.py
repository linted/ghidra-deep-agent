import os
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from ghidra_deep_agent.defaults import config_path


class _ChatDeepSeekFixed(ChatDeepSeek):
    """ChatDeepSeek that round-trips reasoning_content back to the API.

    DeepSeek requires that assistant messages from a reasoning (thinking) model
    include the original reasoning_content on subsequent turns.  The base
    langchain_deepseek package stores it in additional_kwargs but never writes
    it back into the request payload, causing a 400 on multi-turn sessions.
    """

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        reasoning_iter = (
            msg.additional_kwargs.get("reasoning_content")
            for msg in original_messages
            if isinstance(msg, AIMessage)
        )
        for message in payload.get("messages", []):
            if message["role"] == "assistant":
                rc = next(reasoning_iter, None)
                if rc:
                    message["reasoning_content"] = rc

        return payload


def build_embeddings(embed_string: str) -> Embeddings:
    """Return an embeddings instance for the given provider:model string.

    Supported providers:
      ollama:<model>       — OllamaEmbeddings (langchain-ollama, always installed)
      openai:<model>       — OpenAIEmbeddings (requires: uv add langchain-openai)
      huggingface:<model>  — HuggingFaceEmbeddings (uv add langchain-huggingface)
      cohere:<model>       — CohereEmbeddings (requires: uv add langchain-cohere)
      automated:<model>    — AutoEmbeddings; MongoDB Atlas generates embeddings
                             server-side via Voyage AI (requires an Atlas
                             cluster with Voyage AI configured at the project
                             level — see langchain_mongodb.embeddings)
    """
    provider, _, model = embed_string.partition(":")
    if not model:
        raise ValueError(
            f"EMBED_MODEL must be in provider:model format, got {embed_string!r}"
        )
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return cast(Embeddings, HuggingFaceEmbeddings(model_name=model))
    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return cast(Embeddings, CohereEmbeddings(model=model))
    if provider == "automated":
        from langchain_mongodb.embeddings import AutoEmbeddings

        return AutoEmbeddings(model=model)
    raise ValueError(
        f"Unknown embeddings provider {provider!r}. "
        "Supported: ollama, openai, huggingface, cohere, automated"
    )


_OPENROUTER_CONFIG_FILENAME = "openrouter.toml"
# Cache the parsed presets so we read the file once per process.
_openrouter_presets_cache: dict[str, dict[str, Any]] | None = None


def _openrouter_config_path() -> Path:
    """Resolve presets path: ``OPENROUTER_CONFIG`` env, else repo-root TOML."""
    return config_path("OPENROUTER_CONFIG", _OPENROUTER_CONFIG_FILENAME)


def _load_openrouter_presets() -> dict[str, dict[str, Any]]:
    """Load per-model OpenRouter provider-routing presets from TOML.

    The file is optional: a missing default file means "no presets" (every
    ``openrouter:`` model resolves as before). An explicitly-pointed
    ``OPENROUTER_CONFIG`` that is missing/invalid is warned about, not fatal.

    Schema — each model id (the part after ``openrouter:``) maps to OpenRouter's
    ``provider`` routing object::

        [providers."z-ai/glm-5.2"]
        order = ["z-ai", "novita"]
        allow_fallbacks = true
    """
    global _openrouter_presets_cache
    if _openrouter_presets_cache is not None:
        return _openrouter_presets_cache

    path = _openrouter_config_path()
    explicit = bool(os.environ.get("OPENROUTER_CONFIG"))
    presets: dict[str, dict[str, Any]] = {}
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        providers = raw.get("providers", {})
        if isinstance(providers, dict):
            presets = {
                str(model): prefs
                for model, prefs in providers.items()
                if isinstance(prefs, dict)
            }
    except FileNotFoundError:
        if explicit:
            print(
                f"Warning: OPENROUTER_CONFIG file not found at {path}; "
                "ignoring provider presets.",
                file=sys.stderr,
            )
    except tomllib.TOMLDecodeError as exc:
        print(f"Warning: {path} is not valid TOML ({exc}); ignoring.", file=sys.stderr)

    _openrouter_presets_cache = presets
    return presets


# Output-token budget for Anthropic-protocol models langchain-anthropic has no
# profile for (third-party endpoints like z.ai's glm models). Without this,
# ChatAnthropic falls back to max_tokens=4096 (_FALLBACK_MAX_OUTPUT_TOKENS),
# which truncates long tool_use payloads mid-call and silently ends agent runs
# on a bare preamble. Overridable per model via `max_tokens` in subagents.toml.
_UNPROFILED_ANTHROPIC_MAX_TOKENS = 32768


def _anthropic_has_profile(model_name: str) -> bool:
    """Whether langchain-anthropic knows this model's real output cap.

    Uses a private helper; if a future langchain-anthropic removes it, assume a
    profile exists (explicit TOML ``max_tokens`` still applies either way).
    """
    try:
        from langchain_anthropic.chat_models import _get_default_model_profile
    except ImportError:
        return True
    # Unknown models yield an empty profile; a known model without the cap key
    # would hit the same 4096 fallback, so require the key itself.
    profile = _get_default_model_profile(model_name) or {}
    return bool(profile.get("max_output_tokens"))


# z.ai's OpenAI-compatible endpoint — the path their LangChain guide documents
# (docs.z.ai/guides/develop/langchain/introduction). There is no dedicated
# LangChain z.ai integration; ``zai:<model>`` builds ChatOpenAI against this
# URL (override via ZAI_API_URL) with the ZAI_API_KEY credential.
_ZAI_DEFAULT_API_URL = "https://api.z.ai/api/paas/v4/"


def _build_zai_model(model_name: str, max_tokens: int | None) -> BaseChatModel:
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"zai:{model_name} requires the ZAI_API_KEY environment variable "
            "(the same z.ai key used for the Anthropic-compatible endpoint)."
        )
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(
        model=model_name,
        base_url=os.environ.get("ZAI_API_URL", _ZAI_DEFAULT_API_URL),
        api_key=api_key,  # type: ignore[arg-type]  # str coerces to SecretStr
        **kwargs,
    )


def build_model(
    model_string: str, max_tokens: int | None = None
) -> BaseChatModel | str:
    """Return a configured chat model for the given provider:model string.

    For DeepSeek models we return _ChatDeepSeekFixed so reasoning_content is
    correctly round-tripped. For ``openrouter:<model>`` models that have a
    provider-routing preset (see ``openrouter.toml``), we construct
    ``ChatOpenRouter`` directly with that routing; otherwise the string is
    returned as-is for init_chat_model to resolve. ``zai:<model>`` targets
    z.ai's OpenAI-compatible endpoint via ChatOpenAI (their documented
    LangChain path; no dedicated integration exists).

    ``max_tokens`` (from ``max_tokens`` in subagents.toml) caps output tokens.
    Anthropic-protocol models that langchain-anthropic has no profile for get
    ``_UNPROFILED_ANTHROPIC_MAX_TOKENS`` even without an explicit setting —
    the library's 4096 fallback truncates real runs.
    """
    if model_string.startswith("zai:"):
        return _build_zai_model(model_string.split(":", 1)[1], max_tokens)
    if model_string.startswith("deepseek:"):
        model_name = model_string.split(":", 1)[1]
        if max_tokens is not None:
            return _ChatDeepSeekFixed(model=model_name, max_tokens=max_tokens)
        return _ChatDeepSeekFixed(model=model_name)
    if model_string.startswith("openrouter:"):
        model_id = model_string.split(":", 1)[1]
        prefs = _load_openrouter_presets().get(model_id)
        if prefs or max_tokens is not None:
            from langchain_openrouter import ChatOpenRouter

            kwargs: dict[str, Any] = {}
            if prefs:
                kwargs["openrouter_provider"] = prefs
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenRouter(model=model_id, **kwargs)
        return model_string
    if model_string.startswith("anthropic:"):
        model_name = model_string.split(":", 1)[1]
        if max_tokens is None and not _anthropic_has_profile(model_name):
            max_tokens = _UNPROFILED_ANTHROPIC_MAX_TOKENS
        if max_tokens is not None:
            from langchain_anthropic import ChatAnthropic

            # The documented runtime names; mypy's pydantic plugin only sees
            # the field aliases.
            return ChatAnthropic(  # type: ignore[call-arg]
                model=model_name, max_tokens=max_tokens
            )
        return model_string
    if max_tokens is not None:
        # Not every provider accepts a `max_tokens` kwarg (Ollama wants
        # num_predict, etc.), so don't guess — say it's ignored.
        print(
            f"Warning: max_tokens is not supported for {model_string!r}; "
            "ignoring (supported: anthropic, deepseek, openrouter, zai).",
            file=sys.stderr,
        )
    return model_string


def ensure_chat_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model *string* into a real chat model.

    ``build_model`` deliberately returns the string unchanged for every provider
    it doesn't special-case, leaving resolution to deepagents/``init_chat_model``.
    That is fine for handing to ``create_deep_agent``, but callers that *use* the
    model directly need the object: ``.ainvoke`` for the plan/ask context summary
    and ``.profile`` for the context gauge both silently fail on a bare ``str``.
    ``compaction.py`` already guards its own call site this way.
    """
    if isinstance(model, str):
        from deepagents._models import resolve_model

        resolved: BaseChatModel = resolve_model(model)
        return resolved
    return model
