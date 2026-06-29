import os
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


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
    env = os.environ.get("OPENROUTER_CONFIG")
    if env:
        return Path(env).expanduser()
    # models.py -> ghidra_deep_agent -> src -> <repo root>
    return Path(__file__).resolve().parents[2] / _OPENROUTER_CONFIG_FILENAME


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


def build_model(model_string: str) -> BaseChatModel | str:
    """Return a configured chat model for the given provider:model string.

    For DeepSeek models we return _ChatDeepSeekFixed so reasoning_content is
    correctly round-tripped. For ``openrouter:<model>`` models that have a
    provider-routing preset (see ``openrouter.toml``), we construct
    ``ChatOpenRouter`` directly with that routing; otherwise the string is
    returned as-is for init_chat_model to resolve.
    """
    if model_string.startswith("deepseek:"):
        model_name = model_string.split(":", 1)[1]
        return _ChatDeepSeekFixed(model=model_name)
    if model_string.startswith("openrouter:"):
        model_id = model_string.split(":", 1)[1]
        prefs = _load_openrouter_presets().get(model_id)
        if prefs:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(model=model_id, openrouter_provider=prefs)
    return model_string
