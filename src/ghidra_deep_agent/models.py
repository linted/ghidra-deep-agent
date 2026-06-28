import os
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


def build_model(model_string: str) -> BaseChatModel | str:
    """Return a configured chat model for the given provider:model string.

    For DeepSeek models we return _ChatDeepSeekFixed so reasoning_content is
    correctly round-tripped.  For OpenRouter models we return a ChatOpenAI
    instance pointed at OpenRouter's OpenAI-compatible API (configured via
    OPENROUTER_API_KEY and OPENROUTER_BASE_URL).  All other provider strings are
    returned as-is for init_chat_model to resolve.
    """
    if model_string.startswith("deepseek:"):
        model_name = model_string.split(":", 1)[1]
        return _ChatDeepSeekFixed(model=model_name)
    if model_string.startswith("openrouter:"):
        model_name = model_string.split(":", 1)[1]
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set for openrouter: models")
        return ChatOpenAI(
            model=model_name,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            api_key=cast(Any, api_key),
        )
    return model_string
