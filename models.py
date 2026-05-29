from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class _ChatDeepSeekFixed(ChatDeepSeek):
    """ChatDeepSeek that round-trips reasoning_content back to the API.

    DeepSeek requires that assistant messages from a reasoning (thinking) model
    include the original reasoning_content on subsequent turns.  The base
    langchain_deepseek package stores it in additional_kwargs but never writes
    it back into the request payload, causing a 400 on multi-turn sessions.
    """

    def _get_request_payload(self, input_: Any, *, stop=None, **kwargs) -> dict:
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
    raise ValueError(
        f"Unknown embeddings provider {provider!r}. "
        "Supported: ollama, openai, huggingface, cohere"
    )


def build_model(model_string: str):
    """Return a configured chat model for the given provider:model string.

    For DeepSeek models we return _ChatDeepSeekFixed so reasoning_content is
    correctly round-tripped.  All other provider strings are returned as-is for
    init_chat_model to resolve.
    """
    if model_string.startswith("deepseek:"):
        model_name = model_string.split(":", 1)[1]
        return _ChatDeepSeekFixed(model=model_name)
    return model_string
