from typing import Any

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
