"""Seeding a side thread with a summary of the main session."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ghidra_deep_agent.prompt import MARKED_BACKGROUND
from ghidra_deep_agent.seed import (
    MIN_MESSAGES_FOR_SUMMARY,
    SeedError,
    marked_prior_context,
)


class _State:
    def __init__(self, messages: list[Any]) -> None:
        self.values = {"messages": messages}


class _Graph:
    def __init__(self, messages: list[Any] | None = None, fail: bool = False) -> None:
        self.messages = messages or []
        self.fail = fail

    async def aget_state(self, config: Any) -> _State:
        if self.fail:
            raise RuntimeError("mongo down")
        return _State(self.messages)


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class _Model:
    def __init__(self, reply: str = "digest", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> _Reply:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("model down")
        return _Reply(self.reply)


def _run(graph: Any, model: Any) -> str | None:
    return asyncio.run(marked_prior_context(graph, {"configurable": {}}, model))


def test_no_model_or_short_history_means_no_seed() -> None:
    assert _run(_Graph(["a"] * 5), None) is None
    model = _Model()
    assert _run(_Graph(["a"] * (MIN_MESSAGES_FOR_SUMMARY - 1)), model) is None
    assert model.prompts == []


def test_summary_is_wrapped_as_background() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    model = _Model("the parser lives at 0x1000")
    history = [HumanMessage("look"), AIMessage("found it"), HumanMessage("more")]
    out = _run(_Graph(history), model)
    assert out == MARKED_BACKGROUND.format(summary="the parser lives at 0x1000")
    assert "<transcript>" in model.prompts[0] and "found it" in model.prompts[0]
    assert _run(_Graph(history), _Model("   ")) is None  # empty summary: no seed


def test_failures_raise_seed_error_with_a_key() -> None:
    with pytest.raises(SeedError) as info:
        _run(_Graph(fail=True), _Model())
    assert info.value.key == "seed_state"
    from langchain_core.messages import HumanMessage

    with pytest.raises(SeedError) as info:
        _run(_Graph([HumanMessage("x")] * 3), _Model(fail=True))
    assert info.value.key == "seed_summary" and "SUMMARY_MODEL" in str(info.value)
