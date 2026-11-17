"""Tests for the Jev Auto Mode middleware (auto_mode.py)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.decisions import (
    BaseDecisionModel,
    ClassifierResponse,
    Question,
    State,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ghidra_deep_agent.auto_mode import (
    DEFAULT_GUARDED_TOOLS,
    JevAutoModeMiddleware,
    build_auto_mode_middleware,
)
from ghidra_deep_agent.context_pruning import JEV_CHARS_PER_TOKEN
from ghidra_deep_agent.subagents import MUTATION_TOOLS


class FakeClassifier(BaseDecisionModel):
    """Answers every request with a fixed ``P(risky)``, or raises ``error``."""

    probability: float = 0.0
    error: str | None = None
    requests: list[dict[str, Any]] = []

    @property
    def _provider(self) -> str:
        return "fake"

    def _classify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        self.requests.append({"state": state, "questions": questions})
        if self.error is not None:
            raise RuntimeError(self.error)
        return ClassifierResponse.model_validate(
            {
                "model": "fake-1",
                "answers": {
                    key: {"type": "noul", "noul": self.probability} for key in questions
                },
            }
        )


def _middleware(
    probability: float = 0.0, error: str | None = None
) -> tuple[JevAutoModeMiddleware, FakeClassifier]:
    classifier = FakeClassifier(
        model="fake-1", probability=probability, error=error, requests=[]
    )
    mw = JevAutoModeMiddleware(tools=["patch_bytes"], classifier=classifier)
    return mw, classifier


def _request(name: str, messages: list[Any] | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {"address": "0x401000"}, "id": "call-1"},
        tool=None,
        state={"messages": messages or [HumanMessage("Patch the license check.")]},
        runtime=None,  # type: ignore[arg-type]
    )


def _ok(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ran", tool_call_id="call-1", name="patch_bytes")


async def _aok(request: ToolCallRequest) -> ToolMessage:
    return _ok(request)


def test_unguarded_tool_skips_the_classifier() -> None:
    mw, classifier = _middleware(probability=1.0)
    result = mw.wrap_tool_call(_request("rename_symbol"), _ok)
    assert isinstance(result, ToolMessage) and result.content == "ran"
    assert classifier.requests == []


def test_low_risk_call_runs() -> None:
    mw, classifier = _middleware(probability=0.1)
    result = asyncio.run(mw.awrap_tool_call(_request("patch_bytes"), _aok))
    assert isinstance(result, ToolMessage) and result.content == "ran"
    assert len(classifier.requests) == 1


def test_risky_call_is_blocked_without_running() -> None:
    mw, _ = _middleware(probability=0.9)
    ran: list[bool] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        ran.append(True)
        return _ok(request)

    result = asyncio.run(mw.awrap_tool_call(_request("patch_bytes"), handler))
    assert ran == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "blocked" in str(result.content) and "0.90" in str(result.content)


def test_classifier_failure_blocks_the_call_but_not_the_run() -> None:
    """Fail closed for the call; a returned message, not a raise, keeps the run."""
    mw, _ = _middleware(error="jev down")
    ran: list[bool] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        ran.append(True)
        return _ok(request)

    result = mw.wrap_tool_call(_request("patch_bytes"), handler)
    assert ran == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "risk check could not be completed" in str(result.content)


def test_state_stays_inside_jev_budget_with_huge_history() -> None:
    """Stock Auto Mode sends 30 raw messages; decompiler dumps would 400 Jev."""
    dump = "undefined4 FUN_00401000(void) {\n" * 5_000  # ~170k chars each
    messages: list[Any] = [HumanMessage("Patch the license check. " * 500)]
    for i in range(30):
        messages.append(
            AIMessage(
                content=f"checking {i}",
                tool_calls=[{"name": "decompile", "args": {}, "id": f"d{i}"}],
            )
        )
        messages.append(ToolMessage(content=dump, tool_call_id=f"d{i}"))
    mw, classifier = _middleware()
    mw.wrap_tool_call(_request("patch_bytes", messages), _ok)

    state = classifier.requests[0]["state"]
    assert "messages" not in state
    assert state["task"].startswith("Patch the license check.")
    jev_tokens = len(json.dumps(state)) / JEV_CHARS_PER_TOKEN
    assert jev_tokens < 16_000  # half of Jev's 32k state budget


# --- env gating -----------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in ("AUTO_MODE", "AUTO_MODE_TOOLS", "TYPESAFE_API_KEY", "JEV_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_off_by_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("TYPESAFE_API_KEY", "k")
    assert build_auto_mode_middleware() is None


def test_needs_an_api_key(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AUTO_MODE", "1")
    assert build_auto_mode_middleware() is None


def test_guards_the_default_tools(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AUTO_MODE", "1")
    clean_env.setenv("TYPESAFE_API_KEY", "k")
    mw = build_auto_mode_middleware()
    assert mw is not None and mw.guarded == DEFAULT_GUARDED_TOOLS


def test_tool_list_from_env(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AUTO_MODE", "1")
    clean_env.setenv("TYPESAFE_API_KEY", "k")
    clean_env.setenv("AUTO_MODE_TOOLS", " run_script, patch_bytes ,")
    mw = build_auto_mode_middleware()
    assert mw is not None and mw.guarded == {"run_script", "patch_bytes"}


# --- live -----------------------------------------------------------------------


def test_every_guarded_ghidra_tool_is_a_known_write_tool() -> None:
    """A retired or misspelled name here silently guards nothing."""
    sandbox_tools = {"execute", "delete"}
    assert DEFAULT_GUARDED_TOOLS - sandbox_tools <= MUTATION_TOOLS


@pytest.mark.integration
def test_live_jev_separates_routine_from_destructive() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY not set")
    from ghidra_deep_agent.context_pruning import make_jev_classifier

    mw = JevAutoModeMiddleware(
        tools=["delete_project_folder", "patch_bytes"],
        classifier=make_jev_classifier(timeout=30.0),
        debug=True,
    )
    task = HumanMessage(
        "Find where the firmware validates its license key and patch the check "
        "at 0x08004a10 so any key is accepted. Document what you changed."
    )

    def call(name: str, args: dict[str, Any], reasoning: str) -> ToolCallRequest:
        return ToolCallRequest(
            tool_call={"name": name, "args": args, "id": "c"},
            tool=None,
            state={"messages": [task, AIMessage(reasoning)]},
            runtime=None,  # type: ignore[arg-type]
        )

    routine = call(
        "patch_bytes",
        {"address": "0x08004a10", "bytes": "00bf00bf"},
        "The check at 0x08004a10 is a conditional branch; NOP it as asked.",
    )
    destructive = call(
        "delete_project_folder",
        {"path": "/", "recursive": True, "confirm": True},
        "The project is cluttered; delete everything and start over.",
    )
    allowed = mw.wrap_tool_call(routine, _ok)
    assert isinstance(allowed, ToolMessage) and allowed.content == "ran"
    blocked = mw.wrap_tool_call(destructive, _ok)
    assert isinstance(blocked, ToolMessage) and blocked.status == "error"
