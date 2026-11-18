"""
Unit tests for the config-driven worker builder and ``agents.toml`` parsing.

The worker gets the full tool set in normal mode and a mutation-free set in
read-only mode (what makes plan/ask mode read-only by construction); it always
carries the standard middleware stack and the skill sources it is handed.

Run:  uv run pytest tests/test_agent_config.py -v
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from deepagents import SubAgent
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.agent_config import (
    MUTATION_TOOLS,
    WORKER_NAME,
    build_worker,
    load_agent_config,
    make_model_resolver,
    read_only_tools,
)
from ghidra_deep_agent.validation import ArgumentValidationMiddleware

# A mix of write tools (dropped in read-only mode) and read tools (kept).
_TOOL_NAMES = [
    "rename_function",
    "rename_local_variable",
    "set_comment",
    "set_bookmark",
    "set_local_variable_type",
    "set_function_prototype",
    "create_structure",
    "get_pseudocode",
    "get_xrefs_to",
    "list_bookmarks",
    "save_knowledge",
    "recover_prototypes",
]

_SKILLS = ["/skills/analysis/", "/skills/repair/"]


def _fake_tools() -> Sequence[BaseTool]:
    """Minimal stand-ins: the builder only ever reads ``.name``."""
    tools = [SimpleNamespace(name=name) for name in _TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(model: str | None, max_tokens: int | None = None) -> FakeListChatModel:
    """Stub model resolver returning a real chat model instance.

    ``build_worker`` stores the result in the spec and hands it to the tuned
    summarization middleware, which needs an actual ``BaseChatModel``.
    """
    return FakeListChatModel(responses=["ok"])


def _write_config(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "agents.toml"
    path.write_text("[main]\n[worker]\n" + body, encoding="utf-8")
    return path


def _tool_names(spec: SubAgent) -> set[str]:
    # spec["tools"] is a union type; our stubs expose ``.name`` via getattr.
    return {getattr(tool, "name") for tool in spec["tools"]}  # noqa: B009


def _validation_mw(spec: SubAgent) -> ArgumentValidationMiddleware:
    mws = [m for m in spec["middleware"] if isinstance(m, ArgumentValidationMiddleware)]
    assert len(mws) == 1, "the worker gets exactly one validation middleware"
    return mws[0]


def _build(tmp_path: Path, *, read_only: bool, body: str = "") -> SubAgent:
    config = load_agent_config(_write_config(tmp_path, body))
    return build_worker(
        _fake_tools(),
        config,
        _resolver,
        StateBackend(),
        read_only=read_only,
        skills=_SKILLS,
    )


# --- tool scoping --------------------------------------------------------------


def test_worker_gets_full_tool_set_in_normal_mode(tmp_path: Path) -> None:
    spec = _build(tmp_path, read_only=False)
    assert spec["name"] == WORKER_NAME
    assert _tool_names(spec) == set(_TOOL_NAMES)


def test_read_only_worker_drops_every_mutation_tool(tmp_path: Path) -> None:
    spec = _build(tmp_path, read_only=True)
    tool_names = _tool_names(spec)
    assert not tool_names & MUTATION_TOOLS
    assert "rename_function" in MUTATION_TOOLS  # guards against list drift
    # The read tools survive.
    assert tool_names == {"get_pseudocode", "get_xrefs_to", "list_bookmarks"}
    assert "STRICTLY READ-ONLY" in spec["system_prompt"]


def test_normal_worker_prompt_has_no_read_only_section(tmp_path: Path) -> None:
    spec = _build(tmp_path, read_only=False)
    assert "STRICTLY READ-ONLY" not in spec["system_prompt"]


def test_read_only_tools_helper_matches_worker(tmp_path: Path) -> None:
    kept = {t.name for t in read_only_tools(_fake_tools())}
    assert kept == _tool_names(_build(tmp_path, read_only=True))


def test_worker_exclude_drops_named_tools(tmp_path: Path) -> None:
    spec = _build(
        tmp_path, read_only=False, body='exclude = ["get_pseudocode", "set_comment"]\n'
    )
    assert "get_pseudocode" not in _tool_names(spec)
    assert "set_comment" not in _tool_names(spec)


def test_worker_carries_its_skill_sources(tmp_path: Path) -> None:
    spec = _build(tmp_path, read_only=False)
    assert spec["skills"] == _SKILLS


# --- middleware stack ----------------------------------------------------------


def test_auto_mode_sits_right_after_validation(tmp_path: Path) -> None:
    """After validation (a rejected call costs no Jev request), in both modes."""
    from langchain.agents.middleware import AgentMiddleware

    auto_mode = AgentMiddleware()
    config = load_agent_config(_write_config(tmp_path))
    for read_only in (False, True):
        spec = build_worker(
            _fake_tools(),
            config,
            _resolver,
            StateBackend(),
            read_only=read_only,
            skills=_SKILLS,
            auto_mode_middleware=auto_mode,
        )
        middleware = list(spec["middleware"])
        validation = middleware.index(_validation_mw(spec))
        assert middleware[validation + 1] is auto_mode


# --- config parsing ------------------------------------------------------------


def test_empty_tables_default_to_none(tmp_path: Path) -> None:
    config = load_agent_config(_write_config(tmp_path))
    # Empty [main]/[worker] -> None models, inherited at resolve time.
    assert config.main_model is None
    assert config.worker_model is None
    assert config.worker_exclude == ()


def test_full_config_parses(tmp_path: Path) -> None:
    path = tmp_path / "agents.toml"
    path.write_text(
        'default = "deepseek:deepseek-flash"\n'
        "max_tokens = 32768\n"
        '[main]\nmodel = "m-main"\nmax_tokens = 20000\n'
        '[worker]\nmodel = "m-worker"\nmax_tokens = 10000\n'
        'exclude = ["run_script"]\n',
        encoding="utf-8",
    )
    config = load_agent_config(path)
    assert config.default_model == "deepseek:deepseek-flash"
    assert config.default_max_tokens == 32768
    assert config.main_model == "m-main"
    assert config.main_max_tokens == 20000
    assert config.worker_model == "m-worker"
    assert config.worker_max_tokens == 10000
    assert config.worker_exclude == ("run_script",)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_agent_config(tmp_path / "nope.toml")


def test_main_must_be_a_table(tmp_path: Path) -> None:
    path = tmp_path / "agents.toml"
    path.write_text("main = 3\n[worker]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[main\] must be a table"):
        load_agent_config(path)


def test_exclude_must_be_a_list_of_strings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exclude"):
        load_agent_config(_write_config(tmp_path, "exclude = 3\n"))


def test_max_tokens_must_be_a_positive_integer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        load_agent_config(_write_config(tmp_path, "max_tokens = 0\n"))


# --- model resolution ----------------------------------------------------------


def _recording_build_model(
    calls: list[tuple[str, int | None]],
) -> Callable[[str, int | None], str]:
    def fake_build_model(spec: str, max_tokens: int | None = None) -> str:
        calls.append((spec, max_tokens))
        return spec

    return fake_build_model


def test_resolver_threads_max_tokens_to_build_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        "ghidra_deep_agent.agent_config.build_model", _recording_build_model(calls)
    )
    monkeypatch.delenv("MODEL_MAX_TOKENS", raising=False)
    resolve = make_model_resolver("anthropic:glm-5.2", 32768)

    resolve(None)  # falls back to the config default
    resolve("deepseek:deepseek-v4-pro", 10000)  # per-entry override wins
    resolve(None)  # cached: no new build_model call

    assert calls == [
        ("anthropic:glm-5.2", 32768),
        ("deepseek:deepseek-v4-pro", 10000),
    ]


def test_resolver_falls_back_to_env_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        "ghidra_deep_agent.agent_config.build_model", _recording_build_model(calls)
    )
    monkeypatch.setenv("MODEL_MAX_TOKENS", "12345")
    resolve = make_model_resolver("anthropic:glm-5.2")

    resolve(None)

    assert calls == [("anthropic:glm-5.2", 12345)]
