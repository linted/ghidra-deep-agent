"""
Unit tests for the config-driven sub-agent builder, focused on the write-policy
tiers: each tier must drop the right write-only tools from the tool set, hand the
validation middleware the right blocked actions and rename prefix, and append its
generated write-scope text — while ``policy_override`` beats whatever the config
asked for (that override is what makes plan/ask mode read-only by construction).

Run:  uv run pytest tests/test_subagents.py -v
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from deepagents import SubAgent
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

from ghidra_deep_agent.report_guard import SubagentReportGuardMiddleware
from ghidra_deep_agent.subagents import (
    _REPORT_PROTOCOL,
    ALL_WRITE_ACTIONS,
    MUTATION_TOOLS,
    PROVISIONAL_RENAME_PREFIX,
    READ_ONLY_WRITE_POLICY,
    build_subagents,
    load_agent_config,
)
from ghidra_deep_agent.validation import ArgumentValidationMiddleware

# A mix of write-only tools (dropped for restricted agents) and read tools (kept).
_TOOL_NAMES = [
    "rename_symbol",
    "get_code",
    "xrefs",
    "save_knowledge",
    "variables",
    "recover_prototypes",
]


def _fake_tools() -> Sequence[BaseTool]:
    """Minimal stand-ins: build_subagents only ever reads ``.name``."""
    tools = [SimpleNamespace(name=name) for name in _TOOL_NAMES]
    return cast("Sequence[BaseTool]", tools)


def _resolver(spec: str | None) -> FakeListChatModel:
    """Stub model resolver returning a real chat model instance.

    ``build_subagents`` stores the result in the spec and hands it to the tuned
    summarization middleware, which needs an actual ``BaseChatModel``.
    """
    return FakeListChatModel(responses=["ok"])


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "subagents.toml"
    path.write_text(
        '[main]\ntools = ["get_code"]\n\n' + body,
        encoding="utf-8",
    )
    return path


def _tool_names(spec: SubAgent) -> set[str]:
    # spec["tools"] is a union type; our stubs expose ``.name`` via getattr.
    return {getattr(tool, "name") for tool in spec["tools"]}  # noqa: B009


def _validation_mw(spec: SubAgent) -> ArgumentValidationMiddleware:
    mws = [m for m in spec["middleware"] if isinstance(m, ArgumentValidationMiddleware)]
    assert len(mws) == 1, "each sub-agent gets exactly one validation middleware"
    return mws[0]


def _build(
    tmp_path: Path, body: str, *, policy_override: str | None = None
) -> dict[str, SubAgent]:
    config = load_agent_config(_write_config(tmp_path, body))
    specs = build_subagents(
        _fake_tools(),
        config,
        resolve_model=_resolver,
        backend=StateBackend(),
        policy_override=policy_override,
    )
    return {spec["name"]: spec for spec in specs}


def _entry(name: str, *, policy_line: str = "") -> str:
    return (
        f'[[subagents]]\nname = "{name}"\n'
        'description = "d"\n'
        'system_prompt = "base prompt"\n'
        'tools = "*"\n' + policy_line
    )


# --- tier: none (read-only) ----------------------------------------------------


def test_read_only_subagent_drops_write_tools_and_blocks_write_actions(
    tmp_path: Path,
) -> None:
    specs = _build(tmp_path, _entry("research", policy_line="read_only = true\n"))
    spec = specs["research"]

    tool_names = _tool_names(spec)
    # A blocked write-only tool is gone; read tools remain.
    assert "rename_symbol" not in tool_names
    assert "rename_symbol" in MUTATION_TOOLS  # guards against list drift
    assert {"get_code", "xrefs"} <= tool_names

    # The validation middleware rejects every write action on dual read/write tools.
    mw = _validation_mw(spec)
    assert mw._write_actions == ALL_WRITE_ACTIONS
    # No prefix rule: it cannot rename at all, so there is nothing to prefix.
    assert mw._rename_prefix is None


def test_write_policy_none_is_equivalent_to_read_only(tmp_path: Path) -> None:
    by_flag = _build(tmp_path, _entry("a", policy_line="read_only = true\n"))["a"]
    by_name = _build(tmp_path, _entry("a", policy_line='write_policy = "none"\n'))["a"]
    assert _tool_names(by_flag) == _tool_names(by_name)
    assert (
        _validation_mw(by_flag)._write_actions == _validation_mw(by_name)._write_actions
    )


# --- tier: annotations ---------------------------------------------------------


def test_annotations_tier_keeps_cheap_writes_and_drops_heavy_ones(
    tmp_path: Path,
) -> None:
    spec = _build(
        tmp_path, _entry("research", policy_line='write_policy = "annotations"\n')
    )["research"]

    tool_names = _tool_names(spec)
    # Renames and KB writes survive; heavyweight mutation tools do not.
    assert {"rename_symbol", "save_knowledge"} <= tool_names
    assert "recover_prototypes" not in tool_names

    blocked = _validation_mw(spec)._write_actions
    # `variables` keeps rename, loses type/signature surgery.
    assert "rename" not in blocked["variables"]
    assert {"retype", "set_prototype"} <= blocked["variables"]
    # Comments and bookmarks are fully allowed, so they drop out of the map.
    assert "comments" not in blocked
    assert "bookmarks" not in blocked
    # Types and structs stay entirely blocked.
    assert blocked["types"] == ALL_WRITE_ACTIONS["types"]
    assert blocked["struct"] == ALL_WRITE_ACTIONS["struct"]


def test_annotations_tier_requires_the_provisional_rename_prefix(
    tmp_path: Path,
) -> None:
    spec = _build(tmp_path, _entry("a", policy_line='write_policy = "annotations"\n'))[
        "a"
    ]
    assert _validation_mw(spec)._rename_prefix == PROVISIONAL_RENAME_PREFIX


# --- tier: full ----------------------------------------------------------------


def test_normal_subagent_keeps_write_tools_and_allows_actions(tmp_path: Path) -> None:
    spec = _build(tmp_path, _entry("analyst"))["analyst"]

    assert "rename_symbol" in _tool_names(spec)
    mw = _validation_mw(spec)
    # No action blocking and no prefix rule: a full-write agent may settle names.
    assert mw._write_actions == {}
    assert mw._rename_prefix is None


# --- generated prompt / description text ---------------------------------------


def test_policy_text_is_appended_and_differs_per_tier(tmp_path: Path) -> None:
    full = _build(tmp_path, _entry("a"))["a"]
    annotations = _build(
        tmp_path, _entry("a", policy_line='write_policy = "annotations"\n')
    )["a"]
    none = _build(tmp_path, _entry("a", policy_line="read_only = true\n"))["a"]

    # The config's own prompt is preserved in every tier.
    for spec in (full, annotations, none):
        assert "base prompt" in spec["system_prompt"]

    # `full` adds no scope text (only the tier-invariant report protocol); the
    # restricted tiers each add their own scope text.
    assert full["system_prompt"] == "base prompt\n\n" + _REPORT_PROTOCOL + "\n"
    assert full["description"] == "d"
    assert PROVISIONAL_RENAME_PREFIX in annotations["system_prompt"]
    assert "pending-change" in annotations["system_prompt"]
    assert "STRICTLY READ-ONLY" in none["system_prompt"]
    assert annotations["system_prompt"] != none["system_prompt"]
    assert annotations["description"] != none["description"] != "d"


def test_report_protocol_and_guard_apply_to_every_tier(tmp_path: Path) -> None:
    """The final-report protocol and its middleware backstop are tier-invariant.

    Both exist because deepagents forwards only the last non-empty AIMessage's
    text as the sub-agent's report; see report_guard.py.
    """
    body = (
        _entry("full")
        + _entry("annotations", policy_line='write_policy = "annotations"\n')
        + _entry("none", policy_line="read_only = true\n")
    )
    for spec in _build(tmp_path, body).values():
        assert "## Final report" in spec["system_prompt"]
        # The protocol comes after the tier's scope section, closing the prompt.
        assert spec["system_prompt"].rstrip().endswith(_REPORT_PROTOCOL)
        guards = [
            m
            for m in spec["middleware"]
            if isinstance(m, SubagentReportGuardMiddleware)
        ]
        assert len(guards) == 1, spec["name"]


# --- policy_override (what makes plan/ask mode read-only) ----------------------


def test_policy_override_forces_lockdown_regardless_of_config(tmp_path: Path) -> None:
    body = _entry("research", policy_line='write_policy = "annotations"\n') + _entry(
        "analyst"
    )
    specs = _build(tmp_path, body, policy_override=READ_ONLY_WRITE_POLICY)

    for name in ("research", "analyst"):
        spec = specs[name]
        assert not (_tool_names(spec) & MUTATION_TOOLS), (
            f"{name} kept a mutation tool despite the read-only override"
        )
        assert _validation_mw(spec)._write_actions == ALL_WRITE_ACTIONS
        assert "STRICTLY READ-ONLY" in spec["system_prompt"]


def test_unknown_policy_override_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_override"):
        _build(tmp_path, _entry("a"), policy_override="nonsense")


# --- config parsing ------------------------------------------------------------


def test_write_policy_defaults_to_full(tmp_path: Path) -> None:
    config = load_agent_config(
        _write_config(
            tmp_path,
            '[[subagents]]\nname = "a"\ndescription = "d"\ntools = ["get_code"]\n',
        )
    )
    assert config.subagents[0].write_policy == "full"


def test_unknown_write_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown write_policy"):
        load_agent_config(
            _write_config(tmp_path, _entry("a", policy_line='write_policy = "sorta"\n'))
        )


def test_read_only_conflicting_with_write_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        load_agent_config(
            _write_config(
                tmp_path,
                _entry(
                    "a", policy_line='read_only = true\nwrite_policy = "annotations"\n'
                ),
            )
        )


def test_read_only_agreeing_with_write_policy_is_accepted(tmp_path: Path) -> None:
    config = load_agent_config(
        _write_config(
            tmp_path,
            _entry("a", policy_line='read_only = true\nwrite_policy = "none"\n'),
        )
    )
    assert config.subagents[0].write_policy == "none"


def test_read_only_rejects_non_bool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read_only"):
        load_agent_config(
            _write_config(tmp_path, _entry("a", policy_line='read_only = "yes"\n'))
        )
