"""Tests for ArgumentValidationMiddleware (validation.py).

Covers the four behaviors the middleware is responsible for: rejecting calls
that don't match a dict JSON schema, blocking write actions in a restricted
context, requiring the provisional prefix on renames from an annotations-tier
agent, and never blocking on a malformed schema. Also pins the validator cache,
since that is what keeps `check_schema` off the per-call path.

Run:  uv run pytest tests/test_validation.py -v
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from ghidra_deep_agent.validation import (
    ArgumentValidationMiddleware,
    create_argument_validation_middleware,
)


class _FakeTool:
    def __init__(self, name: str, args_schema: Any) -> None:
        self.name = name
        self.args_schema = args_schema


class _FakeRequest:
    """Minimal stand-in for ToolCallRequest (only .tool and .tool_call are read)."""

    def __init__(self, tool: Any, args: Any) -> None:
        self.tool = tool
        self.tool_call = {
            "name": getattr(tool, "name", "?"),
            "id": "call-1",
            "args": args,
        }


RENAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "address": {"type": "string"},
        "new_name": {"type": "string"},
    },
    "required": ["address", "new_name"],
}

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"type": "string"}},
}


def _payload(message: ToolMessage) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(str(message.content))
    return parsed


def _check(mw: ArgumentValidationMiddleware, tool: Any, args: Any) -> Any:
    # _check is the whole decision; wrap_tool_call is a two-line adapter over it.
    request: Any = _FakeRequest(tool, args)
    return mw._check(request)


# ── schema validation ─────────────────────────────────────────────────────────


def test_valid_args_pass_through() -> None:
    mw = create_argument_validation_middleware()
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)
    assert _check(mw, tool, {"address": "0x1000", "new_name": "init"}) is None


def test_missing_required_arg_is_rejected() -> None:
    mw = create_argument_validation_middleware()
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)

    result = _check(mw, tool, {"address": "0x1000"})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = _payload(result)["validation_error"]
    assert payload["tool"] == "rename_symbol"
    assert "new_name" in json.dumps(payload["errors"])


def test_wrong_type_is_rejected_with_path() -> None:
    mw = create_argument_validation_middleware()
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)

    result = _check(mw, tool, {"address": 4096, "new_name": "init"})

    assert isinstance(result, ToolMessage)
    assert _payload(result)["validation_error"]["errors"][0]["path"] == "address"


def test_pydantic_schema_tools_are_left_alone() -> None:
    """Non-dict schemas are validated upstream by ToolNode; we must not re-check."""
    mw = create_argument_validation_middleware()

    class _Model:  # stands in for a pydantic args_schema
        pass

    assert _check(mw, _FakeTool("save_knowledge", _Model), {"anything": 1}) is None


def test_unknown_tool_passes_through() -> None:
    mw = create_argument_validation_middleware()
    assert _check(mw, None, {"a": 1}) is None


# ── read-only write-action blocking ───────────────────────────────────────────


def test_write_action_blocked_in_read_only_context() -> None:
    mw = create_argument_validation_middleware({"variables": frozenset({"rename"})})
    tool = _FakeTool("variables", ACTION_SCHEMA)

    result = _check(mw, tool, {"action": "rename"})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = _payload(result)["read_only_error"]
    assert payload["action"] == "rename"


def test_read_action_still_allowed_on_the_same_tool() -> None:
    mw = create_argument_validation_middleware({"variables": frozenset({"rename"})})
    tool = _FakeTool("variables", ACTION_SCHEMA)
    assert _check(mw, tool, {"action": "list"}) is None


def test_write_actions_not_enforced_without_the_map() -> None:
    mw = create_argument_validation_middleware()
    tool = _FakeTool("variables", ACTION_SCHEMA)
    assert _check(mw, tool, {"action": "rename"}) is None


# ── malformed schemas ─────────────────────────────────────────────────────────


def test_malformed_schema_does_not_block() -> None:
    mw = create_argument_validation_middleware()
    tool = _FakeTool("weird", {"type": 123})  # `type` must be a string/array
    assert _check(mw, tool, {"anything": True}) is None


def test_malformed_schema_still_blocks_write_actions() -> None:
    """A bad schema disables validation, not the read-only guard."""
    mw = create_argument_validation_middleware({"weird": frozenset({"delete"})})
    tool = _FakeTool("weird", {"type": 123})

    result = _check(mw, tool, {"action": "delete"})

    assert isinstance(result, ToolMessage)
    assert _payload(result)["read_only_error"]["action"] == "delete"


# ── provisional-rename prefix ─────────────────────────────────────────────────

BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"renames": {"type": "array"}},
}

VARIABLES_RENAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "function_name": {"type": "string"},
        "old_name": {"type": "string"},
        "new_name": {"type": "string"},
    },
}


def _prefix_mw() -> ArgumentValidationMiddleware:
    return create_argument_validation_middleware(rename_prefix="maybe_")


def test_prefixed_rename_is_allowed() -> None:
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)
    assert (
        _check(_prefix_mw(), tool, {"address": "0x1000", "new_name": "maybe_init"})
        is None
    )


def test_unprefixed_rename_is_rejected() -> None:
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)

    result = _check(_prefix_mw(), tool, {"address": "0x1000", "new_name": "init"})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = _payload(result)["rename_prefix_error"]
    assert payload["required_prefix"] == "maybe_"
    assert payload["rejected_names"] == ["init"]


def test_one_unprefixed_entry_rejects_the_whole_batch() -> None:
    """A batch is all-or-nothing: the tool applies it as a unit."""
    tool = _FakeTool("batch_rename", BATCH_SCHEMA)
    args = {
        "renames": [
            {"address": "0x1", "new_name": "maybe_parse"},
            {"address": "0x2", "new_name": "decrypt_payload"},
        ]
    }

    result = _check(_prefix_mw(), tool, args)

    assert isinstance(result, ToolMessage)
    assert _payload(result)["rename_prefix_error"]["rejected_names"] == [
        "decrypt_payload"
    ]


def test_fully_prefixed_batch_is_allowed() -> None:
    tool = _FakeTool("batch_rename", BATCH_SCHEMA)
    args = {
        "renames": [
            {"address": "0x1", "new_name": "maybe_parse"},
            {"address": "0x2", "new_name": "maybe_decrypt"},
        ]
    }
    assert _check(_prefix_mw(), tool, args) is None


def test_variables_rename_is_checked_but_other_actions_are_not() -> None:
    tool = _FakeTool("variables", VARIABLES_RENAME_SCHEMA)
    mw = _prefix_mw()

    rejected = _check(
        mw,
        tool,
        {"action": "rename", "old_name": "local_10", "new_name": "file_size"},
    )
    assert isinstance(rejected, ToolMessage)

    # A read action names nothing, so the rule must not fire on it.
    assert _check(mw, tool, {"action": "list", "function_name": "main"}) is None


def test_target_identifying_args_do_not_trigger_the_rule() -> None:
    """`function_name`/`old_name` name the existing symbol, not the new one."""
    tool = _FakeTool("variables", VARIABLES_RENAME_SCHEMA)
    args = {
        "action": "rename",
        "function_name": "process_packet",
        "old_name": "local_10",
        "new_name": "maybe_file_size",
    }
    assert _check(_prefix_mw(), tool, args) is None


def test_missing_new_name_argument_fails_closed() -> None:
    """If the new name can't be found, the call is rejected, not waved through."""
    tool = _FakeTool("rename_symbol", {"type": "object"})

    result = _check(_prefix_mw(), tool, {"address": "0x1000", "label": "init"})

    assert isinstance(result, ToolMessage)
    payload = _payload(result)["rename_prefix_error"]
    assert payload["rejected_names"] == []
    assert "cannot be checked" in payload["hint"]


def test_rename_prefix_not_enforced_without_the_rule() -> None:
    """A full-write agent may settle a name; the rule is off for it."""
    mw = create_argument_validation_middleware()
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)
    assert _check(mw, tool, {"address": "0x1000", "new_name": "init"}) is None


def test_non_rename_tools_are_untouched_by_the_rule() -> None:
    tool = _FakeTool("comments", ACTION_SCHEMA)
    assert _check(_prefix_mw(), tool, {"action": "set"}) is None


# ── validator cache ───────────────────────────────────────────────────────────


def test_validator_is_built_once_per_tool(monkeypatch: Any) -> None:
    """`check_schema` meta-validates the schema; it must not run per call."""
    import jsonschema

    builds = 0
    real_validator_for = jsonschema.validators.validator_for

    def counting_validator_for(schema: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal builds
        # check_schema() resolves meta-schemas through the same function, so
        # count only lookups for the tool's own schema object.
        if schema is RENAME_SCHEMA:
            builds += 1
        return real_validator_for(schema, *args, **kwargs)

    monkeypatch.setattr(jsonschema.validators, "validator_for", counting_validator_for)

    mw = create_argument_validation_middleware()
    tool = _FakeTool("rename_symbol", RENAME_SCHEMA)
    for _ in range(5):
        _check(mw, tool, {"address": "0x1000", "new_name": "init"})

    assert builds == 1


def test_cache_is_keyed_per_tool() -> None:
    mw = create_argument_validation_middleware()
    ok = _FakeTool("rename_symbol", RENAME_SCHEMA)
    other = _FakeTool("variables", ACTION_SCHEMA)

    assert _check(mw, ok, {"address": "0x1", "new_name": "n"}) is None
    assert _check(mw, other, {"action": "list"}) is None
    # A second tool must not inherit the first tool's validator.
    assert _check(mw, ok, {"address": "0x1"}) is not None
