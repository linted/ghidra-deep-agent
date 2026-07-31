"""Client-side argument-validation middleware.

MCP tools (e.g. the Ghidra tools) are built by ``langchain_mcp_adapters`` with a
raw JSON-schema ``dict`` as ``args_schema`` rather than a pydantic model. For
dict schemas, ``BaseTool._parse_input`` performs *no* validation, so malformed
arguments are shipped to the server and only fail there — a wasted round-trip
and a noisy error.

This middleware validates dict-schema tool arguments against their JSON schema
*before* execution and, on failure, short-circuits with a compact structured
``{"validation_error": ...}`` ``ToolMessage`` the model can self-correct from.

Pydantic-schema tools (knowledge tools, deepagents built-ins) are left untouched:
``ToolNode`` already validates them and returns a clean error, so re-validating
here would only risk diverging from each tool's own coercion/defaults.

It also enforces **write-action blocking**: GhidrAssistMCP consolidates
read and write operations into single ``action``-based tools (e.g. ``variables``
does both ``action:list`` and ``action:rename``), so a restricted context can't
simply drop the tool. When built with a ``write_actions`` map, this middleware
rejects a call whose ``action`` argument names a blocked write on that tool,
using the same short-circuit path as a schema failure.

Finally it enforces the **provisional-rename prefix**: an ``annotations``-tier
sub-agent may persist renames, but only provisional ones. Rather than trusting
the prompt, this middleware rejects any rename whose new name lacks the required
prefix, so an investigator physically cannot commit a name that reads as settled
(nor strip another agent's prefix — promotion belongs to a full-write agent).
"""

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Any

import jsonschema
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Cap the number of reported errors so a badly-off call can't flood context.
_MAX_REPORTED_ERRORS = 10

# Tools that assign a new symbol name. ``variables`` is included conditionally —
# only for ``action: rename`` (see ``_RENAME_ACTIONS``) — since its other actions
# don't name anything.
_RENAME_TOOLS = frozenset({"rename_symbol", "batch_rename", "variables"})
_RENAME_ACTIONS: dict[str, frozenset[str]] = {"variables": frozenset({"rename"})}
# Argument keys that carry the NEW name in a rename call, matched after
# lowercasing and stripping non-alphanumerics. Deliberately narrow: keys naming
# the *existing* target (``function_name``, ``old_name``, ``variable``) must not
# match, or every call would be rejected for the wrong reason. Nested structures
# are searched too, so a `batch_rename` list of per-symbol entries is covered.
_NEW_NAME_KEYS = frozenset({"newname", "newlabel", "newsymbolname", "rename", "to"})
# How deep to search nested rename arguments before giving up (and failing
# closed). Well past any real batch payload's shape.
_MAX_ARG_DEPTH = 6


def _format_path(error: jsonschema.ValidationError) -> str:
    """Render a JSON-pointer-ish path for a validation error, or '(root)'."""
    if not error.absolute_path:
        return "(root)"
    return ".".join(str(part) for part in error.absolute_path)


def _normalize_key(key: str) -> str:
    """Lowercase a key and drop non-alphanumerics, so ``new_name`` == ``newName``."""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _new_names(value: Any, depth: int = 0) -> Iterator[str]:
    """Yield every new-name string found in a rename call's arguments.

    Walks nested dicts/lists so a batch payload (a list of per-symbol entries) is
    covered without hardcoding its shape.
    """
    if depth > _MAX_ARG_DEPTH:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _normalize_key(key) in _NEW_NAME_KEYS:
                if isinstance(item, str):
                    yield item
                    continue
            yield from _new_names(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _new_names(item, depth + 1)


def _error_message(request: ToolCallRequest, payload: dict[str, Any]) -> ToolMessage:
    """Wrap a structured error payload as an error ``ToolMessage``."""
    return ToolMessage(
        content=json.dumps(payload),
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error",
    )


class ArgumentValidationMiddleware(AgentMiddleware):
    """Validate dict-schema tool arguments (and, optionally, restrict writes)."""

    def __init__(
        self,
        write_actions: Mapping[str, frozenset[str]] | None = None,
        *,
        rename_prefix: str | None = None,
    ) -> None:
        super().__init__()
        # Tool name -> set of `action` values this context may not use. Populated
        # only for restricted tiers; empty means "allow everything the schema does".
        self._write_actions = dict(write_actions or {})
        # When set, every new symbol name must start with this prefix.
        self._rename_prefix = rename_prefix
        # Tool name -> its compiled validator, or None when the schema itself is
        # malformed. Cached because building one meta-schema-validates the
        # schema (`check_schema`), and a tool's args_schema never changes — so
        # every call after the first was paying for the same result.
        self._validators: dict[str, Any | None] = {}

    def _validator_for(self, tool_name: str, schema: dict[str, Any]) -> Any | None:
        """Compiled validator for a tool's schema; ``None`` if it is malformed."""
        if tool_name in self._validators:
            return self._validators[tool_name]
        validator: Any | None
        try:
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
        except jsonschema.exceptions.SchemaError:
            validator = None
        self._validators[tool_name] = validator
        return validator

    def _check(self, request: ToolCallRequest) -> ToolMessage | None:
        """Return a structured error ToolMessage if the call should be rejected.

        Returns ``None`` when the call is allowed, the tool is unknown, the
        schema is not a JSON-schema dict, or the schema itself is malformed — in
        every such case the caller proceeds to execute the tool normally.
        """
        tool = request.tool
        if tool is None:
            return None

        schema = getattr(tool, "args_schema", None)
        args = request.tool_call.get("args", {})
        # Only dict (JSON-schema) tools are unvalidated by the framework; pydantic
        # schemas are already validated upstream.
        if not isinstance(schema, dict) or not isinstance(args, dict):
            return None

        validator = self._validator_for(tool.name, schema)
        # A malformed schema is never grounds to block — let the server decide.
        # The read-only write-action check below still applies.
        errors = (
            []
            if validator is None
            else sorted(
                validator.iter_errors(args), key=lambda e: list(e.absolute_path)
            )
        )

        if errors:
            return _error_message(
                request,
                {
                    "validation_error": {
                        "tool": tool.name,
                        "errors": [
                            {"path": _format_path(err), "message": err.message}
                            for err in errors[:_MAX_REPORTED_ERRORS]
                        ],
                        "hint": (
                            "Arguments did not match the tool's schema. "
                            "Fix them and call the tool again."
                        ),
                    }
                },
            )

        blocked = self._write_actions.get(tool.name)
        if blocked:
            action = args.get("action")
            if isinstance(action, str) and action in blocked:
                return _error_message(
                    request,
                    {
                        "read_only_error": {
                            "tool": tool.name,
                            "action": action,
                            "hint": (
                                f"'{tool.name}' with action '{action}' mutates the "
                                "program and is out of scope for you. Use a read "
                                "action (e.g. list/get) only, and record the change "
                                "you wanted as a `pending-change` bookmark plus a "
                                "PENDING: line in your summary instead of applying "
                                "it. Do not retry this call."
                            ),
                        }
                    },
                )

        return self._check_rename_prefix(request, tool.name, args)

    def _check_rename_prefix(
        self, request: ToolCallRequest, tool_name: str, args: dict[str, Any]
    ) -> ToolMessage | None:
        """Reject a rename whose new name lacks the required prefix.

        Fails closed: if the call renames something but no new-name argument can
        be found, it is rejected rather than let through unchecked, so a schema
        change on the server can't silently disable the rule.
        """
        prefix = self._rename_prefix
        if prefix is None or tool_name not in _RENAME_TOOLS:
            return None
        gated_actions = _RENAME_ACTIONS.get(tool_name)
        if gated_actions is not None and args.get("action") not in gated_actions:
            return None

        names = list(_new_names(args))
        offenders = [name for name in names if not name.startswith(prefix)]
        if not names:
            offenders = []  # nothing found — reported as the fail-closed case below
        if names and not offenders:
            return None

        return _error_message(
            request,
            {
                "rename_prefix_error": {
                    "tool": tool_name,
                    "required_prefix": prefix,
                    "rejected_names": offenders,
                    "hint": (
                        f"Renames from this agent must start with '{prefix}' — the "
                        "name is a provisional conclusion, not a settled one. "
                        f"Re-call with every new name prefixed (e.g. '{prefix}"
                        "parse_header'), or, if you are confident enough that the "
                        "prefix would be wrong, leave the symbol alone and record "
                        "the rename as a `pending-change` bookmark plus a PENDING: "
                        "line for a full-write agent to apply."
                    )
                    if names
                    else (
                        "Could not find the new-name argument in this call, so it "
                        f"cannot be checked against the required '{prefix}' prefix "
                        "and is rejected. Re-call using the tool schema's "
                        "documented new-name field."
                    ),
                }
            },
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        error = self._check(request)
        return error if error is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        error = self._check(request)
        return error if error is not None else await handler(request)


def create_argument_validation_middleware(
    write_actions: Mapping[str, frozenset[str]] | None = None,
    *,
    rename_prefix: str | None = None,
) -> ArgumentValidationMiddleware:
    """Build the argument-validation middleware (factory for ``cli.py``).

    Pass ``write_actions`` (tool name -> blocked ``action`` values) to also
    reject write actions on consolidated read/write tools, and ``rename_prefix``
    to require that prefix on every new symbol name. Both come from the agent's
    ``WritePolicy`` (see subagents.py).
    """
    return ArgumentValidationMiddleware(write_actions, rename_prefix=rename_prefix)
