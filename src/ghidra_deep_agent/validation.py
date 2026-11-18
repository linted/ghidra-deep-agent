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
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import jsonschema
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Cap the number of reported errors so a badly-off call can't flood context.
_MAX_REPORTED_ERRORS = 10


def _format_path(error: jsonschema.ValidationError) -> str:
    """Render a JSON-pointer-ish path for a validation error, or '(root)'."""
    if not error.absolute_path:
        return "(root)"
    return ".".join(str(part) for part in error.absolute_path)


def _error_message(request: ToolCallRequest, payload: dict[str, Any]) -> ToolMessage:
    """Wrap a structured error payload as an error ``ToolMessage``."""
    return ToolMessage(
        content=json.dumps(payload),
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error",
    )


class ArgumentValidationMiddleware(AgentMiddleware):
    """Validate dict-schema tool arguments against their JSON schema."""

    def __init__(self) -> None:
        super().__init__()
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
        if validator is None:
            return None
        errors = sorted(
            validator.iter_errors(args), key=lambda e: list(e.absolute_path)
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

        return None

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


def create_argument_validation_middleware() -> ArgumentValidationMiddleware:
    """Build the argument-validation middleware (factory for ``cli.py``)."""
    return ArgumentValidationMiddleware()
