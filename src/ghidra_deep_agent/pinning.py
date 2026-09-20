"""Pin an agent's Ghidra tool calls to one program.

One GhidrAssistMCP server fronts every program open in Ghidra, and each tool
call targets whichever program ``program_name`` names — or, when it is absent
*or unknown*, the program in the active CodeBrowser window. Several agents
sharing that server therefore must name their program on every call, and must
notice when the server quietly fell back to the active window (it logs a
warning and carries on, which for a rename means editing the wrong binary).

:func:`pin_program` is a ``langchain_mcp_adapters`` tool-call interceptor that
does both: it overrides ``program_name`` with the pinned project path, and it
checks the ``[Context] Operating on: <name>`` line the server prefixes to
every result against that path. A mismatch raises :class:`PinMismatchError`,
which the error-wrapping interceptor above it turns into a failed tool call the
model can see, and reports through ``on_mismatch`` so the owner can stop
scheduling work on that agent.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    ToolCallInterceptor,
)

from ghidra_deep_agent.async_tasks import to_text
from ghidra_deep_agent.program_resolver import ProgramRef

# Tools whose target is not a program. Injecting `program_name` into these is
# at best ignored, and pinning `list_binaries` would hide the other programs a
# caller needs to see.
UNPINNED_TOOLS: frozenset[str] = frozenset(
    {
        "list_binaries",
        "get_task_status",
        "cancel_task",
        "list_tasks",
        "open_program",
        "close_program",
        "import_file",
        "project_files",
    }
)

_CONTEXT_RE = re.compile(
    r"\[Context\]\s*Operating on:\s*(?P<target>.+?)\s*(?:\||$)", re.MULTILINE
)


class PinMismatchError(RuntimeError):
    """The server operated on a different program than the one pinned."""


def parse_operating_on(text: str) -> str | None:
    """The program a result's ``[Context]`` line says it operated on, if any."""
    match = _CONTEXT_RE.search(text)
    return match.group("target").strip() if match else None


def matches_program(display: str, program_path: str) -> bool:
    """Whether a context-line display name refers to ``program_path``.

    The server prints the bare program name (or the project path when the name
    is unknown), so both spellings are accepted.
    """
    return display == program_path or display == program_path.rsplit("/", 1)[-1]


def result_text(result: Any) -> str:
    """Flatten an interceptor result (CallToolResult / ToolMessage / str) to text."""
    content = getattr(result, "content", result)
    return to_text(content)


OnMismatch = Callable[[str, str], None]
Handler = Callable[[MCPToolCallRequest], Awaitable[Any]]


def pin_program(
    program_path: str,
    *,
    verify: bool = True,
    on_mismatch: OnMismatch | None = None,
) -> ToolCallInterceptor:
    """Build the interceptor that binds every program-targeting call to one program.

    ``program_name`` is always overridden, even when the model supplied one: an
    agent instance is bound to a single program by construction, and a
    model-chosen name is exactly the cross-program leak the pin exists to stop.
    """

    async def interceptor(request: MCPToolCallRequest, handler: Handler) -> Any:
        if request.name in UNPINNED_TOOLS:
            return await handler(request)
        pinned = request.override(args={**request.args, "program_name": program_path})
        result = await handler(pinned)
        if verify:
            target = parse_operating_on(result_text(result))
            if target is not None and not matches_program(target, program_path):
                if on_mismatch is not None:
                    on_mismatch(request.name, target)
                raise PinMismatchError(
                    f"Tool '{request.name}' is pinned to {program_path} but Ghidra "
                    f"operated on {target!r}; is the program still open? "
                    "Refusing to continue against the wrong binary."
                )
        return result

    # The adapter's Protocol spells out its result union; this interceptor is
    # transparent to it (it returns whatever the handler returned).
    return cast(ToolCallInterceptor, interceptor)


async def verify_pinned(tools: list[Any], program: ProgramRef) -> None:
    """Probe once through the pinned tools so a bad pin fails at creation.

    Uses ``get_binary_info`` (read-only, cheap). The interceptor raises on a
    mismatch; a result with no context line (an older server) passes, since
    there is nothing to check against.
    """
    probe = next((t for t in tools if t.name == "get_binary_info"), None)
    if probe is None:
        return
    result = await probe.ainvoke({})
    # The error-wrapping interceptor turns the mismatch into a string result,
    # so look for it there as well as trusting a raised exception.
    text = result_text(result)
    if "is pinned to" in text and "operated on" in text:
        raise PinMismatchError(text.strip())
    target = parse_operating_on(text)
    if target is not None and not matches_program(target, program.project_path):
        raise PinMismatchError(
            f"get_binary_info for {program.project_path} answered for {target!r}"
        )
