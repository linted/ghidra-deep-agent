"""Which Ghidra program an agent works on.

GhidrAssistMCP serves every program open in Ghidra through one server. A
program is identified to it by its *project path* (``/folder/name``), which is
what ``list_binaries`` prints beneath each entry and what the ``program_name``
tool argument matches first. The bare name is what the knowledge base, read
cache and session registry are keyed by (see ``ProgramRef``).

This module is TUI-free: when several programs are open and the caller has no
way to ask (the HTTP server), :func:`resolve_program` fails with a message that
names them instead of opening a Textual screen.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ghidra_deep_agent.async_tasks import to_text

# A GhidrAssistMCP ``list_binaries`` entry, e.g. "1. libnloader.so [ACTIVE]".
# Indented detail lines (Project Path / Executable Path / Format / Language) that
# follow each entry do NOT match, so they are handled separately.
_NUMBERED_ENTRY = re.compile(r"^\s*\d+\.\s+(?P<name>.+?)\s*$")
_DETAIL_LINE = re.compile(r"^\s+(?P<key>[A-Za-z ]+?):\s*(?P<value>.*?)\s*$")
_ACTIVE_TAG = re.compile(r"\s*\[ACTIVE\]\s*$")
# A project listing line from ``open_program`` action=list: "  /dir/name  (Program)".
_PROJECT_FILE = re.compile(r"^\s+(?P<path>/\S.*?)\s+\((?P<type>[^)]*)\)\s*$")


@dataclass(frozen=True)
class ProgramInfo:
    """One program open in Ghidra, as ``list_binaries`` reports it."""

    name: str
    project_path: str
    active: bool = False
    format: str = ""
    language: str = ""


@dataclass(frozen=True)
class ProgramRef:
    """The program an agent instance is bound to.

    ``project_path`` is the pin (what every tool call names) and the agent's
    identity; ``name`` is the scope key for knowledge, cache and sessions, kept
    as the bare program name so existing Mongo data stays addressable.
    """

    name: str
    project_path: str

    @property
    def slug(self) -> str:
        """A filesystem-safe form of the project path, for per-agent output dirs."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.project_path.strip("/"))
        return slug.strip("._") or "program"


def parse_program_entries(text: str) -> list[ProgramInfo]:
    """Parse ``list_binaries`` output into structured entries.

    Accepts the numbered text format (with its indented detail lines) as well
    as the JSON shapes older servers produced. Entries without a
    ``Project Path`` line fall back to the bare name as their path, which the
    server resolves too (name match is its second rule).
    """
    entries = _parse_json_entries(text)
    if entries is not None:
        return entries

    entries = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        match = _NUMBERED_ENTRY.match(line)
        if match:
            if current is not None:
                entries.append(_finish_entry(current))
            raw_name = match.group("name")
            current = {
                "name": _ACTIVE_TAG.sub("", raw_name).strip(),
                "active": bool(_ACTIVE_TAG.search(raw_name)),
            }
            continue
        detail = _DETAIL_LINE.match(line)
        if detail and current is not None:
            current[detail.group("key").lower()] = detail.group("value")
    if current is not None:
        entries.append(_finish_entry(current))
    if entries:
        return entries

    # Fallback for other/plain-text formats.
    names = []
    for line in text.splitlines():
        line = line.strip()
        # Headers, the context prefix, and the server's "No programs currently
        # open" message are not program names.
        if not line or line.lower().startswith(
            ("open", "program", "#", "[context]", "no ")
        ):
            continue
        line = re.sub(r"^\d+\.\s*|^[-*]\s*", "", line)
        line = re.sub(r"\s*\(.*?\)\s*$", "", line).strip()
        if line and not line.startswith(("---", "Total:", "NOTE:", "Example:")):
            names.append(line)
    return [ProgramInfo(name, name) for name in names]


def _finish_entry(fields: dict[str, Any]) -> ProgramInfo:
    name = str(fields["name"])
    return ProgramInfo(
        name=name,
        project_path=str(fields.get("project path") or name),
        active=bool(fields.get("active")),
        format=str(fields.get("format", "")),
        language=str(fields.get("language", "")),
    )


def _parse_json_entries(text: str) -> list[ProgramInfo] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    items: Any
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("programs"), list):
        items = data["programs"]
    else:
        return None
    entries = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", item))
            entries.append(ProgramInfo(name, str(item.get("project_path") or name)))
        else:
            entries.append(ProgramInfo(str(item), str(item)))
    return entries


def parse_program_list(result: str) -> list[str]:
    """Parse list_binaries output into a list of program names."""
    return [entry.name for entry in parse_program_entries(result)]


def parse_project_files(text: str) -> list[str]:
    """Parse ``open_program`` action=list output into project paths."""
    return [m.group("path") for m in map(_PROJECT_FILE.match, text.splitlines()) if m]


def _find_tool(tools: list[Any], name: str) -> Any:
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise RuntimeError(
            f"Ghidra MCP does not expose '{name}'. "
            "Set BINARY_NAME or pass --binary-name."
        )
    return tool


async def _invoke_text(tools: list[Any], name: str, args: dict[str, Any]) -> str:
    tool = _find_tool(tools, name)
    try:
        result = await tool.ainvoke(args)
    except Exception as exc:
        raise RuntimeError(f"Ghidra MCP call '{name}' failed: {exc}") from exc
    return to_text(result)


async def list_open_programs(tools: list[Any]) -> list[ProgramInfo]:
    """The programs currently open in Ghidra."""
    text = await _invoke_text(tools, "list_binaries", {})
    return parse_program_entries(text)


async def list_project_files(tools: list[Any], folder: str = "/") -> list[str]:
    """Project paths of every program file in the Ghidra project."""
    text = await _invoke_text(
        tools, "open_program", {"action": "list", "folder": folder}
    )
    return parse_project_files(text)


async def open_project_program(
    tools: list[Any], path: str, *, auto_analyze: bool = False
) -> ProgramInfo:
    """Open a project program in CodeBrowser and return its entry.

    "Already open" counts as success: the server reports it, and the caller
    only cares that the program is available to pin. Returns the entry as
    ``list_binaries`` then reports it, so the caller gets the canonical path.
    """
    text = await _invoke_text(
        tools,
        "open_program",
        {"action": "open", "name": path, "analyze_after_open": auto_analyze},
    )
    lowered = text.lower()
    if "not found" in lowered or "failed to open" in lowered:
        raise RuntimeError(text.strip())
    entry = find_program(await list_open_programs(tools), path)
    if entry is None:
        raise RuntimeError(
            f"Ghidra reported opening {path!r} but list_binaries does not show it: "
            f"{text.strip()}"
        )
    return entry


def find_program(programs: list[ProgramInfo], key: str) -> ProgramInfo | None:
    """Match ``key`` against open programs by project path first, then by name.

    The same precedence GhidrAssistMCP uses for ``program_name`` (exact path,
    exact name, case-insensitive name), so what we pin is what it resolves.
    """
    for entry in programs:
        if entry.project_path == key:
            return entry
    for entry in programs:
        if entry.name == key:
            return entry
    for entry in programs:
        if entry.name.lower() == key.lower():
            return entry
    return None


# Given the open programs, pick one — or None to abort. The TUI passes a Textual
# picker; the server passes nothing and fails instead.
ProgramChooser = Callable[[list[ProgramInfo]], Awaitable[ProgramInfo | None]]


async def resolve_program(
    tools: list[Any],
    override: str | None,
    *,
    choose: ProgramChooser | None = None,
) -> ProgramRef:
    """Decide which open program this agent works on.

    ``override`` (``--binary-name`` / ``BINARY_NAME``) wins: matched against the
    open programs by path or name when possible. An override matching nothing
    stays the knowledge label (that is what the flag has always been) and pins
    the single open program if there is exactly one, else is used verbatim as
    the pin too (the server's name matching may still resolve it). Otherwise a
    single open program is picked automatically; several need ``choose``.
    """
    if override:
        try:
            programs = await list_open_programs(tools)
        except RuntimeError:
            programs = []
        entry = find_program(programs, override)
        if entry is not None:
            return ProgramRef(entry.name, entry.project_path)
        if len(programs) == 1:
            return ProgramRef(override, programs[0].project_path)
        return ProgramRef(override, override)

    programs = await list_open_programs(tools)
    if not programs:
        raise RuntimeError(
            "No open programs found in Ghidra. Open a binary and try again, "
            "or set BINARY_NAME / pass --binary-name."
        )
    if len(programs) == 1:
        selected: ProgramInfo | None = programs[0]
    elif choose is None:
        listing = ", ".join(p.project_path for p in programs)
        raise RuntimeError(
            f"{len(programs)} programs are open in Ghidra ({listing}); "
            "pass --binary-name or set BINARY_NAME to choose one."
        )
    else:
        selected = await choose(programs)
    if selected is None:
        raise RuntimeError("No program selected.")
    return ProgramRef(selected.name, selected.project_path)


async def resolve_binary_name(
    tools: list[Any], override: str | None, *, choose: ProgramChooser | None = None
) -> str:
    """Backwards-compatible name-only form of :func:`resolve_program`."""
    return (await resolve_program(tools, override, choose=choose)).name
