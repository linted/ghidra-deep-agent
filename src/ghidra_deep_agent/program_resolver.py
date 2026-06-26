import json
import re
from typing import Any


def parse_project_files(result: str) -> list[str]:
    """Parse list_project_files output into a list of program names.

    The headless engine returns ``{"files": [{"name": ..., "type": "Program"},
    ...], "folders": [...]}`` (mirroring ``/server/repository/files``). Only
    ``Program`` entries are returned; folders and other domain-object types are
    skipped. Falls back to :func:`parse_program_list` for unexpected shapes.
    """
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return parse_program_list(result)

    if isinstance(data, dict) and isinstance(data.get("files"), list):
        names: list[str] = []
        for item in data["files"]:
            if not isinstance(item, dict):
                names.append(str(item))
                continue
            # Keep Program files; skip folders/other types. Tolerate a missing
            # "type" (treat as a program rather than silently dropping it).
            if item.get("type", "Program") != "Program":
                continue
            name = item.get("name")
            if name:
                names.append(str(name))
        return names

    return parse_program_list(result)


def parse_program_list(result: str) -> list[str]:
    """Parse list_open_programs output into a list of program names."""
    try:
        data = json.loads(result)
        if isinstance(data, list):
            return [
                item.get("name", str(item)) if isinstance(item, dict) else str(item)
                for item in data
            ]
        if isinstance(data, dict):
            programs = data.get("programs", [])
            if isinstance(programs, list):
                return [
                    item.get("name", str(item)) if isinstance(item, dict) else str(item)
                    for item in programs
                ]
    except (json.JSONDecodeError, AttributeError):
        pass

    names = []
    for line in result.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("open", "program", "#")):
            continue
        line = re.sub(r"^\d+\.\s*|^[-*]\s*", "", line)
        line = re.sub(r"\s*\(.*?\)\s*$", "", line).strip()
        if line:
            names.append(line)
    return names


def _find_tool(tools: list[Any], name: str) -> Any:
    """Return the MCP tool named ``name`` or raise a clear RuntimeError."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise RuntimeError(f"Ghidra MCP does not expose '{name}'.")
    return tool


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result to text.

    langchain_mcp_adapters may return a content-block list instead of a plain
    string; dig out the first text block if so.
    """
    if isinstance(result, list):
        raw = next(
            (
                (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                for item in result
                if (isinstance(item, dict) and item.get("type") == "text")
                or (hasattr(item, "type") and item.type == "text")
            ),
            str(result),
        )
        return raw if isinstance(raw, str) else str(result)
    return str(result)


async def _invoke_text(tools: list[Any], name: str, args: dict[str, Any]) -> str:
    """Invoke the MCP tool ``name`` with ``args`` and return its text result."""
    tool = _find_tool(tools, name)
    result = await tool.ainvoke(args)
    return _result_text(result)


async def list_open_programs(tools: list[Any]) -> list[str]:
    """Query Ghidra for the names of all currently open programs.

    Framework-agnostic: callable from both the TUI and the web UI. Raises
    RuntimeError if the MCP server cannot provide a usable program list.
    """
    if not any(t.name == "list_open_programs" for t in tools):
        raise RuntimeError(
            "Ghidra MCP does not expose 'list_open_programs'. "
            "Set BINARY_NAME or pass --binary-name."
        )
    try:
        text = await _invoke_text(tools, "list_open_programs", {})
        return parse_program_list(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to list open programs: {exc}") from exc


async def list_project_programs(tools: list[Any]) -> list[str]:
    """List the programs in the engine's open (shared, server-bound) project.

    Drives the web UI's binary picker: with the engine mounting the shared repo
    as a server-bound project, ``list_project_files`` enumerates the uploaded
    programs available to open. Raises RuntimeError if the listing fails.
    """
    # ``folder`` is a required query param on the engine's list_project_files
    # endpoint; "/" is the project root that holds the shared repo's programs.
    try:
        text = await _invoke_text(tools, "list_project_files", {"folder": "/"})
    except Exception as exc:
        raise RuntimeError(f"Failed to list project files: {exc}") from exc
    # The engine reports its GUI-only gate as an error payload rather than
    # raising; surface that as a RuntimeError so the route returns 502.
    if _is_error_text(text):
        raise RuntimeError(f"Failed to list project files: {text}")
    return parse_project_files(text)


async def is_program_open(tools: list[Any], name: str) -> bool:
    """Return True if a program named ``name`` is already open in the engine."""
    return name in await list_open_programs(tools)


async def open_program(tools: list[Any], path: str) -> None:
    """Open the program at ``path`` into the engine (no auto-analysis).

    ``path`` is repo-relative, e.g. ``/AB5682B_loader.bin``. Raises RuntimeError
    if the engine reports an error.
    """
    try:
        text = await _invoke_text(
            tools, "open_program", {"path": path, "auto_analyze": False}
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to open program '{path}': {exc}") from exc
    if _is_error_text(text):
        raise RuntimeError(f"Failed to open program '{path}': {text}")


def _is_error_text(text: str) -> bool:
    """True if a tool result is a JSON object carrying an ``error`` field."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and bool(data.get("error"))


async def resolve_binary_name(tools: list[Any], override: str | None) -> str:
    if override:
        return override

    programs = await list_open_programs(tools)

    if not programs:
        raise RuntimeError(
            "No open programs found in Ghidra. Open a binary and try again, "
            "or set BINARY_NAME / pass --binary-name."
        )

    if len(programs) == 1:
        return programs[0]

    # Imported lazily so the TUI dependency stays out of the listing path.
    from ghidra_deep_agent.tui import ProgramSelectApp

    selected = await ProgramSelectApp(programs).run_async()
    if not selected:
        raise RuntimeError("No program selected.")
    return selected
