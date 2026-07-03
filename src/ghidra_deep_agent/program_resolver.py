import json
import re
from typing import Any

from ghidra_deep_agent.tui import ProgramSelectApp

# A GhidrAssistMCP ``list_binaries`` entry, e.g. "1. libnloader.so [ACTIVE]".
# Indented detail lines (Project Path / Executable Path / Format / Language) that
# follow each entry do NOT match, so they are ignored.
_NUMBERED_ENTRY = re.compile(r"^\s*\d+\.\s+(?P<name>.+?)\s*$")


def parse_program_list(result: str) -> list[str]:
    """Parse list_binaries output into a list of program names."""
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

    # GhidrAssistMCP renders a numbered list, one program per top-level entry,
    # with indented detail lines beneath each. Prefer parsing those entries.
    numbered = []
    for line in result.splitlines():
        match = _NUMBERED_ENTRY.match(line)
        if not match:
            continue
        name = match.group("name")
        # Strip a trailing status tag like "[ACTIVE]".
        name = re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()
        if name:
            numbered.append(name)
    if numbered:
        return numbered

    # Fallback for other/plain-text formats.
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


async def resolve_binary_name(tools: list[Any], override: str | None) -> str:
    if override:
        return override

    list_tool = next((t for t in tools if t.name == "list_binaries"), None)
    if list_tool is None:
        raise RuntimeError(
            "Ghidra MCP does not expose 'list_binaries'. "
            "Set BINARY_NAME or pass --binary-name."
        )

    try:
        result = await list_tool.ainvoke({})
        # langchain_mcp_adapters may return a content-block list instead of a
        # plain string; dig out the first text block if so.
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
            text: str = raw if isinstance(raw, str) else str(result)
        else:
            text = str(result)
        programs = parse_program_list(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to list open programs: {exc}") from exc

    if not programs:
        raise RuntimeError(
            "No open programs found in Ghidra. Open a binary and try again, "
            "or set BINARY_NAME / pass --binary-name."
        )

    if len(programs) == 1:
        return programs[0]

    selected = await ProgramSelectApp(programs).run_async()
    if not selected:
        raise RuntimeError("No program selected.")
    return selected
