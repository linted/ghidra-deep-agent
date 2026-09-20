"""Program discovery and selection against GhidrAssistMCP's list formats."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ghidra_deep_agent.program_resolver import (
    ProgramInfo,
    ProgramRef,
    find_program,
    list_open_programs,
    list_project_files,
    open_project_program,
    parse_program_entries,
    parse_program_list,
    parse_project_files,
    resolve_binary_name,
    resolve_program,
)

# Captured from a live GhidrAssistMCP (one program open). The `[Context]`
# prefix is added to every result by the server.
ONE_OPEN = (
    "[Context] Operating on: libdxp.so\n\n"
    "Open Programs in Ghidra:\n\n"
    "1. libdxp.so [ACTIVE]\n"
    "   Project Path: /libdxp.so\n"
    "   Executable Path: /Users/linted/github/love/apk/so/libdxp.so\n"
    "   Format: Executable and Linking Format (ELF)\n"
    "   Language: AARCH64:LE:64:v8A\n"
    "\n---\nTotal: 1 program(s) open\n"
)

# The multi-program shape, per ListProgramsTool.java (same entry format plus
# the NOTE tail that suggests using the Project Path as `program_name`).
TWO_OPEN = (
    "[Context] Operating on: app.exe | Active window: app.exe\n\n"
    "Open Programs in Ghidra:\n\n"
    "1. app.exe [ACTIVE]\n"
    "   Project Path: /v1/app.exe\n"
    "   Executable Path: /tmp/v1/app.exe\n"
    "   Format: Portable Executable (PE)\n"
    "   Language: x86:LE:64:default\n"
    "\n"
    "2. app.exe\n"
    "   Project Path: /v2/app.exe\n"
    "   Executable Path: /tmp/v2/app.exe\n"
    "   Format: Portable Executable (PE)\n"
    "   Language: x86:LE:64:default\n"
    "\n---\nTotal: 2 program(s) open\n"
    "\nNOTE: Multiple programs are open. To target a specific program, "
    "use the listed Project Path as the 'program_name' parameter in tool calls.\n"
    'Example: {"program_name": "/v1/app.exe", ...}'
)

PROJECT_LIST = (
    "[Context] Operating on: libdxp.so\n\n"
    "Programs in project:\n\n"
    "  /libdxp.so  (Program)\n"
    "  /firmware/boot.bin  (Program)\n"
    "\nTotal: 2 file(s)\n"
    '\nUse {"action": "open", "name": "<pathname>"} to open one in CodeBrowser.'
)


class FakeTool:
    def __init__(self, name: str, replies: list[Any] | Any) -> None:
        self.name = name
        self._replies = replies if isinstance(replies, list) else [replies]
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        reply = self._replies[0] if len(self._replies) == 1 else self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_parse_entries_one_program() -> None:
    assert parse_program_entries(ONE_OPEN) == [
        ProgramInfo(
            "libdxp.so",
            "/libdxp.so",
            active=True,
            format="Executable and Linking Format (ELF)",
            language="AARCH64:LE:64:v8A",
        )
    ]
    assert parse_program_list(ONE_OPEN) == ["libdxp.so"]


def test_parse_entries_two_programs_same_name() -> None:
    entries = parse_program_entries(TWO_OPEN)
    assert [(e.name, e.project_path, e.active) for e in entries] == [
        ("app.exe", "/v1/app.exe", True),
        ("app.exe", "/v2/app.exe", False),
    ]


def test_parse_entries_json_and_plain_fallbacks() -> None:
    assert parse_program_entries('["a.bin", {"name": "b.bin"}]') == [
        ProgramInfo("a.bin", "a.bin"),
        ProgramInfo("b.bin", "b.bin"),
    ]
    assert parse_program_entries(
        '{"programs": [{"name": "c", "project_path": "/x/c"}]}'
    ) == [ProgramInfo("c", "/x/c")]
    assert parse_program_list("Open programs:\n- one.so (ELF)\n- two.so\n") == [
        "one.so",
        "two.so",
    ]
    assert parse_program_entries("") == []


def test_parse_project_files() -> None:
    assert parse_project_files(PROJECT_LIST) == ["/libdxp.so", "/firmware/boot.bin"]
    assert parse_project_files("No programs found in the project.") == []


def test_content_block_results_are_flattened() -> None:
    tool = FakeTool("list_binaries", [[{"type": "text", "text": ONE_OPEN}]])
    programs = asyncio.run(list_open_programs([tool]))
    assert programs[0].project_path == "/libdxp.so"


def test_find_program_precedence() -> None:
    programs = parse_program_entries(TWO_OPEN)
    assert find_program(programs, "/v2/app.exe") == programs[1]
    assert find_program(programs, "app.exe") == programs[0]  # first exact name
    assert find_program(programs, "APP.EXE") == programs[0]
    assert find_program(programs, "other") is None


def test_program_ref_slug() -> None:
    assert ProgramRef("app.exe", "/v1/app.exe").slug == "v1_app.exe"
    assert ProgramRef("libdxp.so", "/libdxp.so").slug == "libdxp.so"
    assert ProgramRef("x", "/").slug == "program"
    assert ProgramRef("a b", "/dir with space/a b").slug == "dir_with_space_a_b"


def test_resolve_single_program_auto_selects() -> None:
    ref = asyncio.run(resolve_program([FakeTool("list_binaries", ONE_OPEN)], None))
    assert ref == ProgramRef("libdxp.so", "/libdxp.so")
    assert (
        asyncio.run(resolve_binary_name([FakeTool("list_binaries", ONE_OPEN)], None))
        == "libdxp.so"
    )


def test_resolve_override_matches_path_or_name_or_passes_through() -> None:
    tools = [FakeTool("list_binaries", TWO_OPEN)]
    assert asyncio.run(resolve_program(tools, "/v2/app.exe")) == ProgramRef(
        "app.exe", "/v2/app.exe"
    )
    assert asyncio.run(resolve_program(tools, "app.exe")) == ProgramRef(
        "app.exe", "/v1/app.exe"
    )
    # Unknown override with one program open: the label is kept (it is the
    # knowledge-base key) and the open program is what gets pinned.
    one = [FakeTool("list_binaries", ONE_OPEN)]
    assert asyncio.run(resolve_program(one, "my-label")) == ProgramRef(
        "my-label", "/libdxp.so"
    )
    # Unknown override, ambiguous or unlistable: used verbatim (the server's
    # name matching may still resolve it).
    assert asyncio.run(resolve_program(tools, "mystery.bin")) == ProgramRef(
        "mystery.bin", "mystery.bin"
    )
    broken = [FakeTool("list_binaries", RuntimeError("down"))]
    assert asyncio.run(resolve_program(broken, "mystery.bin")) == ProgramRef(
        "mystery.bin", "mystery.bin"
    )


def test_resolve_many_without_chooser_names_them() -> None:
    with pytest.raises(RuntimeError, match=r"/v1/app.exe, /v2/app.exe.*--binary-name"):
        asyncio.run(resolve_program([FakeTool("list_binaries", TWO_OPEN)], None))


def test_resolve_many_uses_chooser() -> None:
    async def choose(programs: list[ProgramInfo]) -> ProgramInfo | None:
        return programs[1]

    ref = asyncio.run(
        resolve_program([FakeTool("list_binaries", TWO_OPEN)], None, choose=choose)
    )
    assert ref == ProgramRef("app.exe", "/v2/app.exe")

    async def decline(programs: list[ProgramInfo]) -> ProgramInfo | None:
        return None

    with pytest.raises(RuntimeError, match="No program selected"):
        asyncio.run(
            resolve_program([FakeTool("list_binaries", TWO_OPEN)], None, choose=decline)
        )


def test_resolve_errors() -> None:
    with pytest.raises(RuntimeError, match="No open programs"):
        asyncio.run(
            resolve_program(
                [FakeTool("list_binaries", "No programs currently open")], None
            )
        )
    with pytest.raises(RuntimeError, match="does not expose 'list_binaries'"):
        asyncio.run(resolve_program([FakeTool("other", "")], None))
    with pytest.raises(RuntimeError, match="failed: boom"):
        asyncio.run(resolve_program([FakeTool("list_binaries", OSError("boom"))], None))


def test_list_project_files_and_open() -> None:
    opener = FakeTool(
        "open_program",
        [
            PROJECT_LIST,
            "Opened 'boot.bin' (/firmware/boot.bin) in CodeBrowser.\nLanguage: ARM",
        ],
    )
    two = ONE_OPEN.replace(
        "\n---", "\n2. boot.bin\n   Project Path: /firmware/boot.bin\n\n---"
    )
    lister = FakeTool("list_binaries", two)
    tools = [opener, lister]
    assert asyncio.run(list_project_files(tools)) == [
        "/libdxp.so",
        "/firmware/boot.bin",
    ]
    assert opener.calls[0] == {"action": "list", "folder": "/"}
    entry = asyncio.run(open_project_program(tools, "/firmware/boot.bin"))
    assert entry == ProgramInfo("boot.bin", "/firmware/boot.bin")
    assert opener.calls[1] == {
        "action": "open",
        "name": "/firmware/boot.bin",
        "analyze_after_open": False,
    }


def test_open_reports_not_found_and_phantom_opens() -> None:
    tools = [
        FakeTool("open_program", "Program not found: '/nope'. Use action 'list'..."),
        FakeTool("list_binaries", ONE_OPEN),
    ]
    with pytest.raises(RuntimeError, match="not found"):
        asyncio.run(open_project_program(tools, "/nope"))
    tools = [
        FakeTool("open_program", "Opened 'ghost' (/ghost) in CodeBrowser."),
        FakeTool("list_binaries", ONE_OPEN),
    ]
    with pytest.raises(RuntimeError, match="does not show it"):
        asyncio.run(open_project_program(tools, "/ghost"))
