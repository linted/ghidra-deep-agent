"""Shared plumbing for the local tools that drive Ghidra-side scripts.

``prototype_tools`` and ``switch_tools`` both work the same way: deploy a Java
*GhidraScript* through GhidrAssistMCP's ``scripts`` executor, run it, resolve the
async task, and pull a JSON manifest out of the script's stdout between two
sentinel markers. That plumbing lives here so the tool modules only hold what is
actually specific to them — the script source, the arguments, and how the
manifest is rendered for the model.

Server prerequisite: GhidrAssistMCP ships the ``scripts`` tool **disabled by
default**. When it is absent, ``find_scripts_tools`` warns and returns ``None``,
and the calling module returns no tools at all.
"""

import json
import os
import re
import sys
from typing import Any

from langchain_core.tools import BaseTool

from ghidra_deep_agent.async_tasks import resolve_async_result, to_text

# A whole-program decompile pass can run for minutes; poll well past the default
# async timeout. One knob covers every script-driving tool.
SCRIPT_TIMEOUT_S = float(os.environ.get("GHIDRA_RECOVER_TIMEOUT", "1800"))

# How much raw script output to echo back when no manifest could be found.
_RAW_TAIL_CHARS = 800


def manifest_pattern(mark_start: str, mark_end: str) -> re.Pattern[str]:
    """Compile the ``MARK_START {json} MARK_END`` pattern a script emits.

    Non-greedy on purpose: a greedy ``.*`` spans from the *first* start marker to
    the *last* end marker, so any output carrying two marker pairs — a redeploy
    that echoes a previous run, or a script that emits twice — captures both
    payloads plus the markers between them and fails to parse.
    """
    return re.compile(
        re.escape(mark_start) + r"\s*(\{.*?\})\s*" + re.escape(mark_end),
        re.DOTALL,
    )


def find_manifest(pattern: re.Pattern[str], raw: str) -> re.Match[str] | None:
    """Return the *last* manifest in ``raw`` — the current run's, if several."""
    matches = list(pattern.finditer(raw))
    return matches[-1] if matches else None


def find_scripts_tools(
    mcp_tools: list[BaseTool], disabled_note: str
) -> tuple[BaseTool, BaseTool | None] | None:
    """Locate the ``scripts`` executor (and ``get_task_status``, if present).

    Returns ``None`` — after warning with ``disabled_note`` — when the server
    doesn't expose ``scripts``, which is the signal to build no tools.
    """
    by_name = {t.name: t for t in mcp_tools}
    scripts_tool = by_name.get("scripts")
    if scripts_tool is None:
        print(
            "Warning: GhidrAssistMCP 'scripts' tool not available "
            f"(enable it server-side); {disabled_note}",
            file=sys.stderr,
        )
        return None
    return scripts_tool, by_name.get("get_task_status")


class GhidraScriptRunner:
    """Deploys and runs Ghidra-side scripts through the MCP ``scripts`` tool."""

    def __init__(
        self,
        scripts_tool: BaseTool,
        status_tool: BaseTool | None,
        *,
        timeout_s: float = SCRIPT_TIMEOUT_S,
    ) -> None:
        self._scripts_tool = scripts_tool
        self._status_tool = status_tool
        self._timeout_s = timeout_s
        # Resolved output of the most recent deploy. Ghidra compiles the whole
        # script directory as one OSGi bundle, so a javac error surfaces here —
        # and only here. Kept so the no-manifest path can show it instead of the
        # generic "the script may have errored".
        self._last_deploy_output = ""

    async def run(
        self, name: str, source: str, run_args: list[str] | None = None
    ) -> str:
        """Redeploy the script and run it, returning its raw output.

        ``overwrite=True`` redeploys the current source each run, so a stale
        older version can't execute and no separate delete step is needed. The
        async task is resolved on both calls — a whole-program pass can take
        minutes.
        """
        create_args: dict[str, Any] = {
            "action": "create",
            "name": name,
            "source": source,
            "overwrite": True,
        }
        create_out = to_text(await self._scripts_tool.ainvoke(create_args))
        self._last_deploy_output = await resolve_async_result(
            create_out, self._status_tool, timeout_s=self._timeout_s
        )

        run_call: dict[str, Any] = {"action": "run", "name": name}
        if run_args:
            run_call["args"] = run_args
        run_out = to_text(await self._scripts_tool.ainvoke(run_call))
        return await resolve_async_result(
            run_out, self._status_tool, timeout_s=self._timeout_s
        )

    async def run_manifest(
        self,
        name: str,
        source: str,
        pattern: re.Pattern[str],
        tool_label: str,
        run_args: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str, str]:
        """Run a script and parse its JSON manifest.

        Returns ``(payload, raw_output, error)``. On success ``error`` is empty;
        otherwise ``payload`` is ``None`` and ``error`` is a message written for
        the model, explaining what to check.
        """
        raw = await self.run(name, source, run_args)
        match = find_manifest(pattern, raw)
        if match is None:
            tail = raw[-_RAW_TAIL_CHARS:] if raw else "(empty result)"
            deploy = self._last_deploy_output.strip()
            # The deploy step is where a compile error lands, and it is by far the
            # likeliest reason a run produced no manifest — so show it rather than
            # leaving the model to guess from the run output alone.
            deploy_note = (
                "\nDeploy (compile) output:\n" + deploy[-_RAW_TAIL_CHARS:]
                if deploy
                else ""
            )
            return (
                None,
                raw,
                f"{tool_label}: no JSON manifest found in the script output. "
                "The `scripts` executor may not return stdout, the `scripts` "
                "tool may be disabled/misconfigured, or the script failed to "
                "compile or errored at runtime. Raw output tail:\n"
                + tail
                + deploy_note,
            )
        try:
            payload: dict[str, Any] = json.loads(match.group(1))
        except ValueError as exc:
            return (
                None,
                raw,
                f"{tool_label}: could not parse manifest JSON ({exc}).",
            )
        return payload, raw, ""
