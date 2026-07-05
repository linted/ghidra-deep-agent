"""The local ``recover_prototypes`` tool.

This is a normal LangChain tool that runs the prototype-recovery *GhidraScript*
(``recover_prototypes_script.SCRIPT_SOURCE``) inside the running Ghidra via
GhidrAssistMCP's ``scripts`` executor, then hands the model back only the
ambiguous cases that still need judgement.

Why a wrapper instead of letting the model drive ``scripts`` directly: the whole
point is to keep the deterministic, mechanical work (decompiler parameter
recovery, comparing committed vs recovered prototypes, and committing the
clear-cut fixes) out of the LLM. The script does that work in one Ghidra-side
pass; this tool just deploys it, runs it, resolves the async task, and shrinks the
JSON manifest to a compact summary + the review list.

Server prerequisite: GhidrAssistMCP ships the ``scripts`` tool **disabled by
default** — it must be enabled server-side for this tool to appear. When it is
absent, ``build_prototype_tools`` returns ``[]`` (with a warning) and the agent
runs exactly as before.

The ``scripts`` tool is action-based (verified against GhidrAssistMCP): ``create``
takes ``name`` + ``source`` (+ ``overwrite``), ``run`` executes a script by name
(+ optional ``args``). The script returns its result on stdout, which the server
surfaces in the run result; if that ever stops being true this tool reports it
plainly (there is no file fallback: the server may be remote, so no shared
filesystem is assumed).
"""

import json
import os
import re
import sys
from typing import Any

from langchain_core.tools import BaseTool, tool

from ghidra_deep_agent.async_tasks import resolve_async_result, to_text
from ghidra_deep_agent.recover_prototypes_script import (
    MARK_END,
    MARK_START,
    SCRIPT_SOURCE,
)

# Fixed name the script is deployed under inside Ghidra. ``.java`` selects
# Ghidra's always-present Java provider; the public class name in SCRIPT_SOURCE
# must match this basename (``gda_recover_prototypes``).
_SCRIPT_NAME = "gda_recover_prototypes.java"

# A program-wide decompile pass can run for minutes; poll well past the default
# async timeout. Overridable for large binaries.
_RECOVER_TIMEOUT_S = float(os.environ.get("GHIDRA_RECOVER_TIMEOUT", "1800"))

_JSON_RE = re.compile(
    re.escape(MARK_START) + r"\s*(\{.*\})\s*" + re.escape(MARK_END),
    re.DOTALL,
)


# --- GhidrAssistMCP `scripts` argument shapes -----------------------------------
def _scripts_create(source: str) -> dict[str, Any]:
    # overwrite=True redeploys the current script each run, so a stale older
    # version can't be executed and no separate delete step is needed.
    return {
        "action": "create",
        "name": _SCRIPT_NAME,
        "source": source,
        "overwrite": True,
    }


def _scripts_run(dry_run: bool) -> dict[str, Any]:
    args: dict[str, Any] = {"action": "run", "name": _SCRIPT_NAME}
    if dry_run:
        args["args"] = ["dry_run"]
    return args


def _format_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {})
    escalate = payload.get("escalate", []) or []
    header = (
        "Prototype recovery DRY RUN (no changes applied) — 'auto_fixed' is what "
        "WOULD be fixed."
        if payload.get("dry_run")
        else "Prototype recovery pass complete."
    )
    lines = [
        header,
        (
            "scanned={scanned}  already_correct={already_correct}  "
            "auto_fixed={fixed}  needs_review={escalate} new "
            "({escalate_known} already flagged)  "
            "decompile_failed={decompile_failed}"
        ).format(
            scanned=counts.get("scanned", 0),
            already_correct=counts.get("already_correct", 0),
            fixed=counts.get("fixed", 0),
            escalate=counts.get("escalate", 0),
            escalate_known=counts.get("escalate_known", 0),
            decompile_failed=counts.get("decompile_failed", 0),
        ),
    ]
    if escalate:
        lines.append("")
        lines.append(
            "Needs manual review — delegate these to the `prototype-fixer` "
            "sub-agent (committed vs decompiler-recovered, plus why):"
        )
        for e in escalate:
            lines.append(
                "- {addr} {name}: committed {committed}  vs  recovered "
                "{recovered}  — {reason}".format(
                    addr=e.get("addr", "?"),
                    name=e.get("name", "?"),
                    committed=e.get("committed", "?"),
                    recovered=e.get("recovered", "?"),
                    reason=e.get("reason", "?"),
                )
            )
    else:
        lines.append("No new functions need manual review.")
    return "\n".join(lines)


def build_prototype_tools(mcp_tools: list[BaseTool]) -> list[BaseTool]:
    """Build the ``recover_prototypes`` tool, or ``[]`` if unsupported.

    Requires the GhidrAssistMCP ``scripts`` tool (disabled by default server-side);
    without it the tool is omitted with a warning. ``get_task_status`` is used to
    resolve the script's async task when present.
    """
    by_name = {t.name: t for t in mcp_tools}
    scripts_tool = by_name.get("scripts")
    status_tool = by_name.get("get_task_status")
    if scripts_tool is None:
        print(
            "Warning: GhidrAssistMCP 'scripts' tool not available "
            "(enable it server-side); 'recover_prototypes' disabled.",
            file=sys.stderr,
        )
        return []

    async def _run_script(dry_run: bool) -> str:
        # Redeploy the current script (overwrite), then run it. Resolve the async
        # task on the run result — a program-wide pass can take minutes.
        create_out = to_text(await scripts_tool.ainvoke(_scripts_create(SCRIPT_SOURCE)))
        create_out = await resolve_async_result(
            create_out, status_tool, timeout_s=_RECOVER_TIMEOUT_S
        )
        run_out = to_text(await scripts_tool.ainvoke(_scripts_run(dry_run)))
        return await resolve_async_result(
            run_out, status_tool, timeout_s=_RECOVER_TIMEOUT_S
        )

    @tool
    async def recover_prototypes(dry_run: bool = False) -> str:
        """Repair function prototypes program-wide using Ghidra's own decompiler.

        Runs a single Ghidra-side pass that, for every function still lacking a
        real signature (the ``param_count=0`` population), compares the committed
        prototype against the one Ghidra's decompiler already recovered,
        AUTO-APPLIES the clear-cut fixes (a clean register/stack argument list,
        not variadic), and flags only the genuinely ambiguous cases (variadics,
        non-standard storage, failed commits) for review.

        Use this INSTEAD of searching for ``param_count=0`` functions and tracing
        prologues and call sites by hand — the decompiler has already done that
        deterministic work. Returns a compact summary and the list of functions
        that still need judgement; hand those to the ``prototype-fixer``
        sub-agent. Safe to re-run: already-correct and previously-flagged
        functions are skipped.

        Cost: this is ONE whole-program pass and is fast — a few seconds on a
        typical binary (roughly linear in the number of unresolved functions; at
        most a couple of minutes on a very large one). Call it ONCE for the whole
        program, not per-function and not repeatedly in a loop.

        Args:
            dry_run: When true, preview only — report what WOULD be auto-fixed and
                flagged without committing any prototype or writing any bookmark.
        """
        raw = await _run_script(dry_run)
        match = _JSON_RE.search(raw)
        if match is None:
            tail = raw[-800:] if raw else "(empty result)"
            return (
                "recover_prototypes: no JSON manifest found in the script output. "
                "The `scripts` executor may not return stdout, the `scripts` tool "
                "may be disabled/misconfigured, or the script errored. "
                "Raw output tail:\n" + tail
            )
        try:
            payload = json.loads(match.group(1))
        except ValueError as exc:
            return f"recover_prototypes: could not parse manifest JSON ({exc})."
        return _format_summary(payload)

    return [recover_prototypes]
