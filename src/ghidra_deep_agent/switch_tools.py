"""The local ``find_unrecovered_switches`` / ``apply_switch_override`` tools.

These are the jump-table siblings of ``recover_prototypes`` (``prototype_tools.py``)
— read that module's docstring for the shared rationale. Each tool runs a Java
*GhidraScript* inside the running Ghidra via GhidrAssistMCP's ``scripts`` executor,
resolves the async task, and shrinks the JSON manifest to a compact result.

Why a wrapper instead of letting the model drive ``scripts`` directly: GhidrAssistMCP
exposes no tool to add references, set data mutability, or set a jump-table override,
so the ONLY way to apply the fix is a Java script using the Ghidra API. The scripts
here own that deterministic, error-prone work (writing the ``JumpTable`` override,
adding ``COMPUTED_JUMP`` refs, disassembling targets, re-decompiling to verify) so
the LLM only supplies the case targets (or the table shape) it worked out from the
disassembly.

Server prerequisite: GhidrAssistMCP ships the ``scripts`` tool **disabled by
default** — it must be enabled server-side for these tools to appear. When it is
absent, ``build_switch_tools`` returns ``[]`` (with a warning) and the agent runs
exactly as before, just without programmatic jump-table recovery.
"""

import json
import os
import re
import sys
from typing import Any

from langchain_core.tools import BaseTool, tool

from ghidra_deep_agent.apply_switch_override_script import (
    MARK_END as APPLY_MARK_END,
)
from ghidra_deep_agent.apply_switch_override_script import (
    MARK_START as APPLY_MARK_START,
)
from ghidra_deep_agent.apply_switch_override_script import (
    SCRIPT_SOURCE as APPLY_SCRIPT_SOURCE,
)
from ghidra_deep_agent.async_tasks import resolve_async_result, to_text
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_END as FIND_MARK_END,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_START as FIND_MARK_START,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    SCRIPT_SOURCE as FIND_SCRIPT_SOURCE,
)

# Fixed names the scripts are deployed under inside Ghidra. ``.java`` selects
# Ghidra's always-present Java provider; each public class name in the source must
# match this basename.
_FIND_SCRIPT_NAME = "gda_find_switches.java"
_APPLY_SCRIPT_NAME = "gda_apply_switch_override.java"

# A whole-program decompile pass can run for minutes; poll well past the default
# async timeout. Shares the prototype pass's env knob for a single control.
_SWITCH_TIMEOUT_S = float(os.environ.get("GHIDRA_RECOVER_TIMEOUT", "1800"))

_FIND_JSON_RE = re.compile(
    re.escape(FIND_MARK_START) + r"\s*(\{.*\})\s*" + re.escape(FIND_MARK_END),
    re.DOTALL,
)
_APPLY_JSON_RE = re.compile(
    re.escape(APPLY_MARK_START) + r"\s*(\{.*\})\s*" + re.escape(APPLY_MARK_END),
    re.DOTALL,
)


# --- GhidrAssistMCP `scripts` argument shapes -----------------------------------
def _scripts_create(name: str, source: str) -> dict[str, Any]:
    # overwrite=True redeploys the current script each run, so a stale older
    # version can't be executed and no separate delete step is needed.
    return {"action": "create", "name": name, "source": source, "overwrite": True}


def _scripts_run(name: str, run_args: list[str] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"action": "run", "name": name}
    if run_args:
        args["args"] = run_args
    return args


def _format_find_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {})
    switches = payload.get("switches", []) or []
    failed = payload.get("failed", []) or []
    lines = [
        "Unrecovered jump-table scan complete.",
        (
            "scanned={scanned}  unrecovered_funcs={ufuncs}  "
            "unrecovered_jumps={ujumps}  review_known={known}  "
            "decompile_failed={failed}"
        ).format(
            scanned=counts.get("scanned", 0),
            ufuncs=counts.get("unrecovered_funcs", 0),
            ujumps=counts.get("unrecovered_jumps", 0),
            known=counts.get("review_known", 0),
            failed=counts.get("decompile_failed", 0),
        ),
    ]
    if switches:
        lines.append("")
        lines.append(
            "Unrecovered indirect jumps (fix each with apply_switch_override):"
        )
        for s in switches:
            hint = s.get("table_hint") or "?"
            lines.append(
                "- {jump} in {func} {name}  ({mnem}; table@{hint})".format(
                    jump=s.get("jump", "?"),
                    func=s.get("func_addr", "?"),
                    name=s.get("name", "?"),
                    mnem=s.get("mnemonic", "?"),
                    hint=hint,
                )
            )
        if payload.get("switches_truncated"):
            lines.append(
                "  (list truncated; unrecovered_jumps above is the true total)"
            )
    else:
        lines.append("No unrecovered jump tables found.")
    if failed:
        lines.append("")
        lines.append("Failed to decompile (no warning could be checked):")
        for f in failed:
            lines.append(
                "- {addr} {name}: {error}".format(
                    addr=f.get("addr", "?"),
                    name=f.get("name", "?"),
                    error=f.get("error", "?"),
                )
            )
    return "\n".join(lines)


def _format_apply_summary(payload: dict[str, Any]) -> str:
    if not payload.get("applied"):
        err = payload.get("error", "unknown error")
        notes = payload.get("notes") or []
        note_s = ("  notes: " + "; ".join(notes)) if notes else ""
        return f"apply_switch_override: NOT applied — {err}.{note_s}"
    cleared = payload.get("warning_cleared")
    header = (
        "Jump-table override applied at {jump} in {func}: {n} destination(s); "
        "warning {state}."
    ).format(
        jump=payload.get("jump", "?"),
        func=payload.get("func", "?"),
        n=payload.get("num_destinations", 0),
        state="CLEARED" if cleared else "still present (revise the table reading)",
    )
    notes = payload.get("notes") or []
    parts = [header]
    if notes:
        parts.append("Notes: " + "; ".join(notes))
    c = payload.get("decompiled_c", "")
    if c:
        trunc = " (truncated)" if payload.get("c_truncated") else ""
        parts.append(f"\nFresh decompilation{trunc}:\n{c}")
    return "\n".join(parts)


def build_switch_tools(mcp_tools: list[BaseTool]) -> list[BaseTool]:
    """Build the jump-table tools, or ``[]`` if unsupported.

    Requires the GhidrAssistMCP ``scripts`` tool (disabled by default server-side);
    without it the tools are omitted with a warning. ``get_task_status`` is used to
    resolve each script's async task when present.
    """
    by_name = {t.name: t for t in mcp_tools}
    scripts_tool = by_name.get("scripts")
    status_tool = by_name.get("get_task_status")
    if scripts_tool is None:
        print(
            "Warning: GhidrAssistMCP 'scripts' tool not available "
            "(enable it server-side); jump-table tools disabled.",
            file=sys.stderr,
        )
        return []

    async def _run_script(
        name: str, source: str, run_args: list[str] | None = None
    ) -> str:
        # Redeploy the current script (overwrite), then run it. Resolve the async
        # task on both calls — a whole-program pass can take minutes.
        create_out = to_text(await scripts_tool.ainvoke(_scripts_create(name, source)))
        await resolve_async_result(create_out, status_tool, timeout_s=_SWITCH_TIMEOUT_S)
        run_out = to_text(await scripts_tool.ainvoke(_scripts_run(name, run_args)))
        return await resolve_async_result(
            run_out, status_tool, timeout_s=_SWITCH_TIMEOUT_S
        )

    @tool
    async def find_unrecovered_switches() -> str:
        """Find every indirect jump whose jump table Ghidra failed to recover.

        Runs a single read-only Ghidra-side pass that decompiles the program and
        reports each function whose decompilation carries the
        ``Could not recover jumptable`` warning (unrecovered switch statements /
        indirect jumps — "Too many branches" / "Treating indirect jump as call").
        For each it returns the jump instruction's address, the containing
        function, the instruction mnemonic, and a table-base hint.

        Use this to enumerate the jump tables that still need repair, then fix each
        with ``apply_switch_override``. Genuine indirect tail calls are NOT
        reported (they lack the decompiler warning), and jumps already flagged as
        unrecoverable dead ends (a ``switch-review`` bookmark) are skipped. Safe to
        re-run — it writes nothing.

        Cost: ONE whole-program pass; a few seconds on a typical binary (only
        functions containing an unresolved indirect jump are decompiled). Call it
        ONCE, not per-function.
        """
        raw = await _run_script(_FIND_SCRIPT_NAME, FIND_SCRIPT_SOURCE)
        match = _FIND_JSON_RE.search(raw)
        if match is None:
            tail = raw[-800:] if raw else "(empty result)"
            return (
                "find_unrecovered_switches: no JSON manifest found in the script "
                "output. The `scripts` executor may not return stdout, the "
                "`scripts` tool may be disabled/misconfigured, or the script "
                "errored. Raw output tail:\n" + tail
            )
        try:
            payload = json.loads(match.group(1))
        except ValueError as exc:
            return f"find_unrecovered_switches: could not parse manifest JSON ({exc})."
        return _format_find_summary(payload)

    @tool
    async def apply_switch_override(
        jump_address: str,
        destinations: list[str] | None = None,
        table_address: str | None = None,
        element_size: int | None = None,
        count: int | None = None,
        base_address: str | None = None,
        relative: bool = False,
        set_rodata_constant: bool = False,
    ) -> str:
        """Recover one unrecovered switch by writing its jump-table override.

        For the indirect jump at ``jump_address`` this writes Ghidra's decompiler
        jump-table override, adds a ``COMPUTED_JUMP`` reference to every case
        target, disassembles any undefined targets, optionally marks the table's
        memory block read-only, then RE-DECOMPILES and reports whether the
        ``Could not recover jumptable`` warning cleared — returning the fresh
        decompilation so you can read the now-correct function immediately.

        Supply the case targets ONE of two ways (compute them from the
        disassembly and the table bytes first):
        - ``destinations``: an explicit list of absolute target addresses (hex
          strings) — best for small or irregular tables.
        - a strided table to decode: ``table_address`` + ``element_size`` (bytes
          per entry: 1/2/4/8) + ``count`` (number of entries), plus
          ``base_address`` and ``relative=True`` when entries are signed offsets
          added to a base (``dest = base + entry``); with ``relative=False``
          entries are absolute pointers. When ``relative`` and ``base_address`` is
          omitted, the table address is used as the base.

        Set ``set_rodata_constant=True`` (with the table-decode form) to also mark
        the table's memory block read-only — retry with this if a ``.rodata`` table
        doesn't clear on the first attempt. If ``warning_cleared`` is false, revise
        the table reading (stride, count, absolute vs relative) and call again.

        Args:
            jump_address: Address of the indirect jump instruction to fix.
            destinations: Explicit absolute case-target addresses (hex strings).
            table_address: Base address of a strided jump table to decode.
            element_size: Bytes per table entry (1, 2, 4, or 8).
            count: Number of table entries.
            base_address: Base added to each entry when ``relative`` (default:
                ``table_address``).
            relative: True when entries are signed offsets from ``base_address``;
                False when they are absolute pointers.
            set_rodata_constant: Also mark the table's memory block read-only.
        """
        has_dests = bool(destinations)
        has_table = table_address is not None
        if has_dests == has_table:
            return (
                "apply_switch_override: provide EITHER `destinations` OR "
                "`table_address` (+ `element_size` + `count`), not both/neither."
            )
        if has_table and (element_size is None or count is None):
            return (
                "apply_switch_override: the table-decode form needs "
                "`table_address`, `element_size`, and `count`."
            )
        payload: dict[str, Any] = {"jump_address": jump_address}
        if has_dests:
            payload["destinations"] = destinations
        else:
            payload["table_address"] = table_address
            payload["element_size"] = element_size
            payload["count"] = count
            payload["relative"] = relative
            if base_address is not None:
                payload["base_address"] = base_address
        if set_rodata_constant:
            payload["set_rodata_constant"] = True

        raw = await _run_script(
            _APPLY_SCRIPT_NAME, APPLY_SCRIPT_SOURCE, [json.dumps(payload)]
        )
        match = _APPLY_JSON_RE.search(raw)
        if match is None:
            tail = raw[-800:] if raw else "(empty result)"
            return (
                "apply_switch_override: no JSON manifest found in the script "
                "output. Raw output tail:\n" + tail
            )
        try:
            result = json.loads(match.group(1))
        except ValueError as exc:
            return f"apply_switch_override: could not parse manifest JSON ({exc})."
        return _format_apply_summary(result)

    return [find_unrecovered_switches, apply_switch_override]
