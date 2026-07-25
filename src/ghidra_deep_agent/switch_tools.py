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
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_END as FIND_MARK_END,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    MARK_START as FIND_MARK_START,
)
from ghidra_deep_agent.find_unrecovered_switches_script import (
    SCRIPT_SOURCE as FIND_SCRIPT_SOURCE,
)
from ghidra_deep_agent.ghidra_script_tools import (
    GhidraScriptRunner,
    find_scripts_tools,
    manifest_pattern,
)
from ghidra_deep_agent.ollvm_deobfuscate_script import (
    MARK_END as OLLVM_MARK_END,
)
from ghidra_deep_agent.ollvm_deobfuscate_script import (
    MARK_START as OLLVM_MARK_START,
)
from ghidra_deep_agent.ollvm_deobfuscate_script import (
    SCRIPT_SOURCE as OLLVM_SCRIPT_SOURCE,
)

# Fixed names the scripts are deployed under inside Ghidra. ``.java`` selects
# Ghidra's always-present Java provider; each public class name in the source must
# match this basename.
_FIND_SCRIPT_NAME = "gda_find_switches.java"
_APPLY_SCRIPT_NAME = "gda_apply_switch_override.java"
# The CFF deobfuscator's public class is ``OllvmDeobfuscator``, so its deployed
# file name must match.
_OLLVM_SCRIPT_NAME = "OllvmDeobfuscator.java"

_FIND_JSON_RE = manifest_pattern(FIND_MARK_START, FIND_MARK_END)
_APPLY_JSON_RE = manifest_pattern(APPLY_MARK_START, APPLY_MARK_END)
_OLLVM_JSON_RE = manifest_pattern(OLLVM_MARK_START, OLLVM_MARK_END)
# The CFF pass rewrites instructions across a whole flattened function; keep the
# readable phase report the agent sees bounded.
_OLLVM_REPORT_CAP = 6000

# Strides Ghidra can actually read a jump-table entry at. Anything else makes the
# script throw once per entry and report a misleading "no valid destinations".
_VALID_ELEMENT_SIZES = frozenset({1, 2, 4, 8})
# Sanity ceiling on table entries. Real switch tables are far smaller; this only
# exists so a hallucinated `count` can't spin the script for millions of rounds.
_MAX_TABLE_ENTRIES = 4096


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
        # The script caps this list and flags when it did; without the flag the
        # listing silently under-reports on a binary with many failures.
        if payload.get("failed_truncated"):
            lines.append("  (list truncated; decompile_failed above is the true total)")
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


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... ({len(text) - limit} chars elided) ...\n{text[-tail:]}"


def _format_ollvm_summary(payload: dict[str, Any], report: str) -> str:
    status = payload.get("status", "?")
    fn = payload.get("function")
    entry = payload.get("entry")
    where = f"{fn} @ {entry}" if fn else f"@ {entry or '?'}"
    if status == "no_function":
        return (
            "deobfuscate_cff: no target function resolved from that argument. "
            "Pass a function name (e.g. FUN_00123456) or an address (e.g. 0x123456)."
        )
    if status == "no_dispatch":
        return (
            f"deobfuscate_cff: no OLLVM CFF dispatch pattern found in {where}. "
            "This function is not control-flow-flattened (or uses a shape this "
            "pass does not recognize) — nothing to do."
        )
    mode = payload.get("mode", "?")
    verb = "would patch" if mode == "dryrun" else "patched"
    header = (
        "CFF deobfuscation [{mode}{force}] {where}: {sites} dispatch site(s), "
        "{trans} transition(s), {verb} {patched}."
    ).format(
        mode=mode,
        force=" +force" if payload.get("force") else "",
        where=where,
        sites=payload.get("dispatch_sites", 0),
        trans=payload.get("transitions", 0),
        verb=verb,
        patched=payload.get("patched", 0),
    )
    tip = (
        "\nDry run — nothing was written. Re-run with apply=True to patch, and "
        "force=True to write patches whose ranges overlap (shared dispatch tails)."
        if mode == "dryrun"
        else "\nPatched — re-decompile the function to verify recovered flow."
    )
    return f"{header}{tip}\n\n{_cap(report.strip(), _OLLVM_REPORT_CAP)}"


def build_switch_tools(mcp_tools: list[BaseTool]) -> list[BaseTool]:
    """Build the jump-table tools, or ``[]`` if unsupported.

    Requires the GhidrAssistMCP ``scripts`` tool (disabled by default server-side);
    without it the tools are omitted with a warning. ``get_task_status`` is used to
    resolve each script's async task when present.
    """
    found = find_scripts_tools(mcp_tools, "jump-table tools disabled.")
    if found is None:
        return []
    runner = GhidraScriptRunner(*found)

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
        payload, _raw, error = await runner.run_manifest(
            _FIND_SCRIPT_NAME,
            FIND_SCRIPT_SOURCE,
            _FIND_JSON_RE,
            "find_unrecovered_switches",
        )
        if payload is None:
            return error
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
        # Bounds checked here rather than script-side: the Java loop turns a bad
        # stride into one note per entry plus a misleading "no valid destination
        # addresses were produced", and an unbounded `count` into millions of
        # iterations. A precise message up front is what lets the model correct.
        if has_table:
            if element_size not in _VALID_ELEMENT_SIZES:
                return (
                    f"apply_switch_override: `element_size` must be one of "
                    f"{sorted(_VALID_ELEMENT_SIZES)} (got {element_size})."
                )
            if not 0 < (count or 0) <= _MAX_TABLE_ENTRIES:
                return (
                    f"apply_switch_override: `count` must be between 1 and "
                    f"{_MAX_TABLE_ENTRIES} (got {count})."
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

        result, _raw, error = await runner.run_manifest(
            _APPLY_SCRIPT_NAME,
            APPLY_SCRIPT_SOURCE,
            _APPLY_JSON_RE,
            "apply_switch_override",
            [json.dumps(payload)],
        )
        if result is None:
            return error
        return _format_apply_summary(result)

    @tool
    async def deobfuscate_cff(
        function: str,
        apply: bool = False,
        force: bool = False,
    ) -> str:
        """Deobfuscate ONE OLLVM control-flow-flattened (CFF) function by rewriting
        its dispatcher into direct branches.

        Use this for the *other* kind of unrecovered indirect jump: an OLLVM CFF
        *dispatcher* (`br` on a state index loaded from a jump table) that
        `apply_switch_override` cannot fix because the decompiler still times out
        on the flattened state machine even after the jump-table override is
        applied (the `switch-review` bookmark reads like
        "CFF dispatcher ... warning does NOT clear: decompiler TIMES OUT"). This
        pass instead reads the table, resolves each case block's real successor
        (seeing through the index range-clamp and folding constant state indices),
        and patches the indirect branches into direct `b`/`b.cond`, so the
        decompiler can recover normal control flow.

        ALWAYS dry-run first (the default): it writes nothing and returns the
        planned per-block transitions and patches so you can sanity-check the
        recovered flow before committing. Then re-run with `apply=True`.

        Args:
            function: The flattened function to work on — a name (e.g.
                "FUN_001e65bc") or an entry/interior address (e.g. "0x1e65bc").
            apply: False (default) = dry run, analyze and report only, write
                nothing. True = patch the listing.
            force: When True, also write patches whose address ranges overlap a
                previous patch. Overlaps are shared dispatch tails that multiple
                case blocks reach with different targets; forcing writes one and
                is generally WRONG for those — leave False unless a dry run shows
                the overlaps are spurious.

        Cost: ONE function per call (not a whole-program pass). Only rewrites the
        indirect branches; interleaved real work is preserved. NOT for ordinary
        switch tables — use `apply_switch_override` for those.
        """
        mode = "apply" if apply else "dryrun"
        args = [function, mode]
        if force:
            args.append("force")
        result, raw, error = await runner.run_manifest(
            _OLLVM_SCRIPT_NAME,
            OLLVM_SCRIPT_SOURCE,
            _OLLVM_JSON_RE,
            "deobfuscate_cff",
            args,
        )
        if result is None:
            return error
        return _format_ollvm_summary(result, raw)

    return [find_unrecovered_switches, apply_switch_override, deobfuscate_cff]
