"""The GhidrAssistMCP-side jump-table *detection* script, as a source string.

``SCRIPT_SOURCE`` is a **Java** GhidraScript (not code that runs in this process).
It is shipped verbatim to the running Ghidra via GhidrAssistMCP's ``scripts`` tool
and compiled/run there — see ``switch_tools.py``. Keeping it as a plain string
means ruff/mypy only ever see a valid Python-3 string literal. This is the exact
sibling of ``recover_prototypes_script.py``; read that file's module docstring for
the shared design (Java-not-Jython, parallel decompile, JSON-between-markers).

What the script does, program-wide and deterministically (no LLM):

* Cheap pre-filter on the script thread: keep only functions that contain at
  least one *computed jump* instruction (``FlowType.isJump() && isComputed()``)
  with **no** ``COMPUTED_JUMP`` reference — i.e. an indirect jump whose targets
  Ghidra never resolved. Functions with none are skipped without decompiling.
* Decompiles the survivors in parallel and gates on the decompiler warning
  ``"Could not recover jumptable"`` — the switch-specific marker. This gate is
  what keeps genuine indirect tail calls (``jmp rax`` that is really a call) out
  of the results: they have the same instruction shape but do not produce that
  warning.
* For each surviving function it re-walks the body and reports every unresolved
  computed-jump instruction: its address, the containing function, the mnemonic,
  and a ``table_hint`` (a data reference from the jump, usually the table base).
* Idempotency: a jump instruction already carrying a ``switch-review`` bookmark
  whose note marks a **dead end** (``unresolvable`` / ``not-a-switch``, written
  by ``switch-fixer``) is skipped and counted as ``review_known`` so re-runs
  don't re-flood adjudicated dead ends. A ``needs-recovery`` breadcrump (written
  by ``function-analyst``) is NOT a dead end and is still reported.

The script writes NOTHING to the program — it is read-only. The result is a
single JSON object between the ``<<<FIND_SWITCHES_JSON>>>`` /
``<<<END_FIND_SWITCHES_JSON>>>`` markers on stdout.

The GhidraScript public class name must match the deployed file name — see
``_FIND_SCRIPT_NAME`` in ``switch_tools.py`` (``gda_find_switches.java``).
"""

MARK_START = "<<<FIND_SWITCHES_JSON>>>"
MARK_END = "<<<END_FIND_SWITCHES_JSON>>>"

SCRIPT_SOURCE: str = r"""// find_unrecovered_switches -- Ghidra Java GhidraScript. Runs inside Ghidra.
// @category DeepAgent
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.parallel.ChunkingParallelDecompiler;
import ghidra.app.decompiler.parallel.DecompilerCallback;
import ghidra.app.decompiler.parallel.ParallelDecompiler;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Bookmark;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.symbol.FlowType;
import ghidra.program.model.symbol.RefType;
import ghidra.program.model.symbol.Reference;
import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

public class gda_find_switches extends GhidraScript {

    static final String MARK_START = "<<<FIND_SWITCHES_JSON>>>";
    static final String MARK_END = "<<<END_FIND_SWITCHES_JSON>>>";
    static final String REVIEW_CATEGORY = "switch-review";
    static final String WARNING = "Could not recover jumptable";
    static final int DECOMP_TIMEOUT = 60;   // seconds per function
    static final int SWITCH_DETAIL_CAP = 400;
    static final int FAILED_DETAIL_CAP = 200;
    static final int CHUNK_SIZE = 64;

    // A single unresolved computed jump.
    private static class Jump {
        String jumpS, mnemonic, tableHint;
    }

    // Worker threads only decompile + read the listing (never write). Every
    // counter / StringBuilder update happens on the script thread in
    // applyResult(), mirroring recover_prototypes_script.
    private static class SwitchResult {
        static final int NO_WARNING = 0;    // decompiled cleanly, not a candidate
        static final int DECOMP_FAILED = 1;
        static final int UNRECOVERED = 2;   // has the warning + concrete jumps

        Function func;
        int kind;
        String addrS, nameS, reason;
        List<Jump> jumps;
    }

    private int scanned = 0, unrecoveredFuncs = 0, unrecoveredJumps = 0;
    private int reviewKnown = 0, decompFailed = 0;
    private int switchDetail = 0, failedDetail = 0;
    private StringBuilder switchJson = new StringBuilder();
    private StringBuilder failJson = new StringBuilder();

    private String addrString(Address a) {
        String s = a.toString();
        int idx = s.indexOf(':');
        if (idx >= 0) {
            s = s.substring(idx + 1);
        }
        return "0x" + s;
    }

    private String firstLine(String msg, String fallback) {
        if (msg == null) {
            return fallback;
        }
        String s = msg.trim();
        int nl = s.indexOf('\n');
        if (nl >= 0) {
            s = s.substring(0, nl).trim();
        }
        if (s.isEmpty()) {
            return fallback;
        }
        return s.length() > 200 ? s.substring(0, 200) : s;
    }

    private String js(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') {
                b.append('\\').append('"');
            } else if (c == '\\') {
                b.append('\\').append('\\');
            } else if (c == '\n') {
                b.append('\\').append('n');
            } else if (c == '\r') {
                b.append('\\').append('r');
            } else if (c == '\t') {
                b.append('\\').append('t');
            } else if (c < 0x20) {
                b.append(String.format("\\u%04x", (int) c));
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    private void appendObj(StringBuilder target, String obj) {
        if (target.length() > 0) {
            target.append(",");
        }
        target.append(obj);
    }

    // True if the instruction is a computed (indirect) jump that Ghidra never
    // resolved: no COMPUTED_JUMP reference flows from it.
    private boolean isUnresolvedComputedJump(Instruction instr) {
        FlowType ft = instr.getFlowType();
        if (ft == null || !ft.isJump() || !ft.isComputed()) {
            return false;
        }
        for (Reference ref : instr.getReferencesFrom()) {
            if (ref.getReferenceType() == RefType.COMPUTED_JUMP) {
                return false;
            }
        }
        return true;
    }

    // A dead-end verdict bookmark (written by switch-fixer) means "skip forever".
    // A `needs-recovery` breadcrumb is NOT a dead end and must still be reported.
    private boolean hasDeadEndBookmark(Address jumpAddr) {
        for (Bookmark bm : currentProgram.getBookmarkManager().getBookmarks(jumpAddr)) {
            if (!REVIEW_CATEGORY.equals(bm.getCategory())) {
                continue;
            }
            String note = bm.getComment();
            if (note != null
                    && (note.contains("unresolvable") || note.contains("not-a-switch"))) {
                return true;
            }
        }
        return false;
    }

    // A data reference from the jump instruction is usually the table base.
    private String tableHint(Instruction instr) {
        for (Reference ref : instr.getReferencesFrom()) {
            RefType rt = ref.getReferenceType();
            if (rt.isData() || rt.isRead()) {
                return addrString(ref.getToAddress());
            }
        }
        return null;
    }

    // Walk the function body; collect unresolved computed jumps (minus dead ends).
    // Returns null when the function has no candidate jump at all.
    private List<Jump> collectJumps(Function func) {
        List<Jump> out = null;
        for (Instruction instr : currentProgram.getListing()
                .getInstructions(func.getBody(), true)) {
            if (!isUnresolvedComputedJump(instr)) {
                continue;
            }
            Address ja = instr.getAddress();
            if (hasDeadEndBookmark(ja)) {
                continue;   // adjudicated dead end; counted by caller
            }
            Jump j = new Jump();
            j.jumpS = addrString(ja);
            j.mnemonic = instr.toString();
            j.tableHint = tableHint(instr);
            if (out == null) {
                out = new ArrayList<>();
            }
            out.add(j);
        }
        return out;
    }

    // Runs on worker threads: decompile READ + listing READ only, never writes.
    private SwitchResult classify(DecompileResults res) {
        SwitchResult r = new SwitchResult();
        r.func = res.getFunction();
        r.addrS = addrString(r.func.getEntryPoint());
        r.nameS = r.func.getName();
        try {
            if (!res.decompileCompleted()) {
                r.kind = SwitchResult.DECOMP_FAILED;
                r.reason = firstLine(res.getErrorMessage(),
                    "decompilation did not complete");
                return r;
            }
            String c = res.getDecompiledFunction() != null
                ? res.getDecompiledFunction().getC() : null;
            if (c == null || !c.contains(WARNING)) {
                r.kind = SwitchResult.NO_WARNING;
                return r;
            }
            List<Jump> jumps = collectJumps(r.func);
            if (jumps == null || jumps.isEmpty()) {
                // Warning present but every unresolved jump is a dead end (or the
                // warning is not tied to a walkable instruction); treat as
                // no-new-work.
                r.kind = SwitchResult.NO_WARNING;
                return r;
            }
            r.kind = SwitchResult.UNRECOVERED;
            r.jumps = jumps;
        } catch (Throwable t) {
            r.kind = SwitchResult.DECOMP_FAILED;
            r.reason = firstLine(t.getMessage(), t.getClass().getSimpleName());
        }
        return r;
    }

    // Script thread only: all counter / JSON bookkeeping.
    private void applyResult(SwitchResult r) {
        if (r.kind == SwitchResult.NO_WARNING) {
            return;
        }
        if (r.kind == SwitchResult.DECOMP_FAILED) {
            decompFailed++;
            if (failedDetail < FAILED_DETAIL_CAP) {
                String err = r.reason != null ? r.reason : "unknown decompiler failure";
                appendObj(failJson,
                    "{\"addr\":\"" + js(r.addrS) + "\",\"name\":\"" + js(r.nameS)
                    + "\",\"error\":\"" + js(err) + "\"}");
                failedDetail++;
            }
            return;
        }
        // UNRECOVERED
        unrecoveredFuncs++;
        for (Jump j : r.jumps) {
            unrecoveredJumps++;
            if (switchDetail < SWITCH_DETAIL_CAP) {
                String hint = j.tableHint != null ? j.tableHint : "";
                appendObj(switchJson,
                    "{\"func_addr\":\"" + js(r.addrS) + "\",\"name\":\"" + js(r.nameS)
                    + "\",\"jump\":\"" + js(j.jumpS) + "\",\"mnemonic\":\"" + js(j.mnemonic)
                    + "\",\"table_hint\":\"" + js(hint) + "\"}");
                switchDetail++;
            }
        }
    }

    @Override
    public void run() throws Exception {
        // Cheap script-thread pre-filter: only functions that actually contain an
        // unresolved computed jump reach the decompiler. This both skips the
        // expensive decompile for the vast majority of functions and counts
        // review_known dead ends without decompiling them.
        List<Function> candidates = new ArrayList<>();
        for (Function func : currentProgram.getFunctionManager().getFunctions(true)) {
            if (func.isThunk() || func.isExternal()) {
                continue;
            }
            scanned++;
            boolean hasCandidate = false;
            boolean hasDeadEndOnly = false;
            for (Instruction instr : currentProgram.getListing()
                    .getInstructions(func.getBody(), true)) {
                if (!isUnresolvedComputedJump(instr)) {
                    continue;
                }
                if (hasDeadEndBookmark(instr.getAddress())) {
                    hasDeadEndOnly = true;
                } else {
                    hasCandidate = true;
                    break;
                }
            }
            if (hasCandidate) {
                candidates.add(func);
            } else if (hasDeadEndOnly) {
                reviewKnown++;
            }
        }

        DecompilerCallback<SwitchResult> callback = new DecompilerCallback<SwitchResult>(
                currentProgram, d -> d.setOptions(new DecompileOptions())) {
            @Override
            public SwitchResult process(DecompileResults results, TaskMonitor m) {
                return classify(results);
            }
        };
        callback.setTimeout(DECOMP_TIMEOUT);

        ChunkingParallelDecompiler<SwitchResult> pd =
            ParallelDecompiler.createChunkingParallelDecompiler(callback, monitor);
        try {
            monitor.initialize(candidates.size());
            for (int i = 0; i < candidates.size() && !monitor.isCancelled();
                    i += CHUNK_SIZE) {
                List<Function> chunk =
                    candidates.subList(i, Math.min(i + CHUNK_SIZE, candidates.size()));
                List<SwitchResult> results = pd.decompileFunctions(chunk);
                results.sort(Comparator.comparing(r -> r.func.getEntryPoint()));
                for (SwitchResult r : results) {
                    applyResult(r);
                    monitor.incrementProgress(1);
                }
            }
        } catch (CancelledException e) {
            // Cancelled mid-run: still emit the partial manifest below.
        } finally {
            pd.dispose();
            callback.dispose();
        }

        StringBuilder out = new StringBuilder();
        out.append("{\"counts\":{")
           .append("\"scanned\":").append(scanned)
           .append(",\"unrecovered_funcs\":").append(unrecoveredFuncs)
           .append(",\"unrecovered_jumps\":").append(unrecoveredJumps)
           .append(",\"review_known\":").append(reviewKnown)
           .append(",\"decompile_failed\":").append(decompFailed)
           .append("},\"switches\":[").append(switchJson).append("]")
           .append(",\"switches_truncated\":")
           .append(unrecoveredJumps > switchDetail ? "true" : "false")
           .append(",\"failed\":[").append(failJson).append("]")
           .append(",\"failed_truncated\":")
           .append(decompFailed > failedDetail ? "true" : "false")
           .append("}");

        println(MARK_START);
        println(out.toString());
        println(MARK_END);
    }
}
"""
