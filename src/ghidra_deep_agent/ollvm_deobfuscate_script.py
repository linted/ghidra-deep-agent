"""The GhidrAssistMCP-side OLLVM control-flow-flattening (CFF) deobfuscator, as a
source string.

``SCRIPT_SOURCE`` is a **Java** GhidraScript (not code that runs in this process).
It is shipped verbatim to the running Ghidra via GhidrAssistMCP's ``scripts`` tool
and compiled/run there — see ``switch_tools.py`` (``deobfuscate_cff``). Keeping it
as a plain string means ruff/mypy only ever see a valid Python-3 string literal.
This is the sibling of ``find_unrecovered_switches_script.py`` /
``apply_switch_override_script.py``; read those for the shared design
(Java-not-Jython, JSON-between-markers).

What the script does, for ONE function (given by name or address as a script
argument, or the interactive cursor):

* Phase 1 — find each CFF dispatch site: an indirect ``br`` fed by
  ``ldr xT, [table, xIdx, LSL #3]`` with the table base recovered from
  ``adrp``+``add``. These are the flattening dispatchers Ghidra could not recover
  as jump tables ("Could not recover jumptable — Too many branches").
* Phase 2 — read the jump table and analyze each case block to find its real
  successor, seeing through the OLLVM index range-clamp
  (``cmp idx,#max`` / ``csel idx,#clamp,idx,gt``) and folding constant state
  indices, so each block resolves to an unconditional (or genuinely conditional)
  edge.
* Phase 3 — patch the indirect branches into direct ``b`` / ``b.cond`` (only the
  ``br`` is rewritten when work is interleaved before it), NOP-ing dead dispatch
  scaffolding, so the decompiler can recover normal control flow.

Modes (script arguments): ``dryrun`` (analyze and report, write nothing — the
default when driven headless/programmatically), ``apply`` (patch), and ``force``
(write patches even when their address ranges overlap). The result ends with a
single JSON object between the ``<<<OLLVM_DEOBF_JSON>>>`` /
``<<<END_OLLVM_DEOBF_JSON>>>`` markers; the human-readable phase report precedes
it.

The GhidraScript public class name must match the deployed file name — see
``_OLLVM_SCRIPT_NAME`` in ``switch_tools.py`` (``OllvmDeobfuscator.java``).
"""

MARK_START = "<<<OLLVM_DEOBF_JSON>>>"
MARK_END = "<<<END_OLLVM_DEOBF_JSON>>>"

SCRIPT_SOURCE: str = r"""//OllvmDeobfuscator.java
// Ghidra script: deobfuscate OLLVM-style control-flow flattening (CFF) in AARCH64.
//
// Ported from ollvm_decode.py, but works at the INSTRUCTION level because Ghidra
// cannot recover the CFF jump tables ("Could not recover jumptable - Too many
// branches"), which prevents the Pcode/HighFunction approach used by the Python
// original.
//
// THREE PHASES:
//   1. Discover CFF dispatch sites: scan for BR preceded by
//      LDR x<n>,[x<tbl>,x<n>,LSL #3]. Resolve table base via ADRP+ADD.
//   2. Disassemble case blocks. Analyze each to find its exit dispatch.
//      Fold opaque predicates (always-zero globals) to resolve CSEL.
//   3. Patch: replace indirect BR with direct B or B.cond + B. NOP the rest.
//
// Usage: place cursor inside a flattened function and run. Alternatively, pass
// the target as a script argument — either a function name (e.g. "FUN_00123456")
// or an address (e.g. "0x123456") — which is required when running headless or
// over MCP where there is no interactive cursor.
//
// DRY RUN: pass "dryrun" (or "dry-run"/"dry") as a script argument to analyze
// and report the planned patches WITHOUT modifying the listing; pass "apply"
// (or "patch"/"write") to force patching. With no argument, an interactive GUI
// session prompts, and a headless/programmatic session defaults to a safe dry
// run so nothing is mutated unless explicitly requested.
//
// FORCE: pass "force" to write patches even when their address ranges overlap a
// previous patch. By default overlapping patches are skipped (to avoid one
// transition clobbering another); "force" overrides that and writes anyway.
//
// @category Deobfuscation

import ghidra.app.script.GhidraScript;
import ghidra.app.plugin.assembler.Assembler;
import ghidra.app.plugin.assembler.Assemblers;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryAccessException;
import ghidra.program.model.scalar.Scalar;
import ghidra.program.model.lang.Register;
import java.util.*;

public class OllvmDeobfuscator extends GhidraScript {

    // A compact machine-readable summary is printed between these markers so a
    // programmatic caller (see switch_tools.deobfuscate_cff) can parse the
    // outcome deterministically; the human-readable phase report is emitted too.
    private static final String MANIFEST_START = "<<<OLLVM_DEOBF_JSON>>>";
    private static final String MANIFEST_END = "<<<END_OLLVM_DEOBF_JSON>>>";

    private Memory mem;
    private Listing listing;
    private final Set<Long> patchedAddrs = new HashSet<>();
    private boolean dryRun;
    private boolean force;

    @Override
    protected void run() throws Exception {
        mem = currentProgram.getMemory();
        listing = currentProgram.getListing();
        dryRun = resolveDryRun();
        force = hasArg("force", "--force");

        Function func = resolveTargetFunction();
        if (func == null) {
            println("No target function. Pass a function name or address as a script "
                    + "argument, or place the cursor inside a function. "
                    + "(cursor was at " + currentAddress + ")");
            emitManifest("no_function", null, 0, 0, 0);
            return;
        }

        println("=== OllvmDeobfuscator ===");
        println("Function: " + func.getName() + " @ " + func.getEntryPoint());
        println("Mode: " + (dryRun ? "DRY RUN (no changes will be written)" : "PATCH")
                + (force ? " [FORCE: overlapping patches will be written anyway]" : ""));
        println("");

        // Phase 1: find dispatch sites
        println("--- Phase 1: Discovering CFF dispatch sites ---");
        List<DispatchSite> sites = findDispatchSites(func);
        if (sites.isEmpty()) {
            println("No CFF dispatch patterns found. Aborting.");
            emitManifest("no_dispatch", func, 0, 0, 0);
            return;
        }
        for (DispatchSite ds : sites) {
            resolveTable(ds, func);
            println("  BR@" + hex(ds.brAddr) + " table=" + hex(ds.tableAddr)
                    + " x" + ds.tableReg + " entries=" + ds.caseTargets.length
                    + " maxIdx=" + ds.maxIndex + " clamp=" + ds.clampIndex);
        }
        println("");

        // Disassemble case targets that aren't yet instructions
        println("--- Phase 1b: Ensuring case blocks are disassembled ---");
        int newInsts = 0;
        for (DispatchSite ds : sites) {
            for (long tgt : ds.caseTargets) {
                newInsts += disasmAt(toAddr(tgt));
            }
        }
        println("  Disassembled " + newInsts + " new code ranges.");
        println("");

        // Phase 2: analyze each case block
        println("--- Phase 2: Analyzing case blocks ---");
        Map<Long, BlockTransition> transitions = new LinkedHashMap<>();
        for (DispatchSite ds : sites) {
            analyzeBlocks(ds, func, transitions);
        }
        for (Map.Entry<Long, BlockTransition> e : transitions.entrySet()) {
            BlockTransition bt = e.getValue();
            String type = bt.conditional ? "COND" : "UNCOND";
            String tgt;
            if (bt.trueTarget < 0 && bt.falseTarget < 0) tgt = "RET/EXIT";
            else if (bt.conditional) tgt = hex(bt.trueTarget) + " / " + hex(bt.falseTarget);
            else tgt = hex(bt.trueTarget);
            println("  " + hex(e.getKey()) + " [" + type + "] -> " + tgt);
        }
        println("  Total: " + transitions.size() + " transitions");
        println("");

        // Phase 3: patch (or, in dry-run, report the planned patches)
        println("--- Phase 3: " + (dryRun ? "Planned patches (dry run)" : "Patching") + " ---");
        int patched = 0;
        for (Map.Entry<Long, BlockTransition> e : transitions.entrySet()) {
            try {
                if (patchTransition(e.getValue())) patched++;
            } catch (Exception ex) {
                println("  FAILED at " + hex(e.getKey()) + ": " + ex.getMessage());
            }
        }
        println("  " + (dryRun ? "Would patch " : "Patched ") + patched + "/" + transitions.size());
        if (dryRun) {
            println("\n=== Dry run complete. No changes written. "
                    + "Re-run with \"apply\" to patch. ===");
        } else {
            println("\n=== Done. Re-decompile to verify. ===");
        }
        emitManifest("ok", func, sites.size(), transitions.size(), patched);
    }

    /**
     * Print the compact JSON result summary between the manifest markers. In dry
     * run, {@code patched} is the count that WOULD be patched.
     */
    private void emitManifest(String status, Function func, int sites,
            int transitions, int patched) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"status\":\"").append(status).append("\"");
        sb.append(",\"function\":")
          .append(func == null ? "null" : "\"" + jsonEsc(func.getName()) + "\"");
        sb.append(",\"entry\":")
          .append(func == null ? "null"
                  : "\"" + hex(func.getEntryPoint().getOffset()) + "\"");
        sb.append(",\"mode\":\"").append(dryRun ? "dryrun" : "patch").append("\"");
        sb.append(",\"force\":").append(force ? "true" : "false");
        sb.append(",\"dispatch_sites\":").append(sites);
        sb.append(",\"transitions\":").append(transitions);
        sb.append(",\"patched\":").append(patched);
        sb.append("}");
        println(MANIFEST_START);
        println(sb.toString());
        println(MANIFEST_END);
    }

    private String jsonEsc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /**
     * Decide dry-run vs. patch. Script args win ("dryrun"/"dry-run"/"dry" ->
     * true; "apply"/"patch"/"write" -> false). With no directive, an interactive
     * GUI session asks; anything else (headless / programmatic, e.g. driven over
     * MCP with no args) defaults to a safe dry run so the listing is never
     * mutated unless explicitly requested.
     */
    private boolean resolveDryRun() {
        for (String a : getScriptArgs()) {
            String s = a.trim().toLowerCase();
            if (s.equals("dryrun") || s.equals("dry-run") || s.equals("dry")) return true;
            if (s.equals("apply") || s.equals("patch") || s.equals("write")) return false;
        }
        try {
            if (!isRunningHeadless()) {
                return askYesNo("OllvmDeobfuscator",
                        "Dry run? (Yes = analyze and report only; No = patch the listing)");
            }
        } catch (Exception e) {
            // askYesNo throws when no interactive UI is attached; fall through.
        }
        return true; // safe default: never mutate unless explicitly told to
    }

    /** True if any script argument (case-insensitive) equals one of {@code names}. */
    private boolean hasArg(String... names) {
        for (String a : getScriptArgs()) {
            String s = a.trim().toLowerCase();
            for (String n : names) {
                if (s.equals(n)) return true;
            }
        }
        return false;
    }

    /** Mode-selecting flags that must not be mistaken for a target token. */
    private boolean isDirective(String low) {
        return low.equals("dryrun") || low.equals("dry-run") || low.equals("dry")
                || low.equals("apply") || low.equals("patch") || low.equals("write")
                || low.equals("force") || low.equals("--force");
    }

    /**
     * Pick the function to work on. A non-directive script argument is treated
     * as a target: first as a function name, then as a hex address. Falls back
     * to the function under the interactive cursor (currentAddress).
     */
    private Function resolveTargetFunction() {
        for (String a : getScriptArgs()) {
            String s = a.trim();
            if (isDirective(s.toLowerCase())) continue; // mode flag, not a target
            Function f = functionFromToken(s);
            if (f != null) return f;
            println("  [warn] Argument \"" + s + "\" did not resolve to a function.");
        }
        Function f = getFunctionContaining(currentAddress);
        if (f == null) f = getFunctionAt(currentAddress);
        return f;
    }

    private Function functionFromToken(String s) {
        List<Function> named = getGlobalFunctions(s);
        if (named != null && !named.isEmpty()) return named.get(0);
        String h = (s.startsWith("0x") || s.startsWith("0X")) ? s.substring(2) : s;
        if (h.matches("[0-9a-fA-F]+")) {
            try {
                Address addr = toAddr(Long.parseLong(h, 16));
                Function f = getFunctionContaining(addr);
                if (f == null) f = getFunctionAt(addr);
                return f;
            } catch (Exception e) {
                // not a usable address; fall through
            }
        }
        return null;
    }

    //=========================================================================
    // Types
    //=========================================================================

    static class DispatchSite {
        long brAddr;
        long tableAddr;
        int  tableReg;
        int  indexReg;
        int  maxIndex   = -1;
        int  clampIndex = -1;
        long[] caseTargets;
    }

    static class BlockTransition {
        boolean conditional;
        long trueTarget  = -1;
        long falseTarget = -1;
        long patchStart;
        long brAddr;
        String cond;
    }

    //=========================================================================
    // Phase 1: Find dispatch sites
    //=========================================================================

    private List<DispatchSite> findDispatchSites(Function func) throws Exception {
        List<DispatchSite> results = new ArrayList<>();
        Instruction inst = getFirstInst(func);
        while (inst != null && isInFunc(inst, func)) {
            if (monitor.isCancelled()) break;
            if ("br".equals(inst.getMnemonicString())) {
                DispatchSite ds = parseDispatch(inst, func);
                if (ds != null) results.add(ds);
            }
            inst = inst.getNext();
        }
        return results;
    }

    private DispatchSite parseDispatch(Instruction brInst, Function func) throws Exception {
        int brReg = getRegNum(brInst, 0);
        if (brReg < 0) return null;

        // Walk backwards to find LDR with scaled register index
        Instruction ldrInst = null;
        int tableReg = -1;
        Instruction cur = brInst.getPrevious();
        for (int i = 0; i < 15 && cur != null; i++) {
            if ("ldr".equals(cur.getMnemonicString()) && ldrInst == null) {
                int[] bi = extractBaseIndexRegs(cur);
                if (bi != null && bi[1] == brReg) {
                    ldrInst = cur;
                    tableReg = bi[0];
                    break;
                }
            }
            cur = cur.getPrevious();
        }
        if (ldrInst == null || tableReg < 0) return null;

        DispatchSite ds = new DispatchSite();
        ds.brAddr   = brInst.getAddress().getOffset();
        ds.tableReg = tableReg;
        ds.indexReg = brReg;

        // Find CMP and CSEL before the LDR
        cur = ldrInst.getPrevious();
        for (int i = 0; i < 10 && cur != null; i++) {
            String mn = cur.getMnemonicString();
            if (mn.equals("cmp") && ds.maxIndex < 0) {
                Scalar s = getScalar(cur, 1);
                if (s != null) ds.maxIndex = (int) s.getValue();
            }
            if (mn.equals("csel")) {
                Instruction p = cur.getPrevious();
                for (int j = 0; j < 4 && p != null; j++) {
                    if ("mov".equals(p.getMnemonicString())) {
                        Scalar s = getScalar(p, 1);
                        if (s != null) {
                            ds.clampIndex = (int) s.getValue();
                            break;
                        }
                    }
                    p = p.getPrevious();
                }
            }
            cur = cur.getPrevious();
        }

        ds.tableAddr = resolveTableBaseReg(func, tableReg, brInst);
        if (ds.tableAddr < 0) {
            println("  [warn] Cannot resolve table base x" + tableReg
                    + " for BR@" + hex(ds.brAddr));
            return null;
        }
        return ds;
    }

    //=========================================================================
    // Phase 1b: Read jump table entries
    //=========================================================================

    private void resolveTable(DispatchSite ds, Function func) throws MemoryAccessException {
        long fStart = func.getEntryPoint().getOffset();
        long fEnd = fStart + 0x10000;
        if (func.getBody().getNumAddresses() > 0) {
            fEnd = func.getBody().getMaxAddress().getOffset();
        }

        int maxScan = ds.maxIndex > 0 ? ds.maxIndex + 2 : 64;
        List<Long> targets = new ArrayList<>();
        for (int i = 0; i < maxScan; i++) {
            long ea = ds.tableAddr + (long) i * 8;
            try {
                long ptr = mem.getLong(toAddr(ea));
                if (ptr >= fStart && ptr <= fEnd + 4) {
                    targets.add(ptr);
                } else {
                    if (!targets.isEmpty()) break;
                }
            } catch (MemoryAccessException e) {
                break;
            }
        }
        ds.caseTargets = new long[targets.size()];
        for (int i = 0; i < targets.size(); i++) {
            ds.caseTargets[i] = targets.get(i);
        }
    }

    private int disasmAt(Address addr) {
        Instruction inst = listing.getInstructionAt(addr);
        if (inst != null) return 0;
        if (dryRun) {
            // Disassembly mutates the program; a dry run stays read-only.
            // Blocks that aren't already code simply won't be analyzed.
            return 0;
        }
        try {
            disassemble(addr);
            return 1;
        } catch (Exception e) {
            return 0;
        }
    }

    //=========================================================================
    // Phase 2: Analyze case blocks
    //=========================================================================

    private void analyzeBlocks(DispatchSite ds, Function func,
                               Map<Long, BlockTransition> out) throws Exception {
        Set<Long> done = new HashSet<>();
        for (int ci = 0; ci < ds.caseTargets.length; ci++) {
            long blockAddr = ds.caseTargets[ci];
            if (done.contains(blockAddr)) continue;
            done.add(blockAddr);
            BlockTransition bt = scanBlock(blockAddr, ds, func);
            if (bt != null) {
                out.put(blockAddr, bt);
            } else {
                println("  [warn] No terminator found for block at " + hex(blockAddr));
            }
        }
    }

    private BlockTransition scanBlock(long start, DispatchSite ds, Function func) throws Exception {
        Instruction inst = listing.getInstructionAt(toAddr(start));
        if (inst == null) inst = listing.getInstructionAfter(toAddr(start));
        if (inst == null) {
            disasmAt(toAddr(start));
            inst = listing.getInstructionAt(toAddr(start));
            if (inst == null) return null;
        }

        Map<Integer, Long> consts = new HashMap<>();
        Set<Integer> opaqueZero = new HashSet<>();
        Map<Long, Long> stackSlots = new HashMap<>(); // stack offset -> value

        for (int i = 0; i < 400 && inst != null; i++) {
            if (monitor.isCancelled()) return null;
            String mn = inst.getMnemonicString();
            long addr = inst.getAddress().getOffset();

            // Track constants
            if (mn.equals("mov")) trackMovConst(inst, consts);
            if (mn.equals("orr")) trackOrrConst(inst, consts);
            if (mn.equals("ldr")) trackOpaqueLoad(inst, opaqueZero, consts);
            if (mn.equals("mul") || mn.equals("msub") || mn.equals("madd"))
                trackMulZero(inst, opaqueZero, consts);

            // Track stack spills: STR x<n>, [sp, #off]
            if (mn.equals("str")) {
                int rd = getRegNum(inst, 0);
                Long off = getStackOffset(inst);
                if (rd >= 0 && off != null && consts.containsKey(rd)) {
                    stackSlots.put(off, consts.get(rd));
                }
            }
            // Track stack reloads: LDR x<n>, [sp, #off]
            if (mn.equals("ldr")) {
                int rd = getRegNum(inst, 0);
                Long off = getStackOffset(inst);
                if (rd >= 0 && off != null && stackSlots.containsKey(off)) {
                    consts.put(rd, stackSlots.get(off));
                }
            }

            // Detect exit dispatch: CSEL -> LDR [table] -> BR
            if (mn.equals("csel")) {
                Instruction n1 = inst.getNext();
                if (n1 != null && "ldr".equals(n1.getMnemonicString())) {
                    int[] bi = extractBaseIndexRegs(n1);
                    if (bi != null && bi[0] == ds.tableReg) {
                        Instruction n2 = n1.getNext();
                        if (n2 != null && "br".equals(n2.getMnemonicString())) {
                            long cmpAddr = findPrevCmp(inst, start);
                            return resolveExit(cmpAddr, inst, n2, ds, consts, opaqueZero);
                        }
                    }
                }
            }

            // LDR [table] -> BR without CSEL (unconditional state update)
            if (mn.equals("ldr")) {
                int[] bi = extractBaseIndexRegs(inst);
                if (bi != null && bi[0] == ds.tableReg) {
                    Instruction n1 = inst.getNext();
                    if (n1 != null && "br".equals(n1.getMnemonicString())) {
                        Long idx = consts.get(bi[1]);
                        if (idx != null) {
                            BlockTransition bt = new BlockTransition();
                            bt.conditional = false;
                            bt.trueTarget = resolveCase(ds, idx);
                            bt.patchStart = addr;
                            bt.brAddr = n1.getAddress().getOffset();
                            return bt;
                        }
                        // Index register value unknown — state may have been
                        // spilled to stack and reloaded. Log for diagnosis.
                        println("  [warn] Unknown index reg x" + bi[1]
                                + " at LDR " + hex(addr)
                                + " (state may be reloaded from stack)");
                        // Try to find CMP+CSEL before this LDR (clamp pattern)
                        // and resolve as opaque if applicable
                        // Otherwise leave as-is (can't resolve)
                    }
                }
            }

            // Already-resolved direct branch
            if (mn.equals("b") && i > 0) {
                Address[] flows = inst.getFlows();
                if (flows.length > 0) {
                    BlockTransition bt = new BlockTransition();
                    bt.conditional = false;
                    bt.trueTarget = flows[0].getOffset();
                    bt.patchStart = addr;
                    bt.brAddr = addr;
                    return bt;
                }
            }

            // Real conditional branch. getFlows() excludes the fall-through,
            // so cbz/cbnz/tbz/tbnz report exactly one flow (the taken target);
            // the not-taken target is the fall-through.
            if (mn.equals("cbz") || mn.equals("cbnz") || mn.equals("tbz") || mn.equals("tbnz")) {
                Address[] flows = inst.getFlows();
                Address fallThrough = inst.getFallThrough();
                if (flows.length >= 1 && fallThrough != null) {
                    BlockTransition bt = new BlockTransition();
                    bt.conditional = true;
                    bt.trueTarget = flows[0].getOffset();
                    bt.falseTarget = fallThrough.getOffset();
                    bt.patchStart = addr;
                    bt.brAddr = addr;
                    bt.cond = mn;
                    return bt;
                }
            }

            // RET
            if (mn.equals("ret")) {
                BlockTransition bt = new BlockTransition();
                bt.trueTarget = -1;
                bt.falseTarget = -1;
                bt.patchStart = addr;
                bt.brAddr = addr;
                return bt;
            }

            inst = inst.getNext();
            if (inst == null) break;
        }
        return null;
    }

    private long findPrevCmp(Instruction from, long lowerBound) {
        Instruction cur = from.getPrevious();
        for (int i = 0; i < 10 && cur != null; i++) {
            if (cur.getAddress().getOffset() < lowerBound) break;
            if ("cmp".equals(cur.getMnemonicString())) {
                return cur.getAddress().getOffset();
            }
            cur = cur.getPrevious();
        }
        return -1;
    }

    private BlockTransition resolveExit(long cmpAddr, Instruction cselInst,
            Instruction brInst, DispatchSite ds,
            Map<Integer, Long> consts, Set<Integer> opaqueZero) {

        // If no CSEL (direct LDR+BR with known index), handle separately
        if (cselInst == null || !"csel".equals(cselInst.getMnemonicString())) {
            return null;
        }

        // CSEL xD, xA, xB, cond  -- operands: [0]=xD [1]=xA [2]=xB [3]=cond
        int regA = getRegNum(cselInst, 1); // selected when cond is true
        int regB = getRegNum(cselInst, 2); // selected when cond is false
        String cond = getCondString(cselInst);

        // Index range-clamp, NOT a real branch:
        //   mov  xIdx, #<state>        ; real next index (a constant, in range)
        //   cmp  xIdx, #<maxIndex>     ; is the index out of range?
        //   csel xIdx, #<clampIndex>, xIdx, gt   ; clamp if so, else keep it
        //   ldr  xT, [table, xIdx, LSL #3] ; br xT
        // In normal flow the index is always in range, so the guard is dead. The
        // real next index is the operand kept when the range check is FALSE
        // (regB for a gt-style condition). Collapse to an unconditional jump and
        // patch ONLY the BR — OLLVM interleaves real work (e.g. struct-zeroing
        // stores) between the LDR and the BR, which must be preserved.
        if (isClampCsel(cmpAddr, ds, cond)) {
            Long realIdx = consts.get(regB);
            if (realIdx == null) {
                println("  [warn] Clamp dispatch with unknown index at "
                        + hex(cselInst.getAddress().getOffset())
                        + " (state may be reloaded from stack) — cannot resolve.");
                return null;
            }
            BlockTransition bt = new BlockTransition();
            bt.conditional = false;
            bt.trueTarget = resolveCase(ds, realIdx);
            bt.patchStart = brInst.getAddress().getOffset();
            bt.brAddr = brInst.getAddress().getOffset();
            return bt;
        }

        Long valA = consts.get(regA);
        Long valB = consts.get(regB);

        if (valA == null && valB == null) {
            println("  [warn] Both CSEL inputs unknown at " + hex(cselInst.getAddress().getOffset()));
            return null;
        }
        if (valA == null) valA = -1L;
        if (valB == null) valB = -1L;

        // Check if the CMP is opaque (fed by zero-derived values)
        boolean opaque = isOpaqueCmp(cmpAddr, opaqueZero, consts);

        BlockTransition bt = new BlockTransition();
        bt.patchStart = cmpAddr >= 0 ? cmpAddr : cselInst.getAddress().getOffset();
        bt.brAddr = brInst.getAddress().getOffset();

        if (opaque) {
            // Opaque: condition is always fixed.
            // ASSUMPTION: the opaque CMP operands both fold to 0, so the CMP
            // yields EQ (0 == 0). This holds for the DAT_0030ca70*DAT_0030ca70
            // always-zero patterns this script targets, but is NOT valid for a
            // general known-constant CMP — a non-zero constant pair sets the
            // flags differently and would flip the selection below.
            // CSEL selects xA when cond matches, xB otherwise.
            long selected;
            if (cond.equals("eq") || cond.equals("le") || cond.equals("ge")
                    || cond.equals("ls") || cond.equals("hs") || cond.equals("cc")) {
                selected = valA; // EQ-like conditions are true
            } else {
                selected = valB; // NE-like conditions are false
            }
            bt.conditional = false;
            bt.trueTarget = resolveCase(ds, selected);
        } else {
            // Real conditional -- both targets live
            bt.conditional = true;
            bt.trueTarget = resolveCase(ds, valA);
            bt.falseTarget = resolveCase(ds, valB);
            bt.cond = cond;
        }
        return bt;
    }

    private boolean isOpaqueCmp(long cmpAddr, Set<Integer> opaqueZero,
            Map<Integer, Long> consts) {
        if (cmpAddr < 0) return false;
        Instruction cmpInst = listing.getInstructionAt(toAddr(cmpAddr));
        if (cmpInst == null) return false;
        int r0 = getRegNum(cmpInst, 0);
        int r1 = getRegNum(cmpInst, 1);
        if (opaqueZero.contains(r0) || opaqueZero.contains(r1)) return true;
        Long v0 = consts.get(r0);
        Long v1 = consts.get(r1);
        // If both inputs are known constants, it's effectively opaque
        if (v0 != null && v1 != null) return true;
        return false;
    }

    /**
     * Is the dispatch CSEL actually the CFF index range-clamp rather than a real
     * data-dependent branch? Signature: the guarding CMP compares against the
     * table's max index ({@code cmp idx, #maxIndex}) and the CSEL condition is an
     * out-of-range test ({@code gt}/{@code hi}/{@code ge}/{@code hs}/{@code cs}).
     * Such a guard is dead in normal flow (the index is always in range).
     */
    private boolean isClampCsel(long cmpAddr, DispatchSite ds, String cond) {
        if (cmpAddr < 0 || ds.maxIndex < 0) return false;
        if (!(cond.equals("gt") || cond.equals("hi") || cond.equals("ge")
                || cond.equals("hs") || cond.equals("cs"))) {
            return false;
        }
        Instruction cmpInst = listing.getInstructionAt(toAddr(cmpAddr));
        if (cmpInst == null) return false;
        Scalar s = getScalar(cmpInst, 1);
        return s != null && (int) s.getValue() == ds.maxIndex;
    }

    private long resolveCase(DispatchSite ds, long idx) {
        int i = (int) idx;
        if (i >= 0 && i < ds.caseTargets.length) return ds.caseTargets[i];
        if (ds.clampIndex >= 0 && ds.clampIndex < ds.caseTargets.length)
            return ds.caseTargets[ds.clampIndex];
        return -1;
    }

    //=========================================================================
    // Phase 3: Patching
    //=========================================================================

    /**
     * Reserve [start, end] (inclusive, 4-byte steps) so independent transitions
     * cannot clobber one another. Returns false — and reserves nothing — if the
     * range is invalid (start > end) or any address in it was already patched.
     */
    private boolean reserveRange(long start, long end) {
        if (start < 0 || end < 0 || start > end) return false;
        boolean overlap = false;
        for (long a = start; a <= end; a += 4) {
            if (patchedAddrs.contains(a)) { overlap = true; break; }
        }
        if (overlap && !force) return false;
        if (overlap) {
            println("  [force] range " + hex(start) + "-" + hex(end)
                    + " overlaps a previous patch — writing anyway");
        }
        for (long a = start; a <= end; a += 4) {
            patchedAddrs.add(a);
        }
        return true;
    }

    private boolean patchTransition(BlockTransition bt) throws Exception {
        if (bt.trueTarget < 0 && bt.falseTarget < 0) return false; // RET

        if (!bt.conditional) {
            if (bt.trueTarget < 0) {
                println("  [warn] Skipping unconditional patch at " + hex(bt.patchStart)
                        + ": unresolved target");
                return false;
            }
            return patchDirect(bt.patchStart, bt.brAddr, bt.trueTarget);
        } else {
            if (bt.trueTarget < 0 || bt.falseTarget < 0) {
                println("  [warn] Skipping conditional patch at " + hex(bt.patchStart)
                        + ": unresolved target(s)");
                return false;
            }
            return patchCond(bt.patchStart, bt.brAddr, bt.trueTarget, bt.falseTarget, bt.cond);
        }
    }

    /**
     * Unconditional: B target at patchStart, NOP everything up to brAddr.
     */
    private boolean patchDirect(long patchStart, long brAddr, long target) {
        if (!reserveRange(patchStart, brAddr)) {
            println("  patchDirect skipped: range " + hex(patchStart) + "-" + hex(brAddr)
                    + " invalid or overlaps a previous patch");
            return false;
        }
        try {
            emit(toAddr(patchStart), "b " + hex(target));
            nopRange(patchStart + 4, brAddr);
            return true;
        } catch (Exception e) {
            println("  patchDirect failed: " + e.getMessage());
            return false;
        }
    }

    /**
     * Conditional from CSEL: B.cond trueTarget; B falseTarget; NOP rest.
     * For cbz/cbnz/tbz/tbnz: already direct branches, leave as-is.
     */
    private boolean patchCond(long patchStart, long brAddr,
            long trueTarget, long falseTarget, String cond) {
        try {
            if (cond != null && (cond.startsWith("cb") || cond.startsWith("tb"))) {
                return false; // already direct, skip
            }
            if (!reserveRange(patchStart, brAddr)) {
                println("  patchCond skipped: range " + hex(patchStart) + "-" + hex(brAddr)
                        + " invalid or overlaps a previous patch");
                return false;
            }
            String bc = cond != null ? cond : "eq";
            emit(toAddr(patchStart), "b." + bc + " " + hex(trueTarget));
            emit(toAddr(patchStart + 4), "b " + hex(falseTarget));
            nopRange(patchStart + 8, brAddr);
            return true;
        } catch (Exception e) {
            println("  patchCond failed: " + e.getMessage());
            return false;
        }
    }

    //=========================================================================
    // Instruction analysis helpers
    //=========================================================================

    private int getRegNum(Instruction inst, int opIdx) {
        Object[] objs = inst.getOpObjects(opIdx);
        for (Object o : objs) {
            if (o instanceof Register) return regToNum((Register) o);
        }
        return -1;
    }

    private int regToNum(Register reg) {
        String name = reg.getName();
        if (name.startsWith("x") || name.startsWith("w")) {
            try { return Integer.parseInt(name.substring(1)); }
            catch (NumberFormatException e) { return -1; }
        }
        if (name.equals("sp")) return 31;
        if (name.equals("xzr") || name.equals("wzr")) return -2;
        return -1;
    }

    private Scalar getScalar(Instruction inst, int opIdx) {
        Object[] objs = inst.getOpObjects(opIdx);
        for (Object o : objs) {
            if (o instanceof Scalar) return (Scalar) o;
        }
        return null;
    }

    /**
     * Extract [baseReg, indexReg] from LDR x<n>, [x<base>, x<idx>, LSL #0x3]
     */
    private int[] extractBaseIndexRegs(Instruction inst) {
        // Try operand objects first
        List<Register> regs = new ArrayList<>();
        for (int i = 0; i < inst.getNumOperands(); i++) {
            Object[] objs = inst.getOpObjects(i);
            for (Object o : objs) {
                if (o instanceof Register) regs.add((Register) o);
            }
        }
        if (regs.size() >= 3) {
            int dest = regToNum(regs.get(0));
            int base = regToNum(regs.get(1));
            int idx  = regToNum(regs.get(2));
            if (base >= 0 && idx >= 0 && dest >= 0) return new int[]{base, idx};
        }
        // Fallback: parse string representation
        String s = inst.toString();
        int bracket = s.indexOf('[');
        int bracketEnd = s.indexOf(']');
        if (bracket < 0 || bracketEnd < 0) return null;
        String inside = s.substring(bracket + 1, bracketEnd);
        String[] parts = inside.split(",");
        if (parts.length < 2) return null;
        int base = parseRegName(parts[0].trim());
        int idx = parseRegName(parts[1].trim());
        if (base >= 0 && idx >= 0) return new int[]{base, idx};
        return null;
    }

    private int parseRegName(String s) {
        s = s.trim();
        int space = s.indexOf(' ');
        if (space > 0) s = s.substring(0, space);
        int star = s.indexOf('*');
        if (star > 0) s = s.substring(0, star).trim();
        if (s.startsWith("x") || s.startsWith("w")) {
            try { return Integer.parseInt(s.substring(1)); }
            catch (NumberFormatException e) { return -1; }
        }
        return -1;
    }

    /**
     * Resolve table base for a register by scanning prologue for ADRP+ADD.
     */
    private long resolveTableBaseReg(Function func, int regNum, Instruction beforeInst) throws Exception {
        Instruction inst = getFirstInst(func);
        long adrpVal = -1;

        // Also scan backwards from the BR site in case prologue is truncated
        // First try forward from function entry
        for (int i = 0; i < 100 && inst != null && isInFunc(inst, func); i++) {
            if (monitor.isCancelled()) return -1;
            String mn = inst.getMnemonicString();
            int r0 = getRegNum(inst, 0);

            if (mn.equals("adrp") && r0 == regNum) {
                Object[] objs = inst.getOpObjects(1);
                if (objs != null && objs.length > 0 && objs[0] instanceof Address) {
                    adrpVal = ((Address) objs[0]).getOffset();
                } else {
                    Scalar s = getScalar(inst, 1);
                    if (s != null) adrpVal = s.getValue();
                }
            }

            if (mn.equals("add") && r0 == regNum && adrpVal >= 0) {
                int r1 = getRegNum(inst, 1);
                if (r1 == regNum) {
                    Scalar s = getScalar(inst, 2);
                    if (s != null) return adrpVal + s.getValue();
                }
            }

            if (inst.getAddress().equals(beforeInst.getAddress())) break;
            inst = inst.getNext();
        }

        // If not found forward, try backward from beforeInst
        if (adrpVal < 0) {
            Instruction cur = beforeInst;
            for (int i = 0; i < 200 && cur != null; i++) {
                String mn = cur.getMnemonicString();
                int r0 = getRegNum(cur, 0);
                if (mn.equals("add") && r0 == regNum) {
                    int r1 = getRegNum(cur, 1);
                    if (r1 == regNum) {
                        // Found the ADD; now find the preceding ADRP
                        Instruction prev = cur.getPrevious();
                        for (int j = 0; j < 30 && prev != null; j++) {
                            if ("adrp".equals(prev.getMnemonicString())) {
                                int ar = getRegNum(prev, 0);
                                if (ar == regNum) {
                                    Object[] objs = prev.getOpObjects(1);
                                    if (objs != null && objs.length > 0 && objs[0] instanceof Address) {
                                        long av = ((Address) objs[0]).getOffset();
                                        Scalar s = getScalar(cur, 2);
                                        if (s != null) return av + s.getValue();
                                    }
                                }
                            }
                            prev = prev.getPrevious();
                        }
                    }
                }
                cur = cur.getPrevious();
            }
        }

        // No complete ADRP+ADD pair matched. Returning the bare ADRP page base
        // here would hand back an address missing its low 12 bits, which reads
        // as a valid table and produces garbage targets. Fail instead.
        return -1;
    }

    //=========================================================================
    // Constant propagation
    //=========================================================================

    private void trackMovConst(Instruction inst, Map<Integer, Long> consts) {
        int rd = getRegNum(inst, 0);
        if (rd < 0) return;
        Scalar s = getScalar(inst, 1);
        if (s != null) {
            consts.put(rd, s.getValue());
        } else {
            int rs = getRegNum(inst, 1);
            if (rs >= 0 && consts.containsKey(rs)) {
                consts.put(rd, consts.get(rs));
            } else {
                consts.remove(rd);
            }
        }
    }

    private void trackOrrConst(Instruction inst, Map<Integer, Long> consts) {
        // ORR xD, xZR, #imm is how MOV imm is sometimes encoded
        int rd = getRegNum(inst, 0);
        int rs = getRegNum(inst, 1);
        if (rd < 0) return;
        if (rs == -2) { // xzr/wzr
            Scalar s = getScalar(inst, 2);
            if (s != null) consts.put(rd, s.getValue());
        }
    }

    private void trackOpaqueLoad(Instruction inst, Set<Integer> opaqueZero,
            Map<Integer, Long> consts) {
        int rd = getRegNum(inst, 0);
        if (rd < 0) return;
        String s = inst.toString();
        // Check for loads from opaque-predicate offsets [xN, #0xa70] or [xN, #0xa74]
        if (s.contains("0xa70") || s.contains("0xa74")) {
            opaqueZero.add(rd);
            consts.put(rd, 0L);
        }
    }

    private void trackMulZero(Instruction inst, Set<Integer> opaqueZero,
            Map<Integer, Long> consts) {
        int rd = getRegNum(inst, 0);
        int ra = getRegNum(inst, 1);
        int rb = getRegNum(inst, 2);
        if (rd < 0) return;
        if (opaqueZero.contains(ra) || opaqueZero.contains(rb)) {
            opaqueZero.add(rd);
            consts.put(rd, 0L);
        }
    }

    private String getCondString(Instruction cselInst) {
        int n = cselInst.getNumOperands();
        if (n < 1) return "eq";
        String rep = cselInst.getDefaultOperandRepresentation(n - 1);
        return rep.toLowerCase().trim();
    }

    /**
     * Extract the stack offset from a STR/LDR with [sp, #offset] addressing.
     * Returns null if not a stack-relative access.
     */
    private Long getStackOffset(Instruction inst) {
        String s = inst.toString();
        int bracket = s.indexOf('[');
        int bracketEnd = s.indexOf(']');
        if (bracket < 0 || bracketEnd < 0) return null;
        String inside = s.substring(bracket + 1, bracketEnd).trim();
        // Match patterns: [sp, #0x90], [x29, #-0x8], [sp]
        // For CFF state tracking, we care about [sp, #off]
        String[] parts = inside.split(",");
        if (parts.length < 1) return null;
        String baseStr = parts[0].trim();
        if (!baseStr.equals("sp") && !baseStr.equals("x29")) return null;
        if (parts.length < 2) return 0L; // [sp] with no offset
        String offStr = parts[1].trim();
        if (offStr.startsWith("#")) offStr = offStr.substring(1);
        try {
            // Handle hex (0x...) and decimal
            if (offStr.startsWith("0x") || offStr.startsWith("-0x")) {
                boolean neg = offStr.startsWith("-");
                if (neg) offStr = offStr.substring(1);
                long val = Long.parseLong(offStr, 16);
                // Use the base register name + offset as a unique key
                // Since we use baseStr in the key, collisions are unlikely
                long key = baseStr.equals("sp") ? val : val + 0x100000;
                if (neg) key = -key;
                return key;
            }
            boolean neg = offStr.startsWith("-");
            if (neg) offStr = offStr.substring(1);
            long val = Long.parseLong(offStr);
            long key = baseStr.equals("sp") ? val : val + 0x100000;
            if (neg) key = -key;
            return key;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    //=========================================================================
    // Utility
    //=========================================================================

    private String hex(long v) {
        if (v < 0) return "INVALID";
        return "0x" + Long.toHexString(v);
    }

    private Instruction getFirstInst(Function func) {
        Instruction inst = listing.getInstructionAt(func.getEntryPoint());
        if (inst == null) inst = listing.getInstructionAfter(func.getEntryPoint());
        return inst;
    }

    private boolean isInFunc(Instruction inst, Function func) {
        return func.getBody().contains(inst.getAddress());
    }

    private void setCode(Address addr, String asm) throws Exception {
        Assembler a = Assemblers.getAssembler(currentProgram);
        a.assembleLine(addr, asm);
    }

    /** Assemble one instruction, or just report it when in dry-run. */
    private void emit(Address addr, String asm) throws Exception {
        if (dryRun) {
            println("    [dry] " + hex(addr.getOffset()) + ": " + asm);
        } else {
            setCode(addr, asm);
        }
    }

    /** NOP the inclusive [start, end] range, or report the count in dry-run. */
    private void nopRange(long start, long end) throws Exception {
        int count = 0;
        for (long a = start; a <= end; a += 4) {
            if (!dryRun) setCode(toAddr(a), "nop");
            count++;
        }
        if (dryRun && count > 0) {
            println("    [dry] " + hex(start) + ".." + hex(end) + ": nop x" + count);
        }
    }
}
"""
