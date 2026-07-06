"""The GhidrAssistMCP-side prototype-recovery script, as a source string.

``SCRIPT_SOURCE`` is a **Java** GhidraScript, not code that executes in this
process. It is shipped verbatim to the running Ghidra via GhidrAssistMCP's
``scripts`` tool and compiled/run there — see ``prototype_tools.py``. Keeping it
as a plain string means ruff/mypy only ever see a valid Python-3 string literal.

Java (not Python/Jython) because the target instances do not necessarily launch
with PyGhidra — the ``scripts`` provider reports "Python is not available" unless
Ghidra is started in PyGhidra mode — whereas Java's ``GhidraScriptProvider`` is
always present. (Trade-off: Ghidra compiles the whole user script directory as one
OSGi bundle, so any *other* broken ``.java`` in that directory will fail the
build; that's an environment concern, not ours.)

What the script does, program-wide and deterministically (no LLM):

* For every function with **no committed signature** (signature source
  ``DEFAULT`` — the stripped/``param_count=0`` population the agent used to hunt
  for by hand), it decompiles the function and reads the prototype Ghidra's
  decompiler already recovered (``HighFunction.getFunctionPrototype()``).
* Functions whose signature anyone has already committed (``USER_DEFINED`` /
  ``IMPORTED`` / ``ANALYSIS`` — including this script's own prior commits and
  ``function-analyst``'s work) are **not** touched or even decompiled. This makes
  the pass a true no-op on re-run and guarantees it never clobbers types someone
  else set.
* When the recovered prototype is clean (a normal register/stack argument
  sequence, not variadic) it **applies it to that
  one function** via ``HighFunctionDBUtil.commitParamsToDatabase`` — the same
  program-DB write ``variables action:set_prototype`` performs, one function at a
  time. It never saves the program to disk.
* Ambiguous cases (variadic, non-standard storage, or a failed commit) are
  **not** changed; they get a ``proto-review`` bookmark and are
  reported for the LLM ``prototype-fixer`` to adjudicate. Any DEFAULT function
  already carrying that bookmark is skipped up front on re-run (counted, not
  re-decompiled and not re-reported), so re-runs don't re-flood the same
  ambiguities.
* Functions the decompiler could not process at all (timeouts, varnode hash
  errors, ...) are reported per-function with the decompiler's error message so
  ``prototype-fixer`` can triage them from the disassembly. The script itself
  writes no bookmark for these; the fixer resolves each by either committing a
  prototype (which moves the signature off ``DEFAULT``) or, for a dead end,
  writing a ``proto-review`` bookmark recording the verdict (not-a-function /
  unrecoverable) — which the skip above then keeps out of every later pass.

Decompilation — the expensive step — runs in parallel via Ghidra's
``ParallelDecompiler``/``DecompilerCallback`` (one ``DecompInterface`` and native
decompiler process per worker; pool defaults to cores + 1). Program-DB writes
(commits, bookmarks) and all bookkeeping happen only on the script thread, with
each chunk's results sorted by address so output stays deterministic.

The result is printed to stdout as a single JSON object between the
``<<<RECOVER_PROTOTYPES_JSON>>>`` / ``<<<END_RECOVER_PROTOTYPES_JSON>>>`` markers
so the caller can extract it from surrounding Ghidra console noise.

The GhidraScript public class name must match the deployed file name — see
``_SCRIPT_NAME`` in ``prototype_tools.py`` (``gda_recover_prototypes.java``).
"""

MARK_START = "<<<RECOVER_PROTOTYPES_JSON>>>"
MARK_END = "<<<END_RECOVER_PROTOTYPES_JSON>>>"

SCRIPT_SOURCE: str = r"""// recover_prototypes -- Ghidra Java GhidraScript. Runs inside Ghidra.
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
import ghidra.program.model.data.DataType;
import ghidra.program.model.listing.Bookmark;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.VariableStorage;
import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighFunctionDBUtil;
import ghidra.program.model.pcode.HighFunctionDBUtil.ReturnCommitOption;
import ghidra.program.model.symbol.SourceType;
import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

public class gda_recover_prototypes extends GhidraScript {

    static final String MARK_START = "<<<RECOVER_PROTOTYPES_JSON>>>";
    static final String MARK_END = "<<<END_RECOVER_PROTOTYPES_JSON>>>";
    static final String REVIEW_CATEGORY = "proto-review";
    static final int DECOMP_TIMEOUT = 60;   // seconds per function
    static final int FIXED_DETAIL_CAP = 200;
    static final int FAILED_DETAIL_CAP = 200;
    static final int CHUNK_SIZE = 64;       // bounds live HighFunctions per chunk

    // Worker threads only decompile and classify (read-only); every program-DB
    // write and every counter/StringBuilder update happens on the script thread
    // in applyResult(). That split is what makes the parallelism safe without
    // any locking of our own.
    private static class ProtoResult {
        static final int ALREADY_CORRECT = 0;
        static final int DECOMP_FAILED = 1;
        static final int CLEAN = 2;
        static final int ESCALATE = 3;

        Function func;
        int kind;
        HighFunction hf;    // CLEAN only; nulled after commit to free memory
        String addrS, nameS, recS, comS, reason;
    }

    private boolean dryRun = false;
    private int scanned = 0, alreadyCorrect = 0, fixed = 0;
    private int escalate = 0, escalateKnown = 0, decompFailed = 0;
    private int fixedDetail = 0, failedDetail = 0;
    private StringBuilder fixedJson = new StringBuilder();
    private StringBuilder escJson = new StringBuilder();
    private StringBuilder failJson = new StringBuilder();

    private String typeName(DataType dt) {
        return dt != null ? dt.getName() : "undefined";
    }

    private String protoString(String ret, List<String> ptypes) {
        String inner = ptypes.isEmpty() ? "void" : String.join(", ", ptypes);
        return ret + " (" + inner + ")";
    }

    private List<String> recParamTypes(FunctionPrototype proto) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < proto.getNumParams(); i++) {
            out.add(typeName(proto.getParam(i).getDataType()));
        }
        return out;
    }

    private List<String> comParamTypes(Parameter[] params) {
        List<String> out = new ArrayList<>();
        for (Parameter p : params) {
            out.add(typeName(p.getDataType()));
        }
        return out;
    }

    private boolean cleanStorage(FunctionPrototype proto) {
        for (int i = 0; i < proto.getNumParams(); i++) {
            VariableStorage st = proto.getParam(i).getStorage();
            if (st == null || !(st.isRegisterStorage() || st.isStackStorage())) {
                return false;
            }
        }
        return true;
    }

    private boolean applyPrototype(HighFunction hf) {
        try {
            HighFunctionDBUtil.commitParamsToDatabase(
                hf, true, ReturnCommitOption.COMMIT, SourceType.ANALYSIS);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    // Decompiler error text can be long multi-line console noise; keep the gist.
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

    private String addrString(Address entry) {
        String s = entry.toString();
        int idx = s.indexOf(':');
        if (idx >= 0) {
            s = s.substring(idx + 1);
        }
        return "0x" + s;
    }

    private boolean hasReviewBookmark(Address entry) {
        for (Bookmark bm : currentProgram.getBookmarkManager().getBookmarks(entry)) {
            if (REVIEW_CATEGORY.equals(bm.getCategory())) {
                return true;
            }
        }
        return false;
    }

    private void setReviewBookmark(Address entry, String reason) {
        currentProgram.getBookmarkManager().setBookmark(
            entry, "Note", REVIEW_CATEGORY, "proto: " + reason);
    }

    private String js(String s) {
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

    // Runs on ParallelDecompiler worker threads: program READS only (the
    // decompile result plus the function's committed signature), never writes.
    private ProtoResult classify(DecompileResults res) {
        ProtoResult r = new ProtoResult();
        r.func = res.getFunction();
        r.kind = ProtoResult.DECOMP_FAILED;
        r.addrS = addrString(r.func.getEntryPoint());
        r.nameS = r.func.getName();
        try {
            if (!res.decompileCompleted()) {
                r.reason = firstLine(res.getErrorMessage(),
                    "decompilation did not complete");
                return r;
            }
            HighFunction hf = res.getHighFunction();
            if (hf == null) {
                r.reason = "decompiler produced no high function";
                return r;
            }
            FunctionPrototype proto = hf.getFunctionPrototype();
            if (proto == null) {
                r.reason = "decompiler recovered no prototype";
                return r;
            }

            Function func = r.func;
            int recCount = proto.getNumParams();
            String recRet = typeName(proto.getReturnType());
            String comRet = typeName(func.getReturnType());
            Parameter[] comParams = func.getParameters();

            if (recCount == 0
                    && (recRet.equals("void") || recRet.equals("undefined"))) {
                r.kind = ProtoResult.ALREADY_CORRECT;
                return r;
            }

            if (recCount == comParams.length && recRet.equals(comRet)) {
                boolean allEq = true;
                for (int i = 0; i < recCount; i++) {
                    if (!typeName(proto.getParam(i).getDataType())
                            .equals(typeName(comParams[i].getDataType()))) {
                        allEq = false;
                        break;
                    }
                }
                if (allEq) {
                    r.kind = ProtoResult.ALREADY_CORRECT;
                    return r;
                }
            }

            boolean isVarargs = false;
            try {
                isVarargs = proto.isVarArg();
            } catch (Throwable t) {
                isVarargs = false;
            }
            boolean clean = !isVarargs && cleanStorage(proto);

            r.recS = protoString(recRet, recParamTypes(proto));
            r.comS = protoString(comRet, comParamTypes(comParams));

            if (clean) {
                r.kind = ProtoResult.CLEAN;
                r.hf = hf;   // DecompileResults are fully decoded; safe to hand
                             // to the script thread for the later commit
            } else {
                r.kind = ProtoResult.ESCALATE;
                r.reason = isVarargs
                    ? "variadic (...) - needs a manual prototype"
                    : "recovered args use non-standard storage";
            }
        } catch (Throwable t) {
            // One bad function must not kill the whole queue.
            r.kind = ProtoResult.DECOMP_FAILED;
            r.hf = null;
            r.reason = firstLine(t.getMessage(), t.getClass().getSimpleName());
        }
        return r;
    }

    // Script thread only: all DB writes and bookkeeping.
    private void applyResult(ProtoResult r) {
        if (r.kind == ProtoResult.DECOMP_FAILED) {
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
        if (r.kind == ProtoResult.ALREADY_CORRECT) {
            alreadyCorrect++;
            return;
        }
        if (r.kind == ProtoResult.CLEAN) {
            boolean committed = dryRun || applyPrototype(r.hf);
            r.hf = null;
            if (committed) {
                fixed++;
                if (fixedDetail < FIXED_DETAIL_CAP) {
                    appendObj(fixedJson,
                        "{\"addr\":\"" + js(r.addrS) + "\",\"name\":\"" + js(r.nameS)
                        + "\",\"old\":\"" + js(r.comS) + "\",\"new\":\"" + js(r.recS)
                        + "\"}");
                    fixedDetail++;
                }
                return;
            }
            escalateResult(r, "decompiler recovery could not be committed");
            return;
        }
        escalateResult(r, r.reason);
    }

    private void escalateResult(ProtoResult r, String reason) {
        // Candidates carrying a prior proto-review bookmark are filtered out in
        // run(), so anything reaching here is a genuinely NEW escalation.
        if (!dryRun) {
            setReviewBookmark(r.func.getEntryPoint(), reason);
        }
        escalate++;
        appendObj(escJson,
            "{\"addr\":\"" + js(r.addrS) + "\",\"name\":\"" + js(r.nameS)
            + "\",\"committed\":\"" + js(r.comS) + "\",\"recovered\":\"" + js(r.recS)
            + "\",\"reason\":\"" + js(reason) + "\"}");
    }

    @Override
    public void run() throws Exception {
        for (String a : getScriptArgs()) {
            if (a != null && a.equalsIgnoreCase("dry_run")) {
                dryRun = true;
            }
        }

        // Cheap pre-filter on the script thread so only real work reaches the
        // decompiler pool. Only ever fill in functions with NO committed
        // signature (DEFAULT source). Anything a human, function-analyst, or
        // Ghidra's analyzer has already committed (USER_DEFINED / IMPORTED /
        // ANALYSIS) is left untouched — so we never clobber someone's types,
        // and re-runs are a true no-op: our own commits use ANALYSIS source and
        // drop out next pass. (A deliberately-set `void f(void)` is
        // USER_DEFINED, so it is safe here even though it has zero parameters.)
        List<Function> candidates = new ArrayList<>();
        for (Function func : currentProgram.getFunctionManager().getFunctions(true)) {
            if (func.isThunk() || func.isExternal()) {
                continue;
            }
            scanned++;
            if (func.getSignatureSource() != SourceType.DEFAULT) {
                alreadyCorrect++;
                continue;
            }
            // A DEFAULT function already carrying a proto-review bookmark was
            // triaged by prototype-fixer on a prior pass (an unresolved
            // escalation, or a decompile failure it judged not-a-function /
            // unrecoverable). Skip it up front: don't re-decompile and don't
            // re-report. This is what stops re-runs re-flooding the same
            // dead-ends, and it also spares the expensive decompile.
            if (hasReviewBookmark(func.getEntryPoint())) {
                escalateKnown++;
                continue;
            }
            candidates.add(func);
        }

        // One DecompInterface (and native decompiler process) per worker
        // thread, pooled by DecompilerCallback; a DecompInterface is not
        // thread-safe and must never be shared.
        DecompilerCallback<ProtoResult> callback = new DecompilerCallback<ProtoResult>(
                currentProgram, d -> d.setOptions(new DecompileOptions())) {
            @Override
            public ProtoResult process(DecompileResults results, TaskMonitor m) {
                return classify(results);
            }
        };
        callback.setTimeout(DECOMP_TIMEOUT);

        ChunkingParallelDecompiler<ProtoResult> pd =
            ParallelDecompiler.createChunkingParallelDecompiler(callback, monitor);
        try {
            monitor.initialize(candidates.size());
            for (int i = 0; i < candidates.size() && !monitor.isCancelled();
                    i += CHUNK_SIZE) {
                List<Function> chunk =
                    candidates.subList(i, Math.min(i + CHUNK_SIZE, candidates.size()));
                List<ProtoResult> results = pd.decompileFunctions(chunk);
                // Results arrive in completion order; sort by address so the
                // JSON arrays and the FIXED_DETAIL_CAP cutoff stay deterministic.
                results.sort(Comparator.comparing(r -> r.func.getEntryPoint()));
                for (ProtoResult r : results) {
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
        out.append("{\"dry_run\":").append(dryRun ? "true" : "false")
           .append(",\"counts\":{")
           .append("\"scanned\":").append(scanned)
           .append(",\"already_correct\":").append(alreadyCorrect)
           .append(",\"fixed\":").append(fixed)
           .append(",\"escalate\":").append(escalate)
           .append(",\"escalate_known\":").append(escalateKnown)
           .append(",\"decompile_failed\":").append(decompFailed)
           .append("},\"fixed\":[").append(fixedJson).append("]")
           .append(",\"fixed_truncated\":").append(fixed > fixedDetail ? "true" : "false")
           .append(",\"escalate\":[").append(escJson).append("]")
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
