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
  sequence, not variadic, within the ABI argument budget) it **applies it to that
  one function** via ``HighFunctionDBUtil.commitParamsToDatabase`` — the same
  program-DB write ``variables action:set_prototype`` performs, one function at a
  time. It never saves the program to disk.
* Ambiguous cases (variadic, non-standard storage, too many args, or a failed
  commit) are **not** changed; they get a ``proto-review`` bookmark and are
  reported for the LLM ``prototype-fixer`` to adjudicate. A function already
  carrying that bookmark is counted but not re-reported, so re-runs don't re-flood
  the same ambiguities.

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
import java.util.List;

import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
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

public class gda_recover_prototypes extends GhidraScript {

    static final String MARK_START = "<<<RECOVER_PROTOTYPES_JSON>>>";
    static final String MARK_END = "<<<END_RECOVER_PROTOTYPES_JSON>>>";
    static final String REVIEW_CATEGORY = "proto-review";
    static final int DECOMP_TIMEOUT = 60;   // seconds per function
    static final int MAX_ARGS = 8;          // ABI argument-register budget
    static final int FIXED_DETAIL_CAP = 200;

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

    @Override
    public void run() throws Exception {
        boolean dryRun = false;
        for (String a : getScriptArgs()) {
            if (a != null && a.equalsIgnoreCase("dry_run")) {
                dryRun = true;
            }
        }

        DecompInterface di = new DecompInterface();
        di.setOptions(new DecompileOptions());
        di.openProgram(currentProgram);

        int scanned = 0, alreadyCorrect = 0, fixed = 0;
        int escalate = 0, escalateKnown = 0, decompFailed = 0;
        int fixedDetail = 0;
        StringBuilder fixedJson = new StringBuilder();
        StringBuilder escJson = new StringBuilder();

        for (Function func : currentProgram.getFunctionManager().getFunctions(true)) {
            if (func.isThunk() || func.isExternal()) {
                continue;
            }
            scanned++;

            // Only ever fill in functions with NO committed signature (DEFAULT
            // source). Anything a human, function-analyst, or Ghidra's analyzer
            // has already committed (USER_DEFINED / IMPORTED / ANALYSIS) is left
            // untouched — so we never clobber someone's types, and re-runs are a
            // true no-op: our own commits use ANALYSIS source and drop out next
            // pass. (A deliberately-set `void f(void)` is USER_DEFINED, so it is
            // safe here even though it has zero parameters.)
            if (func.getSignatureSource() != SourceType.DEFAULT) {
                alreadyCorrect++;
                continue;
            }

            DecompileResults res = di.decompileFunction(func, DECOMP_TIMEOUT, monitor);
            if (res == null || !res.decompileCompleted()) {
                decompFailed++;
                continue;
            }
            HighFunction hf = res.getHighFunction();
            if (hf == null) {
                decompFailed++;
                continue;
            }
            FunctionPrototype proto = hf.getFunctionPrototype();
            if (proto == null) {
                decompFailed++;
                continue;
            }

            int recCount = proto.getNumParams();
            String recRet = typeName(proto.getReturnType());
            String comRet = typeName(func.getReturnType());
            Parameter[] comParams = func.getParameters();

            if (recCount == 0
                    && (recRet.equals("void") || recRet.equals("undefined"))) {
                alreadyCorrect++;
                continue;
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
                    alreadyCorrect++;
                    continue;
                }
            }

            boolean isVarargs = false;
            try {
                isVarargs = proto.isVarArg();
            } catch (Throwable t) {
                isVarargs = false;
            }
            boolean clean = !isVarargs && recCount <= MAX_ARGS && cleanStorage(proto);

            String addrS = addrString(func.getEntryPoint());
            String nameS = func.getName();
            String recS = protoString(recRet, recParamTypes(proto));
            String comS = protoString(comRet, comParamTypes(comParams));

            String reason;
            if (clean) {
                if (dryRun || applyPrototype(hf)) {
                    fixed++;
                    if (fixedDetail < FIXED_DETAIL_CAP) {
                        appendObj(fixedJson,
                            "{\"addr\":\"" + js(addrS) + "\",\"name\":\"" + js(nameS)
                            + "\",\"old\":\"" + js(comS) + "\",\"new\":\"" + js(recS)
                            + "\"}");
                        fixedDetail++;
                    }
                    continue;
                }
                reason = "decompiler recovery could not be committed";
            } else if (isVarargs) {
                reason = "variadic (...) - needs a manual prototype";
            } else {
                reason = "recovered args use non-standard storage or exceed " + MAX_ARGS;
            }

            if (!dryRun && hasReviewBookmark(func.getEntryPoint())) {
                escalateKnown++;
                continue;
            }
            if (!dryRun) {
                setReviewBookmark(func.getEntryPoint(), reason);
            }
            escalate++;
            appendObj(escJson,
                "{\"addr\":\"" + js(addrS) + "\",\"name\":\"" + js(nameS)
                + "\",\"committed\":\"" + js(comS) + "\",\"recovered\":\"" + js(recS)
                + "\",\"reason\":\"" + js(reason) + "\"}");
        }

        di.dispose();

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
           .append(",\"escalate\":[").append(escJson).append("]}");

        println(MARK_START);
        println(out.toString());
        println(MARK_END);
    }
}
"""
