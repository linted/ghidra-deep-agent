"""The GhidrAssistMCP-side jump-table *override* script, as a source string.

``SCRIPT_SOURCE`` is a **Java** GhidraScript (not code that runs in this process).
It is shipped verbatim to the running Ghidra via GhidrAssistMCP's ``scripts`` tool
and compiled/run there — see ``switch_tools.py``. Sibling of
``recover_prototypes_script.py`` / ``find_unrecovered_switches_script.py``; see
those docstrings for the shared design.

This is the deterministic MUTATION for one unrecovered indirect jump. Its single
JSON argument (``getScriptArgs()[0]``, parsed with Ghidra's bundled Gson) selects
one of two contracts:

* ``destinations``: an explicit list of absolute case-target addresses (for
  small / irregular / hand-verified tables), OR
* ``table_address`` + ``element_size`` + ``count`` (+ ``base_address`` +
  ``relative``): a regular strided table the SCRIPT decodes — it reads
  ``count`` entries of ``element_size`` bytes at ``table_address``, respecting
  the program's endianness, sign-extends each when ``relative``, and computes
  ``dest = relative ? base + entry : entry``.

For that jump it then, in order:
  1. clears a stale ``CALL`` / ``CALL_RETURN`` flow override on the jump
     instruction (the "Treating indirect jump as call" cause),
  2. writes the decompiler jump-table override
     (``JumpTable(switchAddr, dests, true, 0).writeOverride(func)`` — the final
     ``0`` is ``EquateSymbol.FORMAT_DEFAULT``, i.e. no case-label display-format
     override; Ghidra 12.x only has the 4-arg constructor),
  3. adds a ``COMPUTED_JUMP`` reference to every destination and disassembles any
     that are still undefined bytes,
  4. optionally marks the table's memory block read-only (``set_rodata_constant``)
     so decompiler constant-propagation can fold the table load,
  5. re-decompiles with a FRESH ``DecompInterface`` and checks whether the
     ``"Could not recover jumptable"`` warning is gone.

Result JSON (between the ``<<<APPLY_SWITCH_JSON>>>`` / ``<<<END_APPLY_SWITCH_JSON>>>``
markers) carries ``applied``, ``warning_cleared``, ``num_destinations``, the fresh
(capped) decompiled C, and any ``notes`` / ``error``.

The GhidraScript public class name must match the deployed file name — see
``_APPLY_SCRIPT_NAME`` in ``switch_tools.py`` (``gda_apply_switch_override.java``).
"""

MARK_START = "<<<APPLY_SWITCH_JSON>>>"
MARK_END = "<<<END_APPLY_SWITCH_JSON>>>"

SCRIPT_SOURCE: str = r"""// apply_switch_override -- Ghidra Java GhidraScript. Runs inside Ghidra.
// @category DeepAgent
import java.util.ArrayList;
import java.util.List;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSpace;
import ghidra.program.model.listing.CodeUnit;
import ghidra.program.model.listing.FlowOverride;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.pcode.JumpTable;
import ghidra.program.model.symbol.RefType;
import ghidra.program.model.symbol.SourceType;
import ghidra.util.DataConverter;

public class gda_apply_switch_override extends GhidraScript {

    static final String MARK_START = "<<<APPLY_SWITCH_JSON>>>";
    static final String MARK_END = "<<<END_APPLY_SWITCH_JSON>>>";
    static final String WARNING = "Could not recover jumptable";
    static final int DECOMP_TIMEOUT = 60;      // seconds
    static final int C_CAP = 20000;            // cap the returned decompiled C

    private List<String> notes = new ArrayList<>();

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

    private String jsonArray(List<String> items) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                b.append(",");
            }
            b.append("\"").append(js(items.get(i))).append("\"");
        }
        return b.append("]").toString();
    }

    private void emit(String body) {
        println(MARK_START);
        println(body);
        println(MARK_END);
    }

    private void emitError(String msg) {
        emit("{\"applied\":false,\"warning_cleared\":false,\"error\":\""
            + js(msg) + "\",\"notes\":" + jsonArray(notes) + "}");
    }

    private String opt(JsonObject o, String k) {
        return (o.has(k) && !o.get(k).isJsonNull()) ? o.get(k).getAsString() : null;
    }

    private Integer optInt(JsonObject o, String k) {
        return (o.has(k) && !o.get(k).isJsonNull()) ? o.get(k).getAsInt() : null;
    }

    private boolean optBool(JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() && o.get(k).getAsBoolean();
    }

    // Decode one strided-table entry into a destination address.
    private Address decodeEntry(Memory mem, DataConverter dc, AddressSpace space,
            Address entryAddr, int size, boolean relative, Address base)
            throws Exception {
        byte[] buf = new byte[size];
        mem.getBytes(entryAddr, buf);
        long value;
        switch (size) {
            case 1:
                value = relative ? (long) buf[0] : (buf[0] & 0xFFL);
                break;
            case 2:
                short sv = dc.getShort(buf);
                value = relative ? (long) sv : (sv & 0xFFFFL);
                break;
            case 4:
                int iv = dc.getInt(buf);
                value = relative ? (long) iv : (iv & 0xFFFFFFFFL);
                break;
            case 8:
                value = dc.getLong(buf);
                break;
            default:
                throw new IllegalArgumentException(
                    "element_size must be 1, 2, 4, or 8 (got " + size + ")");
        }
        return relative ? base.add(value) : space.getAddress(value);
    }

    // Build destinations from either contract; skips out-of-memory targets.
    private ArrayList<Address> buildDestinations(JsonObject in) throws Exception {
        Memory mem = currentProgram.getMemory();
        ArrayList<Address> dests = new ArrayList<>();

        if (in.has("destinations") && !in.get("destinations").isJsonNull()) {
            JsonArray arr = in.getAsJsonArray("destinations");
            for (int i = 0; i < arr.size(); i++) {
                Address d = toAddr(arr.get(i).getAsString());
                if (d != null && mem.contains(d)) {
                    dests.add(d);
                } else {
                    notes.add("dropped out-of-memory destination "
                        + arr.get(i).getAsString());
                }
            }
            return dests;
        }

        String tableS = opt(in, "table_address");
        Integer size = optInt(in, "element_size");
        Integer count = optInt(in, "count");
        if (tableS == null || size == null || count == null) {
            throw new IllegalArgumentException(
                "provide either 'destinations' or "
                + "'table_address'+'element_size'+'count'");
        }
        boolean relative = optBool(in, "relative");
        Address tableAddr = toAddr(tableS);
        String baseS = opt(in, "base_address");
        Address base = baseS != null ? toAddr(baseS) : tableAddr;
        DataConverter dc = DataConverter.getInstance(mem.isBigEndian());
        AddressSpace space = currentProgram.getAddressFactory().getDefaultAddressSpace();

        for (int i = 0; i < count; i++) {
            Address entryAddr = tableAddr.add((long) i * size);
            try {
                Address d = decodeEntry(mem, dc, space, entryAddr, size, relative, base);
                if (mem.contains(d)) {
                    dests.add(d);
                } else {
                    notes.add("dropped out-of-memory entry " + i + " -> " + d);
                }
            } catch (Exception ex) {
                notes.add("skipped entry " + i + ": " + ex.getMessage());
            }
        }
        return dests;
    }

    private boolean freshWarningGone(Function func, String[] cOut) {
        DecompInterface decomp = new DecompInterface();
        try {
            decomp.setOptions(new DecompileOptions());
            if (!decomp.openProgram(currentProgram)) {
                notes.add("verify: could not open program in fresh decompiler");
                cOut[0] = "";
                return false;
            }
            DecompileResults res = decomp.decompileFunction(func, DECOMP_TIMEOUT, monitor);
            if (res == null || !res.decompileCompleted()
                    || res.getDecompiledFunction() == null) {
                notes.add("verify: re-decompile did not complete");
                cOut[0] = "";
                return false;
            }
            String c = res.getDecompiledFunction().getC();
            cOut[0] = c != null ? c : "";
            return c != null && !c.contains(WARNING);
        } finally {
            decomp.dispose();
        }
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 1 || args[0] == null || args[0].isEmpty()) {
            emitError("missing JSON argument");
            return;
        }

        JsonObject in;
        try {
            in = JsonParser.parseString(args[0]).getAsJsonObject();
        } catch (Exception ex) {
            emitError("could not parse JSON argument: " + ex.getMessage());
            return;
        }

        String jumpS = opt(in, "jump_address");
        if (jumpS == null) {
            emitError("'jump_address' is required");
            return;
        }
        Address jumpAddr = toAddr(jumpS);
        if (jumpAddr == null) {
            emitError("could not parse jump_address: " + jumpS);
            return;
        }
        Instruction instr = getInstructionAt(jumpAddr);
        if (instr == null) {
            emitError("no instruction at jump_address " + jumpS);
            return;
        }
        Function func = getFunctionContaining(jumpAddr);
        if (func == null) {
            emitError("no function contains jump_address " + jumpS);
            return;
        }

        ArrayList<Address> dests;
        try {
            dests = buildDestinations(in);
        } catch (Exception ex) {
            emitError(ex.getMessage());
            return;
        }
        if (dests.isEmpty()) {
            emitError("no valid destination addresses were produced");
            return;
        }

        // 1. Clear a stale CALL/CALL_RETURN flow override (the "treating indirect
        //    jump as call" cause) so the decompiler treats it as a jump again.
        FlowOverride fo = instr.getFlowOverride();
        if (fo == FlowOverride.CALL || fo == FlowOverride.CALL_RETURN) {
            instr.setFlowOverride(FlowOverride.NONE);
            notes.add("cleared " + fo + " flow override");
        }

        // 2. Decompiler jump-table override.
        try {
            // Final arg is the case-label display format; 0 == EquateSymbol.FORMAT_DEFAULT
            // (no format override). Ghidra 12.x only has the 4-arg constructor.
            JumpTable jt = new JumpTable(jumpAddr, dests, true, 0);
            jt.writeOverride(func);
        } catch (Exception ex) {
            emitError("writeOverride failed: " + ex.getMessage());
            return;
        }

        // 3. COMPUTED_JUMP references + disassemble undefined targets.
        int disassembled = 0;
        for (Address dest : dests) {
            currentProgram.getReferenceManager().addMemoryReference(
                jumpAddr, dest, RefType.COMPUTED_JUMP, SourceType.USER_DEFINED,
                CodeUnit.MNEMONIC);
            if (getInstructionAt(dest) == null) {
                if (disassemble(dest)) {
                    disassembled++;
                }
            }
        }
        if (disassembled > 0) {
            notes.add("disassembled " + disassembled + " new target(s)");
        }

        // 4. Optional: make the table block read-only so const-prop can fold it.
        if (optBool(in, "set_rodata_constant")) {
            String tableS = opt(in, "table_address");
            if (tableS != null) {
                MemoryBlock blk = currentProgram.getMemory().getBlock(toAddr(tableS));
                if (blk != null && blk.isWrite()) {
                    blk.setWrite(false);
                    notes.add("marked block '" + blk.getName() + "' read-only");
                }
            }
        }

        // 5. Verify with a FRESH decompile (never reuse a pre-write result).
        String[] cOut = new String[1];
        boolean cleared = freshWarningGone(func, cOut);

        String c = cOut[0] != null ? cOut[0] : "";
        boolean truncated = c.length() > C_CAP;
        if (truncated) {
            c = c.substring(0, C_CAP);
        }

        StringBuilder out = new StringBuilder();
        out.append("{\"applied\":true")
           .append(",\"jump\":\"").append(js(jumpS)).append("\"")
           .append(",\"func\":\"").append(js(func.getName())).append("\"")
           .append(",\"warning_cleared\":").append(cleared ? "true" : "false")
           .append(",\"num_destinations\":").append(dests.size())
           .append(",\"decompiled_c\":\"").append(js(c)).append("\"")
           .append(",\"c_truncated\":").append(truncated ? "true" : "false")
           .append(",\"notes\":").append(jsonArray(notes))
           .append("}");
        emit(out.toString());
    }
}
"""
