// UnitAudit -- report how much of a memory block is covered by code units
// (instructions and data) and how many bytes are gaps with no unit at all.
//
// Usage: [blockName]   default ".text". Read-only.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressRange;
import ghidra.program.model.address.AddressRangeImpl;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.listing.CodeUnit;
import ghidra.program.model.listing.CodeUnitIterator;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.mem.MemoryBlock;

import java.util.Map;
import java.util.TreeMap;

public class UnitAudit extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String blockName = args.length > 0 ? args[0] : ".text";
        MemoryBlock b = currentProgram.getMemory().getBlock(blockName);
        if (b == null) {
            println("no block " + blockName);
            return;
        }

        Listing listing = currentProgram.getListing();
        AddressRange range = new AddressRangeImpl(b.getStart(), b.getEnd());
        CodeUnitIterator it = listing.getCodeUnits(new AddressSet(range), true);
        Map<String, long[]> hist = new TreeMap<>();
        long covered = 0;
        long gapBytes = 0;
        Address expected = b.getStart();
        while (it.hasNext()) {
            CodeUnit cu = it.next();
            Address s = cu.getAddress();
            String cls = cu instanceof Instruction ? "Instruction"
                : (cu instanceof Data ? "Data:" + ((Data) cu).getDataType().getName() : "Other");
            long[] v = hist.computeIfAbsent(cls, k -> new long[2]);
            v[0]++;
            v[1] += cu.getLength();
            covered += cu.getLength();
            if (s.compareTo(expected) > 0) {
                gapBytes += s.subtract(expected);
            }
            expected = s.add(cu.getLength());
        }
        if (expected.compareTo(b.getEnd()) <= 0) {
            gapBytes += b.getEnd().subtract(expected) + 1;
        }

        println("block " + blockName + " size=" + b.getSize() + " unitCovered=" + covered +
            " gapBytes=" + gapBytes);
        for (Map.Entry<String, long[]> e : hist.entrySet()) {
            println(String.format("  %-32s units=%d bytes=%d", e.getKey(), e.getValue()[0], e.getValue()[1]));
        }
    }
}
