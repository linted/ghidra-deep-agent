// ListMemoryBlocks -- print every memory block with its range, size,
// initialized flag and permissions. Read-only.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;

public class ListMemoryBlocks extends GhidraScript {
    @Override
    public void run() throws Exception {
        Memory mem = currentProgram.getMemory();
        for (MemoryBlock b : mem.getBlocks()) {
            println(String.format("%-12s %s..%s size=0x%-9x init=%-5s %s%s%s",
                b.getName(), b.getStart(), b.getEnd(), b.getSize(),
                b.isInitialized(),
                b.isRead() ? "R" : "-", b.isWrite() ? "W" : "-",
                b.isExecute() ? "X" : "-"));
        }
        println("total blocks=" + mem.getBlocks().length);
    }
}
