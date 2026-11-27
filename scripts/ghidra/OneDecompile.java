// OneDecompile -- decompile the function at one address and print its C.
//
// Usage: <address>   e.g. 0x401000 (the function must already exist)
// Read-only; prints "DECOMPILE OK" plus the C, or "DECOMPILE FAILED" plus the
// decompiler's error.
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;

public class OneDecompile extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            printerr("usage: OneDecompile <address>");
            return;
        }
        Function f = currentProgram.getFunctionManager().getFunctionAt(toAddr(args[0]));
        if (f == null) {
            println("no function at " + args[0]);
            return;
        }

        DecompInterface di = new DecompInterface();
        di.setOptions(new DecompileOptions());
        if (!di.openProgram(currentProgram)) {
            println("open failed");
            return;
        }
        DecompileResults r = di.decompileFunction(f, 120, monitor);
        if (r.decompileCompleted()) {
            println("DECOMPILE OK: " + f.getName());
            println(r.getDecompiledFunction().getC());
        }
        else {
            println("DECOMPILE FAILED");
            println(r.getErrorMessage());
        }
        di.dispose();
    }
}
