// DecompileFailures -- list functions the decompiler cannot produce output for.
//
// Usage (script args, all optional):
//   [zeroParamsOnly]   only consider functions with 0 parameters (what the
//                      prototype-recovery pass scans). Default: all functions.
//
// Output: one line per failure, "entry name :: first error line", then a
// TOTAL line. Thunks are skipped. Read-only.
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.util.task.ConsoleTaskMonitor;

public class DecompileFailures extends GhidraScript {
    @Override
    public void run() throws Exception {
        boolean zeroParamsOnly = false;
        for (String arg : getScriptArgs()) {
            if (arg.equals("zeroParamsOnly")) {
                zeroParamsOnly = true;
            }
            else {
                printerr("unknown argument: " + arg);
                return;
            }
        }

        DecompInterface di = new DecompInterface();
        di.setOptions(new DecompileOptions());
        if (!di.openProgram(currentProgram)) {
            printerr("decompiler failed to open the program");
            return;
        }

        int checked = 0;
        int failed = 0;
        FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
        while (it.hasNext() && !monitor.isCancelled()) {
            Function f = it.next();
            if (f.isThunk()) {
                continue;
            }
            if (zeroParamsOnly && f.getParameterCount() != 0) {
                continue;
            }
            checked++;
            DecompileResults res = di.decompileFunction(f, 60, new ConsoleTaskMonitor());
            if (!res.decompileCompleted() || res.getHighFunction() == null) {
                String msg = res.getErrorMessage();
                int nl = msg.indexOf('\n');
                if (nl > 0) {
                    msg = msg.substring(0, nl);
                }
                println(f.getEntryPoint() + " " + f.getName() + " :: " + msg);
                failed++;
            }
        }
        di.dispose();
        println("CHECKED=" + checked);
        println("TOTAL_FAILED=" + failed);
    }
}
