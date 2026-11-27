// BookmarkDecompileFailures -- add a NOTE bookmark to every function the
// decompiler cannot produce output for, so a later pass can skip or review them.
//
// Usage (script args, optional):
//   [apply]   write the bookmarks. Default is a dry run that only lists them.
//
// Bookmarks use category "decompile-failure" and skip functions already
// bookmarked, so the script is safe to re-run. Thunks are skipped.
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.BookmarkManager;
import ghidra.program.model.listing.BookmarkType;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.util.task.ConsoleTaskMonitor;

import java.util.ArrayList;
import java.util.List;

public class BookmarkDecompileFailures extends GhidraScript {
    private static final String CATEGORY = "decompile-failure";

    @Override
    public void run() throws Exception {
        boolean apply = false;
        for (String arg : getScriptArgs()) {
            if (arg.equals("apply")) {
                apply = true;
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
        BookmarkManager bm = currentProgram.getBookmarkManager();

        monitor.initialize(currentProgram.getFunctionManager().getFunctionCount());
        int skipped = 0;
        int failed = 0;
        List<String> report = new ArrayList<>();
        FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
        while (it.hasNext() && !monitor.isCancelled()) {
            Function f = it.next();
            monitor.incrementProgress(1);
            if (f.isThunk()) {
                continue;
            }
            Address ep = f.getEntryPoint();
            if (bm.getBookmark(ep, BookmarkType.NOTE, CATEGORY) != null) {
                skipped++;
                continue;
            }
            DecompileResults res = di.decompileFunction(f, 60, new ConsoleTaskMonitor());
            if (res.decompileCompleted() && res.getHighFunction() != null) {
                continue;
            }
            String msg = res.getErrorMessage();
            int nl = msg.indexOf('\n');
            if (nl > 0) {
                msg = msg.substring(0, nl);
            }
            failed++;
            report.add(ep + " " + f.getName() + " :: " + msg);
            if (apply) {
                bm.setBookmark(ep, BookmarkType.NOTE, CATEGORY, "decompiler: " + msg);
            }
        }
        di.dispose();

        for (String line : report) {
            println(line);
        }
        println(apply ? "BOOKMARKED=" + failed : "WOULD_BOOKMARK=" + failed + " (dry run; pass 'apply')");
        println("ALREADY_BOOKMARKED_SKIPPED=" + skipped);
    }
}
