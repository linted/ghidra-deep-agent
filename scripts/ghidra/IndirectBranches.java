// IndirectBranches -- list the register-indirect branches inside a function:
// br, braa, brab (AArch64) and any other instruction whose mnemonic starts with
// "br" or "blr". Read-only. Use it to find the jumps switch-recovery has to fix.
//
// Usage: <address>   any address inside the function
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

public class IndirectBranches extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            printerr("usage: IndirectBranches <address>");
            return;
        }
        Function f = getFunctionContaining(toAddr(args[0]));
        if (f == null) {
            println("no function contains " + args[0]);
            return;
        }

        println("indirect branches in " + f.getName() + " @ " + f.getEntryPoint() + ":");
        int found = 0;
        InstructionIterator it = currentProgram.getListing().getInstructions(f.getBody(), true);
        while (it.hasNext() && !monitor.isCancelled()) {
            Instruction in = it.next();
            String mn = in.getMnemonicString();
            if (mn.startsWith("br") || mn.startsWith("blr")) {
                println("  " + in.getAddress() + "  " + in.toString());
                found++;
            }
        }
        println("total=" + found);
    }
}
