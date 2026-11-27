# Ghidra utility scripts

Generic GhidraScripts for triaging a program: decompile failures, one-function
decompiles, memory layout, and code-unit coverage. None of them hard-code an
address or a target binary; addresses come in as script arguments.

| Script | Arguments | Writes? | What it does |
|---|---|---|---|
| `DecompileFailures.java` | `[zeroParamsOnly]` | no | Lists functions the decompiler cannot produce output for. |
| `BookmarkDecompileFailures.java` | `[apply]` | only with `apply` | Adds a NOTE bookmark (category `decompile-failure`) to each failure. Dry run by default; skips functions already bookmarked. |
| `OneDecompile.java` | `<address>` | no | Decompiles one function and prints its C, or the decompiler's error. |
| `ListMemoryBlocks.java` | none | no | Prints each memory block's range, size, initialized flag and permissions. |
| `UnitAudit.java` | `[blockName]` (default `.text`) | no | Reports code-unit coverage of a block by class, and the bytes with no unit at all. |

## Running

Through GhidrAssistMCP, the agent (or you) can deploy a script's source with
`create_script` and run it with `run_script`, passing the arguments as the
script arguments. Or copy a file into Ghidra's script directory and run it from
the Script Manager.

## Checking the scripts compile

```bash
./scripts/ghidra/compile_check.sh
```

This compiles every script here against the jars of a local Ghidra install. It
needs a JDK that matches the Ghidra version: Ghidra 12.2 ships Java 25 class
files, so the default is `openjdk@25`. Override with `JAVAC=...` and
`GHIDRA_HOME=...`. Ghidra only reports a broken script when it runs, so run this
before deploying.
