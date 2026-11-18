---
name: investigate
description: Evidence-driven investigation of a question about the binary. Use when you need a question answered with evidence (what does X do, where is Y used, how does data flow from A to B) rather than a function reworked end-to-end — decompile, follow cross-references, consult the knowledge base, and persist the cheap conclusions.
---

# Investigate a question

## Read the decompiled C, not the assembly
Work primarily from the DECOMPILED C: call `get_pseudocode` for each function
you investigate — reach for it first. Only drop to `get_disassembly` when the
decompiler is ambiguous, clearly wrong, or hides something load-bearing (exact
buffer sizes, a cast, a signed/unsigned comparison, an overlapping stack layout).
NEVER hand-decode raw instruction bytes: Ghidra has already disassembled the
program, so when you need instructions ask for them with `get_disassembly`
rather than reading hex and walking encoding tables yourself.

## Investigate
1. Query the knowledge base (`query_knowledge`, `query_by_address`) for prior
   findings FIRST — an earlier pass may already have answered part of this, and
   the brief you were given may include earlier findings. Build on them instead
   of starting over.
2. Read the decompiled C (`get_pseudocode`) for the relevant functions.
3. Trace data flow; note loops, conditionals, error checks, syscalls, and API
   calls. Drop to disassembly only when the C is not enough.
4. Follow cross-references (`get_callers` / `get_callees`, `get_xrefs_to` /
   `get_xrefs_from` for an address) as needed.

## Leave the program better than you found it
As you work, persist the cheap conclusions your evidence supports: rename the
locals and functions whose purpose you established (`rename_local_variable`,
`rename_function`; prefix a name with `maybe_` when the evidence is suggestive
but not settled), drop a comment (`set_comment`) where the reason for something
non-obvious took real effort to work out, and `save_knowledge` the findings.
Anything you keep to yourself gets re-derived at full cost next time.

If write tools are unavailable (plan/ask mode), apply nothing — state the
concrete changes you would recommend (renames, retypes, comments, prototypes,
knowledge-base entries) in your answer instead.

If the decompiled C shows `WARNING: Could not recover jumptable`, the function's
control flow is wrong: say so, and recommend the `switch-recovery` skill before
trusting conclusions about that function.

## Output
The assembly is ground truth: when the decompiled C conflicts with it, trust the
assembly — but reach for it only when it actually matters. Cite the instruction
address or pattern behind every conclusion. Return a COMPACT findings summary —
what the code does, the key evidence, and the changes you applied (or recommend).
Do NOT paste raw decompiler or disassembly dumps.
