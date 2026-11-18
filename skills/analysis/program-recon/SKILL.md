---
name: program-recon
description: Reconnaissance of an unfamiliar binary. Use at the start of a session to build a compact brief (format, architecture, entry points, notable imports/exports/strings, prior knowledge) before planning deeper analysis.
---

# Program reconnaissance

Build a quick, high-level picture of an unfamiliar binary so the analysis can be
planned.

Gather, in as few round-trips as possible (batch independent reads in one turn):
- Program info (`get_program_info`): format, architecture, bit-width, base
  address. Entry points from `list_entry_points`.
- The function inventory size (`list_functions`) and any notably named functions.
- Imports and exports (`list_imports`, `list_exports`) that hint at capability
  (networking, crypto, file I/O, process/thread, registry).
- A scan of strings (`search_strings`) for telling artifacts (URLs, paths,
  commands, error messages, format strings).
- Relevant prior knowledge: call `get_knowledge_summary` and `query_knowledge`
  to see what is already known about this binary.

Record the brief with `save_knowledge` (category `overview`) before returning it,
unless `get_knowledge_summary` shows an equivalent entry already exists — a later
session should be able to query it instead of paying for this pass again. If
write tools are unavailable (plan/ask mode), skip the save and say so.

## Output

A single compact brief (a few short sections or bullets) covering
format/arch/entry points, notable imports/exports/strings, counts, a one-line
hypothesis about the binary's purpose, and anything already in the knowledge
base. Do NOT paste full catalog listings or raw tool dumps — summarize.
