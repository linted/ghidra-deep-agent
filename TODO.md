# TODOs

- [ ] **Plan mode for the RE agent**
- [x] **OpenRouter support**

## Plan mode for the RE agent
Add a "plan mode" inspired by Claude Code's plan mode. When invoked, the agent
should reason about a presented problem, produce a **markdown plan for the human
to review** (explicitly asking for feedback), and **write the plan to disk** —
all *before* making any mutating changes to the Ghidra database.

Design thoughts (from how plan mode works):
- **Read-only while planning.** During plan mode the agent must not rename,
  retype, or otherwise mutate the binary — only read assembly/decompiler output
  and query the knowledge base. Mirrors plan mode's "no edits" guarantee.
- **Phased flow:** (1) explore/understand the problem, (2) design an approach,
  (3) write the plan, (4) hand back to the human for approval before execution.
- **Persist the plan to disk** via the existing `FilesystemBackend`
  (see AGENT_OUTPUT_DIR handling in main.py) — e.g. a `plans/` subdirectory —
  so plans survive across sessions like other artifacts.
- **Ask for feedback / approval gate:** end the planning turn by returning the
  markdown and waiting for the human, rather than charging ahead.
- **Likely plug-in points in this codebase:**
  - A `/plan` slash command in the TUI dispatcher
    (src/ghidra_deep_agent/tui/app.py).
  - Either a dedicated planning subagent (deepagents `task` mechanism,
    constrained to read-only Ghidra tools + knowledge query tools) or a
    plan-specific system-prompt variant alongside src/ghidra_deep_agent/prompt.py.
  - Reuse the FilesystemBackend already wired up in main.py for writing the
    plan file.

## OpenRouter support
Add support for using OpenRouter as a model provider. LangChain should have a
library/integration for it (e.g. `langchain-openai` pointed at the OpenRouter
base URL, or a dedicated OpenRouter package) — investigate which is the right
fit and wire it into the model configuration.
