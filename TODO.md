# TODOs

- [ ] **Plan mode for the RE agent**
- [x] **OpenRouter support**

### From optimization report (2026-06-28, 7d window)

Cost
- [ ] **Right-size subagent model & context** — pass only task-specific artifacts to subagents (not full history); route structured/retrieval subagent work to a lighter model
- [ ] **Tune forced compaction** — lower trigger threshold, truncate tool outputs before they enter context, route summarization call to a cheaper/smaller model
- [ ] **Trim per-call prompt bloat** — compress tool descriptions, conditionally inject middleware content (skip filesystem tree / todo list when irrelevant), audit system prompt
- [x] **Conditionally disable `AnthropicPromptCachingMiddleware`** when running non-Anthropic providers (e.g. DeepSeek) — no-op: the middleware isn't wired into this codebase, and the library version already no-ops for non-Anthropic models (isinstance check). Nothing to do.

Errors
- [x] **Harden `update_knowledge`** — retries + backoff, entity-exists guard, return structured warning instead of raising (highest per-tool error rate, 5.6%). Also applied to `save_knowledge` (sibling write tool).
- [ ] **Add retry logic to filesystem tool calls** for transient I/O errors; return structured edit-failure errors so the LLM self-corrects
- [x] **Pydantic argument-validation shim** before tool execution — return `{"validation_error": ...}` for self-correction. Implemented as `ArgumentValidationMiddleware` (validation.py); validates dict-schema MCP tools client-side via jsonschema (pydantic-schema tools already validated by the framework).

Latency
- [ ] **Parallelize the ~118s monolithic analysis tools** (`find_anti_analysis_techniques`, `detect_malware_behaviors`, `extract_iocs_with_context`, `detect_crypto_constants`, `analyze_api_call_chains`) via `asyncio.gather` / LangGraph `Send`
- [ ] **Enable streaming LLM responses** to overlap generation with tool execution
- [ ] **Route routine/structured-output LLM calls to a smaller, faster model** (model-router at middleware layer)
- [x] **Batch independent read-only tool calls** — prompt the agent to call independent read-only tools simultaneously. Added "Batch independent tool calls" section to SYSTEM_PROMPT (prompt.py).

Sub-agent design
- [ ] **`function-analyst` sub-agent (build first)** — decompile/xref/analysis tools behind a delegation boundary so only structured findings return to the main agent
- [ ] **`program-recon` sub-agent (quick win)** — consolidate the "what binary is this" preamble into one delegation returning a compact JSON brief
- [ ] **`threat-hunter` sub-agent (latency isolation)** — fan-out-and-aggregate wrapper for the heavy threat-analysis tools off the main critical path
- [ ] Keep search primitives, knowledge queries, and filesystem tools on the main agent (no sub-agent)

### Backlog (deferred — not now)
- [ ] **Spill large tool outputs to a file instead of re-injecting** — *already implemented in deepagents:* `FilesystemMiddleware` offloads tool results over `tool_token_limit_before_evict` (default 20k tokens / ~80 KB) to `large_tool_results/`, leaving a preview + pointer. The hard part is lowering that threshold: `create_deep_agent` doesn't expose it, hardcodes `FilesystemMiddleware` in 3 places (graph.py:645/720/779), and the clean overrides are blocked — duplicate-instance assertion (factory.py:1080) and `_REQUIRED_MIDDLEWARE` blocks `excluded_middleware` (graph.py:230). Lowering it needs a monkeypatch (subclass + swap `deepagents.graph.FilesystemMiddleware`) or a custom offload middleware (~80 lines). Not worth it now for a non-urgent latency/cost win; revisit if deepagents exposes the knob or context bloat becomes a measured problem.
- [ ] **Add graph-level timeout & error boundary** to top-level LangGraph — wall-clock timeout (~20 min) / recursion limit with graceful early-exit returning partial findings
- [ ] **Bound `task` sub-agents** — max tool-call rounds + wall-clock timeout, return partial results on expiry

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
