# TODOs

- [ ] **Plan mode for the RE agent**
- [x] **OpenRouter support**

### From optimization report (2026-06-28, 7d window)

Cost
- [x] **Right-size subagent model & context** — agents are now defined declaratively in `subagents.toml` (per-agent model + tool allowlist), loaded by `subagents.py`; the coordinator is restricted to orchestration + navigation/search (analysis/mutation tools moved to sub-agents). Per-agent models leverage OpenRouter. "Task-specific artifacts, not full history" is already handled by deepagents' `task` isolation. (The *dynamic* per-call model-router is still the separate Latency item below.)
- [x] **Tune forced compaction** — `create_deep_agent` hard-wires `create_summarization_middleware(model, backend)` with no trigger/model knob, so `compaction.py`'s `install_tuned_summarization` monkeypatches `deepagents.graph.create_summarization_middleware` to return a deepagents `SummarizationMiddleware` with env-tuned `trigger`/`keep` (`COMPACT_TRIGGER_FRACTION`/`_TOKENS`, `COMPACT_KEEP_MESSAGES`/`_FRACTION`; profile-aware — fractions fall back to tokens with a warning when the model has no context profile) and routes the summary call to `SUMMARY_MODEL`. Applies to the main agent and all sub-agents; no-env = deepagents defaults unchanged. Tool-*arg* truncation is already active via deepagents' `truncate_args_settings`; lowering the large-tool-*result* offload threshold remains the deferred backlog item ("Spill large tool outputs to a file").
- [x] **Trim per-call prompt bloat** — audited & compressed `SYSTEM_PROMPT` (prompt.py) ~35% (6.2k→4.0k chars) by removing duplication: the verbose 7-step function-analyst loop (already verbatim in that sub-agent's prompt), the repeated recon/analyze/mutate Workflow section, and per-tool KB prose — every directive (trust-assembly, delegation, batching, KB usage, naming, param-names, never-guess) preserved. Remaining sub-items left for a later pass (lower payoff / need the same deepagents-internal patching deferred under the backlog): conditionally skipping FilesystemMiddleware's filesystem-tree and TodoListMiddleware injections when irrelevant, and overriding built-in tool descriptions (MCP tool descriptions are server-authored, not ours to compress).
- [x] **Conditionally disable `AnthropicPromptCachingMiddleware`** when running non-Anthropic providers (e.g. DeepSeek) — no-op: the middleware isn't wired into this codebase, and the library version already no-ops for non-Anthropic models (isinstance check). Nothing to do.
- [x] **openrouter provider selection** — implemented: optional `openrouter.toml` (path overridable via `OPENROUTER_CONFIG`, see `openrouter.toml.example`) maps each OpenRouter model id to a provider-routing object (`order`/`allow_fallbacks`/`sort`/…). `build_model` (models.py) constructs `ChatOpenRouter(openrouter_provider=...)` when a preset exists, else resolves the string as before.

Errors
- [x] **Harden `update_knowledge`** — retries + backoff, entity-exists guard, return structured warning instead of raising (highest per-tool error rate, 5.6%). Also applied to `save_knowledge` (sibling write tool).
- [x] **Add tool-call retry for transient failures** — implemented in `resilience.py` (`build_tool_retry_middleware`): stock `ToolRetryMiddleware` scoped to the idempotent filesystem tools (`write_file`/`edit_file`/`read_file`), `retry_on=(OSError,)`, `on_failure="continue"`. Wired into the main agent (main.py) and every sub-agent (subagents.py). `TOOL_MAX_RETRIES` env (default 3). (Merged with the 2026-06-29 enrichment note below.)
- [x] **Pydantic argument-validation shim** before tool execution — return `{"validation_error": ...}` for self-correction. Implemented as `ArgumentValidationMiddleware` (validation.py); validates dict-schema MCP tools client-side via jsonschema (pydantic-schema tools already validated by the framework).

Latency
- [x] **Parallelize the ~118s monolithic analysis tools** (`find_anti_analysis_techniques`, `detect_malware_behaviors`, `extract_iocs_with_context`, `detect_crypto_constants`, `analyze_api_call_chains`) — N/A here: these are *server-side* Ghidra MCP tools (no references in `src/`), so the client can't `asyncio.gather`/`Send` their internals. The only client-side lever is batching the independent calls in one turn, which is already done (the `threat-hunter` sub-agent prompt instructs invoking them together, plus the completed "Batch independent tool calls" item). Reopen as a Ghidra-MCP-server task if their internals need parallelizing.
- [x] **Enable streaming LLM responses** — already done: the TUI consumes `astream_events` (tui/app.py) and renders `on_chat_model_stream` token events (tui/events.py). "Overlap generation with tool execution" doesn't apply to the linear ReAct loop (tools run only after the model emits the tool calls).
- [ ] **Route routine/structured-output LLM calls to a smaller, faster model** (model-router at middleware layer)
- [x] **Batch independent read-only tool calls** — prompt the agent to call independent read-only tools simultaneously. Added "Batch independent tool calls" section to SYSTEM_PROMPT (prompt.py).

Sub-agent design — implemented in `src/ghidra_deep_agent/subagents.py` (`build_subagents`), wired via `subagents=` in main.py, delegation guidance in prompt.py. Sub-agents run on `SUBAGENT_MODEL` (defaults to main `MODEL`).
- [x] **`function-analyst` sub-agent (build first)** — full per-function loop: decompile/xref/analysis + applies renames/retypes/comments/prototype + saves findings; returns a compact summary.
- [x] **`program-recon` sub-agent (quick win)** — read-only "what binary is this" delegation returning a compact brief.
- [x] **`threat-hunter` sub-agent (latency isolation)** — isolates the heavy threat-analysis tools off the main critical path; writes findings to the KB, returns a compact summary.
- [x] Keep search primitives, knowledge queries, and filesystem tools on the main agent (no sub-agent) — prompt steers quick searches/KB queries/filesystem reads to the main agent; sub-agent tool allowlists exclude them.

### Backlog (deferred — not now)
- [ ] **Spill large tool outputs to a file instead of re-injecting** — *already implemented in deepagents:* `FilesystemMiddleware` offloads tool results over `tool_token_limit_before_evict` (default 20k tokens / ~80 KB) to `large_tool_results/`, leaving a preview + pointer. The hard part is lowering that threshold: `create_deep_agent` doesn't expose it, hardcodes `FilesystemMiddleware` in 3 places (graph.py:645/720/779), and the clean overrides are blocked — duplicate-instance assertion (factory.py:1080) and `_REQUIRED_MIDDLEWARE` blocks `excluded_middleware` (graph.py:230). Lowering it needs a monkeypatch (subclass + swap `deepagents.graph.FilesystemMiddleware`) or a custom offload middleware (~80 lines). Not worth it now for a non-urgent latency/cost win; revisit if deepagents exposes the knob or context bloat becomes a measured problem.
- [ ] **Add graph-level timeout & error boundary** to top-level LangGraph — wall-clock timeout (~20 min) / recursion limit with graceful early-exit returning partial findings
- [ ] **Bound `task` sub-agents** — max tool-call rounds + wall-clock timeout, return partial results on expiry

### From optimization report (2026-06-29, 6h window)

_Caveats: the report's cost column is broken (all `$0.0000`) and several sub-agents have only 2
runs, so its small-sample "50% error rate" figures are noise. Most recommendations overlap the
2026-06-28 pass above and are already done/tracked — only the items below are net-new. Verified
against the codebase and the LangChain/deepagents docs._

New
- [x] **Add model-call retry + provider fallback middleware** (report Errors #5) — implemented in
  `resilience.py` (`build_model_resilience_middleware`): stock `ModelRetryMiddleware` (transient-only
  via an `_is_transient` predicate: 5xx/429/timeouts, not deterministic 4xx) plus an optional
  `ModelFallbackMiddleware` (outermost) driven by `MODEL_FALLBACK` (comma-separated `provider:model`).
  Wired into the main agent (main.py) and every sub-agent (subagents.py). Env: `MODEL_MAX_RETRIES`
  (default 3), `MODEL_FALLBACK`.
- [x] **Cache immutable read-only MCP tools in MongoDB** (report Latency #1) — implemented as
  `MCPReadCacheMiddleware` (mcp_cache.py): a `wrap_tool_call`/`awrap_tool_call` cache keyed on
  `(binary, tool, args)` (sha256), scoped to a conservative immutable-read allowlist (`search_strings`,
  `list_imports`, `list_exports`, `get_entry_points`, `get_current_program_info` — `list_functions`/
  `search_functions` deliberately excluded since renames change them). Backed by MongoDB
  (`MONGODB_TOOL_CACHE_COLLECTION`, default `tool_cache`) with a TTL index (`MONGODB_TOOL_CACHE_TTL`,
  default 86400). Only successful results are stored; pymongo I/O is offloaded via `asyncio.to_thread`.
  One shared instance across main + sub-agents; `MONGODB_TOOL_CACHE_TOOLS=` disables it. Hit/miss
  counters + `MONGODB_TOOL_CACHE_DEBUG` provide the call-count instrumentation.

Enrichment of existing items
- *(Merged)* The retry-mechanism note has been folded into the single **"Add tool-call retry for
  transient failures"** item in the Errors section above (use built-in `ToolRetryMiddleware`).

Rejected / redundant (recorded so they aren't reconsidered next report)
- **Cost #2 (restructure for Anthropic prompt caching):** N/A — project runs OpenRouter/DeepSeek;
  the caching middleware isn't wired and no-ops for non-Anthropic models (see done item above).
- **Errors #2 / Sub-agent #2 (merge `program-recon` + `threat-hunter`):** reject — rests on 2-run
  "50%" error rates (noise) and contradicts the deliberate latency-isolation split that keeps the
  heavy threat tools off the recon critical path.
- **Sub-agent #3 (new `data-region-analyst`):** defer — the report itself flags "only 2 traces,
  instrument before committing"; those tools already live in `function-analyst` / `general-purpose`.
- **Cost #1/#3/#4/#5, Errors #1/#3/#4, Latency #2/#3/#5, Sub-agent #1/#4/#5:** already done or
  tracked above (per-agent tool allowlists, batched parallel tool calls in sub-agent prompts,
  `ArgumentValidationMiddleware`, "Tune forced compaction", "Route routine LLM calls to a smaller
  model", backlog "graph-level timeout", backlog "Bound `task` sub-agents").

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
