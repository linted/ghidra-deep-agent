# TODOs

- [x] **Plan mode for the RE agent** — implemented as a separate read-only agent
  graph rather than a tool-blocking middleware. A `PLAN_MODE_BLOCKED_TOOLS` denylist
  (`subagents.py`) defines the mutating tools (Ghidra renames/retypes/comments/
  prototypes + `save_knowledge`/`update_knowledge`); read-only is "everything else".
  `build_research_subagent` builds a shared read-only `research` sub-agent (full tool
  set minus the denylist) used by **both** the normal coordinator (delegate
  investigation without applying changes) and a new plan-mode coordinator graph
  (`build_plan_mode_main_tools` + `PLAN_MODE_SYSTEM_PROMPT`), built in main.py and
  passed to the TUI. Both graphs share the checkpointer `thread_id`/backend so
  history + the plan file carry over. TUI (`tui/app.py`): `/plan [goal]` mints a
  fresh timestamped `plans/<ts>-<slug>.md`, flips a magenta **PLAN** status chip, and
  routes every typed message through the plan graph (re-writing/overwriting that file
  each turn and reading it back as the authoritative "Current plan" block); `/approve`
  exits and tells the normal agent to execute; `/plan-cancel` exits without executing.
  Durable plans need `AGENT_OUTPUT_DIR` (FilesystemBackend); otherwise the plan lives
  in agent state and is read back from there.
- [x] **`/resume` — list & resume previous sessions** — implemented: a dedicated
  `sessions` collection (`sessions.py`, `SessionStore`/`build_session_store`,
  `MONGODB_SESSIONS_COLLECTION`) records `{session_id, binary_name, created_at,
  last_active_at, title}` on session start (main.py) and on each turn (TUI
  `_touch_session`). The `/resume` TUI command opens a `SessionSelectScreen`
  modal (`tui/session_select.py`) listing sessions most-recent-first, scoped to
  the open binary by default with an 'a' key to toggle all binaries; picking one
  swaps the checkpointer `thread_id`/`session_id` and clears the log (minimal
  switch — context stays server-side). Degrades gracefully when Mongo is
  unreachable. Cross-binary resume is a documented soft footgun (tools stay bound
  to the open binary).
- [x] **OpenRouter support**
- [ ] **Dynamic subagents: split `research` into planner → parallel workers → synthesizer** —
  see the "Dynamic subagents — split `research`" section below for the full write-up.
  Evidence (`agent_topology`): 80 LLM calls / 5.92M tokens across 2 invocations (~40 calls /
  ~2.96M tokens each, 84:1 prompt:completion). Expected: 40–60% token reduction and latency
  541s → ~120–180s per invocation. Effort: Med. *Caveat:* dynamic subagents were evaluated
  2026-06-29 and parked because the QuickJS interpreter runtime is beta — prior design work in
  `~/.claude/plans/langchain-came-out-with-ticklish-scone.md`; start there.

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
- [ ] **Run the agent under Docker Sandboxes (`docker sbx`)** — assessed 2026-07-13:
  **works**. The agent is a pure network client (MCP-over-HTTP to GhidrAssistMCP, TCP
  to MongoDB, HTTPS to the model API, optional Ollama), so it fits sbx's microVM +
  egress-allowlist model: Ghidra/Ollama on the host stay reachable via
  `host.docker.internal` after `sbx policy allow network localhost:<port>`. One
  caveat: the sbx proxy carries HTTP(S) only — MongoDB's raw-TCP wire protocol likely
  can't reach host/Atlas Mongo, so local mode runs the existing `mongodb/` compose
  stack *inside* the sandbox's own Docker daemon (loopback bypasses the proxy);
  external/Atlas mode is kept but experimental until empirically tested. Zero Python
  changes needed (`main.py` `load_dotenv()` doesn't override exported env). Full
  design — three `scripts/sbx-*.sh` scripts, `.env.sandbox.example`, README section,
  exact policy rules, verification steps — in
  `~/.claude/plans/are-we-able-to-vectorized-floyd.md`; start there.
- [~] **Adopt GhidrAssistMCP MCP resources & prompts** — the new server (see the
  GhidrAssistMCP migration) exposes, beyond tools, **6 MCP resources**
  (`ghidra://program/{name}/info` / `functions` / `strings` / `imports` /
  `exports` / `segments`) and **7 MCP prompts** (`analyze_function`,
  `identify_vulnerability`, `document_function`, `trace_data_flow`,
  `trace_network_data`, `compare_functions`, `reverse_engineer_struct`).
  **Prompt-wording sub-item DONE** (2026-07-05): audited all 7 verbatim server
  templates (upstream `github.com/symgraph/GhidrAssistMCP`,
  `src/main/java/ghidrassistmcp/prompts/*.java`) against our sub-agent prompts.
  Folded the `reverse_engineer_struct` methodology (get_data_at → xrefs →
  get_code → infer-from-access-patterns → typedef → register) into
  `function-analyst`, and the `trace_network_data` network-protocol guidance plus
  the `identify_vulnerability` TOCTOU/race + information-disclosure categories into
  `vuln-hunter` (all in `subagents.toml`). The other 5 (`analyze_function`,
  `document_function`, `trace_data_flow`, `compare_functions`, and the
  `identify_vulnerability` core) already met or beat the server templates —
  nothing borrowed. **Still open / deliberately deferred:** (a) **resources**
  could replace some `program-recon`/coordinator read *tool* calls with cheaper
  resource reads (marginal — data overlaps existing tools; templated URIs must be
  passed explicitly); (b) any runtime **prompt wiring** — retaining
  `MultiServerMCPClient` and exposing `get_prompt`/`get_resources` via TUI slash
  commands or a data-injected sub-agent primer — was scoped out (wording only).
  Low urgency — a capability-upgrade exploration, not a fix.
- [ ] **TUI approval affordance for plan mode** — replace/augment the `/approve`
  command with an interactive popup or buttons to **Approve / Reject / Keep working**
  on the plan (modal in the `SessionSelectScreen` style, `tui/session_select.py`),
  instead of a typed command.
- [ ] **Spill large tool outputs to a file instead of re-injecting** — *already implemented in deepagents:* `FilesystemMiddleware` offloads tool results over `tool_token_limit_before_evict` (default 20k tokens / ~80 KB) to `large_tool_results/`, leaving a preview + pointer. The hard part is lowering that threshold: `create_deep_agent` doesn't expose it, hardcodes `FilesystemMiddleware` in 3 places (graph.py:645/720/779), and the clean overrides are blocked — duplicate-instance assertion (factory.py:1080) and `_REQUIRED_MIDDLEWARE` blocks `excluded_middleware` (graph.py:230). Lowering it needs a monkeypatch (subclass + swap `deepagents.graph.FilesystemMiddleware`) or a custom offload middleware (~80 lines). Not worth it now for a non-urgent latency/cost win; revisit if deepagents exposes the knob or context bloat becomes a measured problem.
- [ ] **Add graph-level timeout & error boundary** to top-level LangGraph — wall-clock timeout (~20 min) / recursion limit with graceful early-exit returning partial findings
- [ ] **Bound `task` sub-agents** — max tool-call rounds + wall-clock timeout, return partial results on expiry
- [ ] **Give `prototype-fixer` a clear/undefine-function tool** — when `recover_prototypes`
  surfaces a decompile failure whose disassembly is plainly *not a real function*
  (data/padding/misaligned, no coherent prologue), the fixer can currently only
  bookmark it `not-a-function` and report it for a human/analyzer to remove — its
  tool set (`variables`, `bookmarks`, read-only nav) has no way to undefine/clear
  the bogus function. Add a Ghidra clear-function capability (e.g. an MCP
  `clear_function`/`remove_function` tool, or a small local tool wrapping
  `Listing.removeFunction` / `ClearFlowAndRepairCmd`) and grant it to
  `prototype-fixer` so it can delete these itself. Destructive, so gate it behind
  the same plan-mode/mutation controls as other write tools. Deferred out of the
  "surface decompile failures" change on purpose.

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

### From optimization report (2026-07-04, 2h window)

_Report: `ghidra-deepagents-20260704T003612Z.md`. Triaged against the code 2026-07-03; only one
item survived. Full implementation plan already written:
`~/.claude/plans/consider-ghidra-deepagents-20260704t0036-federated-alpaca.md`._

New
- [x] **Cache `get_code`/`xrefs`/`get_data_at` with write-invalidation** (report Latency #5) — implemented
  2026-07-03 (PR #14, `289fcfc`) as described below; smoke-tested (both tiers, tiered debug logging,
  failed-mutation no-flush, per-binary isolation, env opt-out, async path). Judge value via a live
  `MONGODB_TOOL_CACHE_DEBUG=1` session: if `INVALIDATE ... cleared N` wipes dominate mutable-tier `HIT`s,
  set `MONGODB_TOOL_CACHE_MUTABLE_TOOLS=` and drop it.
  Original design notes:
  extend `MCPReadCacheMiddleware` (mcp_cache.py) with a second, *mutable* tool tier
  (`get_code`, `xrefs`, `get_data_at`; env `MONGODB_TOOL_CACHE_MUTABLE_TOOLS`, empty = off).
  Invalidation is whole-binary/whole-tier — `delete_many({binary, mutable: true})` after any
  successful Ghidra-mutating tool (`rename_symbol`, `batch_rename`, `variables`, `comments`,
  `types`, `struct`, `create_function`) — because per-address is unsound (renaming A changes the
  decompilation of every caller of A). Docs gain `binary` + `mutable` fields (no migration; old
  docs are all immutable-tier). Instrument with an `invalidations` counter + tiered
  `MONGODB_TOOL_CACHE_DEBUG` `HIT`/`MISS`/`INVALIDATE ... cleared N` logging so one debug session
  shows whether invalidation churn kills the hit rate (traces showed 226 `get_code` calls/window
  at ~0.6 re-fetch probability, but mutation-heavy sessions may wipe the tier constantly — if so,
  disable via env and drop it). Known limitation: Ghidra-GUI edits bypass invalidation; TTL is the
  backstop.

Rejected / redundant (recorded so they aren't reconsidered next report)
- **Cost #2 / Latency #1 / Errors #2 (`get_task_status` "polling spin-loop", cap polls):**
  mostly a misdiagnosis — polling is code-driven inside `AsyncTaskMiddleware` (async_tasks.py) with
  exponential backoff (0.25s→2s) and a 180s timeout, and no LLM round-trip happens per poll (the
  report's own table shows those spans at 0 tokens). But "the LLM never sees `get_task_status`" was
  only *aspirational*: two residual leaks were closed (2026-07-12). `get_task_status` is now in
  `WITHHELD_TOOLS` (was still granted to the `research` and `general-purpose` wildcard agents), and
  on timeout the middleware returns an explicit "did not complete" message instead of a raw
  `Status: RUNNING` stub that a wildcard agent could have started manually polling.
- **Latency #2/#3 (parallelize tool calls / `task` dispatch):** already concurrent — the app runs
  fully async and langgraph's `ToolNode` gathers same-turn tool calls (incl. `atask`) via
  `asyncio.gather`; serial traces mean the *model* emitted one call per turn (prompt guidance for
  batching already exists).
- **Cost #1 (truncate verbose tool outputs):** already handled — deepagents `FilesystemMiddleware`
  offloads tool results over ~20k tokens (~80 KB) to `large_tool_results/`; see the existing
  backlog item about lowering that threshold. Its 13:1 "chain vs LLM tokens" figure is LangSmith
  double-counting parent spans, not real spend.
- **Cost #3 (dedupe sub-agent system prompts):** no client-side action — DeepSeek does automatic
  server-side prefix caching.
- **Cost #4 (gate `AnthropicPromptCachingMiddleware`):** N/A again — registered upstream by
  deepagents with `unsupported_model_behavior="ignore"`, silently no-ops on DeepSeek/OpenRouter
  (already recorded in the 2026-06-29 pass).
- **Sub-agent #1 (route `analyze_function` into `function-analyst`):** already done — the
  coordinator's allowlist (subagents.toml) excludes it; it's scoped to `function-analyst`,
  `prototype-auditor`, and the wildcard agents. The "inline" calls in traces were sub-agent calls.
- **Errors #1/#5 (retry + compaction observability), Errors #3 (sub-agent timeouts):** real gaps
  but declined for now (2026-07-03) — retries are silent until terminal failure and compaction
  logs no token counts, but neither is currently hurting; timeouts already tracked in Backlog
  ("graph-level timeout", "Bound `task` sub-agents").
- **Cost #4 (move coordinator to DeepSeek) / Latency #4 (faster routing model):** config choice,
  not a code task — `[main] model` in subagents.toml is `openrouter:z-ai/glm-5.2`; flip the one
  line if desired.
- **Errors #4 (smoke-test single-call tools), Sub-agent #2/#3 (don't create cluster sub-agents):**
  generic/no-op advice; nothing to change.

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

## `/resume` — list & resume previous sessions
Add a `/resume` slash command (TUI dispatcher in src/ghidra_deep_agent/tui/app.py)
that lists previous sessions sorted most-recent-first and lets the human pick one
to continue. Today sessions can only be resumed by passing an explicit
`--session-id` (main.py:48), with no way to discover what prior session IDs exist
— `/resume` should surface that list interactively.

Design thoughts:
- **Where the data lives.** Sessions are persisted as LangGraph checkpoints via
  `MongoDBSaver` (main.py:120 `MongoDBSaver`, `MONGODB_DB` default
  `checkpointing_db`), keyed by `thread_id` (= our `session_id`, main.py:57).
- **Sorting by recency may need a new collection.** The checkpoint documents are
  not obviously timestamped in a way that's cheap to sort/query by "most recent",
  and the saver's schema is an implementation detail we shouldn't depend on. We
  likely need a dedicated **`sessions` collection** that we write a small record
  to on session start / each turn — e.g. `{session_id, binary_name, created_at,
  last_active_at, title/summary}` — so `/resume` can do a simple
  `find().sort("last_active_at", -1)`. (Confirm first whether the checkpoint docs
  already carry a usable timestamp before adding the collection.)
- **Filter by open binary.** A `/resume` list is most useful scoped to the
  binary currently open in Ghidra (we already track `binary_name` for knowledge
  isolation — main.py:129 `binary_name_override`, `BINARY_NAME`). Default to
  filtering the list to the current binary, and offer an option to show all
  sessions across binaries.
- **Plug-in points:** the `/resume` command in the TUI dispatcher
  (src/ghidra_deep_agent/tui/app.py); session-record writes wired alongside the
  `MongoDBSaver`/`binary_name` setup in main.py; reuse the existing `session_id`
  / `thread_id` plumbing to actually re-attach to the chosen checkpoint thread.

## Dynamic subagents — split `research` into planner → workers → synthesizer

Look into adding LangChain deepagents **dynamic subagents** (docs:
https://docs.langchain.com/oss/python/deepagents/subagents — attach `langchain-quickjs`
`CodeInterpreterMiddleware` so the coordinator writes a small orchestration script that fans
out subagents in parallel via a `task()` global, instead of one native `task` call per turn)
and use them to restructure the `research` sub-agent:

- **Evidence** (`agent_topology`): 80 LLM calls, 5.92M tokens, 84:1 prompt:completion ratio,
  2 invocations at ~40 calls / ~2.96M tokens each. The agent is accumulating enormous context
  across 40 iterations without effective compaction.
- **Proposed structure:**
  - **research-planner** (lightweight step in main agent): decompose the research question
    into 4–6 sub-queries.
  - **research-worker** (spawned N× in parallel, `_ChatDeepSeekFixed`): each handles one
    sub-query with focused tools (`search_strings`, `search_bytes`,
    `search_functions_by_name`, `query_by_address`, `get_code`, `xrefs`, `grep`) and returns
    a compact summary.
  - **research-synthesizer** (single call, stronger model optional): aggregate sub-summaries
    into the final report.
- **Expected impact:** per-invocation tokens drop from ~2.96M to ~500–800K; per-invocation
  LLM calls from ~40 to ~10–15. **40–60% token reduction** and latency cut from 541s to
  ~120–180s.
- **Effort:** Med.
- **Prior art / caveat:** dynamic subagents were already evaluated (2026-06-29) as a strong
  fit for this project but **parked because the QuickJS interpreter runtime is beta** (runs
  in-process, and interpreter-dispatched runs break the TUI's `is_subagent = name == "task"`
  tracking). The full flag-gated design, TUI observability work, and open questions (5s eval
  timeout, dispatched-run event shape) are written up in
  `~/.claude/plans/langchain-came-out-with-ticklish-scone.md` — start there rather than
  re-deriving. If QuickJS is still a blocker, the planner→workers→synthesizer shape can be
  approximated with the existing static `task` tool (batched same-turn parallel `task` calls),
  at the cost of code-driven orchestration.

## OpenRouter support
Add support for using OpenRouter as a model provider. LangChain should have a
library/integration for it (e.g. `langchain-openai` pointed at the OpenRouter
base URL, or a dedicated OpenRouter package) — investigate which is the right
fit and wire it into the model configuration.
