# TODOs

- [x] **Plan mode for the RE agent** — implemented as a separate read-only agent
  graph rather than a tool-blocking middleware. A `PLAN_MODE_BLOCKED_TOOLS` denylist
  (`subagents.py`) defines the mutating tools (Ghidra renames/retypes/comments/
  prototypes + `save_knowledge`/`update_knowledge`); read-only is "everything else".
  `build_research_subagent` builds a shared read-only `research` sub-agent (full tool
  set minus the denylist) used by **both** the normal coordinator (delegate
  investigation without applying changes) and a new plan-mode coordinator graph
  (`build_plan_mode_main_tools` + `PLAN_MODE_SYSTEM_PROMPT`), built in cli.py and
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
  last_active_at, title}` on session start (cli.py) and on each turn (TUI
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
  2026-06-29 and parked because the QuickJS interpreter runtime is beta. (The prior design
  writeup, `~/.claude/plans/langchain-came-out-with-ticklish-scone.md`, no longer exists on
  disk — the "Dynamic subagents — split `research`" section below is the surviving design
  record.) *Update (2026-07-30):* langchain-quickjs 0.3.5 shipped alongside deepagents 0.7
  (which now has a `quickjs` extra), but `CodeInterpreterMiddleware` is still marked
  experimental — stays parked; re-check when deepagents 0.8 lands. *Update (2026-07-30,
  OpenShell criterion):* evaluated whether the interpreter could run inside the
  `SANDBOX=openshell` backend — it cannot: `CodeInterpreterMiddleware` embeds QuickJS in the
  agent process via `quickjs-rs` (docs: "runs in an embedded QuickJS context, not a separate
  VM or process"), exposes no executor/backend parameter, and its `task()`/PTC bridges are
  in-process callbacks into the live agent loop that can't cross into a sandbox — the docs
  treat interpreters and sandboxes as disjoint features. Stays parked until that changes (or
  the in-process constraint is accepted); still beta as of langchain-quickjs 0.3.5.

### From the deepagents 0.7 upgrade (2026-07-30)

deepagents 0.6.12 → 0.7.0 landed the seam this repo was missing: a `middleware=`
(or sub-agent `middleware`) instance whose `.name` matches a built-in now
**replaces** the default instead of tripping the duplicate assertion. The
upgrade itself rewrote `compaction.py` onto that seam (monkeypatch deleted) and
dropped `TodoListMiddleware` with the new lean defaults (0.7 cuts a
default-agent turn's input tokens 65%). Follow-ups it unlocks:

- [ ] **Spill large tool outputs — now unblocked** — pass
  `FilesystemMiddleware(tool_token_limit_before_evict=N)` (default 20k tokens,
  `filesystem.py:1337`) via replace-by-name `middleware=`; no fork or internal
  patching. See the backlog entry above for the original evidence; wire it to an
  env knob like the `COMPACT_*` family if a lower threshold proves out.
- [ ] **Upstream report-extraction quirk, guards reverted** — deepagents 0.7's
  `_return_command_with_state_update` (`middleware/subagents.py:495-505`) still
  picks the sub-agent report as the last non-empty `AIMessage` without checking
  `tool_calls`, so a run ending on a tool call reports only its preamble. The
  local guard layer for it (#57 report guard + #60 sentinel/reply protocols)
  was reverted 2026-08-04: redundant once the real cause was fixed — the 4096
  max_tokens fallback (#58, kept) — and the `zai:` provider (#59) left the
  anthropic-compat path; the guards had started misfiring on good output. If
  preamble-only reports ever recur, the remaining lever is reverting #56
  (provisional-findings persistence); the full bail-out recipe is in git
  history at c3dafd7's version of this file.
- [ ] **Revisit the deferred prompt-trim sub-items** — the TodoListMiddleware
  injection is gone as of this upgrade, and built-in tool descriptions are now
  overridable directly (`FilesystemMiddleware(custom_tool_descriptions={...})`).
  0.7's own 43% tool-description trim may make further trimming moot — measure a
  live turn's token breakdown before doing anything.
- [x] **Evaluate harness profiles** (`deepagents/profiles/`) — per-model/provider
  config bundles (Anthropic, OpenAI Codex, NVIDIA Nemotron 3 Ultra, an OpenRouter
  provider profile) that `create_deep_agent` applies automatically, including
  per-profile middleware exclusion by name. Could replace some hand-rolled
  per-model config in `models.py` / `openrouter.toml`; also worth checking none
  auto-apply unexpectedly to our model specs. *Evaluated 2026-07-30 — no action.*
  Two orthogonal systems: `HarnessProfile` (runtime: prompt suffix, extra
  middleware, tool-description overrides, exclusion by name; matched purely on
  `provider:model`/`provider` strings, no `profile=` param or disable env,
  unmatched models get an empty no-op) and `ProviderProfile` (construction
  kwargs, string specs only). **Nothing auto-applies to our specs:** all 14
  built-in harness profiles are per-model (no provider-wide `anthropic`/
  `openrouter` keys), 2-colon specs like `openrouter:z-ai/glm-5.2:floor` are
  rejected by the lookup, preset-backed models are pre-built instances whose
  identifiers match nothing, and nothing is registered for `deepseek`. Only
  latent match is subagents.py's last-resort `anthropic:claude-sonnet-4-6`
  default → a benign 3-block prompt suffix. No shipped profile excludes any
  middleware, so the replace-by-name compaction seam is unaffected. **Nothing
  to migrate:** profiles carry no token/context/temperature/summarization
  knobs; `openrouter.toml` is per-model *routing* a per-provider
  `ProviderProfile` can't express; `_ChatDeepSeekFixed` is a wire-payload fix
  with no profile hook. Caveats recorded, both currently harmless: (a)
  pre-built `ChatOpenRouter` instances skip the `openrouter` ProviderProfile
  (version floor moot — pyproject pins ≥0.2.6; app-attribution headers
  cosmetic; the `ignore: ["azure"]` workaround moot while every preset pins
  providers via `only` — revisit if a preset without `only`/`order` is added);
  (b) **Nemotron footgun** — `openrouter:nvidia/nemotron-3-ultra-550b-a55b`
  (+7 sibling keys) auto-applies 12 middlewares incl. hard budget caps
  (16 model calls / 48 tool results) that would strangle our loops, for string
  specs *and* pre-built instances; avoid Nemotron ids in subagents.toml
  (registration is additive-merge, no unregister API).

### From optimization report (2026-06-28, 7d window)

Cost
- [x] **Right-size subagent model & context** — agents are now defined declaratively in `subagents.toml` (per-agent model + tool allowlist), loaded by `subagents.py`; the coordinator is restricted to orchestration + navigation/search (analysis/mutation tools moved to sub-agents). Per-agent models leverage OpenRouter. "Task-specific artifacts, not full history" is already handled by deepagents' `task` isolation. (The *dynamic* per-call model-router is still the separate Latency item below.)
- [x] **Tune forced compaction** — a deepagents `SummarizationMiddleware` with env-tuned `trigger`/`keep` (`COMPACT_TRIGGER_FRACTION`/`_TOKENS`, `COMPACT_KEEP_MESSAGES`/`_FRACTION`; profile-aware — fractions fall back to tokens with a warning when the model has no context profile), summary call routed to `SUMMARY_MODEL`. Applies to the main agent and all sub-agents; no-env = deepagents defaults unchanged. *Reworked 2026-07-30:* originally a monkeypatch of `deepagents.graph.create_summarization_middleware` (0.6.x had no other seam); the 0.7 upgrade replaced it with `compaction.py`'s `build_tuned_summarization_middleware`, passed per scope via 0.7's replace-by-name `middleware=`. Tool-*arg* truncation is already active via deepagents' `truncate_args_settings`; lowering the large-tool-*result* offload threshold is now unblocked ("From the deepagents 0.7 upgrade" section).
- [x] **Trim per-call prompt bloat** — audited & compressed `SYSTEM_PROMPT` (prompt.py) ~35% (6.2k→4.0k chars) by removing duplication: the verbose 7-step function-analyst loop (already verbatim in that sub-agent's prompt), the repeated recon/analyze/mutate Workflow section, and per-tool KB prose — every directive (trust-assembly, delegation, batching, KB usage, naming, param-names, never-guess) preserved. Remaining sub-items left for a later pass: *(2026-07-30: mostly overtaken by deepagents 0.7 — TodoListMiddleware is no longer wired at all, and tool-description overrides are now first-class; see "From the deepagents 0.7 upgrade" section)* conditionally skipping FilesystemMiddleware's filesystem-tree injection when irrelevant, and overriding built-in tool descriptions (MCP tool descriptions are server-authored, not ours to compress).
- [x] **Conditionally disable `AnthropicPromptCachingMiddleware`** when running non-Anthropic providers (e.g. DeepSeek) — no-op: the middleware isn't wired into this codebase, and the library version already no-ops for non-Anthropic models (isinstance check). Nothing to do.
- [x] **openrouter provider selection** — implemented: optional `openrouter.toml` (path overridable via `OPENROUTER_CONFIG`, see `openrouter.toml.example`) maps each OpenRouter model id to a provider-routing object (`order`/`allow_fallbacks`/`sort`/…). `build_model` (models.py) constructs `ChatOpenRouter(openrouter_provider=...)` when a preset exists, else resolves the string as before.

Errors
- [x] **Harden `update_knowledge`** — retries + backoff, entity-exists guard, return structured warning instead of raising (highest per-tool error rate, 5.6%). Also applied to `save_knowledge` (sibling write tool).
- [x] **Add tool-call retry for transient failures** — implemented in `resilience.py` (`build_tool_retry_middleware`): stock `ToolRetryMiddleware` scoped to the idempotent filesystem tools (`write_file`/`edit_file`/`read_file`), `retry_on=(OSError,)`, `on_failure="continue"`. Wired into the main agent (cli.py) and every sub-agent (subagents.py). `TOOL_MAX_RETRIES` env (default 3). (Merged with the 2026-06-29 enrichment note below.)
- [x] **Pydantic argument-validation shim** before tool execution — return `{"validation_error": ...}` for self-correction. Implemented as `ArgumentValidationMiddleware` (validation.py); validates dict-schema MCP tools client-side via jsonschema (pydantic-schema tools already validated by the framework).

Latency
- [x] **Parallelize the ~118s monolithic analysis tools** (`find_anti_analysis_techniques`, `detect_malware_behaviors`, `extract_iocs_with_context`, `detect_crypto_constants`, `analyze_api_call_chains`) — N/A here: these are *server-side* Ghidra MCP tools (no references in `src/`), so the client can't `asyncio.gather`/`Send` their internals. The only client-side lever is batching the independent calls in one turn, which is already done (the `threat-hunter` sub-agent prompt instructs invoking them together, plus the completed "Batch independent tool calls" item). Reopen as a Ghidra-MCP-server task if their internals need parallelizing.
- [x] **Enable streaming LLM responses** — already done: the TUI consumes `astream_events` (tui/app.py) and renders `on_chat_model_stream` token events (tui/events.py). "Overlap generation with tool execution" doesn't apply to the linear ReAct loop (tools run only after the model emits the tool calls).
- [ ] **Route routine/structured-output LLM calls to a smaller, faster model** (model-router at middleware layer)
- [x] **Batch independent read-only tool calls** — prompt the agent to call independent read-only tools simultaneously. Added "Batch independent tool calls" section to SYSTEM_PROMPT (prompt.py).

Sub-agent design — implemented in `src/ghidra_deep_agent/subagents.py` (`build_subagents`), wired via `subagents=` in cli.py, delegation guidance in prompt.py. Sub-agents run on `SUBAGENT_MODEL` (defaults to main `MODEL`).
- [x] **`function-analyst` sub-agent (build first)** — full per-function loop: decompile/xref/analysis + applies renames/retypes/comments/prototype + saves findings; returns a compact summary.
- [x] **`program-recon` sub-agent (quick win)** — read-only "what binary is this" delegation returning a compact brief.
- [x] **`threat-hunter` sub-agent (latency isolation)** — isolates the heavy threat-analysis tools off the main critical path; writes findings to the KB, returns a compact summary.
- [x] Keep search primitives, knowledge queries, and filesystem tools on the main agent (no sub-agent) — prompt steers quick searches/KB queries/filesystem reads to the main agent; sub-agent tool allowlists exclude them.

### Backlog (deferred — not now)
- [ ] **Custom OpenShell sandbox image with RE tooling** — the `SANDBOX=openshell`
  backend (shipped) gives the agent a generic isolated shell, but the base OpenShell
  image ships only dev tools (git/python/node/networking) — no RE tooling — and the
  sample binary is not in the sandbox (it lives in the Ghidra project). To make the
  `execute` tool actually useful for reverse engineering, build a custom sandbox image
  preloaded with `binwalk`, `radare2`/`r2pipe`, `capa`, `yara`, `objdump`/`nm`/
  `readelf`/`file`, and unpackers, **and** add a path to get the sample binary into the
  sandbox (e.g. seed it via the sync dir / `upload_files`). Until then the shell is a
  generic scratch/scripting environment, not an RE toolbox. See the tradeoff notes in
  `~/.claude/plans/i-want-to-add-pure-sifakis.md`.
- [ ] **Hard-lock sandbox writes to `/sandbox/output` (optional)** — the agent is
  already steered there (it is the default working directory, and the system prompt
  says durable files go there), but nothing *prevents* a command from writing to an
  absolute path like `/tmp/x`, which is lost on teardown. For a true guarantee, author
  an OpenShell filesystem policy (`openshell policy set`) that makes `/sandbox/output`
  (+ `/tmp`) the only writable paths. Deferred because it is aggressive (can break
  legitimate writes to caches/home) and the policy engine is alpha — needs tuning and
  testing. See `~/.claude/plans/i-want-to-add-pure-sifakis.md`.
- [ ] **Run the agent under Docker Sandboxes (`docker sbx`)** — assessed 2026-07-13:
  **works**. The agent is a pure network client (MCP-over-HTTP to GhidrAssistMCP, TCP
  to MongoDB, HTTPS to the model API, optional Ollama), so it fits sbx's microVM +
  egress-allowlist model: Ghidra/Ollama on the host stay reachable via
  `host.docker.internal` after `sbx policy allow network localhost:<port>`. One
  caveat: the sbx proxy carries HTTP(S) only — MongoDB's raw-TCP wire protocol likely
  can't reach host/Atlas Mongo, so local mode runs the existing `mongodb/` compose
  stack *inside* the sandbox's own Docker daemon (loopback bypasses the proxy);
  external/Atlas mode is kept but experimental until empirically tested. Zero Python
  changes needed (`cli.py` `load_dotenv()` doesn't override exported env). Full
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
- [ ] **Spill large tool outputs to a file instead of re-injecting** — *already implemented in deepagents:* `FilesystemMiddleware` offloads tool results over `tool_token_limit_before_evict` (default 20k tokens / ~80 KB) to `large_tool_results/`, leaving a preview + pointer. *Update (2026-07-30):* the blockers were 0.6.x-era — deepagents 0.7's replace-by-name middleware merge removed the duplicate-instance assertion, so a custom `FilesystemMiddleware(tool_token_limit_before_evict=N)` passed in `middleware=` now just works. Tracked as an unblocked item under "From the deepagents 0.7 upgrade" above; still a non-urgent latency/cost win — do it when context bloat is a measured problem.
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

#### From the repo audit (2026-07-24)
- [~] **Dispatch-table `handle_event` / split `_run_agent`** — the *state* half is
  done (2026-07-25): `handle_event`'s bookkeeping moved into a `RunState`
  (`tui/run_state.py`) the app replaces per turn, which ended its eleven reaches
  into app privates and fixed the leak where a cancelled turn left in-flight
  run_ids behind forever. `_run_agent`'s mode selection collapsed into `SideMode`
  (`tui/side_mode.py`). The **branch tables themselves are still deliberately
  not built**: both read fine as linear routers, and the branch sets share locals
  (`run_id`/`metadata`/`checkpoint_ns`/`is_compaction` in the first) that a table
  would have to thread through a context object — roughly as much scaffolding as
  it removes. Revisit if either grows another branch.

#### From the repo health pass (2026-07-25)

Closed a set of defects that were invisible to ruff and mypy --strict, hence
survived a green CI — see the commits for detail. Notable ones worth remembering
as *classes* of bug rather than one-offs:

- `build_model` returns a `str` for any provider it doesn't special-case, so
  anything that *uses* the model object (`.ainvoke`, `.profile`) silently fails
  on most models. `ensure_chat_model` now resolves at the two call sites that
  need it, but the `BaseChatModel | str` union is still the root cause and will
  keep producing this. Narrowing it is the real fix.
- A tool added to a factory must also be added to `mcp_cache._MUTATING_TOOLS` if
  it mutates Ghidra; `deobfuscate_cff` (#40) wasn't, and served stale
  decompilation for a full TTL. That set is still hand-maintained against
  `subagents.toml` with nothing enforcing the correspondence — a cross-check
  would close the class.
- Textual `@work(exclusive=True)` cancels the whole *group*, so any fire-and-
  forget worker sharing the default group with `_run_agent` dies. Give new
  workers an explicit `group=`.

Still open from that pass:
- [ ] **Share the embedded Java helper preamble** — ~160-180 of ~1,056 non-OLLVM
  Java lines are mechanical copies across the four `*_script.py` modules: `js()`
  (3 verbatim + 1 degenerate `jsonEsc`), `firstLine()`, `addrString()`,
  `appendObj()`, the marker/emit block, and ~30 lines of
  `ChunkingParallelDecompiler` scaffolding duplicated between `find` and
  `recover`. Extract a shared Python constant holding the common preamble,
  concatenated into each `SCRIPT_SOURCE`. **Preserve the args-not-interpolation
  rule** — nothing may be `.format()`ed into the source, which is why there is no
  escaping bug today. Deferred as the largest-diff, lowest-urgency item; the
  compile check (`-m integration`) is what verifies it.
- [ ] **Delete `src/ghidra_deep_agent/web/__pycache__/`** — 33 KB of orphaned
  bytecode for a web subpackage that was never tracked and no longer has any
  source. Local to the primary checkout, invisible to `git status` (ignored).

#### From dependency review (2026-07-20)
- [ ] **Adopt `ToolErrorMiddleware` (langchain 1.3.14)** — *(version gate cleared:
  the deepagents 0.7 upgrade, 2026-07-30, pulled langchain ≥1.3.14)* — evaluate folding the new
  `ToolErrorMiddleware` in alongside our existing `build_tool_retry_middleware`
  (`resilience.py`) and `ArgumentValidationMiddleware` (`validation.py`) for cleaner
  tool-error → self-correction handling. Note 1.3.14 also tightened
  `ToolRetryMiddleware` to only retry *retryable* exceptions — cross-check that our
  `retry_on=(OSError,)` scoping (`resilience.py`) still behaves as intended after the
  upgrade. Low urgency — a robustness cleanup, not a fix.
- [ ] **Evaluate mcp-adapters 0.3.0 MCP error surfacing** — 0.3.0 now surfaces MCP
  tool execution errors as failed tool output. Assess whether this changes how Ghidra
  (GhidrAssistMCP) tool failures reach the agent, and whether it lets us thin out any
  of our own hardening in `resilience.py` / `validation.py` — or, conversely, causes
  double-reporting of the same failure. Behavioral eval against a live server.
- [ ] **pymongo `session.bind()` ergonomics (pymongo 4.17+)** — 4.17 adds a
  context-manager `session.bind()` that scopes all ops to a session without passing it
  explicitly; could simplify session-scoped Mongo work in `sessions.py` / `knowledge.py`.
  *Blocked:* `langgraph-checkpoint-mongodb==0.4.0` (latest) caps `pymongo>=4.12,<4.17`,
  so the resolver holds pymongo at 4.16.0 — 4.17 can't be selected until upstream
  relaxes that ceiling in a newer `langgraph-checkpoint-mongodb` release. Revisit then.
  Low-priority.

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
  Wired into the main agent (cli.py) and every sub-agent (subagents.py). Env: `MODEL_MAX_RETRIES`
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
  (see AGENT_OUTPUT_DIR handling in cli.py) — e.g. a `plans/` subdirectory —
  so plans survive across sessions like other artifacts.
- **Ask for feedback / approval gate:** end the planning turn by returning the
  markdown and waiting for the human, rather than charging ahead.
- **Likely plug-in points in this codebase:**
  - A `/plan` slash command in the TUI dispatcher
    (src/ghidra_deep_agent/tui/app.py).
  - Either a dedicated planning subagent (deepagents `task` mechanism,
    constrained to read-only Ghidra tools + knowledge query tools) or a
    plan-specific system-prompt variant alongside src/ghidra_deep_agent/prompt.py.
  - Reuse the FilesystemBackend already wired up in cli.py for writing the
    plan file.

## `/resume` — list & resume previous sessions
Add a `/resume` slash command (TUI dispatcher in src/ghidra_deep_agent/tui/app.py)
that lists previous sessions sorted most-recent-first and lets the human pick one
to continue. Today sessions can only be resumed by passing an explicit
`--session-id` (cli.py `_parse_args`), with no way to discover what prior session IDs exist
— `/resume` should surface that list interactively.

Design thoughts:
- **Where the data lives.** Sessions are persisted as LangGraph checkpoints via
  `MongoDBSaver` (cli.py `MongoDBSaver.from_conn_string`, `MONGODB_DB` default
  `checkpointing_db`), keyed by `thread_id` (= our `session_id`, cli.py `config`).
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
  isolation — cli.py `binary_name_override`, `BINARY_NAME`). Default to
  filtering the list to the current binary, and offer an option to show all
  sessions across binaries.
- **Plug-in points:** the `/resume` command in the TUI dispatcher
  (src/ghidra_deep_agent/tui/app.py); session-record writes wired alongside the
  `MongoDBSaver`/`binary_name` setup in cli.py; reuse the existing `session_id`
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
  timeout, dispatched-run event shape) were written up in
  `~/.claude/plans/langchain-came-out-with-ticklish-scone.md`, but that file no longer exists
  on disk — this section is the surviving design record; the flag-gating and TUI observability
  work would need re-deriving. If QuickJS is still a blocker, the planner→workers→synthesizer shape can be
  approximated with the existing static `task` tool (batched same-turn parallel `task` calls),
  at the cost of code-driven orchestration.

## OpenRouter support
Add support for using OpenRouter as a model provider. LangChain should have a
library/integration for it (e.g. `langchain-openai` pointed at the OpenRouter
base URL, or a dedicated OpenRouter package) — investigate which is the right
fit and wire it into the model configuration.
