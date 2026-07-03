# ghidra-deep-agent

A reverse engineering agent built on the [LangChain Deep Agents SDK](https://docs.langchain.com/oss/python/deepagents/overview). It connects to a running Ghidra instance through the [GhidrAssistMCP](https://github.com/symgraph/GhidrAssistMCP) MCP server and iteratively analyzes binaries — reading assembly, understanding behavior, and writing its findings back into the Ghidra project as renamed functions, typed variables, and updated prototypes.

## How it works

The agent uses Ghidra's MCP server as its primary toolset. On each turn it can:

- Fetch disassembly and decompiler output for any function or address range
- Rename functions, local variables, and parameters
- Update variable and parameter types
- Set function prototypes
- Add comments at specific addresses

It treats the assembly as ground truth. Every rename or retype it applies is grounded in specific evidence from the disassembly — it won't guess. As it learns more about the binary it refines its earlier work, building up a progressively more readable Ghidra project.

Conversation history is persisted to MongoDB via `langgraph-checkpoint-mongodb`, so sessions survive restarts and the agent can continue where it left off. Findings are also stored in a MongoDB vector collection so the agent can retrieve prior knowledge across sessions.

The knowledge base is scoped per binary — each program analyzed gets its own isolated namespace so findings never bleed between targets. When multiple binaries are open in Ghidra at startup, the agent presents a selection screen. A global semantic search tool is also available when cross-binary comparison is useful.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Ghidra 11.4+ (tested with 12.1) with the [GhidrAssistMCP](https://github.com/symgraph/GhidrAssistMCP) extension installed and its MCP server enabled (HTTP or SSE)
- MongoDB instance (local or remote)
- An Anthropic API key (or any other supported LangChain model provider)
- [Ollama](https://ollama.com/) (for the vector knowledge base embeddings)

## Setup

```bash
git clone https://github.com/linted/ghidra-deep-agent
cd ghidra-deep-agent
uv sync
cp .env.example .env
```

Edit `.env` with your API key, MongoDB URI, and Ghidra MCP server config.

## Configuration

All configuration is done via environment variables (`.env` file or shell exports).

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | API key for Anthropic *(not needed for Ollama)* |
| `MODEL` | `anthropic:claude-sonnet-4-6` | Any `provider:model` string supported by LangChain (also `openrouter:<model-id>` — see [Using OpenRouter](#using-openrouter)) |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter *(required for `openrouter:` models)* |
| `OPENROUTER_CONFIG` | `./openrouter.toml` | Optional TOML of per-model OpenRouter provider-routing presets — see [Pinning providers](#pinning-providers-provider-routing) |
| `SUMMARY_MODEL` | *(main `MODEL`)* | Optional `provider:model` for the conversation-summarization call (manual `/compact` **and** the auto summarizer) — route it to a smaller/cheaper model |
| `COMPACT_TRIGGER_FRACTION` | *(deepagents default ~0.85)* | Auto-compact when context usage reaches this fraction (0-1) — lower compacts earlier |
| `COMPACT_TRIGGER_TOKENS` | *(unset)* | Absolute token trigger for auto-compaction (used if fraction unset) |
| `COMPACT_KEEP_MESSAGES` | *(deepagents default)* | Recent messages to keep after a compaction |
| `MODEL_FALLBACK` | *(unset)* | Comma-separated `provider:model` fallbacks tried, in order, after the primary model's retries are exhausted |
| `MODEL_MAX_RETRIES` | `3` | Retry attempts per model call on transient errors (5xx/429/timeouts) |
| `TOOL_MAX_RETRIES` | `3` | Retry attempts for transient filesystem-tool I/O errors |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL *(only needed if non-default)* |
| `EMBED_MODEL` | `ollama:nomic-embed-text` | `provider:model` for embeddings — supports `ollama`, `openai`, `huggingface`, `cohere`, `automated` (MongoDB Atlas Automated Embeddings via Voyage AI; requires an Atlas cluster with Voyage AI configured at the project level, e.g. `automated:voyage-4`) |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string for checkpoint persistence |
| `MONGODB_DB` | `checkpointing_db` | Database used by the checkpointer and knowledge base |
| `MONGODB_VECTOR_COLLECTION` | `re_knowledge` | Collection for the vector knowledge base |
| `MONGODB_TOOL_CACHE_COLLECTION` | `tool_cache` | Collection caching immutable read-only MCP tool results |
| `MONGODB_TOOL_CACHE_TTL` | `86400` | Cache entry lifetime in seconds (TTL index); sized to a session |
| `MONGODB_TOOL_CACHE_TOOLS` | *(immutable read set)* | Comma-separated allowlist override; empty disables the cache |
| `BINARY_NAME` | *(auto-detected)* | Override the binary name used to scope the knowledge base — see [Binary selection](#binary-selection) |
| `GHIDRA_MCP_TRANSPORT` | `http` | Transport type: `http` or `sse` (GhidrAssistMCP is HTTP-only) |
| `GHIDRA_MCP_URL` | `http://localhost:8080/mcp` | URL of the GhidrAssistMCP server (`/mcp` for http, `/sse` for sse) |
| `AGENT_OUTPUT_DIR` | *(unset)* | Optional directory the agent can read/write files in |
| `RECURSION_LIMIT` | `100` | LangGraph recursion limit for deep analysis sessions |
| `AGENTS_MD` | *(unset)* | Optional path to an `AGENTS.md` memory file |
| `LANGSMITH_API_KEY` | *(unset)* | *(optional)* LangSmith API key to enable run tracing |
| `LANGSMITH_TRACING` | *(unset)* | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_PROJECT` | *(unset)* | LangSmith project name for traces |

### Using Ollama

Set `MODEL` to `ollama:<model-name>` — no API key needed. The model **must support tool calling**; good options for code/RE work:

| Model | Pull command |
|---|---|
| `qwen2.5-coder:32b` | `ollama pull qwen2.5-coder:32b` |
| `devstral` | `ollama pull devstral` |
| `llama3.3` | `ollama pull llama3.3` |

```env
MODEL=ollama:qwen2.5-coder:32b
# OLLAMA_HOST=http://localhost:11434  # only if non-default
```

Ollama must be running (`ollama serve`) and the model must already be pulled before starting the agent.

### Using OpenRouter

[OpenRouter](https://openrouter.ai) provides a single API key for many models across
providers. It is a built-in LangChain provider (via the bundled `langchain-openrouter`
package), so set `MODEL` to `openrouter:<model-id>` (using the model id as it appears
on OpenRouter) and provide `OPENROUTER_API_KEY`:

```env
MODEL=openrouter:anthropic/claude-3.5-sonnet
OPENROUTER_API_KEY=sk-or-...
```

The selected model **must support tool calling**.

#### Pinning providers (provider routing)

OpenRouter can route the same model to different upstream providers. To control
which providers it picks (e.g. pin to one, set an order, or sort by throughput),
copy `openrouter.toml.example` to `openrouter.toml` (or point `OPENROUTER_CONFIG`
at a file) and add a preset per model id:

```toml
[providers."anthropic/claude-3.5-sonnet"]
order = ["Anthropic"]
allow_fallbacks = false
```

With no file present, OpenRouter's default routing is used. See
[provider routing](https://openrouter.ai/docs/features/provider-routing) for all fields.

### Transport options

The agent connects to [GhidrAssistMCP](https://github.com/symgraph/GhidrAssistMCP),
a Ghidra 11.4+/12.1 extension that serves MCP over HTTP. Enable the plugin in
Ghidra (Window → GhidrAssistMCP → turn the MCP server on) with a program open,
then point the agent at it:

**HTTP** (default):
```env
GHIDRA_MCP_TRANSPORT=http
GHIDRA_MCP_URL=http://localhost:8080/mcp
```

**SSE**:
```env
GHIDRA_MCP_TRANSPORT=sse
GHIDRA_MCP_URL=http://localhost:8080/sse
```

> GhidrAssistMCP listens on port `8080` by default. If another service uses that
> port, change it in the GhidrAssistMCP control panel and update `GHIDRA_MCP_URL`.

### Binary selection

At startup the agent calls `list_binaries` on the Ghidra MCP server to determine which binary you are working on. This name is used to scope all knowledge base reads and writes so findings from different binaries never mix.

- **One program open** — selected automatically, printed to console.
- **Multiple programs open** — a selection screen appears before the main TUI; use arrow keys and Enter to choose.
- **Override** — set `BINARY_NAME` in `.env` or pass `--binary-name` on the command line to skip detection entirely (useful for scripting or when the MCP server doesn't expose `list_binaries`).

```env
BINARY_NAME=firmware_v2.bin
```

```bash
uv run python main.py --binary-name firmware_v2.bin
```

### Optional: AGENTS.md memory file

Create an `AGENTS.md` file and point `AGENTS_MD` at it. The agent loads it into its context at the start of every session — useful for recording the binary's architecture, known data structures, and naming conventions:

```markdown
# Binary: firmware.bin
Architecture: ARM Cortex-M4, little-endian
Format: raw binary, base address 0x08000000

## Known structures
- `0x08001234` — interrupt vector table
- `0x08002000` — main application entry

## Naming conventions
- Peripheral drivers: `drv_<peripheral>_<action>`
- ISR handlers: `isr_<source>`
```

```env
AGENTS_MD=./AGENTS.md
```

## Running

```bash
uv run python main.py
```

Pass `--session-id` to resume a previous session:

```bash
uv run python main.py --session-id <your-session-id>
```

Pass `--binary-name` to skip Ghidra's program detection:

```bash
uv run python main.py --binary-name firmware_v2.bin
```

The agent connects to the Ghidra MCP server, loads its tools, and opens an interactive prompt. Enter your analysis task in plain English:

```
> what does the function at 0x401000 do?
> rename all functions related to network I/O with a net_ prefix
> find the main loop and document its structure
> what arguments does sub_403200 take? update the prototype
```

Each run starts a new session with a random UUID unless `--session-id` is supplied. The session ID is printed when you exit — use it to resume later. Press `Ctrl+C` or type `quit` to exit.
