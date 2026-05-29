# ghidra-deep-agent

A reverse engineering agent built on the [LangChain Deep Agents SDK](https://docs.langchain.com/oss/python/deepagents/overview). It connects to a running Ghidra instance through the [ghidra-mcp](https://github.com/bethington/ghidra-mcp) MCP server and iteratively analyzes binaries — reading assembly, understanding behavior, and writing its findings back into the Ghidra project as renamed functions, typed variables, and updated prototypes.

## How it works

The agent uses Ghidra's MCP server as its primary toolset. On each turn it can:

- Fetch disassembly and decompiler output for any function or address range
- Rename functions, local variables, and parameters
- Update variable and parameter types
- Set function prototypes
- Add comments at specific addresses

It treats the assembly as ground truth. Every rename or retype it applies is grounded in specific evidence from the disassembly — it won't guess. As it learns more about the binary it refines its earlier work, building up a progressively more readable Ghidra project.

Conversation history is persisted to MongoDB via `langgraph-checkpoint-mongodb`, so sessions survive restarts and the agent can continue where it left off. Findings are also stored in a MongoDB vector collection so the agent can retrieve prior knowledge across sessions.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A running [ghidra-mcp](https://github.com/bethington/ghidra-mcp) server (stdio, HTTP, or SSE)
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
| `MODEL` | `anthropic:claude-sonnet-4-6` | Any `provider:model` string supported by LangChain |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL *(only needed if non-default)* |
| `EMBED_MODEL` | `ollama:nomic-embed-text` | `provider:model` for embeddings — supports `ollama`, `openai`, `huggingface`, `cohere` |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string for checkpoint persistence |
| `MONGODB_DB` | `checkpointing_db` | Database used by the checkpointer and knowledge base |
| `MONGODB_VECTOR_COLLECTION` | `re_knowledge` | Collection for the vector knowledge base |
| `GHIDRA_MCP_TRANSPORT` | `stdio` | Transport type: `stdio`, `http`, or `sse` |
| `GHIDRA_MCP_COMMAND` | `ghidra-mcp` | *(stdio only)* Command to launch the MCP bridge |
| `GHIDRA_MCP_ARGS` | *(empty)* | *(stdio only)* Extra CLI flags, space-separated |
| `GHIDRA_MCP_URL` | `http://localhost:8080/mcp` | *(http/sse only)* URL of the MCP server |
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

### Transport options

**stdio** (default) — the agent spawns the MCP bridge as a subprocess:
```env
GHIDRA_MCP_TRANSPORT=stdio
GHIDRA_MCP_COMMAND=ghidra-mcp
```

**HTTP / SSE** — connect to an already-running MCP server:
```env
GHIDRA_MCP_TRANSPORT=http
GHIDRA_MCP_URL=http://localhost:8080/mcp
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

The agent connects to the Ghidra MCP server, loads its tools, and opens an interactive prompt. Enter your analysis task in plain English:

```
> what does the function at 0x401000 do?
> rename all functions related to network I/O with a net_ prefix
> find the main loop and document its structure
> what arguments does sub_403200 take? update the prototype
```

Each run starts a new session with a random UUID unless `--session-id` is supplied. The session ID is printed when you exit — use it to resume later. Press `Ctrl+C` or type `quit` to exit.
