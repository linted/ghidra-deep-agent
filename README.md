# ghidra-deep-agent

A reverse engineering agent built on the [LangChain Deep Agents SDK](https://docs.langchain.com/oss/python/deepagents/overview). It connects to a running Ghidra instance through an MCP server and iteratively analyzes binaries — reading assembly, understanding behavior, and writing its findings back into the Ghidra project as renamed functions, typed variables, and updated prototypes.

## How it works

The agent uses Ghidra's MCP server as its primary toolset. On each turn it can:

- Fetch disassembly and decompiler output for any function or address range
- Rename functions, local variables, and parameters
- Update variable and parameter types
- Set function prototypes
- Add comments at specific addresses

It treats the assembly as ground truth. Every rename or retype it applies is grounded in specific evidence from the disassembly — it won't guess. As it learns more about the binary it refines its earlier work, building up a progressively more readable Ghidra project.

Conversation history is persisted to MongoDB via `langgraph-checkpoint-mongodb`, so sessions survive restarts and the agent can continue where it left off.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A running Ghidra MCP server (stdio, HTTP, or SSE)
- MongoDB instance (local or remote)
- An Anthropic API key (or any other supported LangChain model provider)

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
| `ANTHROPIC_API_KEY` | — | API key for Anthropic (or swap for another provider) |
| `MODEL` | `anthropic:claude-sonnet-4-6` | Any `provider:model` string supported by LangChain |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string for checkpoint persistence |
| `GHIDRA_MCP_TRANSPORT` | `stdio` | Transport type: `stdio`, `http`, or `sse` |
| `GHIDRA_MCP_COMMAND` | `ghidra-mcp` | *(stdio only)* Command to launch the MCP bridge |
| `GHIDRA_MCP_ARGS` | *(empty)* | *(stdio only)* Extra CLI flags, space-separated |
| `GHIDRA_MCP_URL` | `http://localhost:8080/mcp` | *(http/sse only)* URL of the MCP server |
| `AGENTS_MD` | *(unset)* | Optional path to an `AGENTS.md` memory file |

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

The agent connects to the Ghidra MCP server, loads its tools, and opens an interactive prompt. Enter your analysis task in plain English:

```
> what does the function at 0x401000 do?
> rename all functions related to network I/O with a net_ prefix
> find the main loop and document its structure
> what arguments does sub_403200 take? update the prototype
```

Each session uses the thread ID `re-session`. To start a fresh session (e.g. for a new binary), change the `thread_id` in `main.py` or clear the relevant MongoDB collection.

Press `Ctrl+C` or type `quit` to exit.
