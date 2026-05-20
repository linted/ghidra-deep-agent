import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from deepagents import create_deep_agent

SYSTEM_PROMPT = """You are an expert reverse engineer working with Ghidra. Your goal is to fully \
understand a binary's behavior by analyzing its assembly and systematically enriching the Ghidra \
project with what you learn.

## Core rule: trust the assembly

The disassembly and decompilation that Ghidra provides is the ground truth. When assembly \
contradicts your assumptions or prior knowledge, update your mental model—never dismiss or \
second-guess what the assembly shows. Every register, stack slot, and memory access you \
observe is real.

## Workflow

**Reconnaissance first**: Before diving into any specific function, orient yourself:
- List all functions and their addresses
- Check imports, exports, and strings for hints about purpose
- Note the binary format, architecture, and calling convention

**Analyze systematically**: For each function you investigate:
1. Get the disassembly and/or decompiler output
2. Identify the calling convention and argument count from the prologue
3. Trace data flow: follow values through registers and stack across the function body
4. Identify patterns—loops, conditionals, comparisons, error checks, syscalls, API calls
5. Note cross-references: what calls this function and what does it call

**Apply learnings immediately**: As soon as you understand something, commit it back to Ghidra:
- Rename variables and parameters to reflect their purpose (e.g., `local_10` → `file_size`)
- Set correct types for variables and parameters (e.g., change `int` to `FILE *`)
- Rename functions based on their behavior (e.g., `FUN_00401000` → `parse_config_file`)
- Update function prototypes to match the real signature
- Add inline comments at key instructions to explain non-obvious behavior

**Track your progress**: Use the built-in to-do list and filesystem to record:
- Which functions you've analyzed
- Key data structures you've identified
- Your current working hypothesis about the binary's purpose
- Areas that still need investigation

## Naming conventions

Use lowercase snake_case for all names unless the binary itself uses another convention. \
Prefer descriptive names over abbreviated ones. If you are uncertain about a name, prefix \
it with `maybe_` and refine it as you learn more.

## Never guess without evidence

Do not rename or retype anything based on speculation alone. Every change you make to Ghidra \
must be grounded in specific evidence from the assembly—cite the instruction address or \
pattern that led you to that conclusion.
"""

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_DIM = "\033[2m"


def get_mcp_config() -> dict:
    transport = os.environ.get("GHIDRA_MCP_TRANSPORT", "stdio").lower()

    if transport in ("http", "streamable-http", "streamable_http"):
        url = os.environ.get("GHIDRA_MCP_URL", "http://localhost:8080/mcp")
        return {"ghidra": {"transport": "http", "url": url}}

    if transport == "sse":
        url = os.environ.get("GHIDRA_MCP_URL", "http://localhost:8080/mcp")
        return {"ghidra": {"transport": "sse", "url": url}}

    # stdio (default): launch ghidra MCP bridge as a subprocess
    command = os.environ.get("GHIDRA_MCP_COMMAND", "ghidra-mcp")
    args_raw = os.environ.get("GHIDRA_MCP_ARGS", "")
    args = args_raw.split() if args_raw.strip() else []
    return {"ghidra": {"transport": "stdio", "command": command, "args": args}}


async def stream_response(agent, user_input: str, config: dict) -> None:
    """Stream the agent's response, showing tool calls and text tokens in real time."""
    print()

    in_text = False
    active_tool: str | None = None

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            name = event.get("name", "")
            # Skip deep-agent internal harness tools from the display
            internal = {"write_todos", "read_file", "write_file", "edit_file",
                        "ls", "glob", "grep", "task"}
            if name and name not in internal:
                if in_text:
                    print()
                    in_text = False
                active_tool = name
                print(f"{ANSI_YELLOW}⚙ {name}{ANSI_RESET}", end="  ", flush=True)

        elif kind == "on_tool_end":
            if active_tool:
                print(f"{ANSI_GREEN}✓{ANSI_RESET}", flush=True)
                active_tool = None

        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            content = chunk.content
            if isinstance(content, str) and content:
                if not in_text:
                    print()
                    in_text = True
                print(content, end="", flush=True)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            if not in_text:
                                print()
                                in_text = True
                            print(text, end="", flush=True)

    if in_text:
        print()
    print()


async def main() -> None:
    load_dotenv()

    mcp_config = get_mcp_config()
    model = os.environ.get("MODEL", "anthropic:claude-sonnet-4-6")
    agents_md = os.environ.get("AGENTS_MD", "")

    transport_desc = mcp_config["ghidra"].get("transport", "stdio")
    if transport_desc == "stdio":
        cmd = mcp_config["ghidra"].get("command", "ghidra-mcp")
        print(f"Connecting to Ghidra MCP server [stdio: {cmd}]...")
    else:
        url = mcp_config["ghidra"].get("url", "")
        print(f"Connecting to Ghidra MCP server [{transport_desc}: {url}]...")

    try:
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()
    except Exception as exc:
        print(f"{ANSI_RED}Failed to connect to Ghidra MCP server: {exc}{ANSI_RESET}",
              file=sys.stderr)
        print("Set GHIDRA_MCP_TRANSPORT, GHIDRA_MCP_URL, or GHIDRA_MCP_COMMAND as needed.",
              file=sys.stderr)
        sys.exit(1)

    if not tools:
        print(f"{ANSI_YELLOW}Warning: no tools loaded from Ghidra MCP server.{ANSI_RESET}",
              file=sys.stderr)
    else:
        names = ", ".join(t.name for t in tools)
        print(f"Loaded {len(tools)} Ghidra tool(s): {ANSI_DIM}{names}{ANSI_RESET}")

    print()

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db = os.environ.get("MONGODB_DB", "checkpointing_db")
    with MongoDBSaver.from_conn_string(mongodb_uri, db_name=mongodb_db) as checkpointer:
        agent_kwargs: dict = dict(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        if agents_md:
            agent_kwargs["memory"] = [agents_md]

        agent = create_deep_agent(**agent_kwargs)

        # All turns within one interactive session share a single thread so the
        # agent accumulates context across messages.
        recursion_limit = int(os.environ.get("RECURSION_LIMIT", "100"))
        config = {"configurable": {"thread_id": "re-session"}, "recursion_limit": recursion_limit}

        print(f"Ghidra Reverse Engineering Agent ready  {ANSI_DIM}[model: {model}]{ANSI_RESET}")
        print("Enter your analysis task. Type 'quit' or press Ctrl+C to exit.")
        print()

        while True:
            try:
                user_input = input(f"{ANSI_CYAN}> {ANSI_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            try:
                await stream_response(agent, user_input, config)
            except KeyboardInterrupt:
                print(f"\n{ANSI_DIM}[interrupted]{ANSI_RESET}")
            except Exception as exc:
                print(f"\n{ANSI_RED}[Error: {exc}]{ANSI_RESET}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
