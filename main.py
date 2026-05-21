import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from deepagents import create_deep_agent

from knowledge import build_knowledge_tools
from mcp import get_mcp_config
from prompt import SYSTEM_PROMPT
from streaming import (
    ANSI_CYAN, ANSI_DIM, ANSI_RED, ANSI_RESET, ANSI_YELLOW,
    recover_from_tool_error,
    stream_response,
)


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
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    try:
        knowledge_tools = build_knowledge_tools(mongodb_uri, mongodb_db, embed_model)
        print(f"Knowledge base ready  {ANSI_DIM}[embed: {embed_model}]{ANSI_RESET}")
    except Exception as exc:
        print(f"{ANSI_YELLOW}Warning: knowledge base unavailable ({exc}){ANSI_RESET}",
              file=sys.stderr)
        knowledge_tools = []

    with MongoDBSaver.from_conn_string(mongodb_uri, db_name=mongodb_db) as checkpointer:
        agent_kwargs: dict = dict(
            model=model,
            tools=knowledge_tools + tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        if agents_md:
            agent_kwargs["memory"] = [agents_md]

        agent = create_deep_agent(**agent_kwargs)

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
                print(f"\n{ANSI_RED}[Tool error: {exc}]{ANSI_RESET}", file=sys.stderr)
                await recover_from_tool_error(agent, config, exc)


if __name__ == "__main__":
    asyncio.run(main())
