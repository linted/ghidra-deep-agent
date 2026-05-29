import argparse
import asyncio
import os
import sys
import uuid

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import (
    create_summarization_tool_middleware,
)
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.checkpoint.mongodb import MongoDBSaver

from ghidra_transport import get_mcp_config
from knowledge import build_knowledge_tools
from models import build_model
from prompt import SYSTEM_PROMPT
from tui import GhidraAgentApp


def _parse_token_count(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    if value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    return int(value)


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ghidra deep agent")
    parser.add_argument(
        "--session-id", default=None, help="Resume a previous session by ID"
    )
    args = parser.parse_args()
    session_id = args.session_id or str(uuid.uuid4())

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

    async def handle_mcp_errors(request: MCPToolCallRequest, handler):
        try:
            return await handler(request)
        except Exception as exc:
            return f"Tool '{request.name}' failed: {exc}"

    try:
        client = MultiServerMCPClient(mcp_config, tool_interceptors=[handle_mcp_errors])
        tools = await client.get_tools()
    except Exception as exc:
        print(f"Failed to connect to Ghidra MCP server: {exc}", file=sys.stderr)
        print(
            "Set GHIDRA_MCP_TRANSPORT, GHIDRA_MCP_URL, or "
            "GHIDRA_MCP_COMMAND as needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not tools:
        print("Warning: no tools loaded from Ghidra MCP server.", file=sys.stderr)
    else:
        names = ", ".join(t.name for t in tools)
        print(f"Loaded {len(tools)} Ghidra tool(s): {names}")

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db = os.environ.get("MONGODB_DB", "checkpointing_db")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    try:
        knowledge_tools = build_knowledge_tools(mongodb_uri, mongodb_db, embed_model)
        print(f"Knowledge base ready  [embed: {embed_model}]")
    except Exception as exc:
        print(f"Warning: knowledge base unavailable ({exc})", file=sys.stderr)
        knowledge_tools = []

    built_model = build_model(model)
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "100"))
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
    if output_dir:
        backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)
    else:
        backend = StateBackend()

    with MongoDBSaver.from_conn_string(mongodb_uri, db_name=mongodb_db) as checkpointer:
        agent_kwargs: dict = dict(
            model=built_model,
            tools=knowledge_tools + tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
            middleware=[create_summarization_tool_middleware(built_model, backend)],
            backend=backend,
            memory=[agents_md] if agents_md else None,
        )

        agent = create_deep_agent(**agent_kwargs)

        app = GhidraAgentApp(
            agent=agent, config=config, model=model, session_id=session_id
        )
        await app.run_async()

    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    asyncio.run(main())
