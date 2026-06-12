import argparse
import asyncio
import os
import sys
import uuid
from typing import Any

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
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.ghidra_transport import get_mcp_config
from ghidra_deep_agent.knowledge import build_knowledge_tools
from ghidra_deep_agent.models import build_embeddings, build_model
from ghidra_deep_agent.program_resolver import resolve_binary_name
from ghidra_deep_agent.prompt import SYSTEM_PROMPT
from ghidra_deep_agent.tui import GhidraAgentApp


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ghidra deep agent")
    parser.add_argument(
        "--session-id", default=None, help="Resume a previous session by ID"
    )
    parser.add_argument(
        "--binary-name",
        default=None,
        help="Binary name to use for knowledge isolation (overrides auto-detection)",
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

    async def handle_mcp_errors(request: MCPToolCallRequest, handler: Any) -> Any:
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

    # MCP server errors arrive as isError=True results, which langchain_mcp_adapters
    # converts to ToolException. Without handle_tool_error=True, ToolException bypasses
    # LangGraph's ToolNode default handler (which only catches ToolInvocationError) and
    # propagates all the way up through sub-agents to the TUI.
    for tool in tools:
        tool.handle_tool_error = True

    if not tools:
        print("Warning: no tools loaded from Ghidra MCP server.", file=sys.stderr)
    else:
        names = ", ".join(t.name for t in tools)
        print(f"Loaded {len(tools)} Ghidra tool(s): {names}")

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db = os.environ.get("MONGODB_DB", "checkpointing_db")

    # EMBED_MODEL takes precedence; fall back to legacy OLLAMA_EMBED_MODEL.
    _ollama_fallback = (
        f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
    )
    embed_string = os.environ.get("EMBED_MODEL", _ollama_fallback)

    binary_name_override = args.binary_name or os.environ.get("BINARY_NAME")
    try:
        binary_name = await resolve_binary_name(tools, binary_name_override)
        print(f"Analyzing binary: {binary_name}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        embeddings = build_embeddings(embed_string)
        knowledge_tools = build_knowledge_tools(
            mongodb_uri, mongodb_db, embeddings, binary_name
        )
        print(f"Knowledge base ready  [embed: {embed_string}]")
    except Exception as exc:
        print(f"Warning: knowledge base unavailable ({exc})", file=sys.stderr)
        knowledge_tools = []

    built_model = build_model(model)
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "10000"))
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    output_dir = os.environ.get("AGENT_OUTPUT_DIR", "")
    backend: Any
    if output_dir:
        backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)
    else:
        backend = StateBackend()

    try:
        with MongoDBSaver.from_conn_string(
            mongodb_uri, db_name=mongodb_db
        ) as checkpointer:
            agent_kwargs: dict[str, Any] = dict(
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
                agent=agent,
                config=config,
                model=model,
                session_id=session_id,
                mcp_ok=True,
                db_ok=True,
            )
            await app.run_async()
    except ServerSelectionTimeoutError as e:
        print(
            f"Error: could not connect to MongoDB — {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Session ID: {session_id}")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
