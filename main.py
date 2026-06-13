import argparse
import asyncio
import os
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.errors import ServerSelectionTimeoutError

from ghidra_deep_agent.ghidra_transport import get_mcp_config
from ghidra_deep_agent.knowledge import build_knowledge_tools
from ghidra_deep_agent.models import build_embeddings, build_model
from ghidra_deep_agent.program_resolver import resolve_binary_name
from ghidra_deep_agent.runtime import (
    build_agent,
    connect_mcp_tools,
    context_window,
    describe_mcp_transport,
    log_loaded_tools,
    make_backend,
    resolve_settings,
)
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

    settings = resolve_settings()

    transport = describe_mcp_transport(get_mcp_config())
    print(f"Connecting to Ghidra MCP server [{transport}]...")
    try:
        tools = await connect_mcp_tools()
    except Exception as exc:
        print(f"Failed to connect to Ghidra MCP server: {exc}", file=sys.stderr)
        print(
            "Set GHIDRA_MCP_TRANSPORT, GHIDRA_MCP_URL, or "
            "GHIDRA_MCP_COMMAND as needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_loaded_tools(tools)

    binary_name_override = args.binary_name or os.environ.get("BINARY_NAME")
    try:
        binary_name = await resolve_binary_name(tools, binary_name_override)
        print(f"Analyzing binary: {binary_name}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        embeddings = build_embeddings(settings.embed_string)
        knowledge_tools = build_knowledge_tools(
            settings.mongodb_uri, settings.mongodb_db, embeddings, binary_name
        )
        print(f"Knowledge base ready  [embed: {settings.embed_string}]")
    except Exception as exc:
        print(f"Warning: knowledge base unavailable ({exc})", file=sys.stderr)
        knowledge_tools = []

    built_model = build_model(settings.model)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }

    backend: Any = make_backend(settings.output_dir)

    try:
        with MongoDBSaver.from_conn_string(
            settings.mongodb_uri, db_name=settings.mongodb_db
        ) as checkpointer:
            agent = build_agent(
                built_model,
                knowledge_tools + tools,
                checkpointer,
                backend,
                settings.agents_md,
            )

            app = GhidraAgentApp(
                agent=agent,
                config=config,
                model=settings.model,
                session_id=session_id,
                mcp_ok=True,
                db_ok=True,
                max_context_tokens=context_window(built_model),
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
