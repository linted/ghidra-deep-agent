"""Shared agent-runtime bootstrap.

Pulls the MCP connection, settings resolution, backend selection, and deep-agent
construction out of the TUI entry point so both the TUI ([main.py](main.py)) and the
web UI ([web/service.py](web/service.py)) build the agent the same way.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import (
    create_summarization_tool_middleware,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from ghidra_deep_agent.ghidra_transport import get_mcp_config
from ghidra_deep_agent.prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class Settings:
    """Runtime configuration resolved from the environment."""

    mongodb_uri: str
    mongodb_db: str
    model: str
    embed_string: str
    recursion_limit: int
    output_dir: str
    agents_md: str
    shared_repo: str


def resolve_settings() -> Settings:
    """Read all runtime configuration from environment variables."""
    # EMBED_MODEL takes precedence; fall back to legacy OLLAMA_EMBED_MODEL.
    ollama_fallback = (
        f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
    )
    return Settings(
        mongodb_uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db=os.environ.get("MONGODB_DB", "checkpointing_db"),
        model=os.environ.get("MODEL", "anthropic:claude-sonnet-4-6"),
        embed_string=os.environ.get("EMBED_MODEL", ollama_fallback),
        recursion_limit=int(os.environ.get("RECURSION_LIMIT", "10000")),
        output_dir=os.environ.get("AGENT_OUTPUT_DIR", ""),
        agents_md=os.environ.get("AGENTS_MD", ""),
        shared_repo=os.environ.get("GHIDRA_DEFAULT_REPOSITORY", "agent-shared"),
    )


def describe_mcp_transport(mcp_config: dict[str, Any]) -> str:
    """Human-readable description of the configured Ghidra MCP transport."""
    ghidra = mcp_config.get("ghidra", {})
    transport = ghidra.get("transport", "stdio")
    if transport == "stdio":
        return f"stdio: {ghidra.get('command', 'ghidra-mcp')}"
    return f"{transport}: {ghidra.get('url', '')}"


async def connect_mcp_tools() -> list[Any]:
    """Connect to the Ghidra MCP server and return its tools.

    Wraps tool calls so MCP errors are returned to the model as text rather
    than propagating, and flips ``handle_tool_error`` so ToolExceptions are
    caught by LangGraph instead of bubbling up through sub-agents.
    """
    mcp_config = get_mcp_config()

    async def handle_mcp_errors(request: MCPToolCallRequest, handler: Any) -> Any:
        try:
            return await handler(request)
        except Exception as exc:
            return f"Tool '{request.name}' failed: {exc}"

    client = MultiServerMCPClient(mcp_config, tool_interceptors=[handle_mcp_errors])
    tools = await client.get_tools()

    # MCP server errors arrive as isError=True results, which
    # langchain_mcp_adapters converts to ToolException. Without
    # handle_tool_error=True, ToolException bypasses LangGraph's ToolNode
    # default handler (which only catches ToolInvocationError) and propagates
    # all the way up through sub-agents to the UI.
    for tool in tools:
        tool.handle_tool_error = True

    return tools


def make_backend(output_dir: str) -> Any:
    """Pick the deep-agent backend based on whether an output dir is set."""
    if output_dir:
        return FilesystemBackend(root_dir=output_dir, virtual_mode=True)
    return StateBackend()


def build_agent(
    model: Any,
    tools: list[Any],
    checkpointer: Any,
    backend: Any,
    agents_md: str,
) -> Any:
    """Construct the deep agent. ``tools`` should already include knowledge tools."""
    agent_kwargs: dict[str, Any] = dict(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[create_summarization_tool_middleware(model, backend)],
        backend=backend,
        memory=[agents_md] if agents_md else None,
    )
    return create_deep_agent(**agent_kwargs)


def context_window(model: Any) -> int:
    """Max input tokens for the model, falling back to MAX_CONTEXT_TOKENS/200k."""
    profile = getattr(model, "profile", None) or {}
    return profile.get("max_input_tokens") or int(
        os.environ.get("MAX_CONTEXT_TOKENS", "200000")
    )


def log_loaded_tools(tools: list[Any]) -> None:
    """Print a short summary of the loaded Ghidra tools (TUI startup banner)."""
    if not tools:
        print("Warning: no tools loaded from Ghidra MCP server.", file=sys.stderr)
    else:
        names = ", ".join(t.name for t in tools)
        print(f"Loaded {len(tools)} Ghidra tool(s): {names}")
