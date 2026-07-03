import os
from typing import Any


def get_mcp_config() -> dict[str, Any]:
    # GhidrAssistMCP serves MCP over HTTP only: streamable-http at /mcp (default)
    # or SSE at /sse. There is no stdio bridge.
    transport = os.environ.get("GHIDRA_MCP_TRANSPORT", "http").lower()

    if transport == "sse":
        url = os.environ.get("GHIDRA_MCP_URL", "http://localhost:8080/sse")
        return {"ghidra": {"transport": "sse", "url": url}}

    url = os.environ.get("GHIDRA_MCP_URL", "http://localhost:8080/mcp")
    return {"ghidra": {"transport": "http", "url": url}}
