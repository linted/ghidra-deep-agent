import os


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
