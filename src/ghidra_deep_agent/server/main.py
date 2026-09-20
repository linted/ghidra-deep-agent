"""Entry point: ``ghidra-deep-agent-server``.

Configuration (env, on top of the agent's own ``.env``):
  WEB_HOST     bind address (default 127.0.0.1)
  WEB_PORT     port (default 8000)
  WEB_TOKEN    optional bearer token required on every route but /health
  AGENT_OUTPUT_DIR  root for per-agent output dirs (``<root>/<program-slug>``)
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn
from dotenv import load_dotenv

from ghidra_deep_agent.runtime import StartupError, open_shared_runtime
from ghidra_deep_agent.server.app import create_app
from ghidra_deep_agent.server.events import build_run_store
from ghidra_deep_agent.server.registry import AgentRegistry


async def serve() -> None:
    shared = await open_shared_runtime()
    try:
        store = build_run_store(shared.mongo.uri, shared.mongo.db)
        registry = AgentRegistry(
            shared, store, output_root=os.environ.get("AGENT_OUTPUT_DIR", "")
        )
        app = create_app(registry, token=os.environ.get("WEB_TOKEN") or None)
        host = os.environ.get("WEB_HOST", "127.0.0.1")
        port = int(os.environ.get("WEB_PORT", "8000"))
        print(f"Serving on http://{host}:{port}  (docs at /docs)")
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        await uvicorn.Server(config).serve()
    finally:
        shared.close()


def run() -> None:
    load_dotenv()
    try:
        asyncio.run(serve())
    except StartupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
