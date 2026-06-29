"""MongoDB-backed cache for immutable read-only Ghidra MCP tools.

Some Ghidra MCP reads are the slowest frequent tools in a session
(``search_strings`` measured at p50 ~64s / p99 ~186s) yet their output is
**immutable for a program within a session** — the strings table, import/export
tables, entry points, and program metadata don't change as the agent renames or
retypes functions. Repeat calls with the same arguments are therefore pure
latency waste.

This middleware intercepts calls to an allowlist of *provably-immutable* read
tools, keys them on ``(binary, tool, args)``, and serves repeats from MongoDB.
Mutation tools are never cached, and only successful results are stored. The
cache collection carries a **TTL index** so Mongo's background monitor expires
entries (sized to a typical session) with no local memory or manual eviction.

Configuration (env):
  MONGODB_TOOL_CACHE_COLLECTION   collection name (default ``tool_cache``)
  MONGODB_TOOL_CACHE_TTL          entry lifetime in seconds (default 86400)
  MONGODB_TOOL_CACHE_TOOLS        comma-separated allowlist override; empty
                                  disables caching (default: the immutable set
                                  below)
  MONGODB_TOOL_CACHE_DEBUG        set to log each hit/miss to stderr
"""

import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pymongo import MongoClient
from pymongo.collection import Collection

# Read tools whose output is invariant for a program regardless of any renames /
# retypes / comments the agent applies during the session. Deliberately
# conservative: tools like ``list_functions`` / ``search_functions`` are excluded
# because renaming a function changes their output mid-session.
_DEFAULT_CACHED_TOOLS = (
    "search_strings",
    "list_imports",
    "list_exports",
    "get_entry_points",
    "get_current_program_info",
)

_TTL_INDEX_NAME = "created_at_ttl"


def _allowlist() -> frozenset[str]:
    override = os.environ.get("MONGODB_TOOL_CACHE_TOOLS")
    if override is None:
        return frozenset(_DEFAULT_CACHED_TOOLS)
    return frozenset(name.strip() for name in override.split(",") if name.strip())


def _ensure_ttl_index(collection: Collection[dict[str, Any]], ttl_seconds: int) -> None:
    """Create (or re-sync) a TTL index on ``created_at`` for ``ttl_seconds``."""
    existing = collection.index_information().get(_TTL_INDEX_NAME)
    if existing is not None:
        if existing.get("expireAfterSeconds") != ttl_seconds:
            collection.database.command(
                "collMod",
                collection.name,
                index={
                    "keyPattern": {"created_at": 1},
                    "expireAfterSeconds": ttl_seconds,
                },
            )
        return
    collection.create_index(
        "created_at", name=_TTL_INDEX_NAME, expireAfterSeconds=ttl_seconds
    )


class MCPReadCacheMiddleware(AgentMiddleware):
    """Serve repeat immutable-read MCP calls from MongoDB instead of re-running them."""

    def __init__(
        self,
        collection: Collection[dict[str, Any]],
        binary_name: str,
        cached_tools: frozenset[str],
        *,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._collection = collection
        self._binary = binary_name
        self._cached = cached_tools
        self._debug = debug
        self.hits = 0
        self.misses = 0

    # --- helpers ---------------------------------------------------------------

    def _key(self, name: str, args: Any) -> str:
        blob = json.dumps(
            {"b": self._binary, "t": name, "a": args}, sort_keys=True, default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _lookup(self, key: str) -> dict[str, Any] | None:
        return self._collection.find_one({"_id": key})

    def _store(self, key: str, name: str, content: Any, status: str) -> None:
        self._collection.replace_one(
            {"_id": key},
            {
                "_id": key,
                "tool": name,
                "content": content,
                "status": status,
                "created_at": datetime.now(UTC),
            },
            upsert=True,
        )

    def _log(self, kind: str, name: str) -> None:
        if self._debug:
            print(f"[mcp-cache] {kind} {name}", file=sys.stderr)

    def _cached_message(
        self, doc: dict[str, Any], request: ToolCallRequest
    ) -> ToolMessage:
        return ToolMessage(
            content=doc["content"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status=doc.get("status", "success"),
        )

    def _should_store(self, result: ToolMessage | Command[Any]) -> bool:
        return isinstance(result, ToolMessage) and result.status != "error"

    # --- hooks -----------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        if name not in self._cached:
            return handler(request)
        key = self._key(name, request.tool_call.get("args", {}))
        doc = self._lookup(key)
        if doc is not None:
            self.hits += 1
            self._log("HIT", name)
            return self._cached_message(doc, request)
        self.misses += 1
        self._log("MISS", name)
        result = handler(request)
        if self._should_store(result):
            assert isinstance(result, ToolMessage)
            self._store(key, name, result.content, result.status)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        if name not in self._cached:
            return await handler(request)
        key = self._key(name, request.tool_call.get("args", {}))
        # pymongo is synchronous; offload to a thread so cache I/O doesn't block
        # the event loop while other tool calls run.
        doc = await asyncio.to_thread(self._lookup, key)
        if doc is not None:
            self.hits += 1
            self._log("HIT", name)
            return self._cached_message(doc, request)
        self.misses += 1
        self._log("MISS", name)
        result = await handler(request)
        if self._should_store(result):
            assert isinstance(result, ToolMessage)
            await asyncio.to_thread(
                self._store, key, name, result.content, result.status
            )
        return result


def build_mcp_cache_middleware(
    mongodb_uri: str, mongodb_db: str, binary_name: str
) -> MCPReadCacheMiddleware | None:
    """Build the read-cache middleware, or ``None`` if disabled/unavailable.

    Returns ``None`` (caching off) when the allowlist is empty or MongoDB can't
    be reached — the agent then runs exactly as before, just without the cache.
    """
    cached_tools = _allowlist()
    if not cached_tools:
        return None

    ttl_seconds = int(os.environ.get("MONGODB_TOOL_CACHE_TTL", "86400"))
    coll_name = os.environ.get("MONGODB_TOOL_CACHE_COLLECTION", "tool_cache")
    debug = bool(os.environ.get("MONGODB_TOOL_CACHE_DEBUG"))

    try:
        client: MongoClient[dict[str, Any]] = MongoClient(mongodb_uri)
        collection = client[mongodb_db][coll_name]
        _ensure_ttl_index(collection, ttl_seconds)
    except Exception as exc:  # pragma: no cover - environmental
        print(f"Warning: MCP read cache disabled ({exc})", file=sys.stderr)
        return None

    return MCPReadCacheMiddleware(collection, binary_name, cached_tools, debug=debug)
