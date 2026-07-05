"""MongoDB-backed cache for read-only Ghidra MCP tools.

Some Ghidra MCP reads are the slowest frequent tools in a session
(``search_strings`` measured at p50 ~64s / p99 ~186s) yet their output is
**immutable for a program within a session** — the strings table, import/export
tables, entry points, and program metadata don't change as the agent renames or
retypes functions. Repeat calls with the same arguments are therefore pure
latency waste.

This middleware intercepts calls to two allowlists of read tools, keys them on
``(binary, tool, args)``, and serves repeats from MongoDB:

- an **immutable** tier (strings/imports/exports/entry points/metadata) whose
  entries only ever expire by TTL, and
- a **mutable** tier (``get_code``/``xrefs``/``get_data_at``) whose output
  *does* change when the agent renames/retypes/comments — a rename of function
  A changes the decompilation of every caller of A, so per-address invalidation
  is unsound and the whole tier is flushed for the binary whenever any Ghidra
  mutation tool succeeds.

Mutation tools are never cached, and only successful results are stored. The
cache collection carries a **TTL index** so Mongo's background monitor expires
entries (sized to a typical session) with no local memory or manual eviction.
Mutations made directly in the Ghidra GUI bypass the MCP tools and therefore
bypass invalidation — the TTL is the only backstop; disable the mutable tier
if you edit in Ghidra while the agent runs.

Configuration (env):
  MONGODB_TOOL_CACHE_COLLECTION     collection name (default ``tool_cache``)
  MONGODB_TOOL_CACHE_TTL            entry lifetime in seconds (default 86400)
  MONGODB_TOOL_CACHE_TOOLS          comma-separated immutable-tier override;
                                    empty disables the tier (default: the
                                    immutable set below)
  MONGODB_TOOL_CACHE_MUTABLE_TOOLS  comma-separated mutable-tier override;
                                    empty disables the tier (default: the
                                    mutable set below)
  MONGODB_TOOL_CACHE_DEBUG          set to log each hit/miss/invalidation to
                                    stderr
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
# conservative: tools like ``get_functions`` / ``search_functions_by_name`` are
# excluded because renaming a function changes their output mid-session.
_DEFAULT_CACHED_TOOLS = (
    "search_strings",
    "get_imports",
    "get_exports",
    "get_entry_points",
    "get_binary_info",
)

# Read tools whose output changes as the agent mutates the program (renames
# propagate into callers' decompilation, comments/retypes appear inline).
# Cacheable only because every mutation path goes through the MCP tools below,
# which flush this tier for the binary.
_DEFAULT_MUTABLE_CACHED_TOOLS = (
    "get_code",
    "xrefs",
    "get_data_at",
)

# Ghidra-mutating tools (see the function-analyst allowlist in subagents.toml).
# Any successful call invalidates the mutable tier. Knowledge-base writes are
# excluded — they don't touch Ghidra state.
_MUTATING_TOOLS = frozenset(
    {
        "rename_symbol",
        "batch_rename",
        "variables",
        "comments",
        "types",
        "struct",
        "create_function",
        # Local tool (prototype_tools.py) that commits recovered prototypes via a
        # Ghidra script. It's the name seen in the graph — the coordinator calls
        # it, not `scripts` — so invalidation keys on it here.
        "recover_prototypes",
    }
)

_TTL_INDEX_NAME = "created_at_ttl"


def _env_allowlist(var: str, default: tuple[str, ...]) -> frozenset[str]:
    override = os.environ.get(var)
    if override is None:
        return frozenset(default)
    return frozenset(name.strip() for name in override.split(",") if name.strip())


def _allowlist() -> frozenset[str]:
    return _env_allowlist("MONGODB_TOOL_CACHE_TOOLS", _DEFAULT_CACHED_TOOLS)


def _mutable_allowlist() -> frozenset[str]:
    return _env_allowlist(
        "MONGODB_TOOL_CACHE_MUTABLE_TOOLS", _DEFAULT_MUTABLE_CACHED_TOOLS
    )


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
    """Serve repeat read-only MCP calls from MongoDB instead of re-running them."""

    def __init__(
        self,
        collection: Collection[dict[str, Any]],
        binary_name: str,
        cached_tools: frozenset[str],
        mutable_tools: frozenset[str] = frozenset(),
        *,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._collection = collection
        self._binary = binary_name
        # A tool listed in both tiers is treated as mutable (the safe reading).
        self._mutable = mutable_tools
        self._cached = (cached_tools | mutable_tools) - _MUTATING_TOOLS
        self._debug = debug
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

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
                "binary": self._binary,
                "tool": name,
                "mutable": name in self._mutable,
                "content": content,
                "status": status,
                "created_at": datetime.now(UTC),
            },
            upsert=True,
        )

    def _invalidate(self, name: str) -> None:
        """Flush the binary's mutable-tier entries after a successful mutation."""
        deleted = self._collection.delete_many(
            {"binary": self._binary, "mutable": True}
        ).deleted_count
        self.invalidations += 1
        if self._debug:
            print(
                f"[mcp-cache] INVALIDATE {name} cleared {deleted} mutable entries",
                file=sys.stderr,
            )

    def _log(self, kind: str, name: str) -> None:
        if self._debug:
            tier = "mutable" if name in self._mutable else "immutable"
            print(f"[mcp-cache] {kind} {name} ({tier})", file=sys.stderr)

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

    def _mutation_succeeded(self, result: ToolMessage | Command[Any]) -> bool:
        # Commands don't carry a status; assume the mutation happened (a spurious
        # flush only costs cache misses, a skipped flush serves stale code).
        return not (isinstance(result, ToolMessage) and result.status == "error")

    # --- hooks -----------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        if name in _MUTATING_TOOLS:
            result = handler(request)
            if self._mutation_succeeded(result):
                self._invalidate(name)
            return result
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
        if name in _MUTATING_TOOLS:
            result = await handler(request)
            if self._mutation_succeeded(result):
                await asyncio.to_thread(self._invalidate, name)
            return result
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

    Returns ``None`` (caching off) when both allowlists are empty or MongoDB
    can't be reached — the agent then runs exactly as before, just without the
    cache.
    """
    cached_tools = _allowlist()
    mutable_tools = _mutable_allowlist()
    if not cached_tools and not mutable_tools:
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

    return MCPReadCacheMiddleware(
        collection, binary_name, cached_tools, mutable_tools, debug=debug
    )
