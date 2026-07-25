"""Unit tests for the MCP read cache (``mcp_cache.py``).

The cache is only sound if two things hold: every tool that mutates Ghidra
flushes the mutable tier, and no read that was already in flight when a flush
happened is allowed to write its now-stale result. Both are exercised here
against a fake collection, so no MongoDB is needed.

Run:  uv run pytest tests/test_mcp_cache.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.messages import ToolMessage

from ghidra_deep_agent.mcp_cache import _MUTATING_TOOLS, MCPReadCacheMiddleware

CACHED = frozenset({"get_binary_info"})
MUTABLE = frozenset({"get_code"})


class FakeCollection:
    """The slice of the pymongo Collection API the cache actually uses."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return self.docs.get(query["_id"])

    def replace_one(
        self, query: dict[str, Any], doc: dict[str, Any], upsert: bool = False
    ) -> None:
        self.docs[query["_id"]] = doc

    def delete_many(self, query: dict[str, Any]) -> Any:
        doomed = [
            k
            for k, d in self.docs.items()
            if d["binary"] == query["binary"] and d["mutable"] == query["mutable"]
        ]
        for k in doomed:
            del self.docs[k]
        return type("R", (), {"deleted_count": len(doomed)})()


def _middleware() -> tuple[MCPReadCacheMiddleware, FakeCollection]:
    coll = FakeCollection()
    mw = MCPReadCacheMiddleware(cast(Any, coll), "firmware.bin", CACHED, MUTABLE)
    return mw, coll


def _request(name: str, args: dict[str, Any] | None = None) -> Any:
    return type(
        "Req", (), {"tool_call": {"name": name, "id": "call-1", "args": args or {}}}
    )()


def _ok(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id="call-1")


# ── the mutating-tool set ─────────────────────────────────────────────────────


def test_every_listing_mutating_local_tool_is_registered() -> None:
    """The local Ghidra-script tools that write must all flush the cache.

    `deobfuscate_cff` patches indirect branches into direct ones with apply=True;
    it was added (#40) without being registered here, so `get_code` kept serving
    pre-patch decompilation for the whole TTL.
    """
    for name in ("recover_prototypes", "apply_switch_override", "deobfuscate_cff"):
        assert name in _MUTATING_TOOLS


def test_read_only_switch_tool_is_not_registered() -> None:
    assert "find_unrecovered_switches" not in _MUTATING_TOOLS


# ── caching + invalidation ────────────────────────────────────────────────────


def test_repeated_read_is_served_from_cache() -> None:
    mw, _ = _middleware()
    calls = []

    def handler(req: Any) -> ToolMessage:
        calls.append(req)
        return _ok("get_code", "decompiled v1")

    mw.wrap_tool_call(_request("get_code"), handler)
    result = mw.wrap_tool_call(_request("get_code"), handler)

    assert len(calls) == 1
    assert mw.hits == 1
    assert isinstance(result, ToolMessage)
    assert result.content == "decompiled v1"


def test_mutation_flushes_the_mutable_tier() -> None:
    mw, coll = _middleware()
    mw.wrap_tool_call(_request("get_code"), lambda r: _ok("get_code", "v1"))
    assert coll.docs

    mw.wrap_tool_call(
        _request("deobfuscate_cff"), lambda r: _ok("deobfuscate_cff", "patched")
    )

    assert coll.docs == {}
    assert mw.invalidations == 1


def test_failed_mutation_does_not_flush() -> None:
    mw, coll = _middleware()
    mw.wrap_tool_call(_request("get_code"), lambda r: _ok("get_code", "v1"))

    def failing(req: Any) -> ToolMessage:
        msg = _ok("deobfuscate_cff", "boom")
        msg.status = "error"
        return msg

    mw.wrap_tool_call(_request("deobfuscate_cff"), failing)

    assert coll.docs != {}
    assert mw.invalidations == 0


# ── the store-after-invalidate race ───────────────────────────────────────────


def test_read_flushed_while_in_flight_is_not_stored() -> None:
    """A slow read must not resurrect pre-mutation output after a flush.

    One middleware instance is shared by the coordinator and every sub-agent, so
    a `get_code` taking tens of seconds can return *after* a concurrent rename
    flushed the tier. Without the generation check its stale result would be
    written back and outlive the flush for the full TTL.
    """
    mw, coll = _middleware()

    def slow_read(req: Any) -> ToolMessage:
        # A mutation lands while this read is still running.
        mw.wrap_tool_call(
            _request("rename_symbol"), lambda r: _ok("rename_symbol", "renamed")
        )
        return _ok("get_code", "pre-rename decompilation")

    mw.wrap_tool_call(_request("get_code"), slow_read)

    assert coll.docs == {}, "stale read was stored after the flush"


def test_immutable_tier_read_is_stored_even_across_a_flush() -> None:
    """Immutable output can't be invalidated by a Ghidra mutation."""
    mw, coll = _middleware()

    def slow_read(req: Any) -> ToolMessage:
        mw.wrap_tool_call(
            _request("rename_symbol"), lambda r: _ok("rename_symbol", "renamed")
        )
        return _ok("get_binary_info", "ELF 32-bit ARM")

    mw.wrap_tool_call(_request("get_binary_info"), slow_read)

    assert len(coll.docs) == 1


def test_async_path_also_skips_a_flushed_read() -> None:
    mw, coll = _middleware()

    async def slow_read(req: Any) -> ToolMessage:
        await asyncio.to_thread(
            mw.wrap_tool_call,
            _request("apply_switch_override"),
            lambda r: _ok("apply_switch_override", "applied"),
        )
        return _ok("get_code", "pre-patch decompilation")

    asyncio.run(mw.awrap_tool_call(_request("get_code"), slow_read))

    assert coll.docs == {}
