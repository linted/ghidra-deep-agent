"""MongoDB-backed registry of agent sessions for discovery and resume.

LangGraph persists each session as a checkpoint thread keyed by ``thread_id``
(= our ``session_id``), but the checkpoint documents are not cheaply queryable
by recency and their schema is an implementation detail we shouldn't depend on.

This module keeps a small, dedicated ``sessions`` collection — one record per
session — that the TUI's ``/resume`` command can list most-recent-first and
optionally scope to the binary currently open in Ghidra. The record is written
on session start and touched on each turn; the heavy conversation state stays in
the checkpointer.

Configuration (env):
  MONGODB_SESSIONS_COLLECTION   collection name (default ``sessions``)
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    ExecutionTimeout,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WaitQueueTimeoutError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Transient MongoDB failures worth retrying — network blips, server-selection
# timeouts, primary step-downs. Mirrors knowledge.py's classification.
_TRANSIENT_MONGO_ERRORS = (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WaitQueueTimeoutError,
    ExecutionTimeout,
)

_RECENCY_INDEX_NAME = "last_active_at_desc"
_TITLE_MAX_CHARS = 80


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception_type(_TRANSIENT_MONGO_ERRORS),
)
def _mongo_write_with_retry[T](fn: Callable[[], T]) -> T:
    """Run a MongoDB write, retrying transient failures with backoff."""
    return fn()


def _ensure_recency_index(collection: Collection[dict[str, Any]]) -> None:
    """Create a descending index on ``last_active_at`` for the resume sort."""
    if collection.index_information().get(_RECENCY_INDEX_NAME) is not None:
        return
    collection.create_index([("last_active_at", -1)], name=_RECENCY_INDEX_NAME)


def _truncate_title(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= _TITLE_MAX_CHARS:
        return cleaned
    return cleaned[: _TITLE_MAX_CHARS - 1].rstrip() + "…"


class SessionStore:
    """Read/write the ``sessions`` collection backing the ``/resume`` picker."""

    def __init__(self, collection: Collection[dict[str, Any]]) -> None:
        self._collection = collection

    # --- sync core -------------------------------------------------------------

    def record_start(self, session_id: str, binary_name: str) -> None:
        """Register a (new or resumed) session, bumping its activity time.

        Idempotent: ``$setOnInsert`` keeps ``created_at`` and any existing
        ``title`` intact when the session already exists.
        """
        now = datetime.now(UTC)

        def _write() -> None:
            self._collection.update_one(
                {"_id": session_id},
                {
                    "$setOnInsert": {
                        "session_id": session_id,
                        "binary_name": binary_name,
                        "created_at": now,
                    },
                    "$set": {"last_active_at": now},
                },
                upsert=True,
            )

        _mongo_write_with_retry(_write)

    def touch(self, session_id: str, first_prompt: str | None = None) -> None:
        """Update ``last_active_at``; set ``title`` from the first prompt once."""
        now = datetime.now(UTC)

        def _write() -> None:
            self._collection.update_one(
                {"_id": session_id}, {"$set": {"last_active_at": now}}
            )
            if first_prompt:
                # Only set the title if one isn't already recorded.
                self._collection.update_one(
                    {"_id": session_id, "title": {"$exists": False}},
                    {"$set": {"title": _truncate_title(first_prompt)}},
                )

        _mongo_write_with_retry(_write)

    def list_sessions(
        self, binary_name: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return session records most-recent-first, optionally binary-scoped."""
        query: dict[str, Any] = {}
        if binary_name is not None:
            query["binary_name"] = binary_name
        return list(
            self._collection.find(query).sort("last_active_at", -1).limit(limit)
        )

    # --- async wrappers --------------------------------------------------------
    # pymongo is synchronous; offload to a thread so the TUI event loop isn't
    # blocked on Mongo I/O.

    async def arecord_start(self, session_id: str, binary_name: str) -> None:
        await asyncio.to_thread(self.record_start, session_id, binary_name)

    async def atouch(self, session_id: str, first_prompt: str | None = None) -> None:
        await asyncio.to_thread(self.touch, session_id, first_prompt)

    async def alist_sessions(
        self, binary_name: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.list_sessions, binary_name, limit)


def build_session_store(mongodb_uri: str, mongodb_db: str) -> SessionStore | None:
    """Build the session registry, or ``None`` if MongoDB can't be reached.

    Returns ``None`` (registry off) on any connection/index error — the agent
    then runs exactly as before, and ``/resume`` reports nothing to resume.
    """
    coll_name = os.environ.get("MONGODB_SESSIONS_COLLECTION", "sessions")
    try:
        client: MongoClient[dict[str, Any]] = MongoClient(mongodb_uri)
        collection = client[mongodb_db][coll_name]
        _ensure_recency_index(collection)
    except Exception as exc:  # pragma: no cover - environmental
        print(f"Warning: session registry disabled ({exc})", file=sys.stderr)
        return None

    return SessionStore(collection)
