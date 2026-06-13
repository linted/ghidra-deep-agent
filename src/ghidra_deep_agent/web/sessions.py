"""Mongo-backed registry of web sessions.

Each session's ``session_id`` doubles as the LangGraph ``thread_id``, so resume
works directly against the existing checkpointer — this store only tracks the
metadata the web UI needs to list and switch between sessions.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient

COLLECTION = "web_sessions"


@dataclass
class Session:
    session_id: str
    binary_name: str
    title: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    """CRUD over the ``web_sessions`` collection."""

    def __init__(self, mongodb_uri: str, mongodb_db: str) -> None:
        client: MongoClient[Any] = MongoClient(mongodb_uri)
        self._collection = client[mongodb_db][COLLECTION]

    def create(self, binary_name: str, title: str = "") -> Session:
        now = _now()
        session = Session(
            session_id=str(uuid.uuid4()),
            binary_name=binary_name,
            title=title or binary_name,
            created_at=now,
            updated_at=now,
        )
        self._collection.insert_one(session.to_dict())
        return session

    def list(self) -> list[Session]:
        docs = self._collection.find({}, {"_id": False}).sort("updated_at", -1)
        return [Session(**doc) for doc in docs]

    def get(self, session_id: str) -> Session | None:
        doc = self._collection.find_one({"session_id": session_id}, {"_id": False})
        return Session(**doc) if doc else None

    def delete(self, session_id: str) -> bool:
        result = self._collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0

    def touch(self, session_id: str) -> None:
        self._collection.update_one(
            {"session_id": session_id}, {"$set": {"updated_at": _now()}}
        )

    def rename(self, session_id: str, title: str) -> None:
        self._collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": title, "updated_at": _now()}},
        )
