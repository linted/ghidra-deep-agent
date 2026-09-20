"""Run events on the wire and at rest.

A run's events are the typed stream events (:mod:`ghidra_deep_agent.stream`)
plus the lifecycle events defined here. Each is serialized once into a
``Persisted`` document with a per-run sequence number, appended to a
``RunStore`` and fanned out to live subscribers. The sequence number is the
cursor a client resumes from, over SSE (``Last-Event-ID``) or JSON polling
(``?after=``): a caller can drop mid-run and pick up without a gap.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypedDict

from pymongo.collection import Collection

from ghidra_deep_agent.mongo_util import get_mongo_client, mongo_write_with_retry
from ghidra_deep_agent.stream import EVENT_TYPE, StreamEvent

# --- lifecycle events ---------------------------------------------------------


@dataclass(frozen=True)
class Started:
    agent_id: str
    session_id: str
    thread_id: str
    mode: str
    resume: bool


@dataclass(frozen=True)
class RunWarning:
    """A toast raised inside the run (out of credits, sandbox sync, ...)."""

    message: str
    title: str
    severity: str


@dataclass(frozen=True)
class Final:
    reply: str
    input_tokens: int
    output_tokens: int
    status: Literal["done"] = "done"


@dataclass(frozen=True)
class Paused:
    """A provider usage limit outlasted the retries; resumable with continue."""

    message: str


@dataclass(frozen=True)
class Failed:
    message: str


@dataclass(frozen=True)
class Cancelled:
    pass


LifecycleEvent = Started | RunWarning | Final | Paused | Failed | Cancelled
AnyEvent = StreamEvent | LifecycleEvent

LIFECYCLE_TYPE: dict[type[LifecycleEvent], str] = {
    Started: "started",
    RunWarning: "warning",
    Final: "final",
    Paused: "paused",
    Failed: "error",
    Cancelled: "cancelled",
}

# After one of these an SSE stream closes and a poller stops.
TERMINAL_TYPES: frozenset[str] = frozenset({"final", "paused", "error", "cancelled"})

WIRE_TYPE: dict[type[Any], str] = {**EVENT_TYPE, **LIFECYCLE_TYPE}


def event_type(event: AnyEvent) -> str:
    """The wire name of an event."""
    name = WIRE_TYPE.get(type(event))
    if name is None:
        raise TypeError(f"not an event: {event!r}")
    return name


class Persisted(TypedDict):
    """One stored event; also the JSON shape a poller receives."""

    run_id: str
    seq: int
    ts: str
    type: str
    data: dict[str, Any]


def serialize(run_id: str, seq: int, event: AnyEvent) -> Persisted:
    return Persisted(
        run_id=run_id,
        seq=seq,
        ts=datetime.now(UTC).isoformat(),
        type=event_type(event),
        data=dataclasses.asdict(event),
    )


def sse_frame(doc: Persisted) -> dict[str, str]:
    """The keyword arguments for one ``ServerSentEvent``.

    The id is the sequence number so a reconnecting client's ``Last-Event-ID``
    is directly a cursor; the event name is the wire type.
    """
    return {
        "id": str(doc["seq"]),
        "event": doc["type"],
        "data": json.dumps(doc["data"], ensure_ascii=False),
    }


# --- storage ------------------------------------------------------------------


class RunStore(Protocol):
    """Where runs and their events live. Synchronous: callers ``to_thread`` it."""

    def upsert_run(self, doc: dict[str, Any]) -> None: ...

    def load_run(self, run_id: str) -> dict[str, Any] | None: ...

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def append_event(self, doc: Persisted) -> None: ...

    def events_after(self, run_id: str, after: int, limit: int) -> list[Persisted]: ...

    def mark_interrupted(self, message: str) -> int:
        """Fail every non-terminal run (the server restarted). Returns the count."""
        ...


class MemoryRunStore:
    """In-process store: tests, and the fallback when MongoDB is unreachable."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[Persisted]] = {}

    def upsert_run(self, doc: dict[str, Any]) -> None:
        self.runs[doc["id"]] = dict(doc)

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        doc = self.runs.get(run_id)
        return dict(doc) if doc is not None else None

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = [
            dict(r)
            for r in self.runs.values()
            if (agent_id is None or r["agent_id"] == agent_id)
            and (session_id is None or r["session_id"] == session_id)
        ]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows[:limit]

    def append_event(self, doc: Persisted) -> None:
        self.events.setdefault(doc["run_id"], []).append(doc)

    def events_after(self, run_id: str, after: int, limit: int) -> list[Persisted]:
        return [e for e in self.events.get(run_id, []) if e["seq"] > after][:limit]

    def mark_interrupted(self, message: str) -> int:
        count = 0
        for doc in self.runs.values():
            if doc["status"] in ("queued", "running"):
                doc["status"] = "error"
                doc["error"] = message
                count += 1
        return count


class MongoRunStore:
    """``runs`` and ``run_events`` collections, following ``sessions.py``."""

    def __init__(
        self,
        runs: Collection[dict[str, Any]],
        events: Collection[dict[str, Any]],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._runs = runs
        self._events = events
        # One document per (run, seq); the unique index makes a duplicate
        # append a loud error rather than a silent gap in the cursor.
        events.create_index([("run_id", 1), ("seq", 1)], unique=True, name="run_seq")
        if ttl_seconds:
            events.create_index(
                [("ts_at", 1)], expireAfterSeconds=ttl_seconds, name="events_ttl"
            )
        runs.create_index([("created_at", -1)], name="created_desc")

    def upsert_run(self, doc: dict[str, Any]) -> None:
        body = dict(doc)
        run_id = body.pop("id")
        mongo_write_with_retry(
            lambda: self._runs.update_one({"_id": run_id}, {"$set": body}, upsert=True)
        )

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        doc = self._runs.find_one({"_id": run_id})
        if doc is None:
            return None
        doc["id"] = doc.pop("_id")
        return doc

    def list_runs(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if agent_id is not None:
            query["agent_id"] = agent_id
        if session_id is not None:
            query["session_id"] = session_id
        rows = []
        for doc in self._runs.find(query).sort("created_at", -1).limit(limit):
            doc["id"] = doc.pop("_id")
            rows.append(doc)
        return rows

    def append_event(self, doc: Persisted) -> None:
        body: dict[str, Any] = {
            "_id": f"{doc['run_id']}:{doc['seq']:08d}",
            **doc,
            # A real datetime for the TTL index; `ts` stays the ISO string.
            "ts_at": datetime.fromisoformat(doc["ts"]),
        }
        mongo_write_with_retry(lambda: self._events.insert_one(body))

    def events_after(self, run_id: str, after: int, limit: int) -> list[Persisted]:
        cursor = (
            self._events.find({"run_id": run_id, "seq": {"$gt": after}})
            .sort("seq", 1)
            .limit(limit)
        )
        return [
            Persisted(
                run_id=d["run_id"],
                seq=d["seq"],
                ts=d["ts"],
                type=d["type"],
                data=d["data"],
            )
            for d in cursor
        ]

    def mark_interrupted(self, message: str) -> int:
        result = self._runs.update_many(
            {"status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "error", "error": message}},
        )
        return int(result.modified_count)


def build_run_store(mongodb_uri: str, mongodb_db: str) -> RunStore:
    """The Mongo-backed store, or an in-memory one (with a warning) if Mongo is down.

    Falling back keeps the server usable for live runs; only reconnect-after-
    restart history is lost.
    """
    runs_name = os.environ.get("MONGODB_RUNS_COLLECTION", "runs")
    events_name = os.environ.get("MONGODB_RUN_EVENTS_COLLECTION", "run_events")
    ttl_raw = os.environ.get("RUN_EVENTS_TTL", "")
    ttl = int(ttl_raw) if ttl_raw.isdigit() else None
    try:
        db = get_mongo_client(mongodb_uri)[mongodb_db]
        return MongoRunStore(db[runs_name], db[events_name], ttl_seconds=ttl)
    except Exception as exc:  # pragma: no cover - environmental
        print(f"Warning: run history disabled, using memory ({exc})", file=sys.stderr)
        return MemoryRunStore()
