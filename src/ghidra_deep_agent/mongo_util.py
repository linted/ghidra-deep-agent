"""Shared MongoDB helpers.

`knowledge.py` and `sessions.py` both write to Mongo and both need the same
transient-failure classification, so it lives here rather than being mirrored in
each module. The three Mongo-backed subsystems (knowledge base, session registry,
MCP read cache) also all connect to the same URI, so the client they share and the
teardown that closes it live here too.
"""

from collections.abc import Callable
from typing import Any

from pymongo import MongoClient
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
# timeouts, primary step-downs. Persistent errors (bad query, auth) fall through
# to the caller's PyMongoError handler immediately.
TRANSIENT_MONGO_ERRORS = (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WaitQueueTimeoutError,
    ExecutionTimeout,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception_type(TRANSIENT_MONGO_ERRORS),
)
def mongo_write_with_retry[T](fn: Callable[[], T]) -> T:
    """Run a MongoDB write, retrying transient failures with backoff."""
    return fn()


# One client per URI, reused by every subsystem. Each MongoClient owns a
# connection pool (100 sockets by default), so building three against the same
# server — as the knowledge base, session registry, and read cache each used to —
# triples the pools for no benefit.
_CLIENTS: dict[str, MongoClient[Any]] = {}


def get_mongo_client(mongodb_uri: str) -> MongoClient[Any]:
    """Return the shared client for ``mongodb_uri``, creating it on first use."""
    client = _CLIENTS.get(mongodb_uri)
    if client is None:
        client = MongoClient(mongodb_uri)
        _CLIENTS[mongodb_uri] = client
    return client


def close_mongo_clients() -> None:
    """Close every shared client. Called once on shutdown."""
    while _CLIENTS:
        _, client = _CLIENTS.popitem()
        try:
            client.close()
        except Exception:  # teardown must not mask whatever is unwinding
            pass
