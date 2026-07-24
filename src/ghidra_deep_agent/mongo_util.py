"""Shared MongoDB helpers.

`knowledge.py` and `sessions.py` both write to Mongo and both need the same
transient-failure classification, so it lives here rather than being mirrored in
each module.
"""

from collections.abc import Callable

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
