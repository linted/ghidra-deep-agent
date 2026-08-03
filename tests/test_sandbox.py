"""
Unit tests for the sandbox teardown helpers in ghidra_deep_agent.sandbox.

Focus: a connection-loss teardown failure is classified as retryable, the retry
path deletes the sandbox with fresh short-timeout clients and swallows every
failure into False, and the warning fallback is a single readable line instead
of grpc's multi-line _InactiveRpcError debug repr. No network, no sandbox.

Run:  uv run pytest tests/test_sandbox.py -v
"""

from __future__ import annotations

from typing import Any

import pytest

from ghidra_deep_agent.sandbox import (
    _is_connectivity_error,
    _retry_delete,
    _teardown_warning,
)

# grpcio is a transitive dep of the openshell extras; skip cleanly without it.
grpc = pytest.importorskip("grpc")


class FakeRpcError(grpc.RpcError):  # type: ignore[misc, name-defined]
    """Mimics _InactiveRpcError: code()/details() plus the multi-line repr."""

    def __init__(self, code: Any, details: str = "boom") -> None:
        self._code = code
        self._details = details

    def code(self) -> Any:
        return self._code

    def details(self) -> str:
        return self._details

    def __str__(self) -> str:
        return (
            "<_InactiveRpcError of RPC that terminated with:\n"
            f"\tstatus = {self._code}\n"
            f'\tdetails = "{self._details}"\n>'
        )


class StubClient:
    """Records delete calls; raises delete_exc if set; tracks close()."""

    def __init__(self, delete_exc: Exception | None = None) -> None:
        self.delete_exc = delete_exc
        self.delete_calls: list[tuple[str, str]] = []
        self.closed = False

    def delete(self, name: str, *, workspace: str) -> bool:
        self.delete_calls.append((name, workspace))
        if self.delete_exc is not None:
            raise self.delete_exc
        return True

    def close(self) -> None:
        self.closed = True


def _factory(clients: list[StubClient]) -> Any:
    """Client factory handing out ``clients`` in order; raises when exhausted."""
    it = iter(clients)

    def make() -> StubClient:
        return next(it)

    return make


class TestIsConnectivityError:
    def test_unavailable_rpc_error(self) -> None:
        exc = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Stream removed")
        assert _is_connectivity_error(exc)

    def test_other_rpc_error(self) -> None:
        exc = FakeRpcError(grpc.StatusCode.PERMISSION_DENIED)
        assert not _is_connectivity_error(exc)

    def test_substring_fallback_without_rpc_error(self) -> None:
        exc = RuntimeError("rpc failed: StatusCode.UNAVAILABLE, stream gone")
        assert _is_connectivity_error(exc)

    def test_unrelated_exception(self) -> None:
        assert not _is_connectivity_error(RuntimeError("boom"))


class TestTeardownWarning:
    def test_rpc_error_is_condensed_to_one_line(self) -> None:
        exc = FakeRpcError(
            grpc.StatusCode.UNAVAILABLE,
            "Stream removed (recvmsg:Connection reset by peer (54))",
        )
        message = _teardown_warning(exc)
        assert "\n" not in message
        assert "_InactiveRpcError" not in message
        assert message.startswith("Warning: OpenShell sandbox teardown failed: ")
        assert "UNAVAILABLE" in message
        assert "Stream removed" in message

    def test_plain_exception_keeps_original_format(self) -> None:
        message = _teardown_warning(RuntimeError("boom"))
        assert message == "Warning: OpenShell sandbox teardown failed: boom"


class TestRetryDelete:
    def test_success_on_first_attempt(self) -> None:
        client = StubClient()
        assert _retry_delete("sb-1", "ws", client_factory=_factory([client]))
        assert client.delete_calls == [("sb-1", "ws")]
        assert client.closed

    def test_second_attempt_succeeds_after_connectivity_failure(self) -> None:
        first = StubClient(delete_exc=FakeRpcError(grpc.StatusCode.UNAVAILABLE))
        second = StubClient()
        assert _retry_delete("sb-1", "ws", client_factory=_factory([first, second]))
        assert first.closed
        assert second.closed
        assert second.delete_calls == [("sb-1", "ws")]

    def test_all_attempts_fail_returns_false(self) -> None:
        clients = [
            StubClient(delete_exc=FakeRpcError(grpc.StatusCode.UNAVAILABLE)),
            StubClient(delete_exc=FakeRpcError(grpc.StatusCode.UNAVAILABLE)),
        ]
        assert not _retry_delete("sb-1", "ws", client_factory=_factory(clients))
        assert all(c.closed for c in clients)

    def test_not_found_counts_as_success(self) -> None:
        client = StubClient(delete_exc=FakeRpcError(grpc.StatusCode.NOT_FOUND))
        assert _retry_delete("sb-1", "ws", client_factory=_factory([client]))
        assert client.closed

    def test_factory_failure_is_swallowed(self) -> None:
        def broken_factory() -> StubClient:
            raise RuntimeError("gateway config missing")

        assert not _retry_delete("sb-1", "ws", client_factory=broken_factory)
