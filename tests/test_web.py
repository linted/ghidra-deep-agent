"""Tests for the web UI: event translation, program parsing, and HTTP/WS routes.

The route tests drive a stub AgentService (no MongoDB or Ghidra MCP), mirroring
the stub-agent approach used in [test_tui.py](test_tui.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ghidra_deep_agent.importer_client import ImportResult
from ghidra_deep_agent.program_resolver import parse_program_list
from ghidra_deep_agent.web import server
from ghidra_deep_agent.web.event_stream import event_to_payloads
from ghidra_deep_agent.web.sessions import Session

# ---- event_to_payloads -----------------------------------------------------


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text


class _LLMOutput:
    def __init__(self) -> None:
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


def _types(payloads: list[dict[str, Any]]) -> list[str]:
    return [p["type"] for p in payloads]


def test_tool_start_emits_start_and_count() -> None:
    payloads = event_to_payloads(
        {
            "event": "on_tool_start",
            "run_id": "r1",
            "name": "decompile",
            "data": {"input": {"description": "look at main"}},
            "metadata": {"langgraph_checkpoint_ns": "tools:a"},
        }
    )
    assert _types(payloads) == ["tool_start", "tool_count"]
    assert payloads[0]["preview"] == "look at main"
    assert payloads[0]["is_subagent"] is False
    assert payloads[1]["delta"] == 1


def test_task_tool_is_subagent() -> None:
    payloads = event_to_payloads(
        {"event": "on_tool_start", "run_id": "r", "name": "task", "data": {}}
    )
    assert payloads[0]["is_subagent"] is True


def test_tool_end_error_carries_snippet() -> None:
    payloads = event_to_payloads(
        {
            "event": "on_tool_end",
            "run_id": "r1",
            "data": {"error": True, "output": "boom failed"},
        }
    )
    assert _types(payloads) == ["tool_end", "tool_count"]
    assert payloads[0]["error"] is True
    assert payloads[0]["snippet"] == "boom failed"
    assert payloads[1]["delta"] == -1


def test_chat_model_end_emits_done_and_tokens_and_context() -> None:
    payloads = event_to_payloads(
        {
            "event": "on_chat_model_end",
            "run_id": "r1",
            "metadata": {"langgraph_checkpoint_ns": ""},
            "data": {"output": _LLMOutput()},
        }
    )
    assert _types(payloads) == ["llm_done", "token_update", "context_update"]
    assert payloads[1]["input"] == 10
    assert payloads[1]["output"] == 5
    assert payloads[2]["current_input"] == 10


def test_subagent_chat_end_skips_context_update() -> None:
    payloads = event_to_payloads(
        {
            "event": "on_chat_model_end",
            "run_id": "r1",
            "metadata": {"langgraph_checkpoint_ns": "tools:a|tools:b"},
            "data": {"output": _LLMOutput()},
        }
    )
    assert "context_update" not in _types(payloads)


def test_compaction_events_become_status_flashes() -> None:
    start = event_to_payloads(
        {
            "event": "on_chat_model_start",
            "run_id": "r",
            "metadata": {"lc_source": "summarization"},
        }
    )
    assert start[0]["type"] == "status_flash"
    stream = event_to_payloads(
        {
            "event": "on_chat_model_stream",
            "run_id": "r",
            "metadata": {"lc_source": "summarization"},
            "data": {"chunk": _Chunk("ignored")},
        }
    )
    assert stream == []


def test_token_stream() -> None:
    payloads = event_to_payloads(
        {
            "event": "on_chat_model_stream",
            "run_id": "r",
            "metadata": {},
            "data": {"chunk": _Chunk("hello")},
        }
    )
    assert payloads == [{"type": "token", "text": "hello"}]


# ---- program parsing -------------------------------------------------------


def test_parse_program_list_json() -> None:
    assert parse_program_list('[{"name": "a.bin"}, {"name": "b.bin"}]') == [
        "a.bin",
        "b.bin",
    ]


def test_parse_program_list_lines() -> None:
    assert parse_program_list("Open programs:\n1. a.bin\n2. b.bin (x86)") == [
        "a.bin",
        "b.bin",
    ]


# ---- routes & websocket (stub service) -------------------------------------


class _Settings:
    model = "test-model"


class _StubSessions:
    def __init__(self) -> None:
        self._items: dict[str, Session] = {}

    def create(self, binary_name: str) -> Session:
        s = Session(
            session_id="sess-1",
            binary_name=binary_name,
            title=binary_name,
            created_at="t0",
            updated_at="t0",
        )
        self._items[s.session_id] = s
        return s

    def list(self) -> list[Session]:
        return list(self._items.values())

    def get(self, sid: str) -> Session | None:
        return self._items.get(sid)

    def delete(self, sid: str) -> bool:
        return self._items.pop(sid, None) is not None

    def rename(self, sid: str, title: str) -> None:
        s = self._items.get(sid)
        if s is not None:
            s.title = title


class _StubService:
    def __init__(self) -> None:
        self.settings = _Settings()
        self.max_context_tokens = 200000
        self.mcp_ok = True
        self.db_ok = True
        self.sessions = _StubSessions()
        self._running: set[str] = set()

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def list_programs(self) -> list[str]:
        return ["a.bin", "b.bin"]

    def create_session(self, binary_name: str) -> Session:
        return self.sessions.create(binary_name)

    def is_running(self, sid: str) -> bool:
        return sid in self._running

    def cancel(self, sid: str) -> bool:
        self._running.discard(sid)
        return True

    async def history(self, sid: str) -> list[dict[str, Any]]:
        if self.sessions.get(sid) is None:
            raise KeyError(sid)
        return [{"role": "user", "content": "hi"}]

    async def stream(self, sid: str, query: str) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "token", "text": f"echo: {query}"}
        yield {"type": "agent_done"}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "service", _StubService())
    return TestClient(server.app)


def test_config_route(client: TestClient) -> None:
    with client:
        data = client.get("/api/config").json()
    assert data["model"] == "test-model"
    assert data["mcp_ok"] is True


def test_programs_route(client: TestClient) -> None:
    with client:
        data = client.get("/api/programs").json()
    assert data["programs"] == ["a.bin", "b.bin"]


def test_session_crud_and_history(client: TestClient) -> None:
    with client:
        created = client.post("/api/sessions", json={"binary_name": "a.bin"}).json()
        assert created["binary_name"] == "a.bin"
        listed = client.get("/api/sessions").json()["sessions"]
        assert any(s["session_id"] == created["session_id"] for s in listed)
        hist = client.get(f"/api/sessions/{created['session_id']}/history").json()
        assert hist["messages"][0]["content"] == "hi"
        deleted = client.delete(f"/api/sessions/{created['session_id']}").json()
        assert deleted["deleted"] is True


def test_session_rename(client: TestClient) -> None:
    with client:
        created = client.post("/api/sessions", json={"binary_name": "a.bin"}).json()
        sid = created["session_id"]
        resp = client.patch(f"/api/sessions/{sid}", json={"title": "  renamed  "})
        assert resp.status_code == 200
        assert resp.json()["title"] == "renamed"
        listed = client.get("/api/sessions").json()["sessions"]
        assert next(s for s in listed if s["session_id"] == sid)["title"] == "renamed"


def test_session_rename_empty_title_400(client: TestClient) -> None:
    with client:
        created = client.post("/api/sessions", json={"binary_name": "a.bin"}).json()
        resp = client.patch(
            f"/api/sessions/{created['session_id']}", json={"title": "   "}
        )
    assert resp.status_code == 400


def test_history_unknown_session_404(client: TestClient) -> None:
    with client:
        resp = client.get("/api/sessions/nope/history")
    assert resp.status_code == 404


def test_websocket_streams_query(client: TestClient) -> None:
    with client:
        client.post("/api/sessions", json={"binary_name": "a.bin"})
        with client.websocket_connect("/api/sessions/sess-1/stream") as ws:
            ws.send_json({"type": "query", "text": "ping"})
            first = ws.receive_json()
            second = ws.receive_json()
    assert first == {"type": "token", "text": "echo: ping"}
    assert second == {"type": "agent_done"}


# ---- upload route ----------------------------------------------------------


def _stub_import(
    result: ImportResult | None = None, exc: Exception | None = None
) -> Any:
    """Build an async stand-in for ``server.import_binary``."""

    async def _imp(
        name: str, data: bytes, repo: str | None = None, **_hints: Any
    ) -> ImportResult:
        if exc is not None:
            raise exc
        assert result is not None
        return result

    return _imp


def test_upload_success_passes_through_importer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server,
        "import_binary",
        _stub_import(ImportResult(200, {"status": "imported", "name": "true"})),
    )
    with client:
        resp = client.post("/api/upload", params={"name": "true"}, content=b"\x7fELF")
    assert resp.status_code == 200
    assert resp.json()["name"] == "true"


def test_upload_duplicate_is_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server,
        "import_binary",
        _stub_import(ImportResult(409, {"error": "a program named 'true' exists"})),
    )
    with client:
        resp = client.post("/api/upload", params={"name": "true"}, content=b"x")
    assert resp.status_code == 409


def test_upload_invalid_name_is_400(client: TestClient) -> None:
    # 'ev!l' survives basename but fails the allowlist; importer is never called.
    with client:
        resp = client.post("/api/upload", params={"name": "ev!l"}, content=b"x")
    assert resp.status_code == 400


def test_upload_traversal_name_reduced_to_basename(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    async def _imp(
        name: str, data: bytes, repo: str | None = None, **_hints: Any
    ) -> ImportResult:
        seen["name"] = name
        return ImportResult(200, {"name": name})

    monkeypatch.setattr(server, "import_binary", _imp)
    with client:
        resp = client.post(
            "/api/upload", params={"name": "../../etc/passwd"}, content=b"x"
        )
    assert resp.status_code == 200
    assert seen["name"] == "passwd"


def test_upload_empty_body_is_400(client: TestClient) -> None:
    with client:
        resp = client.post("/api/upload", params={"name": "true"}, content=b"")
    assert resp.status_code == 400


def test_upload_oversize_is_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    with client:
        resp = client.post("/api/upload", params={"name": "true"}, content=b"12345")
    assert resp.status_code == 413


def test_upload_service_down_is_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server, "import_binary", _stub_import(exc=RuntimeError("unreachable"))
    )
    with client:
        resp = client.post("/api/upload", params={"name": "true"}, content=b"x")
    assert resp.status_code == 502


def test_upload_forwards_import_hints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def _imp(
        name: str, data: bytes, repo: str | None = None, **hints: Any
    ) -> ImportResult:
        seen.update(hints)
        return ImportResult(200, {"name": name})

    monkeypatch.setattr(server, "import_binary", _imp)
    with client:
        resp = client.post(
            "/api/upload",
            params={
                "name": "blob.bin",
                "processor": "ARM:LE:32:v8",
                "cspec": "default",
                "base": "0x8000",
            },
            content=b"x",
        )
    assert resp.status_code == 200
    assert seen == {
        "loader": None,
        "processor": "ARM:LE:32:v8",
        "cspec": "default",
        "base": "0x8000",
    }


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("processor", "ARM LE 32"),  # space not allowed
        ("base", "nothex"),
        ("loader", "Binary Loader"),  # space not allowed
    ],
)
def test_upload_invalid_hint_is_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, param: str, value: str
) -> None:
    called = {"hit": False}

    async def _imp(name: str, data: bytes, **_k: Any) -> ImportResult:
        called["hit"] = True
        return ImportResult(200, {})

    monkeypatch.setattr(server, "import_binary", _imp)
    with client:
        resp = client.post(
            "/api/upload", params={"name": "blob", param: value}, content=b"x"
        )
    assert resp.status_code == 400
    assert called["hit"] is False


def test_languages_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _langs() -> ImportResult:
        return ImportResult(200, {"languages": [{"id": "ARM:LE:32:v8"}], "count": 1})

    monkeypatch.setattr(server, "list_languages", _langs)
    with client:
        resp = client.get("/api/languages")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
