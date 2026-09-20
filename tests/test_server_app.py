"""The HTTP API, driven in-process against a stub registry."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from server_stubs import GraphPlan, HangingGraph, StubGraph, make_registry

from ghidra_deep_agent.server.app import create_app
from ghidra_deep_agent.server.registry import AgentRegistry


class _Msg:
    def __init__(self, kind: str, content: str) -> None:
        self.type = kind
        self.content = content


@asynccontextmanager
async def client(
    registry: AgentRegistry, *, token: str | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(registry, token=token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", headers=headers
        ) as c:
            yield c


def test_auth_and_health() -> None:
    async def run() -> None:
        registry, _, _ = make_registry()
        async with client(registry, token="secret") as c:
            assert (await c.get("/health")).json()["status"] == "ok"
            bare = httpx.AsyncClient(transport=c._transport, base_url="http://t")
            assert (await bare.get("/health")).status_code == 200
            r = await bare.get("/agents")
            assert r.status_code == 401 and r.json()["error"] == "unauthorized"
            assert (await c.get("/agents")).status_code == 200
            assert (await c.get("/openapi.json")).status_code == 200

    asyncio.run(run())


def test_programs_and_open() -> None:
    async def run() -> None:
        registry, _, _ = make_registry()
        async with client(registry) as c:
            r = await c.get("/programs")
            assert r.status_code == 200
            assert [p["project_path"] for p in r.json()["programs"]] == [
                "/v1/app.exe",
                "/libs/lib.so",
            ]
            assert r.json()["programs"][0]["active"] is True
            r = await c.post("/programs/open", json={"path": "/v1/app.exe"})
            assert r.status_code == 200 and r.json()["project_path"] == "/v1/app.exe"
            registry.shared.probe_tools[1].reply = "Program not found: '/zz'."
            r = await c.post("/programs/open", json={"path": "/zz"})
            assert r.status_code == 404 and r.json()["error"] == "program_not_open"
            registry.shared.probe_tools[0].reply = RuntimeError("ghidra down")
            r = await c.get("/programs")
            assert r.status_code == 502 and r.json()["error"] == "ghidra_error"

    asyncio.run(run())


def test_agents_crud() -> None:
    async def run() -> None:
        registry, _, _ = make_registry()
        async with client(registry) as c:
            r = await c.post("/agents", json={"program": "/v1/app.exe"})
            assert r.status_code == 201
            agent = r.json()
            assert agent["status"] == "ready" and agent["program"]["name"] == "app.exe"
            assert agent["tool_count"] == 3
            r = await c.post("/agents", json={"program": "app.exe"})
            assert r.status_code == 200 and r.json()["id"] == agent["id"]
            r = await c.post("/agents", json={"program": "/missing"})
            assert r.status_code == 404
            assert (await c.get(f"/agents/{agent['id']}")).json()["id"] == agent["id"]
            assert len((await c.get("/agents")).json()["agents"]) == 1
            assert (await c.delete(f"/agents/{agent['id']}")).status_code == 204
            assert (await c.get(f"/agents/{agent['id']}")).status_code == 404

    asyncio.run(run())


async def _agent(c: httpx.AsyncClient, program: str = "/v1/app.exe") -> str:
    r = await c.post("/agents", json={"program": program})
    assert r.status_code in (200, 201), r.text
    return str(r.json()["id"])


def test_run_start_wait_and_get() -> None:
    async def run() -> None:
        plan = GraphPlan()
        plan.by_path["/v1/app.exe"] = StubGraph("the answer")
        registry, _, _ = make_registry(plan)
        async with client(registry) as c:
            agent_id = await _agent(c)
            r = await c.post(f"/agents/{agent_id}/runs", json={"prompt": "hi"})
            assert r.status_code == 202, r.text
            body = r.json()
            run_id = body["id"]
            assert body["links"]["events"] == f"/runs/{run_id}/events"
            r = await c.post(f"/runs/{run_id}/wait", json={"timeout": 5})
            assert r.status_code == 200 and r.json()["status"] == "done"
            assert r.json()["reply"] == "the answer"
            assert r.json()["usage"] == {"input_tokens": 10, "output_tokens": 5}
            r = await c.get(f"/runs/{run_id}")
            assert r.json()["status"] == "done"
            r = await c.get("/runs", params={"agent_id": agent_id})
            assert [x["id"] for x in r.json()["runs"]] == [run_id]
            assert (await c.get("/runs/nope")).status_code == 404
            # Validation: exactly one of prompt / continue.
            r = await c.post(f"/agents/{agent_id}/runs", json={})
            assert r.status_code == 422
            r = await c.post(
                f"/agents/{agent_id}/runs", json={"prompt": "x", "continue": True}
            )
            assert r.status_code == 422
            r = await c.post(f"/agents/{agent_id}/runs", json={"continue": True})
            assert r.status_code == 422  # needs session_id
            r = await c.post(
                f"/agents/{agent_id}/runs",
                json={"continue": True, "session_id": body["session_id"]},
            )
            assert r.status_code == 202 and r.json()["resume"] is True

    asyncio.run(run())


def test_thread_busy_wait_202_and_cancel() -> None:
    async def run() -> None:
        plan = GraphPlan()
        plan.by_path["/v1/app.exe"] = HangingGraph()
        registry, _, _ = make_registry(plan)
        async with client(registry) as c:
            agent_id = await _agent(c)
            r = await c.post(
                f"/agents/{agent_id}/runs", json={"prompt": "x", "session_id": "s"}
            )
            run_id = r.json()["id"]
            r = await c.post(
                f"/agents/{agent_id}/runs", json={"prompt": "y", "session_id": "s"}
            )
            assert r.status_code == 409 and r.json()["error"] == "thread_busy"
            assert r.json()["extra"] == {"active_run_id": run_id}
            r = await c.post(f"/runs/{run_id}/wait", json={"timeout": 0.05})
            assert r.status_code == 202 and r.json()["status"] == "running"
            r = await c.post(f"/runs/{run_id}/cancel")
            assert r.json() == {"cancelled": True}
            assert (await c.get(f"/runs/{run_id}")).json()["status"] == "cancelled"

    asyncio.run(run())


def test_events_json_cursor_and_long_poll() -> None:
    async def run() -> None:
        plan = GraphPlan()
        plan.by_path["/v1/app.exe"] = StubGraph(delay=0.1)
        registry, _, _ = make_registry(plan)
        async with client(registry) as c:
            agent_id = await _agent(c)
            run_id = (
                await c.post(f"/agents/{agent_id}/runs", json={"prompt": "x"})
            ).json()["id"]
            await c.post(f"/runs/{run_id}/wait", json={"timeout": 5})
            r = await c.get(f"/runs/{run_id}/events")
            page = r.json()
            types = [e["type"] for e in page["events"]]
            assert types[0] == "started" and types[-1] == "final"
            assert page["terminal"] is True and page["next_after"] == len(types)
            assert [e["seq"] for e in page["events"]] == list(range(1, len(types) + 1))
            r = await c.get(f"/runs/{run_id}/events", params={"after": 2, "limit": 2})
            assert [e["seq"] for e in r.json()["events"]] == [3, 4]
            assert r.json()["terminal"] is False and r.json()["next_after"] == 4
            r = await c.get(f"/runs/{run_id}/events", params={"after": 999, "wait": 1})
            assert r.json()["events"] == [] and r.json()["terminal"] is True

            # Long-poll on a live run returns as soon as an event lands.
            run_id = (
                await c.post(f"/agents/{agent_id}/runs", json={"prompt": "y"})
            ).json()["id"]
            r = await c.get(f"/runs/{run_id}/events", params={"after": 1, "wait": 5})
            assert r.json()["events"], "long-poll returned nothing"
            await c.post(f"/runs/{run_id}/wait", json={"timeout": 5})

    asyncio.run(run())


def test_events_sse_replay_then_live_with_last_event_id() -> None:
    async def run() -> None:
        plan = GraphPlan()
        plan.by_path["/v1/app.exe"] = StubGraph(delay=0.2)
        registry, _, _ = make_registry(plan)
        async with client(registry) as c:
            agent_id = await _agent(c)
            run_id = (
                await c.post(f"/agents/{agent_id}/runs", json={"prompt": "x"})
            ).json()["id"]
            frames: list[tuple[str, str, dict[str, Any]]] = []
            async with c.stream(
                "GET", f"/runs/{run_id}/events", headers={"Accept": "text/event-stream"}
            ) as resp:
                assert resp.headers["content-type"].startswith("text/event-stream")
                cur: dict[str, str] = {}
                async for line in resp.aiter_lines():
                    if not line:
                        if "event" in cur:
                            frames.append(
                                (cur["id"], cur["event"], json.loads(cur["data"]))
                            )
                        cur = {}
                        continue
                    key, _, value = line.partition(":")
                    cur[key] = value.strip()
            types = [f[1] for f in frames]
            assert types[0] == "started" and types[-1] == "final"
            assert "tool_start" in types and "token" not in types
            assert [int(f[0]) for f in frames] == list(range(1, len(frames) + 1))

            # Resume from a cursor: only later frames, and the stream closes.
            async with c.stream(
                "GET",
                f"/runs/{run_id}/events",
                headers={"Accept": "text/event-stream", "Last-Event-ID": "2"},
            ) as resp:
                ids = [
                    int(line[3:].strip())
                    async for line in resp.aiter_lines()
                    if line.startswith("id:")
                ]
            assert ids == list(range(3, len(frames) + 1))

    asyncio.run(run())


def test_sessions_and_history() -> None:
    class FakeSessions:
        async def alist_sessions(
            self, binary: str | None, limit: int
        ) -> list[dict[str, Any]]:
            return [
                {"_id": "s1", "session_id": "s1", "binary_name": binary or "app.exe"}
            ]

        async def arecord_start(self, *a: Any) -> None:
            pass

        async def atouch(self, *a: Any, **k: Any) -> None:
            pass

    async def run() -> None:
        plan = GraphPlan()
        graph = StubGraph()
        graph.state_values = {
            "messages": [_Msg("human", "hi"), _Msg("ai", "hello"), _Msg("tool", "x")]
        }
        plan.by_path["/v1/app.exe"] = graph
        registry, _, _ = make_registry(plan, session_store=FakeSessions())
        async with client(registry) as c:
            r = await c.get("/sessions/s1/history")
            assert r.status_code == 409  # no agent yet
            agent_id = await _agent(c)
            r = await c.get("/sessions", params={"binary": "app.exe"})
            assert r.json()["sessions"][0]["session_id"] == "s1"
            r = await c.get("/sessions/s1/history", params={"agent_id": agent_id})
            assert r.json() == {
                "session_id": "s1",
                "thread_id": "s1",
                "messages": [
                    {"role": "user", "text": "hi"},
                    {"role": "assistant", "text": "hello"},
                ],
            }
            assert graph.state_configs[-1]["configurable"]["thread_id"] == "s1"

    asyncio.run(run())


def test_files_filesystem_and_state(tmp_path: Path) -> None:
    async def run() -> None:
        plan = GraphPlan()
        registry, _, _ = make_registry(plan)
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "a.md").write_text("# A")
        (tmp_path / "top.txt").write_text("hello")
        async with client(registry) as c:
            agent_id = await _agent(c)
            engine = registry.get_agent(agent_id).engine
            assert engine is not None
            engine.output_dir = str(tmp_path)
            r = await c.get(f"/agents/{agent_id}/files")
            assert r.json()["backend"] == "filesystem"
            assert [(e["path"], e["is_dir"]) for e in r.json()["entries"]] == [
                ("/notes", True),
                ("/top.txt", False),
            ]
            r = await c.get(f"/agents/{agent_id}/files", params={"path": "/notes"})
            assert r.json()["entries"][0]["path"] == "/notes/a.md"
            r = await c.get(f"/agents/{agent_id}/files/notes/a.md")
            assert r.status_code == 200 and r.text == "# A"
            assert (await c.get(f"/agents/{agent_id}/files/nope")).status_code == 404
            r = await c.get(f"/agents/{agent_id}/files/../../etc/passwd")
            assert r.status_code in (400, 404)
            r = await c.get(f"/agents/{agent_id}/files", params={"path": "../.."})
            assert r.status_code == 400

            # State backend: files live in the thread state.
            engine.output_dir = None
            graph = engine.graphs.main
            graph.state_values = {
                "files": {"/plan.md": {"content": ["# Plan", "- step"]}}
            }
            r = await c.get(f"/agents/{agent_id}/files")
            assert r.status_code == 400 and r.json()["error"] == "session_required"
            r = await c.get(f"/agents/{agent_id}/files", params={"session_id": "s"})
            assert r.json() == {
                "backend": "state",
                "entries": [{"path": "/plan.md", "is_dir": False, "size": 13}],
            }
            r = await c.get(
                f"/agents/{agent_id}/files/plan.md", params={"session_id": "s"}
            )
            assert r.text == "# Plan\n- step"

    asyncio.run(run())


def test_server_package_does_not_import_textual() -> None:
    code = (
        "import sys, ghidra_deep_agent.server.app, ghidra_deep_agent.server.main; "
        "sys.exit(1 if 'textual' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("path", ["/agents", "/programs"])
def test_unknown_agent_is_404(path: str) -> None:
    async def run() -> None:
        registry, _, _ = make_registry()
        async with client(registry) as c:
            r = await c.get("/agents/zzz")
            assert r.status_code == 404 and r.json()["error"] == "unknown_agent"
            r = await c.post("/agents/zzz/runs", json={"prompt": "x"})
            assert r.status_code == 404
            assert (await c.get(path)).status_code == 200

    asyncio.run(run())
