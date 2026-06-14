"""Unit tests for the web-side import client (no network).

Exercises response decoding, URL resolution, and the urllib error mapping that
lets the FastAPI route distinguish a normal rejection (4xx/5xx with a JSON body)
from the import service being unreachable (RuntimeError).
"""

from __future__ import annotations

import asyncio
import io
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from ghidra_deep_agent import importer_client as ic


class _Resp:
    """Minimal stand-in for an http.client.HTTPResponse / urlopen result."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


# ---- url & decode ----------------------------------------------------------


def test_importer_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GHIDRA_IMPORTER_URL", raising=False)
    assert ic._importer_url() == "http://ghidra-mcp:8082/import"


def test_importer_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHIDRA_IMPORTER_URL", "http://host:9/import")
    assert ic._importer_url() == "http://host:9/import"


def test_decode_valid_object() -> None:
    result = ic._decode(200, b'{"status": "imported", "name": "t"}')
    assert result.status_code == 200
    assert result.ok is True
    assert result.payload["name"] == "t"


def test_decode_non_json_becomes_error() -> None:
    result = ic._decode(502, b"kaboom")
    assert result.status_code == 502
    assert result.ok is False
    assert result.payload == {"error": "kaboom"}


def test_decode_non_object_json_wrapped() -> None:
    result = ic._decode(200, b"[1, 2]")
    assert result.payload == {"result": [1, 2]}


def test_decode_empty_body() -> None:
    result = ic._decode(500, b"")
    assert result.payload == {"error": "non-JSON response"}


# ---- _post error mapping ---------------------------------------------------


def test_post_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        return _Resp(200, b'{"name": "t"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ic._post("http://h/import", {"name": "t"}, b"data")
    assert result.status_code == 200
    assert result.payload["name"] == "t"


def test_post_httperror_surfaces_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        raise urllib.error.HTTPError(
            "http://h/import",
            409,
            "Conflict",
            Message(),
            io.BytesIO(b'{"error": "exists"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ic._post("http://h/import", {"name": "t"}, b"data")
    assert result.status_code == 409
    assert result.payload == {"error": "exists"}


def test_post_urlerror_raises_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="unreachable"):
        ic._post("http://h/import", {"name": "t"}, b"data")


# ---- async wrapper ---------------------------------------------------------


def test_import_binary_runs_post_off_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, params: dict[str, str], data: bytes) -> ic.ImportResult:
        captured["params"] = params
        captured["data"] = data
        return ic.ImportResult(200, {"ok": True})

    monkeypatch.setattr(ic, "_post", fake_post)
    result = asyncio.run(ic.import_binary("t", b"bytes", repo="r"))
    assert result.status_code == 200
    assert captured["params"] == {"name": "t", "repo": "r"}
    assert captured["data"] == b"bytes"
