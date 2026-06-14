"""Unit tests for the ghidra-mcp container's import service (no Ghidra, no net).

importer.py is a standalone script deployed into the ghidra-mcp image rather than
part of the package, so it is loaded by path. We cover the security-relevant name
allowlist, the analyzeHeadless output -> HTTP-status mapping, and the duplicate
pre-check's best-effort behavior.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import pytest


def _load_importer() -> Any:
    path = pathlib.Path(__file__).parent / "docker" / "ghidra-mcp" / "importer.py"
    spec = importlib.util.spec_from_file_location("ghidra_importer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = _load_importer()


class _Resp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Proc:
    def __init__(self, stdout: str, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class _FakeSubprocess:
    def __init__(self, proc: _Proc) -> None:
        self._proc = proc

    def run(self, *args: object, **kwargs: object) -> _Proc:
        return self._proc


# ---- name allowlist (defense in depth alongside server._safe_program_name) --


@pytest.mark.parametrize("name", ["ls", "ls.bin", "a_b-c.1", "X" * 128])
def test_name_regex_accepts(name: str) -> None:
    assert importer.NAME_RE.match(name)


@pytest.mark.parametrize(
    "name", ["a/b", "../x", "ev!l", "a b", "x;rm -rf", "", "Y" * 129]
)
def test_name_regex_rejects(name: str) -> None:
    assert importer.NAME_RE.match(name) is None


# ---- analyzeHeadless output -> status --------------------------------------


def test_run_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    monkeypatch.setattr(
        importer,
        "subprocess",
        _FakeSubprocess(_Proc("INFO REPORT: Added file to repository: /x")),
    )
    result = importer._run_import("/tmp/x", "repo", "x")
    assert result["status"] == "imported"
    assert result["analyzed"] is False
    assert result["name"] == "x"


def test_run_import_already_exists_is_409(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    monkeypatch.setattr(
        importer,
        "subprocess",
        _FakeSubprocess(_Proc("ERROR file /x already exists in repository")),
    )
    with pytest.raises(importer.ImportFailure) as excinfo:
        importer._run_import("/tmp/x", "repo", "x")
    assert excinfo.value.status == 409


def test_run_import_other_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    monkeypatch.setattr(
        importer, "subprocess", _FakeSubprocess(_Proc("ERROR boom unexpected"))
    )
    with pytest.raises(importer.ImportFailure) as excinfo:
        importer._run_import("/tmp/x", "repo", "x")
    assert excinfo.value.status == 502


def test_run_import_missing_password_is_500(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "")
    with pytest.raises(importer.ImportFailure) as excinfo:
        importer._run_import("/tmp/x", "repo", "x")
    assert excinfo.value.status == 500


# ---- duplicate pre-check ---------------------------------------------------


def test_repo_has_program_detects_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        return _Resp(200, b'{"files": [{"name": "existing"}]}')

    monkeypatch.setattr(importer.urllib.request, "urlopen", fake_urlopen)
    assert importer._repo_has_program("repo", "existing") is True
    assert importer._repo_has_program("repo", "other") is False


def test_repo_has_program_unreachable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        raise importer.urllib.error.URLError("connection refused")

    monkeypatch.setattr(importer.urllib.request, "urlopen", fake_urlopen)
    assert importer._repo_has_program("repo", "x") is None
