"""Unit tests for the ghidra-server container's import service (no Ghidra, no net).

importer.py is a standalone script deployed into the ghidra-server image rather
than part of the package, so it is loaded by path. We cover the security-relevant
name allowlist, the loader/processor hint-arg construction, the analyzeHeadless
output -> HTTP-status mapping, and the .ldefs language enumeration.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import pytest


def _load_importer() -> Any:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    path = repo_root / "docker" / "ghidra-server" / "importer.py"
    spec = importlib.util.spec_from_file_location("ghidra_importer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = _load_importer()


class _Proc:
    def __init__(self, stdout: str, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class _FakeSubprocess:
    """Stub for the ``subprocess`` module that records the command it was given."""

    def __init__(self, proc: _Proc) -> None:
        self._proc = proc
        self.cmd: list[str] | None = None

    def run(self, cmd: list[str], *args: object, **kwargs: object) -> _Proc:
        self.cmd = cmd
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


# ---- import-hint argument construction -------------------------------------


def test_hint_args_empty_when_no_hints() -> None:
    assert importer._import_hint_args(None, None, None, None) == []


def test_hint_args_processor_implies_binary_loader() -> None:
    args = importer._import_hint_args(None, "ARM:LE:32:v8", None, None)
    assert args == ["-loader", "BinaryLoader", "-processor", "ARM:LE:32:v8"]


def test_hint_args_full_set() -> None:
    args = importer._import_hint_args(None, "ARM:LE:32:v8", "default", "0x8000")
    assert args == [
        "-loader",
        "BinaryLoader",
        "-processor",
        "ARM:LE:32:v8",
        "-cspec",
        "default",
        "-loader-baseAddr",
        "0x8000",
    ]


def test_hint_args_base_ignored_for_non_binary_loader() -> None:
    # Image base only applies to BinaryLoader; an explicit ElfLoader drops it.
    args = importer._import_hint_args("ElfLoader", "x86:LE:64:default", None, "0x1000")
    assert "-loader-baseAddr" not in args
    assert args[:2] == ["-loader", "ElfLoader"]


# ---- analyzeHeadless output -> status --------------------------------------


def test_run_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    fake = _FakeSubprocess(_Proc("INFO REPORT: Added file to repository: /x"))
    monkeypatch.setattr(importer, "subprocess", fake)
    result = importer._run_import("/tmp/x", "repo", "x", processor="ARM:LE:32:v8")
    assert result["status"] == "imported"
    assert result["analyzed"] is False
    assert result["name"] == "x"
    # The hint flags made it into the analyzeHeadless command line.
    assert fake.cmd is not None
    assert "-processor" in fake.cmd and "ARM:LE:32:v8" in fake.cmd


@pytest.mark.parametrize(
    "output",
    [
        "ERROR file /x already exists in repository",
        # The phrase analyzeHeadless emits on a shared-repo commit collision.
        "ERROR REPORT: Found conflicting program file in project: /x",
    ],
)
def test_run_import_duplicate_is_409(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    monkeypatch.setattr(importer, "subprocess", _FakeSubprocess(_Proc(output)))
    with pytest.raises(importer.ImportFailure) as excinfo:
        importer._run_import("/tmp/x", "repo", "x")
    assert excinfo.value.status == 409


def test_run_import_no_load_spec_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "SERVER_PASSWORD", "pw")
    monkeypatch.setattr(
        importer,
        "subprocess",
        _FakeSubprocess(_Proc("ERROR No load spec found for import file: blob.bin")),
    )
    with pytest.raises(importer.ImportFailure) as excinfo:
        importer._run_import("/tmp/x", "repo", "x")
    assert excinfo.value.status == 422
    assert "processor" in excinfo.value.message


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


# ---- language enumeration (.ldefs parsing) ---------------------------------


def test_load_languages_parses_ldefs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    ldir = tmp_path / "Ghidra" / "Processors" / "ARM" / "data" / "languages"
    ldir.mkdir(parents=True)
    (ldir / "ARM.ldefs").write_text(
        """<language_definitions>
          <language id="ARM:LE:32:v8">
            <compiler name="default" spec="ARM.cspec" id="default"/>
            <compiler name="cdecl" spec="ARM.cspec" id="cdecl"/>
          </language>
          <language id="ARM:BE:32:v8"/>
        </language_definitions>"""
    )
    monkeypatch.setattr(importer, "GHIDRA_HOME", str(tmp_path))
    monkeypatch.setattr(importer, "_LANG_CACHE", None)

    langs = importer.load_languages()
    assert langs["ARM:LE:32:v8"] == ["default", "cdecl"]
    assert langs["ARM:BE:32:v8"] == []
