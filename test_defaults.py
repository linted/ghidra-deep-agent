"""Tests for the shared fallbacks and env parsing (``defaults.py``).

These knobs are read at startup from ``.env``. A typo used to raise a bare
``ValueError`` mid-construction, killing the app with a traceback that named
neither the variable nor the bad value; a zero or negative value was accepted and
turned into a busy-poll or a never-backing-off retry loop.

Run:  uv run pytest test_defaults.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghidra_deep_agent.defaults import config_path, env_float, env_int


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GDA_TEST_INT", "GDA_TEST_FLOAT", "GDA_TEST_CONFIG"):
        monkeypatch.delenv(var, raising=False)


# ── env_int / env_float ───────────────────────────────────────────────────────


def test_unset_returns_the_default() -> None:
    assert env_int("GDA_TEST_INT", 7) == 7
    assert env_float("GDA_TEST_FLOAT", 1.5) == 1.5


def test_valid_value_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GDA_TEST_INT", "42")
    monkeypatch.setenv("GDA_TEST_FLOAT", "0.25")
    assert env_int("GDA_TEST_INT", 7) == 42
    assert env_float("GDA_TEST_FLOAT", 1.5) == 0.25


def test_surrounding_whitespace_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GDA_TEST_INT", "  42  ")
    assert env_int("GDA_TEST_INT", 7) == 42


def test_empty_string_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An `FOO=` line in .env is "unset", not "zero"."""
    monkeypatch.setenv("GDA_TEST_INT", "")
    assert env_int("GDA_TEST_INT", 7) == 7


@pytest.mark.parametrize("raw", ["180s", "abc", "1_0_0x", "1.5"])
def test_garbage_falls_back_instead_of_raising(
    raw: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GDA_TEST_INT", raw)
    assert env_int("GDA_TEST_INT", 7) == 7
    # The warning must name the variable, or the user can't find the typo.
    assert "GDA_TEST_INT" in capsys.readouterr().err


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_non_positive_is_rejected_by_default(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero means "busy-loop" for a poll interval and "cache nothing" for a TTL."""
    monkeypatch.setenv("GDA_TEST_INT", raw)
    monkeypatch.setenv("GDA_TEST_FLOAT", raw)
    assert env_int("GDA_TEST_INT", 7) == 7
    assert env_float("GDA_TEST_FLOAT", 1.5) == 1.5


def test_non_positive_allowed_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GDA_TEST_INT", "0")
    assert env_int("GDA_TEST_INT", 7, positive=False) == 0


# ── config_path ───────────────────────────────────────────────────────────────


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.toml"
    monkeypatch.setenv("GDA_TEST_CONFIG", str(target))
    assert config_path("GDA_TEST_CONFIG", "subagents.toml") == target


def test_repo_checkout_resolves_to_the_repo_root() -> None:
    """In a source checkout the file sits next to pyproject.toml."""
    found = config_path("GDA_TEST_CONFIG", "subagents.toml")
    assert found.name == "subagents.toml"
    assert found.exists()


def test_missing_file_falls_back_to_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed as a wheel, `parents[2]` is the dir holding site-packages.

    No config lives there, so the console script has to look somewhere a user
    would plausibly keep one — the directory they ran it from.
    """
    found = config_path("GDA_TEST_CONFIG", "definitely-not-here.toml")
    assert found == Path.cwd() / "definitely-not-here.toml"
