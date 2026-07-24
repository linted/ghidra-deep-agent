"""Offline tests for the pure helpers in knowledge.py.

The tool bodies need a live MongoDB (see test_knowledge.py); these cover the
query-shaping logic that does not, so the regex-escaping and result-cap
behavior is checked on every run.

Run:  uv run pytest test_knowledge_helpers.py -v
"""

from __future__ import annotations

import re

from ghidra_deep_agent.knowledge import (
    QUERY_RESULT_CAP,
    _address_prefix_filter,
    _cap_note,
)

# ── address prefix filter ─────────────────────────────────────────────────────


def test_plain_address_matches_as_prefix() -> None:
    pattern = _address_prefix_filter("0x0800")["$regex"]
    assert re.match(pattern, "0x08001234")
    assert not re.match(pattern, "0x10800000")


def test_metacharacters_are_escaped() -> None:
    """An unescaped '(' is a regex compile error, not a literal."""
    pattern = _address_prefix_filter("sub_(bad")["$regex"]
    compiled = re.compile(pattern)  # would raise if '(' leaked through
    assert compiled.match("sub_(bad_thing")


def test_star_does_not_become_a_wildcard() -> None:
    pattern = _address_prefix_filter("0x10*")["$regex"]
    # Literal '*', so "0x1" must NOT match (it would with an unescaped quantifier).
    assert not re.match(pattern, "0x1")
    assert re.match(pattern, "0x10*_label")


def test_match_is_case_insensitive() -> None:
    assert _address_prefix_filter("0xDEAD")["$options"] == "i"


# ── result cap note ───────────────────────────────────────────────────────────


def test_no_note_when_under_the_cap() -> None:
    assert _cap_note([{}] * (QUERY_RESULT_CAP - 1)) == ""


def test_note_when_cap_is_hit() -> None:
    note = _cap_note([{}] * QUERY_RESULT_CAP)
    assert str(QUERY_RESULT_CAP) in note
    assert "narrow the query" in note


def test_no_note_for_empty_results() -> None:
    assert _cap_note([]) == ""
