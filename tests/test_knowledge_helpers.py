"""Offline tests for the pure helpers in knowledge.py.

The tool bodies need a live MongoDB (see test_knowledge.py); these cover the
query-shaping logic that does not, so the regex-escaping and result-cap
behavior is checked on every run.

Run:  uv run pytest tests/test_knowledge_helpers.py -v
"""

from __future__ import annotations

import asyncio
import re

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from pymongo.errors import OperationFailure

from ghidra_deep_agent.knowledge import (
    QUERY_RESULT_CAP,
    Tags,
    _address_prefix_filter,
    _cap_note,
    _safe_retriever_tool,
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


# ── safe retriever tool ───────────────────────────────────────────────────────


class _FakeRetriever(BaseRetriever):
    fail: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.fail:
            # What a $vectorSearch against a not-yet-built index raises.
            raise OperationFailure(
                "cannot query vector index while in state NOT_STARTED", 8
            )
        return [Document(page_content="a"), Document(page_content="b")]


def test_retriever_tool_formats_documents() -> None:
    qk = _safe_retriever_tool(_FakeRetriever(), "query_knowledge", "desc")
    assert qk.invoke({"query": "x"}) == "a\n\nb"
    assert asyncio.run(qk.ainvoke({"query": "x"})) == "a\n\nb"


def test_retriever_tool_reports_mongo_errors() -> None:
    """A Mongo error must come back as text: raised, it kills the whole run."""
    qk = _safe_retriever_tool(_FakeRetriever(fail=True), "query_knowledge", "desc")
    for result in (
        qk.invoke({"query": "x"}),
        asyncio.run(qk.ainvoke({"query": "x"})),
    ):
        assert result.startswith("Warning: semantic search is unavailable")
        assert "NOT_STARTED" in result


# ── tags coercion ─────────────────────────────────────────────────────────────


@tool
def _tagged(tags: Tags | None = None) -> str:
    """Echo the validated tags."""
    return repr(tags)


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("['state-block', 'cff']", ["state-block", "cff"]),
        ('["vgtrace"]', ["vgtrace"]),
        ("crypto, loop", ["crypto", "loop"]),
        (["already", "a-list"], ["already", "a-list"]),
        (None, None),
    ],
)
def test_tags_sent_as_a_string_become_a_list(sent: object, expected: object) -> None:
    """GLM sometimes serializes the list; strict validation rejected the call."""
    assert _tagged.invoke({"tags": sent}) == repr(expected)
