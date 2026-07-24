"""
End-to-end pytest tests for the MongoDB knowledge base.

Checks each layer independently so it's clear exactly where things break:
  - MongoDB connectivity and direct write/read
  - Vector search index presence
  - Embedding model
  - Full round-trip: save_knowledge → query_knowledge (vector)
  - Direct query tools: query_by_address, query_by_category, list_all_knowledge
  - Tag filtering across all query tools
  - update_knowledge: rename, confidence/tags update, no-op, unknown address
  - list_analyzed_binaries

These are integration tests: they need a reachable MongoDB (with vector search)
and a working embedding model. They are marked ``integration`` and skip cleanly
when either is unavailable, so CI runs ``pytest -m "not integration"``.

Run:  uv run pytest test_knowledge.py -v
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest
from dotenv import dotenv_values

pytestmark = pytest.mark.integration

# ── config ────────────────────────────────────────────────────────────────────

COLLECTION = "re_knowledge_test"

TEST_MARKER = "__test_knowledge_pytest__"
TEST_ADDRESS = "0xDEADBEEF"
UPDATE_ADDRESS = "0xBEEFBEEF"


@dataclass(frozen=True)
class Config:
    mongodb_uri: str
    mongodb_db: str
    embed_model: str
    binary_name: str


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _dotenv_applied() -> Generator[None, None, None]:
    """Apply ``.env`` for the duration of this module's run only.

    Calling ``load_dotenv()`` at import time leaked the developer's ``.env``
    into every *other* test module — pytest imports all of them at collection
    time — which broke the tests that assert deepagents' built-in compaction
    defaults. Reading the file into a dict and applying it through monkeypatch
    keeps the values available here and reverts them afterwards, so the suite
    no longer depends on collection order.
    """
    mp = pytest.MonkeyPatch()
    for key, value in dotenv_values().items():
        # Mirrors load_dotenv()'s default: a real environment wins over .env.
        if value is not None and key not in os.environ:
            mp.setenv(key, value)
    # The collection name the code under test reads; kept off the real one.
    mp.setenv("MONGODB_VECTOR_COLLECTION", COLLECTION)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def config(_dotenv_applied: None) -> Config:
    ollama_fallback = (
        f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
    )
    return Config(
        mongodb_uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db=os.environ.get("MONGODB_DB", "checkpointing_db"),
        embed_model=os.environ.get("EMBED_MODEL", ollama_fallback),
        binary_name=os.environ.get("BINARY_NAME", "test_binary"),
    )


@pytest.fixture(scope="session")
def mongo_client(config: Config) -> Generator[Any, None, None]:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    client: Any = MongoClient(config.mongodb_uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        client.close()
        pytest.skip(f"MongoDB unavailable at {config.mongodb_uri}: {exc}")
    yield client
    client[config.mongodb_db][COLLECTION].drop()
    client.close()


@pytest.fixture(scope="session")
def mongo_collection(config: Config, mongo_client: Any) -> Any:
    return mongo_client[config.mongodb_db][COLLECTION]


@pytest.fixture(scope="session")
def embeddings_model(config: Config) -> Any:
    from ghidra_deep_agent.models import build_embeddings

    try:
        emb = build_embeddings(config.embed_model)
        emb.embed_query("test")
        return emb
    except Exception as exc:
        pytest.skip(f"Embedding model unavailable: {exc}")


@pytest.fixture(scope="session")
def tools_map(config: Config, mongo_client: Any, embeddings_model: Any) -> Any:
    from ghidra_deep_agent.knowledge import build_knowledge_tools

    tm = {
        t.name: t
        for t in build_knowledge_tools(
            config.mongodb_uri,
            config.mongodb_db,
            embeddings_model,
            config.binary_name,
        )
    }

    # Seed the primary read-only test document.
    tm["save_knowledge"].invoke(
        {
            "content": f"Test function at {TEST_ADDRESS}: marker={TEST_MARKER}",
            "category": "function",
            "address": TEST_ADDRESS,
            "function_name": TEST_MARKER,
            "confidence": "high",
            "tags": ["test", "crypto"],
        }
    )
    # Seed a separate document owned by update_knowledge tests.
    tm["save_knowledge"].invoke(
        {
            "content": (
                f"Update test function at {UPDATE_ADDRESS}: marker={TEST_MARKER}"
            ),
            "category": "function",
            "address": UPDATE_ADDRESS,
            "function_name": TEST_MARKER,
            "confidence": "medium",
            "tags": ["test"],
        }
    )

    time.sleep(1)  # let the vector index catch up
    return tm


# ── 1. connectivity ───────────────────────────────────────────────────────────


class TestMongoConnectivity:
    def test_ping(self, mongo_client: Any) -> None:
        mongo_client.admin.command("ping")


# ── 2. direct write + read ────────────────────────────────────────────────────


class TestDirectWriteRead:
    def test_insert_and_read_back(self, config: Config, mongo_collection: Any) -> None:
        doc_id = mongo_collection.insert_one(
            {
                "text": "Direct insert by test_knowledge.py",
                "binary_name": config.binary_name,
                "category": "test",
                "address": "0x00000000",
                "function_name": TEST_MARKER,
                "confidence": "high",
                "tags": ["test"],
            }
        ).inserted_id

        found = mongo_collection.find_one({"_id": doc_id})
        assert found is not None
        assert found["function_name"] == TEST_MARKER
        assert found["tags"] == ["test"]


# ── 3. vector search index ────────────────────────────────────────────────────


class TestVectorSearchIndex:
    def test_search_indexes_accessible(self, mongo_collection: Any) -> None:
        try:
            indexes = list(mongo_collection.list_search_indexes())
        except Exception as exc:
            pytest.skip(
                f"list_search_indexes() not available on this deployment: {exc}"
            )
        # We just assert the call succeeded; the index may not exist yet
        assert isinstance(indexes, list)


# ── 4. embedding model ────────────────────────────────────────────────────────


class TestEmbeddingModel:
    def test_embed_query_returns_vector(self, embeddings_model: Any) -> None:
        vec = embeddings_model.embed_query("test embedding")
        assert len(vec) > 0


# ── 5. full round-trip ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_query_knowledge_scoped(self, tools_map: Any) -> None:
        result = tools_map["query_knowledge"].invoke({"query": TEST_MARKER})
        assert TEST_MARKER in result

    def test_query_knowledge_global(self, tools_map: Any) -> None:
        result = tools_map["query_knowledge_global"].invoke({"query": TEST_MARKER})
        assert TEST_MARKER in result


# ── 6. direct query tools ─────────────────────────────────────────────────────


class TestDirectQueryTools:
    def test_query_by_address(self, tools_map: Any) -> None:
        r = tools_map["query_by_address"].invoke({"address": TEST_ADDRESS})
        assert TEST_MARKER in r

    def test_query_by_category(self, tools_map: Any) -> None:
        r = tools_map["query_by_category"].invoke({"category": "function"})
        assert TEST_MARKER in r

    def test_list_all_knowledge(self, tools_map: Any) -> None:
        r = tools_map["list_all_knowledge"].invoke({})
        assert TEST_MARKER in r


# ── 7. tag filtering ──────────────────────────────────────────────────────────


class TestTagFiltering:
    def test_query_by_tags_match(self, tools_map: Any) -> None:
        r = tools_map["query_by_tags"].invoke({"tags": ["crypto"]})
        assert TEST_MARKER in r

    def test_query_by_tags_no_match(self, tools_map: Any) -> None:
        r = tools_map["query_by_tags"].invoke({"tags": ["__nonexistent__"]})
        assert TEST_MARKER not in r

    def test_query_by_address_tag_match(self, tools_map: Any) -> None:
        r = tools_map["query_by_address"].invoke(
            {"address": TEST_ADDRESS, "tags": ["test"]}
        )
        assert TEST_MARKER in r

    def test_query_by_address_tag_no_match(self, tools_map: Any) -> None:
        r = tools_map["query_by_address"].invoke(
            {"address": TEST_ADDRESS, "tags": ["__nonexistent__"]}
        )
        assert TEST_MARKER not in r

    def test_query_by_category_tag_match(self, tools_map: Any) -> None:
        r = tools_map["query_by_category"].invoke(
            {"category": "function", "tags": ["crypto"]}
        )
        assert TEST_MARKER in r

    def test_list_all_knowledge_tag_match(self, tools_map: Any) -> None:
        r = tools_map["list_all_knowledge"].invoke({"tags": ["test"]})
        assert TEST_MARKER in r

    def test_list_all_knowledge_tag_no_match(self, tools_map: Any) -> None:
        r = tools_map["list_all_knowledge"].invoke({"tags": ["__nonexistent__"]})
        assert TEST_MARKER not in r


# ── 8. update_knowledge ───────────────────────────────────────────────────────


class TestUpdateKnowledge:
    def test_rename(self, tools_map: Any) -> None:
        r = tools_map["update_knowledge"].invoke(
            {"address": UPDATE_ADDRESS, "function_name": "does_a_thing"}
        )
        assert "Updated" in r

    def test_rename_visible_in_query(self, tools_map: Any) -> None:
        r = tools_map["query_by_address"].invoke({"address": UPDATE_ADDRESS})
        assert "does_a_thing" in r

    def test_update_confidence_and_tags(self, tools_map: Any) -> None:
        r = tools_map["update_knowledge"].invoke(
            {
                "address": UPDATE_ADDRESS,
                "confidence": "high",
                "tags": ["crypto", "renamed"],
            }
        )
        assert "Updated" in r

    def test_updated_tags_visible(self, tools_map: Any) -> None:
        r = tools_map["query_by_tags"].invoke({"tags": ["renamed"]})
        assert "does_a_thing" in r

    def test_noop_no_fields(self, tools_map: Any) -> None:
        r = tools_map["update_knowledge"].invoke({"address": UPDATE_ADDRESS})
        assert "Nothing to update" in r

    def test_unknown_address(self, tools_map: Any) -> None:
        r = tools_map["update_knowledge"].invoke(
            {"address": "0xFFFFFFFF", "function_name": "ghost"}
        )
        assert "No entries found" in r


# ── 9. list_analyzed_binaries ─────────────────────────────────────────────────


class TestListAnalyzedBinaries:
    def test_binary_listed(self, config: Config, tools_map: Any) -> None:
        r = tools_map["list_analyzed_binaries"].invoke({})
        assert config.binary_name in r

    def test_current_binary_marked(self, tools_map: Any) -> None:
        r = tools_map["list_analyzed_binaries"].invoke({})
        assert "← current" in r


# ── 10. get_knowledge_summary tool ───────────────────────────────────────────


class TestGetKnowledgeSummary:
    def test_summary_contains_seeded_data(self, config: Config, tools_map: Any) -> None:
        # tools_map fixture has already seeded two findings via save_knowledge.
        summary = tools_map["get_knowledge_summary"].invoke({})
        assert config.binary_name in summary
        assert "Totals:" in summary
        assert TEST_MARKER in summary  # function name appears in analyzed list

    def test_summary_respects_function_cap(
        self,
        config: Config,
        tools_map: Any,
        mongo_collection: Any,
        embeddings_model: Any,
    ) -> None:
        from ghidra_deep_agent.knowledge import (
            SUMMARY_FUNCTION_CAP,
            build_knowledge_tools,
        )

        cap_binary = "__cap_test_binary__"
        docs = [
            {
                "text": f"cap test fn {i}",
                "binary_name": cap_binary,
                "category": "function",
                "address": f"0x{i:08x}",
                "function_name": f"cap_fn_{i:03d}",
                "confidence": "medium",
                "tags": [],
            }
            for i in range(SUMMARY_FUNCTION_CAP + 5)
        ]
        try:
            mongo_collection.insert_many(docs)
            # Build a separate tools map scoped to the cap_binary.
            cap_tools = {
                t.name: t
                for t in build_knowledge_tools(
                    config.mongodb_uri,
                    config.mongodb_db,
                    embeddings_model,
                    cap_binary,
                )
            }
            summary = cap_tools["get_knowledge_summary"].invoke({})
            assert f"({SUMMARY_FUNCTION_CAP} shown)" in summary
            assert "cap_fn_000" in summary
            # An entry beyond the cap should not appear.
            assert f"cap_fn_{SUMMARY_FUNCTION_CAP + 3:03d}" not in summary
        finally:
            mongo_collection.delete_many({"binary_name": cap_binary})
