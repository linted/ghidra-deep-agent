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

Run:  uv run pytest test_knowledge.py -v
"""

import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "checkpointing_db")
COLLECTION = "re_knowledge_test"
os.environ["MONGODB_VECTOR_COLLECTION"] = COLLECTION
_ollama_fallback = f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
EMBED_MODEL = os.environ.get("EMBED_MODEL", _ollama_fallback)
BINARY_NAME = os.environ.get("BINARY_NAME", "test_binary")

TEST_MARKER = "__test_knowledge_pytest__"
TEST_ADDRESS = "0xDEADBEEF"
UPDATE_ADDRESS = "0xBEEFBEEF"


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def mongo_client():
    from pymongo import MongoClient

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    yield client
    client[MONGODB_DB][COLLECTION].drop()
    client.close()


@pytest.fixture(scope="session")
def mongo_collection(mongo_client):
    return mongo_client[MONGODB_DB][COLLECTION]


@pytest.fixture(scope="session")
def embeddings_model():
    from models import build_embeddings

    try:
        emb = build_embeddings(EMBED_MODEL)
        emb.embed_query("test")
        return emb
    except Exception as exc:
        pytest.skip(f"Embedding model unavailable: {exc}")


@pytest.fixture(scope="session")
def tools_map(embeddings_model):
    from knowledge import build_knowledge_tools

    tm = {
        t.name: t
        for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings_model, BINARY_NAME)
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
            "content": f"Update test function at {UPDATE_ADDRESS}: marker={TEST_MARKER}",
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
    def test_ping(self, mongo_client):
        mongo_client.admin.command("ping")


# ── 2. direct write + read ────────────────────────────────────────────────────


class TestDirectWriteRead:
    def test_insert_and_read_back(self, mongo_collection):
        doc_id = mongo_collection.insert_one(
            {
                "text": "Direct insert by test_knowledge.py",
                "binary_name": BINARY_NAME,
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
    def test_search_indexes_accessible(self, mongo_collection):
        try:
            indexes = list(mongo_collection.list_search_indexes())
        except Exception as exc:
            pytest.skip(f"list_search_indexes() not available on this deployment: {exc}")
        # We just assert the call succeeded; the index may not exist yet
        assert isinstance(indexes, list)


# ── 4. embedding model ────────────────────────────────────────────────────────


class TestEmbeddingModel:
    def test_embed_query_returns_vector(self, embeddings_model):
        vec = embeddings_model.embed_query("test embedding")
        assert len(vec) > 0


# ── 5. full round-trip ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_query_knowledge_scoped(self, tools_map):
        result = tools_map["query_knowledge"].invoke({"query": TEST_MARKER})
        assert TEST_MARKER in result

    def test_query_knowledge_global(self, tools_map):
        result = tools_map["query_knowledge_global"].invoke({"query": TEST_MARKER})
        assert TEST_MARKER in result


# ── 6. direct query tools ─────────────────────────────────────────────────────


class TestDirectQueryTools:
    def test_query_by_address(self, tools_map):
        r = tools_map["query_by_address"].invoke({"address": TEST_ADDRESS})
        assert TEST_MARKER in r

    def test_query_by_category(self, tools_map):
        r = tools_map["query_by_category"].invoke({"category": "function"})
        assert TEST_MARKER in r

    def test_list_all_knowledge(self, tools_map):
        r = tools_map["list_all_knowledge"].invoke({})
        assert TEST_MARKER in r


# ── 7. tag filtering ──────────────────────────────────────────────────────────


class TestTagFiltering:
    def test_query_by_tags_match(self, tools_map):
        r = tools_map["query_by_tags"].invoke({"tags": ["crypto"]})
        assert TEST_MARKER in r

    def test_query_by_tags_no_match(self, tools_map):
        r = tools_map["query_by_tags"].invoke({"tags": ["__nonexistent__"]})
        assert TEST_MARKER not in r

    def test_query_by_address_tag_match(self, tools_map):
        r = tools_map["query_by_address"].invoke(
            {"address": TEST_ADDRESS, "tags": ["test"]}
        )
        assert TEST_MARKER in r

    def test_query_by_address_tag_no_match(self, tools_map):
        r = tools_map["query_by_address"].invoke(
            {"address": TEST_ADDRESS, "tags": ["__nonexistent__"]}
        )
        assert TEST_MARKER not in r

    def test_query_by_category_tag_match(self, tools_map):
        r = tools_map["query_by_category"].invoke(
            {"category": "function", "tags": ["crypto"]}
        )
        assert TEST_MARKER in r

    def test_list_all_knowledge_tag_match(self, tools_map):
        r = tools_map["list_all_knowledge"].invoke({"tags": ["test"]})
        assert TEST_MARKER in r

    def test_list_all_knowledge_tag_no_match(self, tools_map):
        r = tools_map["list_all_knowledge"].invoke({"tags": ["__nonexistent__"]})
        assert TEST_MARKER not in r


# ── 8. update_knowledge ───────────────────────────────────────────────────────


class TestUpdateKnowledge:
    def test_rename(self, tools_map):
        r = tools_map["update_knowledge"].invoke(
            {"address": UPDATE_ADDRESS, "function_name": "does_a_thing"}
        )
        assert "Updated" in r

    def test_rename_visible_in_query(self, tools_map):
        r = tools_map["query_by_address"].invoke({"address": UPDATE_ADDRESS})
        assert "does_a_thing" in r

    def test_update_confidence_and_tags(self, tools_map):
        r = tools_map["update_knowledge"].invoke(
            {"address": UPDATE_ADDRESS, "confidence": "high", "tags": ["crypto", "renamed"]}
        )
        assert "Updated" in r

    def test_updated_tags_visible(self, tools_map):
        r = tools_map["query_by_tags"].invoke({"tags": ["renamed"]})
        assert "does_a_thing" in r

    def test_noop_no_fields(self, tools_map):
        r = tools_map["update_knowledge"].invoke({"address": UPDATE_ADDRESS})
        assert "Nothing to update" in r

    def test_unknown_address(self, tools_map):
        r = tools_map["update_knowledge"].invoke(
            {"address": "0xFFFFFFFF", "function_name": "ghost"}
        )
        assert "No entries found" in r


# ── 9. list_analyzed_binaries ─────────────────────────────────────────────────


class TestListAnalyzedBinaries:
    def test_binary_listed(self, tools_map):
        r = tools_map["list_analyzed_binaries"].invoke({})
        assert BINARY_NAME in r

    def test_current_binary_marked(self, tools_map):
        r = tools_map["list_analyzed_binaries"].invoke({})
        assert "← current" in r
