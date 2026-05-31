"""
End-to-end test for the MongoDB knowledge base.

Checks each layer independently so it's clear exactly where things break:
  1. MongoDB connectivity
  2. Direct write + read (no search required)
  3. Vector search index existence / creation
  4. Embedding model (Ollama)
  5. Full round-trip: save_knowledge → query_knowledge (vector)
  6. Direct query tools: query_by_address, query_by_category, list_all_knowledge
  7. Tag filtering: query_by_tags, and tag param on query_by_address /
     query_by_category / list_all_knowledge
  8. update_knowledge: rename, confidence/tags update, no-op, unknown address
  9. list_analyzed_binaries

Run:  uv run python test_knowledge.py
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[2m·\033[0m"

_test_doc_id = None


def step(label: str) -> None:
    print(f"\n{label}")
    print("─" * len(label))


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


def info(msg: str) -> None:
    print(f"  {INFO}  \033[2m{msg}\033[0m")


# ── config ────────────────────────────────────────────────────────────────────

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "checkpointing_db")
COLLECTION = os.environ.get("MONGODB_VECTOR_COLLECTION", "re_knowledge")
_ollama_fallback = f"ollama:{os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}"
EMBED_MODEL = os.environ.get("EMBED_MODEL", _ollama_fallback)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BINARY_NAME = os.environ.get("BINARY_NAME", "test_binary")

TEST_MARKER = "__test_knowledge_script__"

print("\nKnowledge base test")
print("=" * 40)
info(f"URI:        {MONGODB_URI[:40]}...")
info(f"DB:         {MONGODB_DB}")
info(f"Collection: {COLLECTION}")
info(f"Embed:      {EMBED_MODEL}")
info(f"Binary:     {BINARY_NAME}")


# ── 1. connectivity ───────────────────────────────────────────────────────────

step("1. MongoDB connectivity")
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    ok("Connected and ping succeeded")
except ConnectionFailure as exc:
    fail(f"Cannot reach MongoDB: {exc}")
    sys.exit(1)
except OperationFailure as exc:
    fail(f"Auth / operation failure: {exc}")
    sys.exit(1)
except Exception as exc:
    fail(f"Unexpected error: {exc}")
    sys.exit(1)


# ── 2. direct write + read ────────────────────────────────────────────────────

step("2. Direct write + read (no search required)")
try:
    collection = client[MONGODB_DB][COLLECTION]

    result = collection.insert_one(
        {
            "text": "Test document inserted by test_knowledge.py",
            "binary_name": BINARY_NAME,
            "category": "test",
            "address": "0x00000000",
            "function_name": TEST_MARKER,
            "confidence": "high",
            "tags": ["test"],
        }
    )
    _test_doc_id = result.inserted_id
    ok(f"Inserted document  _id={_test_doc_id}")

    found = collection.find_one({"_id": _test_doc_id})
    assert found is not None, "Document not found after insert"
    assert found["function_name"] == TEST_MARKER
    assert found["tags"] == ["test"], f"Expected tags=['test'], got {found['tags']!r}"
    ok("Read back inserted document — content and tags match")
except Exception as exc:
    fail(f"Direct write/read failed: {exc}")
    sys.exit(1)


# ── 3. vector search index ────────────────────────────────────────────────────

step("3. Vector search index")
index_ok = False
try:
    indexes = list(collection.list_search_indexes())
    if indexes:
        names = [idx.get("name") for idx in indexes]
        ok(f"Found {len(indexes)} search index(es): {names}")
        index_ok = True
    else:
        info(
            "No search indexes found yet — auto_create_index=True will "
            "attempt creation on first use"
        )
except Exception as exc:
    fail(f"list_search_indexes() failed: {exc}")
    info(
        "This usually means the Atlas Search (mongot) service is not "
        "running on this deployment."
    )
    info("Vector similarity queries will not work until mongot is available.")


# ── 4. embedding model ────────────────────────────────────────────────────────

step("4. Embedding model")
embeddings = None
try:
    from models import build_embeddings

    _emb = build_embeddings(EMBED_MODEL)
    vec = _emb.embed_query("test embedding")
    ok(f"embed_query returned vector of length {len(vec)}  [{EMBED_MODEL}]")
    embeddings = _emb
except Exception as exc:
    fail(f"embed_query failed: {exc}")
    embeddings = None


# ── 5. full round-trip via knowledge tools ────────────────────────────────────

step("5. Full round-trip via knowledge tools")
if embeddings is None:
    info("Skipping — embedding model unavailable")
else:
    try:
        from knowledge import build_knowledge_tools

        tools_map = {
            t.name: t
            for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings, BINARY_NAME)
        }

        # save
        save = tools_map["save_knowledge"]
        result = save.invoke(
            {
                "content": f"Test function at 0xDEADBEEF: marker={TEST_MARKER}",
                "category": "function",
                "address": "0xDEADBEEF",
                "function_name": TEST_MARKER,
                "confidence": "high",
                "tags": ["test", "crypto"],
            }
        )
        ok(f"save_knowledge: {result}")

        time.sleep(1)  # give the index a moment to catch up

        # vector query (binary-scoped)
        query = tools_map["query_knowledge"]
        results = query.invoke({"query": TEST_MARKER})
        if TEST_MARKER in results:
            ok("query_knowledge (vector, binary-scoped): test document retrieved")
        else:
            fail("query_knowledge (vector): test document NOT found in results")
            info(f"Raw result: {results[:200]}")

        # vector query (global)
        global_query = tools_map["query_knowledge_global"]
        results = global_query.invoke({"query": TEST_MARKER})
        if TEST_MARKER in results:
            ok("query_knowledge_global (vector, all binaries): test document retrieved")
        else:
            fail("query_knowledge_global: test document NOT found in results")
            info(f"Raw result: {results[:200]}")

    except Exception as exc:
        fail(f"Round-trip failed: {exc}")


# ── 6. direct query tools ─────────────────────────────────────────────────────

step("6. Direct query tools")
if embeddings is None:
    info("Skipping — embedding model unavailable")
else:
    try:
        from knowledge import build_knowledge_tools

        tools_map = {
            t.name: t
            for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings, BINARY_NAME)
        }

        r1 = tools_map["query_by_address"].invoke({"address": "0xDEADBEEF"})
        if TEST_MARKER in r1:
            ok("query_by_address: found test document")
        else:
            fail(f"query_by_address: test document not found\n    {r1[:200]}")

        r2 = tools_map["query_by_category"].invoke({"category": "function"})
        if TEST_MARKER in r2:
            ok("query_by_category: found test document")
        else:
            fail(f"query_by_category: test document not found\n    {r2[:200]}")

        r3 = tools_map["list_all_knowledge"].invoke({})
        if TEST_MARKER in r3:
            ok("list_all_knowledge: found test document")
        else:
            fail(f"list_all_knowledge: test document not found\n    {r3[:200]}")

    except Exception as exc:
        fail(f"Direct query tools failed: {exc}")


# ── 7. tag filtering ──────────────────────────────────────────────────────────

step("7. Tag filtering")
if embeddings is None:
    info("Skipping — embedding model unavailable")
else:
    try:
        from knowledge import build_knowledge_tools

        tools_map = {
            t.name: t
            for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings, BINARY_NAME)
        }

        # query_by_tags — matching tag
        r = tools_map["query_by_tags"].invoke({"tags": ["crypto"]})
        if TEST_MARKER in r:
            ok("query_by_tags(['crypto']): found test document")
        else:
            fail(f"query_by_tags(['crypto']): test document not found\n    {r[:200]}")

        # query_by_tags — non-matching tag returns nothing
        r = tools_map["query_by_tags"].invoke({"tags": ["__nonexistent_tag__"]})
        if TEST_MARKER not in r:
            ok("query_by_tags(['__nonexistent_tag__']): correctly returned no results")
        else:
            fail("query_by_tags with non-matching tag unexpectedly returned test document")

        # query_by_address with matching tag filter
        r = tools_map["query_by_address"].invoke(
            {"address": "0xDEADBEEF", "tags": ["test"]}
        )
        if TEST_MARKER in r:
            ok("query_by_address + tags=['test']: found test document")
        else:
            fail(f"query_by_address + tags filter: test document not found\n    {r[:200]}")

        # query_by_address with non-matching tag filter
        r = tools_map["query_by_address"].invoke(
            {"address": "0xDEADBEEF", "tags": ["__nonexistent_tag__"]}
        )
        if TEST_MARKER not in r:
            ok("query_by_address + non-matching tag: correctly returned no results")
        else:
            fail("query_by_address with non-matching tag unexpectedly returned test document")

        # query_by_category with matching tag filter
        r = tools_map["query_by_category"].invoke(
            {"category": "function", "tags": ["crypto"]}
        )
        if TEST_MARKER in r:
            ok("query_by_category + tags=['crypto']: found test document")
        else:
            fail(f"query_by_category + tags filter: test document not found\n    {r[:200]}")

        # list_all_knowledge with matching tag filter
        r = tools_map["list_all_knowledge"].invoke({"tags": ["test"]})
        if TEST_MARKER in r:
            ok("list_all_knowledge + tags=['test']: found test document")
        else:
            fail(f"list_all_knowledge + tags filter: test document not found\n    {r[:200]}")

        # list_all_knowledge with non-matching tag filter
        r = tools_map["list_all_knowledge"].invoke({"tags": ["__nonexistent_tag__"]})
        if TEST_MARKER not in r:
            ok("list_all_knowledge + non-matching tag: correctly returned no results")
        else:
            fail("list_all_knowledge with non-matching tag unexpectedly returned test document")

    except Exception as exc:
        fail(f"Tag filtering tests failed: {exc}")


# ── 8. update_knowledge ───────────────────────────────────────────────────────

step("8. update_knowledge")
if embeddings is None:
    info("Skipping — embedding model unavailable")
else:
    try:
        from knowledge import build_knowledge_tools

        tools_map = {
            t.name: t
            for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings, BINARY_NAME)
        }

        # rename the function
        r = tools_map["update_knowledge"].invoke(
            {"address": "0xDEADBEEF", "function_name": "does_a_thing"}
        )
        if "Updated" in r and "0xDEADBEEF" in r:
            ok(f"update_knowledge (rename): {r}")
        else:
            fail(f"update_knowledge (rename) unexpected response: {r}")

        # verify the new name is visible via query_by_address
        r = tools_map["query_by_address"].invoke({"address": "0xDEADBEEF"})
        if "does_a_thing" in r:
            ok("query_by_address: updated function_name visible after rename")
        else:
            fail(f"query_by_address: updated name not found\n    {r[:200]}")

        # update confidence and tags together
        r = tools_map["update_knowledge"].invoke(
            {"address": "0xDEADBEEF", "confidence": "high", "tags": ["crypto", "renamed"]}
        )
        if "Updated" in r:
            ok(f"update_knowledge (confidence + tags): {r}")
        else:
            fail(f"update_knowledge (confidence + tags) unexpected response: {r}")

        # verify tags update is reflected in query_by_tags
        r = tools_map["query_by_tags"].invoke({"tags": ["renamed"]})
        if "does_a_thing" in r:
            ok("query_by_tags(['renamed']): updated tags visible after update")
        else:
            fail(f"query_by_tags: updated tags not found\n    {r[:200]}")

        # no-op: no fields provided
        r = tools_map["update_knowledge"].invoke({"address": "0xDEADBEEF"})
        if "Nothing to update" in r:
            ok("update_knowledge (no fields): correctly returned no-op message")
        else:
            fail(f"update_knowledge (no fields) unexpected response: {r}")

        # unknown address
        r = tools_map["update_knowledge"].invoke(
            {"address": "0xFFFFFFFF", "function_name": "ghost"}
        )
        if "No entries found" in r:
            ok("update_knowledge (unknown address): correctly returned not-found message")
        else:
            fail(f"update_knowledge (unknown address) unexpected response: {r}")

    except Exception as exc:
        fail(f"update_knowledge tests failed: {exc}")


# ── 9. list_analyzed_binaries ─────────────────────────────────────────────────

step("9. list_analyzed_binaries")
if embeddings is None:
    info("Skipping — embedding model unavailable")
else:
    try:
        from knowledge import build_knowledge_tools

        tools_map = {
            t.name: t
            for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, embeddings, BINARY_NAME)
        }

        r = tools_map["list_analyzed_binaries"].invoke({})
        if BINARY_NAME in r:
            ok(f"list_analyzed_binaries: '{BINARY_NAME}' listed")
        else:
            fail(f"list_analyzed_binaries: '{BINARY_NAME}' not found\n    {r[:200]}")

        if "← current" in r:
            ok("list_analyzed_binaries: current binary marked correctly")
        else:
            fail("list_analyzed_binaries: current binary not marked with '← current'")

    except Exception as exc:
        fail(f"list_analyzed_binaries failed: {exc}")


# ── cleanup ───────────────────────────────────────────────────────────────────

step("Cleanup")
try:
    deleted = collection.delete_many(
        {"$or": [{"function_name": TEST_MARKER}, {"binary_name": BINARY_NAME}]}
    )
    ok(f"Removed {deleted.deleted_count} test document(s)")
except Exception as exc:
    fail(f"Cleanup failed (manual removal may be needed): {exc}")

print()
