"""
End-to-end test for the MongoDB knowledge base.

Checks each layer independently so it's clear exactly where things break:
  1. MongoDB connectivity
  2. Direct write + read (no search required)
  3. Vector search index existence / creation
  4. Embedding model (Ollama)
  5. Full round-trip: save_knowledge → query_knowledge (vector)
  6. Direct query tools: query_by_address, query_by_category, list_all_knowledge

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
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

TEST_MARKER = "__test_knowledge_script__"

print("\nKnowledge base test")
print("=" * 40)
info(f"URI:        {MONGODB_URI[:40]}...")
info(f"DB:         {MONGODB_DB}")
info(f"Collection: {COLLECTION}")
info(f"Embed:      {EMBED_MODEL}  (via {OLLAMA_HOST})")


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

    result = collection.insert_one({
        "text": "Test document inserted by test_knowledge.py",
        "category": "test",
        "address": "0x00000000",
        "function_name": TEST_MARKER,
        "confidence": "high",
        "tags": "test",
    })
    _test_doc_id = result.inserted_id
    ok(f"Inserted document  _id={_test_doc_id}")

    found = collection.find_one({"_id": _test_doc_id})
    assert found is not None, "Document not found after insert"
    assert found["function_name"] == TEST_MARKER
    ok("Read back inserted document — content matches")
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
        info("No search indexes found yet — auto_create_index=True will attempt creation on first use")
except Exception as exc:
    fail(f"list_search_indexes() failed: {exc}")
    info("This usually means the Atlas Search (mongot) service is not running on this deployment.")
    info("Vector similarity queries will not work until mongot is available.")


# ── 4. embedding model ────────────────────────────────────────────────────────

step("4. Ollama embedding model")
embeddings = None
try:
    import httpx
    r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    if any(EMBED_MODEL in m for m in models):
        ok(f"'{EMBED_MODEL}' is available in Ollama")
    else:
        fail(f"'{EMBED_MODEL}' not found in Ollama. Available: {models or '(none)'}")
        info(f"Run:  ollama pull {EMBED_MODEL}")
except Exception as exc:
    fail(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}")
    info("Vector search requires Ollama to embed queries. Other features still work.")

try:
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vec = embeddings.embed_query("test embedding")
    ok(f"embed_query returned vector of length {len(vec)}")
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

        tools_map = {t.name: t for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, EMBED_MODEL)}

        # save
        save = tools_map["save_knowledge"]
        result = save.invoke({
            "content": f"Test function at 0xDEADBEEF: marker={TEST_MARKER}",
            "category": "function",
            "address": "0xDEADBEEF",
            "function_name": TEST_MARKER,
            "confidence": "high",
            "tags": "test",
        })
        ok(f"save_knowledge: {result}")

        time.sleep(1)  # give the index a moment to catch up

        # vector query
        query = tools_map["query_knowledge"]
        results = query.invoke({"query": TEST_MARKER})
        if TEST_MARKER in results:
            ok("query_knowledge (vector): test document retrieved")
        else:
            fail("query_knowledge (vector): test document NOT found in results")
            info(f"Raw result: {results[:200]}")

    except Exception as exc:
        fail(f"Round-trip failed: {exc}")


# ── 6. direct query tools ─────────────────────────────────────────────────────

step("6. Direct query tools")
try:
    from knowledge import build_knowledge_tools

    tools_map = {t.name: t for t in build_knowledge_tools(MONGODB_URI, MONGODB_DB, EMBED_MODEL)}

    r1 = tools_map["query_by_address"].invoke({"address": "0xDEADBEEF"})
    if TEST_MARKER in r1:
        ok(f"query_by_address: found test document")
    else:
        fail(f"query_by_address: test document not found\n    {r1[:200]}")

    r2 = tools_map["query_by_category"].invoke({"category": "function"})
    if TEST_MARKER in r2:
        ok(f"query_by_category: found test document")
    else:
        fail(f"query_by_category: test document not found\n    {r2[:200]}")

    r3 = tools_map["list_all_knowledge"].invoke({})
    if TEST_MARKER in r3:
        ok(f"list_all_knowledge: found test document")
    else:
        fail(f"list_all_knowledge: test document not found\n    {r3[:200]}")

except Exception as exc:
    fail(f"Direct query tools failed: {exc}")


# ── cleanup ───────────────────────────────────────────────────────────────────

step("Cleanup")
try:
    deleted = collection.delete_many({"function_name": TEST_MARKER})
    ok(f"Removed {deleted.deleted_count} test document(s)")
except Exception as exc:
    fail(f"Cleanup failed (manual removal may be needed): {exc}")

print()
