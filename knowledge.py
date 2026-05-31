import os
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


def build_knowledge_tools(
    mongodb_uri: str, mongodb_db: str, embeddings: Embeddings, binary_name: str
) -> list:
    client = MongoClient(mongodb_uri)
    collection_name = os.environ.get("MONGODB_VECTOR_COLLECTION", "re_knowledge")
    collection = client[mongodb_db][collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="re_knowledge_index",
        auto_create_index=False,
    )

    # Ensure the index exists and has binary_name as a filterable field.
    # auto_create_index=True only creates a bare vector index; pre_filter on
    # binary_name requires it to be explicitly declared as a filter field.
    try:
        existing = list(collection.list_search_indexes("re_knowledge_index"))
        fields = existing[0].get("latestDefinition", {}).get("fields", []) if existing else []
        has_filter = any(
            f.get("type") == "filter" and f.get("path") == "binary_name"
            for f in fields
        )
        if not has_filter:
            dims = len(embeddings.embed_query(""))
            vector_store.create_vector_search_index(
                dimensions=dims,
                filters=["binary_name"],
                update=bool(existing),
                wait_until_complete=60,
            )
    except Exception as exc:
        import sys
        print(f"Warning: vector search index setup failed ({exc}); direct query tools still work", file=sys.stderr)

    @tool
    def save_knowledge(
        content: str,
        category: str = "finding",
        address: str = "",
        function_name: str = "",
        confidence: str = "medium",
        tags: list[str] | None = None,
    ) -> str:
        """Save a NEW reverse engineering finding to the long-term knowledge base.

        Call this the first time you record a function, data structure, string, or
        hypothesis — when no entry for it exists yet. Write content as a clear,
        self-contained statement so it makes sense without context later. To update
        an entry that was already saved (rename, confidence change, tag update),
        use update_knowledge instead.

        Args:
            content: The finding as a self-contained statement.
            category: One of 'function', 'structure', 'string', 'hypothesis',
                'rename', or 'finding'.
            address: Ghidra address this relates to, e.g. '0x08000100'.
            function_name: The function name (original or renamed),
                e.g. 'parse_config_file'.
            confidence: 'high', 'medium', or 'low'.
            tags: Keywords for later filtering, e.g. ['crypto', 'loop', 'syscall'].
        """
        doc = Document(
            page_content=content,
            metadata={
                "binary_name": binary_name,
                "category": category,
                "address": address,
                "function_name": function_name,
                "confidence": confidence,
                "tags": tags or [],
                "saved_at": datetime.now(UTC).isoformat(),
            },
        )
        vector_store.add_documents([doc])
        label = function_name or address or category
        return f"Saved: [{binary_name} | {label}]"

    @tool
    def update_knowledge(
        address: str,
        function_name: str | None = None,
        confidence: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Update metadata on EXISTING knowledge entries for a given address.

        Use this when knowledge already saved for an address needs to change —
        a function is renamed, confidence level shifts, or tags need updating.
        Only the fields you provide are changed; omit a field to leave it unchanged.
        To record a brand-new finding with no prior entry, use save_knowledge instead.

        Args:
            address: Exact address of the entries to update, e.g. '0x08000100'.
            function_name: New function name, e.g. 'does_a_thing'.
            confidence: New confidence level: 'high', 'medium', or 'low'.
            tags: Replacement tags list, e.g. ['crypto', 'hash'].
        """
        updates: dict = {}
        if function_name is not None:
            updates["function_name"] = function_name
        if confidence is not None:
            updates["confidence"] = confidence
        if tags is not None:
            updates["tags"] = tags

        if not updates:
            return "Nothing to update — no fields provided."

        result = collection.update_many(
            {"binary_name": binary_name, "address": address},
            {"$set": updates},
        )
        if result.matched_count == 0:
            return f"No entries found for address '{address}'."
        return (
            f"Updated {result.modified_count} of {result.matched_count} "
            f"entry/entries for '{address}'."
        )

    @tool
    def query_by_address(address: str, tags: list[str] | None = None) -> str:
        """Retrieve all stored findings for a given address or address prefix.

        Use this before working on a specific function or data location to surface
        everything already known about it.

        Args:
            address: Full or partial address string, e.g. '0x08000100' or '0x0800'.
            tags: Optional list of tags to filter by; returns findings that have
                at least one matching tag, e.g. ['crypto', 'loop'].
        """
        query: dict = {
            "binary_name": binary_name,
            "address": {"$regex": f"^{address}", "$options": "i"},
        }
        if tags:
            query["tags"] = {"$in": tags}
        docs = list(
            collection.find(
                query,
                {
                    "text": 1,
                    "category": 1,
                    "address": 1,
                    "function_name": 1,
                    "confidence": 1,
                    "tags": 1,
                    "_id": 0,
                },
            )
        )
        if not docs:
            return f"No findings for address '{address}'."
        lines = [
            f"[{d.get('category', '')} | {d.get('confidence', '')}] "
            f"{d.get('address', '')} {d.get('function_name', '')} — {d.get('text', '')}"
            for d in docs
        ]
        return f"{len(docs)} finding(s):\n" + "\n".join(lines)

    @tool
    def query_by_category(category: str, tags: list[str] | None = None) -> str:
        """Retrieve all stored findings of a given category.

        Useful for reviewing every known function, every hypothesis, or every rename
        decision made so far.

        Args:
            category: One of 'function', 'structure', 'string', 'hypothesis',
                'rename', 'finding'.
            tags: Optional list of tags to filter by; returns findings that have
                at least one matching tag, e.g. ['crypto', 'loop'].
        """
        query: dict = {"binary_name": binary_name, "category": category}
        if tags:
            query["tags"] = {"$in": tags}
        docs = list(
            collection.find(
                query,
                {
                    "text": 1,
                    "address": 1,
                    "function_name": 1,
                    "confidence": 1,
                    "_id": 0,
                },
            )
        )
        if not docs:
            return f"No findings for category '{category}'."
        lines = [
            f"[{d.get('address', '')} {d.get('function_name', '')} | "
            f"{d.get('confidence', '')}] {d.get('text', '')}"
            for d in docs
        ]
        return f"{len(docs)} finding(s):\n" + "\n".join(lines)

    @tool
    def list_all_knowledge(tags: list[str] | None = None) -> str:
        """List a summary of every entry in the knowledge base for the current binary.

        Call this at the start of a session to orient yourself — see which
        functions have been analyzed, what hypotheses exist, and where gaps remain.

        Args:
            tags: Optional list of tags to filter by; returns findings that have
                at least one matching tag, e.g. ['crypto', 'loop'].
        """
        query: dict = {"binary_name": binary_name}
        if tags:
            query["tags"] = {"$in": tags}
        docs = list(
            collection.find(
                query,
                {
                    "text": 1,
                    "category": 1,
                    "address": 1,
                    "function_name": 1,
                    "confidence": 1,
                    "tags": 1,
                    "saved_at": 1,
                    "_id": 0,
                },
            ).sort("category", 1)
        )
        if not docs:
            return f"Knowledge base is empty for '{binary_name}'."
        lines = []
        for d in docs:
            parts = filter(
                None,
                [
                    d.get("category", ""),
                    d.get("address", ""),
                    d.get("function_name", ""),
                    d.get("confidence", ""),
                ],
            )
            label = " | ".join(parts)
            snippet = d.get("text", "")[:100]
            lines.append(f"[{label}]  {snippet}")
        return f"{len(docs)} total findings for '{binary_name}':\n" + "\n".join(lines)

    @tool
    def query_by_tags(tags: list[str]) -> str:
        """Retrieve all stored findings that match any of the given tags.

        Use this to surface everything related to a theme across the whole binary,
        e.g. all crypto-related findings or all syscall sites.

        Args:
            tags: One or more tags to search for, e.g. ['crypto', 'hash'].
                Returns findings that match at least one tag.
        """
        docs = list(
            collection.find(
                {"binary_name": binary_name, "tags": {"$in": tags}},
                {
                    "text": 1,
                    "category": 1,
                    "address": 1,
                    "function_name": 1,
                    "confidence": 1,
                    "tags": 1,
                    "_id": 0,
                },
            ).sort("category", 1)
        )
        if not docs:
            return f"No findings tagged with {tags}."
        lines = [
            f"[{d.get('category', '')} | {d.get('confidence', '')}] "
            f"{d.get('address', '')} {d.get('function_name', '')} "
            f"[{', '.join(d.get('tags', []))}] — {d.get('text', '')}"
            for d in docs
        ]
        return f"{len(docs)} finding(s) for tags {tags}:\n" + "\n".join(lines)

    @tool
    def list_analyzed_binaries() -> str:
        """List all binaries that have findings stored in the knowledge base.

        Use this to see what other binaries have been analyzed and whether
        cross-binary queries might surface relevant findings.
        """
        names = collection.distinct("binary_name")
        if not names:
            return "Knowledge base is empty."
        lines = []
        for name in sorted(names):
            count = collection.count_documents({"binary_name": name})
            marker = " ← current" if name == binary_name else ""
            lines.append(f"  {name}: {count} finding(s){marker}")
        return "Analyzed binaries:\n" + "\n".join(lines)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "pre_filter": {"binary_name": {"$eq": binary_name}},
        }
    )
    query_knowledge = create_retriever_tool(
        retriever,
        name="query_knowledge",
        description=(
            f"Semantic search across the knowledge base for '{binary_name}'. "
            "Call this before analyzing any function or structure to "
            "recall prior conclusions. Query with natural language, "
            "function names, addresses, or behavioral descriptions."
        ),
    )

    global_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    query_knowledge_global = create_retriever_tool(
        global_retriever,
        name="query_knowledge_global",
        description=(
            "Semantic search across the knowledge base for ALL analyzed binaries. "
            "Use this to find patterns, shared code, or related findings from other "
            "binaries when cross-binary comparison may be relevant. "
            "Results are labeled with their source binary."
        ),
    )

    return [
        save_knowledge,
        update_knowledge,
        query_knowledge,
        query_knowledge_global,
        query_by_address,
        query_by_category,
        query_by_tags,
        list_all_knowledge,
        list_analyzed_binaries,
    ]
