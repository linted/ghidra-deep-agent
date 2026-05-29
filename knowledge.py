import os
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


def build_knowledge_tools(
    mongodb_uri: str, mongodb_db: str, embeddings: Embeddings
) -> list:
    client = MongoClient(mongodb_uri)
    collection = client[mongodb_db][
        os.environ.get("MONGODB_VECTOR_COLLECTION", "re_knowledge")
    ]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="re_knowledge_index",
        auto_create_index=True,
        auto_index_timeout=60,
    )

    @tool
    def save_knowledge(
        content: str,
        category: str = "finding",
        address: str = "",
        function_name: str = "",
        confidence: str = "medium",
        tags: str = "",
    ) -> str:
        """Save a reverse engineering finding to the long-term knowledge base.

        Call this after EVERY function analyzed, every rename or retype decision, every
        identified data structure, and every hypothesis — even uncertain ones. Write
        content as a clear, self-contained statement so it makes sense without context
        later.

        Args:
            content: The finding as a self-contained statement.
            category: One of 'function', 'structure', 'string', 'hypothesis',
                'rename', or 'finding'.
            address: Ghidra address this relates to, e.g. '0x08000100'.
            function_name: The function name (original or renamed),
                e.g. 'parse_config_file'.
            confidence: 'high', 'medium', or 'low'.
            tags: Comma-separated keywords for later filtering,
                e.g. 'crypto,loop,syscall'.
        """
        doc = Document(
            page_content=content,
            metadata={
                "category": category,
                "address": address,
                "function_name": function_name,
                "confidence": confidence,
                "tags": tags,
                "saved_at": datetime.now(UTC).isoformat(),
            },
        )
        vector_store.add_documents([doc])
        label = function_name or address or category
        return f"Saved: [{label}]"

    @tool
    def query_by_address(address: str) -> str:
        """Retrieve all stored findings for a given address or address prefix.

        Use this before working on a specific function or data location to surface
        everything already known about it.

        Args:
            address: Full or partial address string, e.g. '0x08000100' or '0x0800'.
        """
        docs = list(
            collection.find(
                {"address": {"$regex": f"^{address}", "$options": "i"}},
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
    def query_by_category(category: str) -> str:
        """Retrieve all stored findings of a given category.

        Useful for reviewing every known function, every hypothesis, or every rename
        decision made so far.

        Args:
            category: One of 'function', 'structure', 'string', 'hypothesis',
                'rename', 'finding'.
        """
        docs = list(
            collection.find(
                {"category": category},
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
    def list_all_knowledge() -> str:
        """List a summary of every entry in the knowledge base.

        Call this at the start of a session to orient yourself — see which
        functions have been analyzed, what hypotheses exist, and where gaps remain.
        """
        docs = list(
            collection.find(
                {},
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
            return "Knowledge base is empty."
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
        return f"{len(docs)} total findings:\n" + "\n".join(lines)

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    query_knowledge = create_retriever_tool(
        retriever,
        name="query_knowledge",
        description=(
            "Semantic search across the long-term knowledge base. "
            "Call this before analyzing any function or structure to "
            "recall prior conclusions. Query with natural language, "
            "function names, addresses, or behavioral descriptions."
        ),
    )

    return [
        save_knowledge,
        query_knowledge,
        query_by_address,
        query_by_category,
        list_all_knowledge,
    ]
