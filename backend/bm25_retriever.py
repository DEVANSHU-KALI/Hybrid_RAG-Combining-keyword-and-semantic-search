# Imports rank_bm25 library for computing Okapi BM25 keyword matching scores
from rank_bm25 import BM25Okapi

# Imports AsyncQdrantClient for non-blocking asynchronous Qdrant database queries
from qdrant_client import AsyncQdrantClient

# Imports Qdrant filtering models for multi-tenant data pre-filtering
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Initialize asynchronous Qdrant client pointing to local Docker container on port 6333
client = AsyncQdrantClient(host="localhost", port=6333)

# Qdrant collection target name where vector & payload documents are stored
COLLECTION_NAME = "rag_docs"


# CONCEPT: Multi-Tenant & User-Scoped Lexical BM25 Search
# Security pre-filtering applies to keyword search as well as vector search!
# Passing security_filter=Filter(must=[tenant_id, user_id]) into client.scroll() 
# restricts BM25 index construction exclusively to document chunks belonging to the active tenant_id AND user_id.

async def bm25_search(
    query: str,                              # User prompt string to score against keywords
    tenant_id: str = "tenant_default",       # Tenant ID filter for multi-tenant data isolation
    user_id: str = "usr_admin",              # User ID filter for user-level data isolation
    limit: int = 10                          # Number of top candidate chunks to return
) -> list[dict]:

    # 1. Dual Security Filter Construction (Tenant + User Isolation)
    # FieldCondition checks Qdrant payload: only match points where tenant_id == tenant_id AND user_id == user_id.
    security_filter = Filter(
        must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
    )

    # 2. Retrieve Tenant & User Scoped Documents from Qdrant
    # client.scroll downloads points matching both tenant_id and user_id filters.
    points, _ = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=security_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    # Return empty list immediately if no matching documents exist for this tenant and user
    if not points:
        return []

    # Extract raw payload fields into a clean list of document dictionaries
    documents = []
    for point in points:
        documents.append({
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", "unknown"),
            "chunk_id": point.payload.get("chunk_id", point.id),
            "tenant_id": point.payload.get("tenant_id", tenant_id),
            "user_id": point.payload.get("user_id", user_id),
            "page_number": point.payload.get("page_number", 1)
        })

    # 3. Tokenize Documents & Build BM25 Index
    tokenized_docs = [doc["text"].lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # 4. Tokenize Query & Score Documents
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # 5. Assemble & Sort Results
    results = []
    for i in range(len(documents)):
        results.append({
            "text": documents[i]["text"],
            "source": documents[i]["source"],
            "chunk_id": documents[i]["chunk_id"],
            "tenant_id": documents[i]["tenant_id"],
            "user_id": documents[i]["user_id"],
            "page_number": documents[i]["page_number"],
            "score": float(scores[i])
        })

    # Sort results in descending order (highest BM25 score first)
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # Return top K candidates
    return results[:limit]



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from rank_bm25 import BM25Okapi
# from qdrant_client import AsyncQdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue
# 
# client = AsyncQdrantClient(host="localhost", port=6333)
# COLLECTION_NAME = "rag_docs"
# 
# async def bm25_search(
#     query: str,
#     tenant_id: str = "tenant_default",
#     limit: int = 10
# ) -> list[dict]:
#     tenant_filter = Filter(
#         must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
#     )
#     points, _ = await client.scroll(
#         collection_name=COLLECTION_NAME,
#         scroll_filter=tenant_filter,
#         limit=1000,
#         with_payload=True,
#         with_vectors=False
#     )
#     ...
# =====================================================================
