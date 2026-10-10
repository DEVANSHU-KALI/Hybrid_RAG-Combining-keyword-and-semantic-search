# Imports AsyncQdrantClient for non-blocking asynchronous vector database queries
from qdrant_client import AsyncQdrantClient

# Imports Qdrant filtering models for multi-tenant data pre-filtering
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Imports pre-configured 768D HuggingFace embedding model (BAAI/bge-base-en-v1.5)
from .embedding_model import embedding_model

# Initialize asynchronous Qdrant client pointing to local Docker container on port 6333
client = AsyncQdrantClient(
    host="localhost",
    port=6333
)

# Qdrant collection target name where vector & payload documents are stored
COLLECTION_NAME = "rag_docs"


# CONCEPT: Multi-Tenant & User-Level Data Isolation Pre-Filtering Scoping
# In production enterprise RAG, security isolation MUST happen inside the database query.
# Passing both tenant_id and user_id in query_filter=Filter(must=[...]) ensures Qdrant's 
# HNSW engine mathematically ignores chunks belonging to other tenants OR other users 
# BEFORE vector similarity is computed, preventing cross-tenant and cross-user data leakage.

async def retrieve_chunks(
    query: str,                              # User prompt string to vector search against
    tenant_id: str = "tenant_default",       # Tenant ID filter for multi-tenant isolation
    user_id: str = "usr_admin",              # User ID filter for strict user-level data isolation
    limit: int = 10                          # Number of top vector candidates to return
) -> list[dict]:

    # 1. 768D Query Vectorization: converts text string to 768 float coordinates using BAAI/bge-base-en-v1.5
    query_vector = embedding_model.embed_query(query)

    # 2. Build Dual Pre-Filter Scoping (Tenant + User Level Isolation)
    # Restricts search to points matching BOTH payload["tenant_id"] == tenant_id AND payload["user_id"] == user_id
    security_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            ),
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )

    # 3. Query Qdrant with Dual Pre-Filtering
    # Runs HNSW graph search ONLY on vectors matching active tenant_id AND user_id
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=security_filter,
        limit=limit
    )

    # 4. Extract Payload Data & Standardize Output Objects
    results_list = []
    for point in results.points:
        results_list.append({
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", "unknown"),
            "chunk_id": point.payload.get("chunk_id", point.id),
            "tenant_id": point.payload.get("tenant_id", tenant_id),
            "user_id": point.payload.get("user_id", user_id),
            "page_number": point.payload.get("page_number", 1),
            "score": point.score             # Cosine similarity score float (0.0 to 1.0)
        })
        
    return results_list



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from qdrant_client import AsyncQdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue
# from .embedding_model import embedding_model
# 
# client = AsyncQdrantClient(host="localhost", port=6333)
# COLLECTION_NAME = "rag_docs"
# 
# async def retrieve_chunks(
#     query: str,
#     tenant_id: str = "tenant_default",
#     user_id: str = "usr_admin",
#     limit: int = 10
# ) -> list[dict]:
#     query_vector = embedding_model.embed_query(query)
#     tenant_filter = Filter(
#         must=[
#             FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
#         ]
#     )
#     results = await client.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_vector,
#         query_filter=tenant_filter,
#         limit=limit
#     )
#     results_list = []
#     for point in results.points:
#         results_list.append({
#             "text": point.payload.get("text", ""),
#             "source": point.payload.get("source", "unknown"),
#             "chunk_id": point.payload.get("chunk_id", point.id),
#             "tenant_id": point.payload.get("tenant_id", tenant_id),
#             "page_number": point.payload.get("page_number", 1),
#             "score": point.score
#         })
#     return results_list
# =====================================================================