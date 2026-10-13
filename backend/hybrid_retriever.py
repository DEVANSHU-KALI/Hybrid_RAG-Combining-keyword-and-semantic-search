# Imports asyncio for running asynchronous coroutines concurrently in parallel
import asyncio

# Imports semantic_retriever module for 768D dense vector search in Qdrant
from .semantic_retriever import retrieve_chunks

# Imports bm25_retriever module for sparse lexical keyword search
from .bm25_retriever import bm25_search


# Score Normalization Helper
# Scales raw scores onto a standard 0.0 to 1.0 range using Min-Max Normalization:
# Normalized Score = (Score - Min) / (Max - Min)
def normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return []
    scores = [result["score"] for result in results]
    max_score = max(scores)
    min_score = min(scores)

    # Division-by-Zero Guard: If max == min (e.g. all scores identical), set normalized scores to 1.0 to prevent crash
    if max_score == min_score:
        for result in results:
            result["score"] = 1.0
        return results

    # Apply Min-Max scaling formula
    for result in results:
        result["score"] = (result["score"] - min_score) / (max_score - min_score)
    return results
'''
Individual search scores: Always between 0.0 and 1.0 (never crosses).
Final combined score: Can reach up to 2.0 if both vector search and keyword search agree on the same chunk!
'''


# CONCEPT: Asynchronous Concurrent Hybrid Retrieval with Dual Isolation (Tenant + User)
# In production systems, calling semantic search and BM25 search sequentially adds cumulative network latency.
# By wrapping them in asyncio.gather(), both database search calls execute concurrently while enforcing 
# tenant_id AND user_id security scoping on both engines.

async def hybrid_search(
    query: str,                              # User prompt string to retrieve against
    tenant_id: str = "tenant_default",       # Tenant ID for multi-tenant data isolation pre-filtering
    user_id: str = "usr_admin",              # User ID for strict user-level data isolation
    limit: int = 10                          # Top K candidate chunks to return
) -> list[dict]:

    # 1. Concurrent Parallel Search Execution with Dual Pre-Filtering (tenant_id + user_id)
    semantic_task = retrieve_chunks(query, tenant_id=tenant_id, user_id=user_id, limit=15)
    bm25_task = bm25_search(query, tenant_id=tenant_id, user_id=user_id, limit=15)

    # Executes both database tasks concurrently on the async event loop
    semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)

    # 2. Independent Score Normalization
    semantic_results = normalize_scores(semantic_results)
    bm25_results = normalize_scores(bm25_results)

    # 3. Hybrid Score Fusion & Deduplication Engine
    combined_results = {}

    # Loop 1: Populate dictionary with normalized semantic results
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        combined_results[chunk_id] = {
            "text": result["text"],
            "source": result["source"],
            "chunk_id": chunk_id,
            "tenant_id": result.get("tenant_id", tenant_id),
            "user_id": result.get("user_id", user_id),
            "page_number": result.get("page_number", 1),
            "semantic_score": result["score"],
            "bm25_score": 0.0,
            "final_score": result["score"]
        }

    # Loop 2: Merge normalized BM25 keyword results into dictionary
    for result in bm25_results:
        chunk_id = result["chunk_id"]
        
        if chunk_id in combined_results:
            combined_results[chunk_id]["bm25_score"] = result["score"]
            combined_results[chunk_id]["final_score"] += result["score"]
            
        else:
            combined_results[chunk_id] = {
                "text": result["text"],
                "source": result["source"],
                "chunk_id": chunk_id,
                "tenant_id": result.get("tenant_id", tenant_id),
                "user_id": result.get("user_id", user_id),
                "page_number": result.get("page_number", 1),
                "semantic_score": 0.0,
                "bm25_score": result["score"],
                "final_score": result["score"]
            }

    # Converts combined dictionary values into a list of fused chunk objects
    final_results = list(combined_results.values())
    
    # Sorts fused candidates in descending order by final_score
    final_results = sorted(
        final_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Slices and returns top K best fused candidates
    return final_results[:limit]



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from .semantic_retriever import retrieve_chunks
# from .bm25_retriever import bm25_search
# 
# async def hybrid_search(
#     query: str,
#     tenant_id: str = "tenant_default",
#     user_id: str = "usr_admin",
#     limit: int = 10
# ) -> list[dict]:
#     semantic_task = retrieve_chunks(query, tenant_id=tenant_id, user_id=user_id, limit=15)
#     bm25_task = bm25_search(query, tenant_id=tenant_id, limit=15) # Missing user_id parameter in BM25
#     ...
# =====================================================================
