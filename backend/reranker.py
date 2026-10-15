from sentence_transformers import CrossEncoder

# Load Cross-Encoder Model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# CONCEPT: Two-Stage Re-ranking Precision Filter
# Bi-encoder vector search (Stage 1) is fast but evaluates query and document independently.
# Cross-encoder reranking (Stage 2) passes concatenated (query, text) pairs through transformer 
# multi-head attention layers, re-scoring candidates for deep semantic alignment before LLM generation.

async def rerank_results(query: str, retrieved_chunks: list, top_k: int = 3) -> list[dict]:
    if not retrieved_chunks:
        return []

    # 1. Create Query-Chunk Pairs
    pairs = [
        [query, chunk["text"]]
        for chunk in retrieved_chunks
    ]

    # 2. Batch Cross-Encoder Inference
    scores = reranker_model.predict(pairs)

    # 3. Attach Reranker Scores
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]["reranker_score"] = float(scores[i])

    # 4. Sort & Slice Top Precision Results
    reranked_results = sorted(
        retrieved_chunks,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]


# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from sentence_transformers import CrossEncoder
# 
# reranker_model = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2"
# )
# 
# async def rerank_results(query: str, retrieved_chunks: list):
#     pairs = []
#     for chunk in retrieved_chunks:
#         pairs.append([query, chunk["text"]])
#     scores = reranker_model.predict(pairs)
#     for i in range(len(retrieved_chunks)):
#         retrieved_chunks[i]["reranker_score"] = float(scores[i])
#     reranked_results = sorted(
#         retrieved_chunks,
#         key=lambda x: x["reranker_score"],
#         reverse=True
#     )
#     return reranked_results[:3]
# =====================================================================
