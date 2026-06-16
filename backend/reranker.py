from sentence_transformers import CrossEncoder

# Load Cross-Encoder Model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Reranking Function
async def rerank_results(query: str, retrieved_chunks: list):

    # Create Query-Chunk Pairs
    pairs = []

    for chunk in retrieved_chunks:
        pairs.append(
            [
                query,
                chunk["text"]
            ]
        )

    # Generate Reranker Scores
    scores = reranker_model.predict(pairs)

    # Attach Reranker Scores
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]["reranker_score"] = float(scores[i])

    # Sort By Reranker Score
    reranked_results = sorted(
        retrieved_chunks,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # Return Top 3 Chunks
    return reranked_results[:3]