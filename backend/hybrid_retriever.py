from .semantic_retriever import retrieve_chunks
from .bm25_retriever import bm25_search

# Normalize Scores Function
def normalize_scores(results):
    scores = [
        result["score"]
        for result in results
    ]

    max_score = max(scores)
    min_score = min(scores)

    # Avoid Division By Zero
    if max_score == min_score:
        for result in results:
            result["score"] = 1.0
        return results

    # Min-Max Normalization
    for result in results:
        result["score"] = (
            result["score"] - min_score
        ) / (
            max_score - min_score
        )
    return results

# Hybrid Retrieval Function
async def hybrid_search(query: str):

    # Semantic Retrieval
    semantic_results = await retrieve_chunks(query)

    # BM25 Retrieval
    bm25_results = await bm25_search(query)

    # Normalize Scores
    semantic_results = normalize_scores(semantic_results)
    bm25_results = normalize_scores(bm25_results)

    # Combine Results
    combined_results = {}

    # Add Semantic Results
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        combined_results[chunk_id] = {
            "text": result["text"],
            "source": result["source"],
            "chunk_id": chunk_id,
            "semantic_score": result["score"],
            "bm25_score": 0.0,
            "final_score": result["score"]
        }

    # Add BM25 Results
    for result in bm25_results:
        chunk_id = result["chunk_id"]

        # If Chunk Already Exists
        if chunk_id in combined_results:
            combined_results[chunk_id][
                "bm25_score"
            ] = result["score"]

            combined_results[chunk_id][
                "final_score"
            ] += result["score"]

        # New BM25 Chunk
        else:
            combined_results[chunk_id] = {
                "text": result["text"],
                "source": result["source"],
                "chunk_id": chunk_id,
                "semantic_score": 0.0,
                "bm25_score": result["score"],
                "final_score": result["score"]
            }

    # Convert Dict To List
    final_results = list(
        combined_results.values()
    )

    # Sort By Final Score
    final_results = sorted(
        final_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Return Top Candidates
    return final_results[:5]