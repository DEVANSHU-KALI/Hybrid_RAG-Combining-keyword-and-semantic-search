from rank_bm25 import BM25Okapi

from qdrant-client import AsyncQdrantClient


# Connect to Qdrant
client = AsyncQdrantClient(
    host="localhost",
    port=6333
)

# Collection Name
COLLECTION_NAME = "rag_docs"


# BM25 Retrieval Function
async def bm25_search(query: str):

    # Get All Points From Qdrant
    points, _ = await client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    # Store Documents
    documents = []

    # Extract Chunk Text
    for point in points:
        documents.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_id": point.payload["chunk_id"]
        })

    # Tokenize Documents
    tokenized_docs = [
        doc["text"].lower().split()
        for doc in documents
    ]

    # Create BM25 Index
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize Query
    tokenized_query = query.lower().split()

    # Calculate BM25 Scores
    scores = bm25.get_scores(tokenized_query)

    # Combine Docs + Scores
    results = []
    for i in range(len(documents)):
        results.append({
            "text": documents[i]["text"],
            "source": documents[i]["source"],
            "chunk_id": documents[i]["chunk_id"],
            "score": scores[i]
        })

    # Sort By BM25 Score
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # Return Top 5 Results
    return results[:10]
