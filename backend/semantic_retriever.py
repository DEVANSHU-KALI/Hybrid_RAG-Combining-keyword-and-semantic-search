from qdrant_client import AsyncQdrantClient

from .embedding_model import embedding_model

# -----------------------------
# Connect to Qdrant
# -----------------------------
client = AsyncQdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "rag_docs"


# -----------------------------
# Semantic Retrieval Function
# -----------------------------
async def retrieve_chunks(query: str):

    # Query to Embedding
    query_vector = embedding_model.embed_query(query)

    # retrieving Similar Chunks
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    )

    # Store Retrieved Results
    results_list = []

    # Extract Payload Data
    for point in results.points:
        results_list.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_id": point.payload["chunk_id"],
            "score": point.score
        })
        
    return results_list