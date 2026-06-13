# Script Explanation: `7) semantic_retriever.md`

## 1. Overview
The primary role of the `semantic_retriever.py` script is to run the **semantic search** component of our retrieval pipeline. It connects asynchronously to our Qdrant vector database. When a user asks a question, this script converts the query string into a dense vector embedding using our preloaded embedding model, performs a vector search against the database, extracts the relevant fields (text, source file, chunk ID, and mathematical similarity score) from Qdrant's response, and returns them as a structured list.

---

## 2. Code Walkthrough

### Imports and Database Connection
```python
from qdrant_client import AsyncQdrantClient

from .embedding_model import embedding_model
```
- **Lines 1–3**:
  - We import `AsyncQdrantClient` to perform non-blocking asynchronous calls to the Qdrant database server.
  - We import our preloaded SentenceTransformer model (`embedding_model`) from our local folder.

```python
# Connect to Qdrant
client = AsyncQdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "rag_docs"
```
- **Lines 8–13**: We establish an asynchronous connection client to the Qdrant service running at `localhost:6333` and store our collection name target as `"rag_docs"`.

---

### Retrieval Logic
```python
async def retrieve_chunks(query: str):

    # Query to Embedding
    query_vector = embedding_model.embed_query(query)
```
- **Lines 19–22**:
  - We define the asynchronous function `retrieve_chunks(query: str)`.
  - We call `embedding_model.embed_query(query)` to convert the user's plain-text question into a list of 384 numbers representing its semantic position in vector space.

```python
    # retrieving Similar Chunks
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    )
```
- **Lines 25–29**:
  - We invoke `client.query_points()` asynchronously using `await`.
  - We query our target collection `"rag_docs"`.
  - We pass our query embedding vector (`query_vector`) to search against all document vectors stored in the collection.
  - `limit=10`: We restrict Qdrant to return only the top 10 closest document points based on their vector positions.
