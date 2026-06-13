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