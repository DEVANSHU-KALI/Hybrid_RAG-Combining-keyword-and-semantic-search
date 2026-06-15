# Script Explanation: `5) reset_qdrant.md`

## 1. Overview
The primary role of the `reset_qdrant.py` script is to wipe the vector database collection named `"rag_docs"`. During development, testing, or database schema changes, it is common to re-ingest documents from scratch. Running this script ensures that old document chunks, obsolete vector coordinates, and metadata are deleted, returning the database to a clean, empty state.

---

## 2. Code Walkthrough

### Imports and Database Connection
```python
from qdrant_client import QdrantClient

# Connect To Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)
```