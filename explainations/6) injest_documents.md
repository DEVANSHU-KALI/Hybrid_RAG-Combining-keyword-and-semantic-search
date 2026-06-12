# Script Explanation: `6) injest_documents.md`

## 1. Overview
The primary role of the `injest_documents.py` script (spelled with a typo in the filename) is to perform document ingestion. It reads raw text files from the local directory `data/rag_concepts/`, processes their text through our semantic chunker, batches the resulting text segments to generate dense vector embeddings, maps them to database record structures (called **Points** in Qdrant), and uploads them to our Qdrant vector database collection.

---

## 2. Code Walkthrough

### Imports and Configuration
```python
import os

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from .embedding_model import embedding_model
from .text_chunker import text_splitter
```
- **Lines 1–7**:
  - We import `os` to browse local file directories.
  - We import `QdrantClient` for database uploads and `PointStruct` to construct database rows.
  - We load the embedding model (`embedding_model`) and semantic splitter (`text_splitter`) from our local folder.
