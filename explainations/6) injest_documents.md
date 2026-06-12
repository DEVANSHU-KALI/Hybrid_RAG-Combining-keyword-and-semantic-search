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

```python
client = QdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "rag_docs"
```
- **Lines 13–21**: Connects to the local Qdrant instance and sets our target collection name to `"rag_docs"`.

---

### Ingestion Function - Part 1: Reading and Chunking
```python
def ingest_documents(folder_path: str):
    documents = []
    ids = []
    sources = []
    counter = 0

    # Read TXT Files
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(
            folder_path,
            filename
        )
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Semantic Chunking
        chunks = text_splitter.create_documents([text])

        # Store Chunk Data
        for chunk in chunks:
            documents.append(chunk.page_content)
            ids.append(counter)
            sources.append(filename)
            counter += 1
```
