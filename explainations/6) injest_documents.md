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
- **Lines 26–57**:
  1. We initialize three storage lists: `documents` (chunk texts), `ids` (chunk IDs), and **`sources`** (chunk file sources), along with a sequential ID `counter`.
  2. We browse the directory using `os.listdir(folder_path)`. If a file doesn't end with `".txt"`, we ignore it.
  3. We construct the full path using `os.path.join()` and open the file in read mode with UTF-8 encoding.
  4. We call our `text_splitter` (the semantic chunker) to divide the text file into semantic chunks.
  5. We loop through the generated chunks, appending their text to `documents`, assigning a unique sequential ID from the `counter`, appending the current `filename` to `sources`, and incrementing the counter. This maintains an exact index correlation between text, ID, and source file.

---

### Ingestion Function - Part 2: Embedding Generation
```python
    # Print Chunks
    print("\n======= DOCUMENT CHUNKS =======\n")
    for doc in documents:
        print(doc)
        print("-------------")

    # Generate Embeddings
    embeddings = embedding_model.embed_documents(documents)
    print(
        "\nEmbedding vector size:",
        len(embeddings[0])
    )
```
- **Lines 62–74**:
  - We print the raw chunk texts to the terminal for debugging.
  - We invoke `embedding_model.embed_documents(documents)` to calculate vector embeddings.
    - *Why this way?* We pass the entire list of chunks at once. This performs **batch embedding**, sending all documents in a single process rather than embedding them one by one. This is faster and computationally efficient.

---

### Ingestion Function - Part 3: Point Construction & Database Upload
```python
    # Create Qdrant Points
    points = []

    for i in range(len(documents)):
        points.append(
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "text": documents[i],
                    "source": sources[i],
                    "chunk_id": ids[i]
                }
            )
        )
```
- **Lines 79–93**:
  - We loop through the indices of our chunks to build `PointStruct` items.
  - **Payload Mapping**: We map `"source": sources[i]`. This matches each chunk with the exact name of the file it was extracted from (retrieved from the `sources` index mapping), ensuring citations are accurate in downstream search queries.

> [!NOTE]
> **Developer Lesson Learned (Scope Leak Guard)**: 
> In a previous iteration of this script, the payload was built using `"source": filename`. Because Python loops do not create block scopes, the variable `filename` leaked into the function scope and held the value of the *last file processed* by the outer loop. This caused all uploaded chunks to cite the last file read. We resolved this by introducing the `sources` list to explicitly track and align filenames to each text chunk index.

```python
    # Upload Points to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(
        f"\n✅ Documents successfully ingested into '{COLLECTION_NAME}'"
    )
```
- **Lines 97–103**: Calls `client.upsert` to upload the list of points to Qdrant. If a point has an ID that already exists in the collection, Qdrant overwrites it; otherwise, it inserts a new point.

---
