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
## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     Folder Path: data/rag_concepts
                                │
                                ▼
                       Loop Through Files
                       (Extract filename)
                                │
                                ▼
                        Read File Content
                                │
                                ▼
                        Semantic Chunking
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
       Append Text to     Append ID to     Append filename
        documents[]          ids[]          to sources[]
                │               │               │
                └───────────────┼───────────────┘
                                ▼
                       Loop Finished?
                 ├── No  ──► Process next file
                 └── Yes ──► Batch Embed Documents
                                │
                                ▼
                     Create PointStruct Items:
                      id=ids[i], vector=embeddings[i],
                      payload={"text", "source": sources[i]}
                                │
                                ▼
                       Upload (Upsert) to Qdrant
```

---

### Input and Output Specifications
* **Input**: `folder_path` (Type: `str`) - The path to the folder containing source documents.
* **Output**: Writes points containing IDs, embeddings, and payloads directly to Qdrant. Prints progress to the console.

---

### Step-by-Step Variable Trace Walkthrough
Assume `data/rag_concepts/` contains two files: `overfitting.txt` and `bias.txt`.

1. **Loop File 1 (`overfitting.txt`)**:
   - Split into 2 chunks: Chunk A, Chunk B.
   - `documents` becomes: `["Chunk A text", "Chunk B text"]`.
   - `ids` becomes: `[0, 1]`.
   - `sources` becomes: `["overfitting.txt", "overfitting.txt"]`.
   - `counter` becomes: `2`.

2. **Loop File 2 (`bias.txt`)**:
   - Split into 1 chunk: Chunk C.
   - `documents` becomes: `["Chunk A text", "Chunk B text", "Chunk C text"]`.
   - `ids` becomes: `[0, 1, 2]`.
   - `sources` becomes: `["overfitting.txt", "overfitting.txt", "bias.txt"]`.
   - `counter` becomes: `3`.
   - **File-reading loop completes.**

3. **Batch Embedding**:
   - `embedding_model.embed_documents(documents)` runs, returning a list of 3 vectors: `[VecA, VecB, VecC]`.

4. **Point Struct Loop ($i = 0$ to $2$):**
   - **Index $i = 0$**: Creates `PointStruct(id=0, vector=VecA, payload={"text": "Chunk A text", "source": "overfitting.txt", ...})`.
   - **Index $i = 1$**: Creates `PointStruct(id=1, vector=VecB, payload={"text": "Chunk B text", "source": "overfitting.txt", ...})`.
   - **Index $i = 2$**: Creates `PointStruct(id=2, vector=VecC, payload={"text": "Chunk C text", "source": "bias.txt", ...})`.
   - *Result*: All chunks are correctly mapped to their respective source files!

5. **Upload**: Upserts these 3 points to Qdrant.

---

## 4. Deep Technical Concepts

### Batch Embeddings
Generating embeddings is computationally expensive. Sending text to an embedding model one string at a time introduces significant latency overhead (each call has processing initialization and transfer overhead). **Batch Embedding** bundles a list of texts into a single request, allowing the model (especially on GPUs) to process them in parallel.

### Qdrant PointStruct Schema
A **Point** in Qdrant represents a single data record. It is defined by:
* `id`: A unique integer or UUID.
* `vector`: The dense embedding array of floats.
* `payload`: A JSON object storing metadata (like raw text, source filename, chunk ID). Qdrant indexes payloads, allowing vector searches to be filtered by metadata fields (e.g., searching only within a specific source file).

---

## 5. Architectural Choices and Alternatives

### Why Batch Upsert?
We construct a list of all points and perform a single `client.upsert()` call. This minimizes network round-trips between our application server and Qdrant.

#### Alternatives and Trade-offs

| Ingestion Method | Strategy | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Batch Upsert** *(Chosen)* | Reads all files, chunks them, embeds them in one batch, and performs one upload. | • High throughput.<br>• Minimal API network overhead. | • Higher memory usage (all documents and embeddings are loaded in RAM simultaneously). |
| **Stream Processing** | Reads, chunks, embeds, and uploads one file (or one chunk) at a time. | • Low memory footprint.<br>• Suitable for large-scale migrations. | • High latency due to repeated network calls. |
| **Queue-Based ingestion (Celery/RabbitMQ)** | Files are added to a queue, and background workers process them asynchronously. | • Highly scalable.<br>• Handles failures gracefully. | • Significant complexity and setup overhead. |
