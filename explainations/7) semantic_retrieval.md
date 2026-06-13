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

```python
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
```
- **Lines 32–43**:
  - We initialize an empty list `results_list`.
  - We iterate through all retrieved records in `results.points`.
  - For each point, we extract the metadata stored in its payload (`text`, `source`, and `chunk_id`) along with its similarity score (`point.score`, which represents the Cosine Similarity between the query vector and the document vector).
  - We append these key-value mappings to `results_list` and return the finalized search results.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     Input Query: "How does RAG work?"
                                │
                                ▼
                       Embed Query Vector
                   (all-MiniLM -> 384 floats)
                                │
                                ▼
               Async Database Query (query_points)
                     limit=10, Cosine Distance
                                │
                                ▼
                    Qdrant Cosine Similarity
                           Evaluation
                                │
                                ▼
                    Extract Matches & Payload
                       (text, source, score)
                                │
                                ▼
                    Output List of Dict Items
```

---

### Input and Output Specifications
* **Input**: `query` (Type: `str`) - The plain-text user question (e.g., `"How does RAG work?"`).
* **Output**: A list containing up to 10 dictionary objects (Type: `list[dict]`). Each dictionary contains:
  * `"text"`: The retrieved chunk's text content.
  * `"source"`: The source filename.
  * `"chunk_id"`: The database integer ID of the chunk.
  * `"score"`: The Cosine Similarity score ($0.0$ to $1.0$).

---

### Step-by-Step Variable Trace Walkthrough
Let's trace what happens when we execute `await retrieve_chunks("What is overfitting?")`:

1. **Query Input**: The function receives `query = "What is overfitting?"`.
2. **Embedding Conversion**: `embedding_model.embed_query(query)` maps the text to a dense vector:
   * `query_vector = [0.081, -0.023, ..., 0.045]` (length 384).
3. **Database Query**: `client.query_points` executes:
   - Connects asynchronously to Qdrant's query endpoint.
   - Qdrant compares `query_vector` with all embedded vectors stored in `"rag_docs"` by calculating Cosine Similarity:
     $$\text{Cosine Similarity} = \frac{\mathbf{query\_vector} \cdot \mathbf{document\_vector}}{\|\mathbf{query\_vector}\| \|\mathbf{document\_vector}\|}$$
   - Returns the top 10 vectors with the highest cosine values.
4. **Data Extraction Loop**:
   - Loop starts over Qdrant points. For the first point:
     * `point.payload` is `{"text": "Overfitting occurs when...", "source": "concepts.txt", "chunk_id": 1}`.
     * `point.score` is `0.875`.
     * Appends `{"text": "Overfitting occurs when...", "source": "concepts.txt", "chunk_id": 1, "score": 0.875}` to `results_list`.
5. **Output**: Returns the list of 10 parsed dictionary objects.

---

## 4. Deep Technical Concepts

### Semantic Similarity Search (Nearest Neighbor)
Traditional databases rely on indexing keywords (e.g., searching for "overfitting" will not match "generalization failures" or "memorization of training data"). **Semantic Similarity Search** performs a mathematical nearest-neighbor search by looking at coordinates. By checking which document vectors are closest to the query vector in high-dimensional vector space, it finds conceptually related articles even if they use completely different vocabularies.

### Asynchronous Database I/O Concurrency
In typical synchronous programming, when a script requests data from a database, the entire thread blocks (waits) for the database server to process the request and send a response. This script uses **Asynchronous Concurrency** via `AsyncQdrantClient` and `await`. When querying Qdrant, the python runtime yields the active thread to process other incoming API calls, maximizing backend performance and API capacity.

---

## 5. Architectural Choices and Alternatives

### Why AsyncQdrantClient?
Using Qdrant's asynchronous client API is highly suited for web service environments (like FastAPI) since it prevents database lookups from blocking the main event loop, significantly increasing API capacity and server responsiveness.

#### Alternatives and Trade-offs

| Interface Client | Strategy | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **AsyncQdrantClient** *(Chosen)* | Non-blocking database calls using `async`/`await`. | • Excellent for high-throughput APIs.<br>• Integrates natively with FastAPI's async endpoints. | • Requires coding within Python's async event loop framework. |
| **Synchronous QdrantClient** | Normal blocking requests: `client.query_points(...)`. | • Simple, linear code flow (no need for `async`/`await`). | • Blocks the server thread during database requests, reducing backend concurrent user limits. |
| **HTTP REST Client** | Standard HTTP requests using `httpx` or `requests`. | • Requires zero database library dependencies. | • High manual implementation overhead (building raw JSON endpoints and parsing manually). |

