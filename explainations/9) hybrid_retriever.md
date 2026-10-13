# Script Explanation: `9) hybrid_retriever.md`

## 1. Overview
The primary role of the `hybrid_retriever.py` script is to combine the results of two fundamentally different information retrieval approaches into a unified, high-performance search system:
1. **768D Dense Semantic Retrieval (`retrieve_chunks`)**: Finds candidate document chunks based on conceptual meaning, synonyms, and deep contextual semantics in Qdrant (even if completely different words are used).
2. **Sparse Lexical Retrieval (`bm25_search`)**: Finds candidate document chunks based on exact term frequency and keyword matching.

Since these two retrieval methods produce scores on completely different mathematical scales (e.g., semantic cosine similarity is bounded between `0.0` and `1.0`, whereas BM25 scores are unbounded positive numbers like `14.8` or `2.3`), they cannot be compared or summed directly.

This script is responsible for:
* **Concurrent Execution (`asyncio.gather`)**: Executing both dense vector search and sparse BM25 search in parallel on the event loop to reduce latency by ~50%.
* **Dual Security Scoping**: Enforcing strict `tenant_id` and `user_id` pre-filtering across both retrieval engines to eliminate data leakage.
* **Score Normalization**: Scaling raw scores independently to a common $[0.0, 1.0]$ range using Min-Max Normalization.
* **Deduplication & Linear Score Fusion**: Merging duplicate chunks identified by both systems and adding their normalized scores together (reaching combined scores up to $2.0$).
* **Candidate Pool Slicing**: Sorting the unified candidate list and returning the top $K$ candidates (default `limit=10`).

---

## 2. Code Walkthrough

### Part A: Imports & Helper Functions
```python
import asyncio
from .semantic_retriever import retrieve_chunks
from .bm25_retriever import bm25_search
```
- **`import asyncio`**: Python standard asynchronous I/O library. Used to run both search queries concurrently in parallel via `asyncio.gather()`.
- **`from .semantic_retriever import retrieve_chunks`**: Imports the 768D dense vector search function.
- **`from .bm25_retriever import bm25_search`**: Imports the sparse lexical keyword search function.

---

### Part B: Score Normalization Function (`normalize_scores`)
```python
def normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return []
    scores = [result["score"] for result in results]
    max_score = max(scores)
    min_score = min(scores)
```
- **Lines 14–20**: Takes a list of search result dictionaries. Uses list comprehension to extract only the raw score from each result into a `scores` list, then finds `max_score` and `min_score`.

```python
    # Division-by-Zero Guard
    if max_score == min_score:
        for result in results:
            result["score"] = 1.0
        return results
```
- **Lines 21–25**: **Division-by-Zero Safeguard**:
  - *The Problem*: If all retrieved candidates have the exact same score (e.g. `[8.0, 8.0, 8.0]`), then `max_score - min_score = 0`. Dividing by zero causes a crash (`ZeroDivisionError`).
  - *The Solution*: If `max_score == min_score`, all candidates are equally relevant. We bypass the formula and set all normalized scores to `1.0`.

```python
    # Min-Max Normalization Formula
    for result in results:
        result["score"] = (result["score"] - min_score) / (max_score - min_score)
    return results
```
- **Lines 27–30**: If scores vary, we apply the standard **Min-Max Normalization formula**:
  $$\text{Normalized Score} = \frac{\text{Score} - \text{Min}}{\text{Max} - \text{Min}}$$
  This transforms all raw scores into a clean range between `0.0` (lowest score) and `1.0` (highest score).

---

### Part C: Hybrid Search Core Function (`hybrid_search`)
```python
async def hybrid_search(
    query: str,                              # User prompt string to retrieve against
    tenant_id: str = "tenant_default",       # Tenant ID for multi-tenant data isolation pre-filtering
    user_id: str = "usr_admin",              # User ID for strict user-level data isolation
    limit: int = 10                          # Top K candidate chunks to return
) -> list[dict]:
```
- **Lines 39–44**: Defines the main asynchronous function.
  - Takes `query`, `tenant_id`, and `user_id` to guarantee multi-tenant and user-level data boundary isolation.
  - `limit: int = 10` provides a default candidate return count.

```python
    # 1. Concurrent Parallel Search Execution
    semantic_task = retrieve_chunks(query, tenant_id=tenant_id, user_id=user_id, limit=15)
    bm25_task = bm25_search(query, tenant_id=tenant_id, user_id=user_id, limit=15)

    semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)
```
- **Lines 48–52**:
  - **Over-Fetching Strategy (`limit=15`)**: Fetches 15 dense candidates and 15 sparse candidates (up to 30 candidates total) to ensure the fusion engine has a rich candidate variety.
  - **`asyncio.gather()` Concurrency**:
    - *Sequential Latency*: $\text{Latency} = \text{Latency}_{\text{Dense}} + \text{Latency}_{\text{Sparse}}$
    - *Parallel Latency*: $\text{Latency} = \max(\text{Latency}_{\text{Dense}}, \text{Latency}_{\text{Sparse}})$
    - Running both search calls concurrently cuts retrieval wait time by nearly **50%**!

```python
    # 2. Independent Score Normalization
    semantic_results = normalize_scores(semantic_results)
    bm25_results = normalize_scores(bm25_results)
```
- **Lines 56–57**: Normalizes dense cosine scores and sparse BM25 scores independently to $[0.0, 1.0]$.

```python
    # 3. Hybrid Score Fusion & Deduplication Engine
    combined_results = {}

    # Loop 1: Populate dictionary with normalized semantic results
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        combined_results[chunk_id] = {
            "text": result["text"],
            "source": result["source"],
            "chunk_id": chunk_id,
            "tenant_id": result.get("tenant_id", tenant_id),
            "user_id": result.get("user_id", user_id),
            "page_number": result.get("page_number", 1),
            "semantic_score": result["score"],
            "bm25_score": 0.0,
            "final_score": result["score"]
        }
```
- **Lines 60–77**: Initializes `combined_results = {}`.
  - Using a dictionary keyed by `chunk_id` allows $O(1)$ constant-time duplicate detection.
  - Inserts semantic chunks, setting `semantic_score = score`, `bm25_score = 0.0`, and `final_score = score`.
  - Preserves metadata fields (`tenant_id`, `user_id`, `page_number`).

```python
    # Loop 2: Merge normalized BM25 keyword results into dictionary
    for result in bm25_results:
        chunk_id = result["chunk_id"]
        
        # If Chunk Already Exists (DUPLICATE CHUNK)
        if chunk_id in combined_results:
            combined_results[chunk_id]["bm25_score"] = result["score"]
            combined_results[chunk_id]["final_score"] += result["score"]
            
        # New BM25 Chunk
        else:
            combined_results[chunk_id] = {
                "text": result["text"],
                "source": result["source"],
                "chunk_id": chunk_id,
                "tenant_id": result.get("tenant_id", tenant_id),
                "user_id": result.get("user_id", user_id),
                "page_number": result.get("page_number", 1),
                "semantic_score": 0.0,
                "bm25_score": result["score"],
                "final_score": result["score"]
            }
```
- **Lines 80–103**: Iterates through BM25 results.
  - **`if chunk_id in combined_results`**: If found by both systems, we update `bm25_score` and add it to `final_score` (**Simple Addition Fusion**). Chunks verified by both engines receive a score boost up to **$2.0$**!
  - **`else`**: If found only by BM25, inserts it as a new entry with `semantic_score = 0.0` and `final_score = bm25_score`.

```python
    final_results = list(combined_results.values())
    
    final_results = sorted(
        final_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return final_results[:limit]
```
- **Lines 105–115**: Converts dictionary values back to a list, sorts in descending order by `final_score`, and slices the top $K$ candidates (`[:limit]`, default 10).

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
              User Query + tenant_id + user_id
                             │
                  ┌──────────┴──────────┐
                  │ asyncio.gather(...) │
                  ▼                     ▼
          Semantic Retrieval      BM25 Retrieval
          (768D Dense Vector)   (Okapi Term Match)
                  │                     │
                  ▼                     ▼
           Normalize Scores      Normalize Scores
          (Scale: 0.0 - 1.0)    (Scale: 0.0 - 1.0)
                  │                     │
                  └──────────┬──────────┘
                             ▼
                    Merge Duplicate Chunks
                   (Aggregated by chunk_id)
                             │
                             ▼
                    Linear Addition Fusion
                (semantic_score + bm25_score)
                             │
                             ▼
                    Sort By Final Score
                    (Highest score first)
                             │
                             ▼
                  Return Top K Candidates
                    (Sliced to [:limit])
```

---

### Input and Output Specifications
* **Inputs**:
  * `query` (`str`): The search prompt (e.g., `"what is hybrid retrieval?"`).
  * `tenant_id` (`str`): Multi-tenant organization identity tag (default: `"tenant_default"`).
  * `user_id` (`str`): Authenticated user identity tag (default: `"usr_admin"`).
  * `limit` (`int`): Top candidate slice count (default: `10`).
* **Output**: A list containing up to `limit` dictionary objects (`list[dict]`). Each dictionary contains:
  * `"text"`: Raw text string of the document chunk.
  * `"source"`: Origin filename for citations.
  * `"chunk_id"`: Unique integer ID of the chunk.
  * `"tenant_id"`: Scoped tenant organization identifier.
  * `"user_id"`: Scoped user identifier.
  * `"page_number"`: Source PDF/text page number.
  * `"semantic_score"`: Normalized dense similarity score ($0.0$ to $1.0$).
  * `"bm25_score"`: Normalized lexical match score ($0.0$ to $1.0$).
  * `"final_score"`: Combined fused score ($0.0$ to $2.0$).

---

### Step-by-Step Variable Trace Walkthrough
Sample Query: `"what is hybrid retrieval?"` with `tenant_id="tenant_default"`, `user_id="usr_admin"`.

#### Step 1: Raw Search Retrieval via `asyncio.gather()`
Both retrievers execute concurrently and return candidate lists:
```python
semantic_results = [
    {"chunk_id": 1, "text": "Hybrid retrieval combines BM25 and semantic search", "score": 0.95},
    {"chunk_id": 2, "text": "BM25 is a ranking algorithm", "score": 0.85},
    {"chunk_id": 3, "text": "Football players run fast", "score": 0.85}
]

bm25_results = [
    {"chunk_id": 1, "text": "Hybrid retrieval combines BM25 and semantic search", "score": 8.0},
    {"chunk_id": 4, "text": "Keyword retrieval improves exact matching", "score": 8.0},
    {"chunk_id": 5, "text": "Rerankers improve retrieval quality", "score": 8.0}
]
```

#### Step 2: Normalizing Semantic Scores
Calling `normalize_scores(semantic_results)`:
* `scores = [0.95, 0.85, 0.85]`, `max_score = 0.95`, `min_score = 0.85`.
* Chunk 1: $(0.95 - 0.85) / (0.95 - 0.85) = 1.0$
* Chunk 2: $(0.85 - 0.85) / (0.95 - 0.85) = 0.0$
* Chunk 3: $(0.85 - 0.85) / (0.95 - 0.85) = 0.0$

#### Step 3: Normalizing BM25 Scores
Calling `normalize_scores(bm25_results)`:
* `scores = [8.0, 8.0, 8.0]`, `max_score = 8.0`, `min_score = 8.0`.
* **Division-by-zero check triggers:** `max_score == min_score`. All BM25 scores are set to `1.0`.

#### Step 4: Merging Results & Score Fusion
* **Chunk 1** is found by **BOTH** systems:
  * `semantic_score = 1.0`
  * `bm25_score = 1.0`
  * `final_score = 1.0 + 1.0 = 2.0` (Maximum boosted score!)
* **Chunks 2 and 3** (Semantic only): `final_score = 0.0 + 0.0 = 0.0`
* **Chunks 4 and 5** (BM25 only): `final_score = 0.0 + 1.0 = 1.0`

#### Step 5: Sorting & Slicing
```python
# Sorted Results (Descending by final_score):
[
    {"chunk_id": 1, "final_score": 2.0, "text": "Hybrid retrieval combines..."}, # Dual Agreement!
    {"chunk_id": 4, "final_score": 1.0, "text": "Keyword retrieval improves..."},
    {"chunk_id": 5, "final_score": 1.0, "text": "Rerankers improve..."},
    {"chunk_id": 2, "final_score": 0.0, "text": "BM25 is a ranking algorithm"},
    {"chunk_id": 3, "final_score": 0.0, "text": "Football players run fast"}
]
```

### Core Design Insight: Priority of Duplicates
Observe how **Chunk 1** ranks at the top with a score of `2.0`. Because it was validated independently by both semantic AI and exact keyword matching, the hybrid fusion design naturally prioritizes agreement between search engines.

---

## 4. Deep Technical Concepts

### 1. Concurrent Parallelism (`asyncio.gather`)
In standard sequential code, running two network/database calls takes $\text{Latency}_1 + \text{Latency}_2$. By packaging `semantic_task` and `bm25_task` into `asyncio.gather()`, Python dispatches both asynchronous tasks to the Qdrant server at the exact same millisecond. The event loop waits only for the slowest of the two, reducing total retrieval latency by up to 50%.

### 2. Candidate Over-Fetching
Stage-1 hybrid search requests 15 candidates from each retriever (up to 30 candidates total). This intentionally over-fetches candidates so the fusion engine has a sufficiently rich pool to detect overlaps before slicing the top 10 for Stage-2 Cross-Encoder reranking.

### 3. Linear Score Fusion
Combines normalized scores via simple addition:
$$\text{final\_score} = \text{semantic\_score} + \text{bm25\_score}$$
Chunks matched by both retrievers receive scores in $[1.0, 2.0]$, while chunks matched by only one retriever receive scores in $[0.0, 1.0]$.

---

## 5. Architectural Choices and Alternatives

| Strategy | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Min-Max Score Fusion** *(Chosen)* | Normalizes raw scores of both algorithms to $[0.0, 1.0]$, then sums them together. | • Preserves relative performance gaps between candidates.<br>• Boosts overlapping chunks up to $2.0$. | • Sensitive to extreme outliers in score distributions. |
| **Reciprocal Rank Fusion (RRF)** | Merges lists using rank position: $RRF = \sum \frac{1}{60 + \text{Rank}}$. Ignores raw scores. | • Independent of score distributions.<br>• Completely immune to outliers. | • Discards the magnitude of score differences. |
| **Weighted Linear Fusion** | Applies explicit weights: $w \cdot \text{Semantic} + (1-w) \cdot \text{BM25}$. | • Allows fine-tuning domain priorities (e.g. 70% dense, 30% sparse). | • Requires manual parameter tuning. |
