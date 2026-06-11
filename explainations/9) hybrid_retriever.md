# Script Explanation: `9) hybrid_retriever.md`

## 1. Overview
The primary role of the `hybrid_retriever.py` script is to combine the results of two different information retrieval approaches into a unified search system:
1. **Dense Semantic Retrieval**: Finds candidate document chunks based on conceptual meaning and context (even if different words are used).
2. **Sparse Lexical (BM25) Retrieval**: Finds candidate document chunks based on exact keyword matching.

Since these two retrieval methods produce scores on completely different mathematical scales (e.g., semantic cosine similarity is bounded between `-1` and `1`, whereas BM25 scores are unbounded positive numbers), they cannot be combined directly. 

This script is responsible for:
* Fetching candidates from both semantic and BM25 search systems.
* Normalizing their scores to a common scale ($0.0$ to $1.0$) so they can be merged fairly.
* Merging duplicate chunks (chunks identified by both systems) and fusing their scores together.
* Sorting the unified candidate list and returning the top 5 candidates.

---

## 2. Code Walkthrough

### Imports
```python
from .semantic_retriever import retrieve_chunks
from .bm25_retriever import bm25_search
```
- **What it does**: Imports the two search retrieval functions: `retrieve_chunks` (dense vector search in Qdrant) and `bm25_search` (sparse keyword search). The dot (`.`) denotes a relative import from the current directory.

---

### Score Normalization Function
```python
def normalize_scores(results):
    scores = [
        result["score"]
        for result in results
    ]
```
- **Lines 6–9**: We receive a list of search result dictionaries. We use a **list comprehension** (a compact loop structure) to extract only the raw score from each search result and save them to a list named `scores`.

```python
    max_score = max(scores)
    min_score = min(scores)
```
- **Lines 11–12**: We identify the highest value (`max()`) and the lowest value (`min()`) in the score list.

```python
    # Avoid Division By Zero
    if max_score == min_score:
        for result in results:
            result["score"] = 1.0
        return results
```
- **Lines 14–18**: This is our custom safeguard to handle the **division-by-zero edge case**.
  - *The Problem*: If all retrieved candidates have the exact same score (e.g., `[8.0, 8.0, 8.0]`), then `max_score` and `min_score` are identical. In the min-max formula, the denominator becomes `max_score - min_score = 0`. Dividing by zero would crash the program (`ZeroDivisionError`).
  - *The Solution*: If `max_score == min_score`, we bypass the formula, set all scores to `1.0` (which represents maximum normalized relevance), and return the results immediately. We treat all candidates as equally relevant.

```python
    # Min-Max Normalization
    for result in results:
        result["score"] = (
            result["score"] - min_score
        ) / (
            max_score - min_score
        )
    return results
```
- **Lines 20–27**: If the scores are not all identical, we apply the standard **Min-Max Normalization formula**:
  $$\text{Normalized Score} = \frac{\text{Current Score} - \text{Minimum Score}}{\text{Maximum Score} - \text{Minimum Score}}$$
  This transforms all scores into a clean range between `0.0` (for the lowest raw score) and `1.0` (for the highest raw score). For example, original scores of `[8.0, 4.0, 2.0]` scale to `[1.0, 0.33, 0.0]`.

---

### Hybrid Search Core Function
```python
async def hybrid_search(query: str):
```
- **Line 30**: Defines the main asynchronous function `hybrid_search`. The `async` keyword allows this function to execute concurrently with other tasks instead of blocking the main thread during database or calculation delays.

```python
    # Semantic Retrieval
    semantic_results = await retrieve_chunks(query)

    # BM25 Retrieval
    bm25_results = await bm25_search(query)
```
- **Lines 33–36**:
  - Calls `retrieve_chunks` to fetch semantic search candidates.
  - Calls `bm25_search` to fetch keyword relevance candidates.
  - The `await` keyword tells Python to yield CPU execution to other tasks while waiting for these functions to finish their database and file processing.

```python
    # Normalize Scores
    semantic_results = normalize_scores(semantic_results)
    bm25_results = normalize_scores(bm25_results)
```
- **Lines 39–40**: We normalize the raw scores of both retrieval lists independently using our `normalize_scores` helper function.

```python
    # Combine Results
    combined_results = {}
```
- **Line 43**: We initialize an empty dictionary `combined_results` to aggregate the results.
  - *Why a dictionary?* Because we must check for and merge duplicate chunks. Using a dictionary allows us to perform fast key-lookups in $O(1)$ constant time (checking if `chunk_id` is already in the dictionary).

```python
    # Add Semantic Results
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        combined_results[chunk_id] = {
            "text": result["text"],
            "source": result["source"],
            "chunk_id": chunk_id,
            "semantic_score": result["score"],
            "bm25_score": 0.0,
            "final_score": result["score"]
        }
```
- **Lines 46–55**: We loop through all the normalized semantic search results.
  - **Line 47**: We extract the `chunk_id`, which serves as our unique identifier for deduplication.
  - **Line 48**: We insert the chunk into the dictionary. We set its `semantic_score` and initial `final_score` to the normalized semantic score.
  - **Line 53**: Since we haven't processed the BM25 results yet, we set `bm25_score` to `0.0`.

```python
    # Add BM25 Results
    for result in bm25_results:
        chunk_id = result["chunk_id"]

        # If Chunk Already Exists
        if chunk_id in combined_results:
            combined_results[chunk_id][
                "bm25_score"
            ] = result["score"]

            combined_results[chunk_id][
                "final_score"
            ] += result["score"]
```
- **Lines 58–69**: We iterate through the BM25 search results.
  - **Line 62**: We perform our **duplicate check** (`if chunk_id in combined_results`).
  - If the chunk already exists (meaning it was retrieved by both semantic search and BM25 search):
    - We update its `bm25_score` with its normalized BM25 score.
    - We add the normalized BM25 score to the existing `final_score` (**Simple Addition Fusion**). For example, if a chunk has a semantic score of `0.8` and a BM25 score of `0.7`, its fused `final_score` becomes `1.5`.
