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
