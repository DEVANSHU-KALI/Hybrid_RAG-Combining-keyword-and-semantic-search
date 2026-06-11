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

```python
        # New BM25 Chunk
        else:
            combined_results[chunk_id] = {
                "text": result["text"],
                "source": result["source"],
                "chunk_id": chunk_id,
                "semantic_score": 0.0,
                "bm25_score": result["score"],
                "final_score": result["score"]
            }
```
- **Line 72**: The `else` statement handles new BM25-only chunks (chunks that were not found by the semantic search). We add them as new entries in the dictionary with `semantic_score = 0.0` and `final_score = bm25_score`. This ensures both retrieval systems contribute candidates.

```python
    # Convert Dict To List
    final_results = list(
        combined_results.values()
    )
```
- **Line 83**: We extract the dictionary values (the chunk records) and convert them back to a standard Python list.

```python
    # Sort By Final Score
    final_results = sorted(
        final_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Return Top Candidates
    return final_results[:5]
```
- **Lines 88–95**: We sort the list of merged chunks by their `final_score` in descending order (highest score first) using a lambda function as the sort key. We use list slicing `[:5]` to return only the top 5 highest-ranking candidates.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
              User Query
                  │
         ┌────────┴────────┐
         ▼                 ▼
 Semantic Retrieval    BM25 Retrieval
   (Cosine: Cosine)    (BM25: Unbounded)
         │                 │
         ▼                 ▼
 Normalize Scores      Normalize Scores
   (Scale: 0.0 - 1.0)    (Scale: 0.0 - 1.0)
         │                 │
         └────────┬────────┘
                  ▼
         Merge Duplicate Chunks
          (Aggregated by chunk_id)
                  │
                  ▼
         Simple Addition Fusion
          (semantic_score + bm25_score)
                  │
                  ▼
         Sort By Final Score
         (Highest score first)
                  │
                  ▼
         Return Top 5 Candidates
```

---

### Input and Output Specifications
* **Input**: `query` (Type: `str`) - The search prompt typed by the user (e.g., `"what is hybrid retrieval?"`).
* **Output**: A list containing up to 5 dictionary objects (Type: `list[dict]`). Each dictionary contains:
  * `"text"`: The actual text string of the document chunk.
  * `"source"`: The filename where the text originated.
  * `"chunk_id"`: The unique integer ID of the chunk.
  * `"semantic_score"`: The normalized semantic similarity score ($0.0$ to $1.0$).
  * `"bm25_score"`: The normalized keyword match score ($0.0$ to $1.0$).
  * `"final_score"`: The combined score ($0.0$ to $2.0$).

---

### Step-by-Step Variable Trace Walkthrough
Let's trace the variables and execution state for a sample query: `"what is hybrid retrieval?"`.

#### Step 1: Raw Search Retrieval
Our retrievers return the following lists:
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
*(Notice that all raw BM25 scores are identical at `8.0`. This will trigger our division-by-zero safeguard).*

#### Step 2: Normalizing Semantic Scores
We call `normalize_scores(semantic_results)`:
1. **Extract Scores**: `scores = [0.95, 0.85, 0.85]`
2. **Find Bounds**: `max_score = 0.95`, `min_score = 0.85`.
3. **Division Check**: `0.95 == 0.85` is `False`, so we use the normalization formula:
   * **Chunk 1**: `(0.95 - 0.85) / (0.95 - 0.85) = 0.10 / 0.10 = 1.0`
   * **Chunk 2**: `(0.85 - 0.85) / (0.95 - 0.85) = 0.0 / 0.10 = 0.0`
   * **Chunk 3**: `(0.85 - 0.85) / (0.95 - 0.85) = 0.0 / 0.10 = 0.0`
4. **Result**:
```python
semantic_results = [
    {"chunk_id": 1, "score": 1.0, ...},
    {"chunk_id": 2, "score": 0.0, ...},
    {"chunk_id": 3, "score": 0.0, ...}
]
```

#### Step 3: Normalizing BM25 Scores
We call `normalize_scores(bm25_results)`:
1. **Extract Scores**: `scores = [8.0, 8.0, 8.0]`
2. **Find Bounds**: `max_score = 8.0`, `min_score = 8.0`.
3. **Division Check**: `8.0 == 8.0` is `True`! 
4. **Trigger Edge-Case Handling**: The script avoids dividing by zero by setting every score to `1.0` (representing maximum normalized relevance).
5. **Result**:
```python
bm25_results = [
    {"chunk_id": 1, "score": 1.0, ...},
    {"chunk_id": 4, "score": 1.0, ...},
    {"chunk_id": 5, "score": 1.0, ...}
]
```

#### Step 4: Merging Results (The Fusing Stage)
We initialize `combined_results = {}`.

1. **Adding Semantic Candidates**:
   * **Chunk 1** is added:
     ```python
     combined_results[1] = {"semantic_score": 1.0, "bm25_score": 0.0, "final_score": 1.0, ...}
     ```
   * **Chunk 2** and **Chunk 3** are added:
     ```python
     combined_results[2] = {"semantic_score": 0.0, "bm25_score": 0.0, "final_score": 0.0, ...}
     combined_results[3] = {"semantic_score": 0.0, "bm25_score": 0.0, "final_score": 0.0, ...}
     ```

2. **Processing BM25 Candidates & Merging Duplicates**:
   * **Chunk 1** is processed. The script checks `if 1 in combined_results`. Since it exists, it updates and merges scores:
     * `bm25_score = 1.0`
     * `final_score += 1.0` (Updated `final_score` becomes `1.0 + 1.0 = 2.0`)
   * **Chunk 4** is processed. `if 4 in combined_results` is `False`. It is added as a new entry:
     ```python
     combined_results[4] = {"semantic_score": 0.0, "bm25_score": 1.0, "final_score": 1.0, ...}
     ```
   * **Chunk 5** is processed and added as a new entry similarly.

#### Step 5: Sorting and Slicing
The finalized dictionary values are converted to a list:
```python
final_results = [
    {"chunk_id": 1, "final_score": 2.0},
    {"chunk_id": 2, "final_score": 0.0},
    {"chunk_id": 3, "final_score": 0.0},
    {"chunk_id": 4, "final_score": 1.0},
    {"chunk_id": 5, "final_score": 1.0}
]
```
We sort them in descending order by `final_score` and slice the top 5:
```python
# Sorted List:
[
    {"chunk_id": 1, "final_score": 2.0},
    {"chunk_id": 4, "final_score": 1.0},
    {"chunk_id": 5, "final_score": 1.0},
    {"chunk_id": 2, "final_score": 0.0},
    {"chunk_id": 3, "final_score": 0.0}
]
```

### Core Design Insight: Priority of Duplicates
Observe how **Chunk 1** ended up at the top of the list with a perfect score of `2.0`. Because it was found independently by both semantic retrieval (conceptually relevant) and lexical retrieval (keyword matching), the system treats it as highly relevant. This hybrid retrieval design naturally prioritizes agreement between search engines.

---

## 4. Deep Technical Concepts

### Dense Vector Similarity Search
This method retrieves information by converting text into a **dense vector** (a list of continuous floating-point numbers representing features in a high-dimensional space). This representation captures the conceptual meaning of words rather than their exact letters. Similarity is measured geometrically (typically using Cosine Similarity, which measures the angle between two vectors).

### Sparse Lexical Search (BM25)
This method retrieves information using a **sparse vector** (a large list of numbers representing a global vocabulary, where almost all entries are zero except for the exact words present in the document). BM25 (Best Match 25) is a statistical ranking formula that evaluates term frequency (how many times a word occurs in a document), document length normalization (penalizing long, verbose documents), and inverse document frequency (prioritizing rare words over common ones).

### Score Normalization
Because vector similarity scores and BM25 scores represent different mathematical distributions and ranges, summing them directly would heavily bias results toward the strategy producing larger raw values (usually BM25). Score normalization transforms scores from disparate retrieval runs onto a matching, comparable scale (e.g., $0.0$ to $1.0$).

### Linear Score Fusion
Linear Score Fusion is the process of combining normalized scores from multiple retrievers using a simple weighted sum:
$$\text{Final Score} = w \cdot \text{Score}_{\text{semantic}} + (1 - w) \cdot \text{Score}_{\text{BM25}}$$
In this script, an equal weighting ($w = 0.5$, simplified to addition without dividing by 2) is used:
$$\text{Final Score} = \text{Score}_{\text{semantic}} + \text{Score}_{\text{BM25}}$$

---

## 5. Architectural Choices and Alternatives

### Why Min-Max Fusion?
The script utilizes **Min-Max Score Fusion** to merge the candidate lists. While simple and intuitive, it is sensitive to outliers. For example, a single document with an extremely high BM25 score will push the scores of other documents close to zero.

#### Alternatives and Trade-offs

| Strategy | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Min-Max Score Fusion** *(Chosen)* | Normalizes raw scores of both algorithms to a $[0.0, 1.0]$ range, then sums them together. | • Preserves the relative performance gaps between retrieved candidates. | • Highly sensitive to outliers (e.g., one exceptionally high score compresses all other scores). |
| **Reciprocal Rank Fusion (RRF)** | Merges lists by ignoring raw scores entirely. It uses the rank positions (1st, 2nd, etc.) of candidates: $RRF = \sum \frac{1}{60 + \text{Rank}}$. | • Completely independent of score distributions.<br>• Extremely stable and robust against outliers. | • Discards the magnitude of score differences (e.g., treats a marginal victory the same as a landslide). |
| **Weighted Linear Fusion** | Applies explicit importance weights (e.g., $0.7 \times \text{Semantic} + 0.3 \times \text{BM25}$) to prioritize one retriever. | • Allows fine-tuning the search behavior based on domain needs. | • Requires manual parameter tuning and evaluation to find the ideal weight. |
