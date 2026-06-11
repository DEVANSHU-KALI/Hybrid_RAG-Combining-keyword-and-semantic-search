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
