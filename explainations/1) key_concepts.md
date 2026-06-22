# Conceptual Guide: `1) key_concepts.md`

This document serves as a high-level guide to the core software engineering, database, and machine learning concepts implemented throughout this Hybrid RAG project.

---

## 1. Retrieval-Augmented Generation (RAG)

### Definition
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that combines search-based retrieval with generative language models. Large language models (LLMs) have static knowledge locked at their training cutoff date and are prone to "hallucinations" (generating confident but incorrect facts). RAG resolves this by first searching a private database for text passages relevant to a user's query, and then feeding those passages to the LLM as grounding context. This forces the LLM to construct answers backed by verified source materials.

### Architecture Diagram
```
    User Query ─────────► [ Retrieval Stage ] ─────────► [ Augmentation Stage ] ─────────► [ Generation Stage ]
                             │                              │                                 │
                             ▼                              ▼                                 ▼
                     Scan Vector DB &                Inject retrieved                 Local Qwen model
                     Keyword Indexes                 document text into               generates grounded
                     for matching chunks             prompt context template          response
```

---

## 2. Dense Semantic vs. Sparse Lexical Retrieval

Information retrieval in modern RAG systems uses two main paradigms:

### A. Dense Semantic Retrieval
* **Concept**: Converts queries and document chunks into dense floating-point vectors (embeddings) where conceptual meaning is represented by spatial coordinates. Search matches concepts rather than character patterns.
* **Metric (Cosine Similarity)**: Measures the directional overlap (angle) between the query vector ($\mathbf{A}$) and document vector ($\mathbf{B}$):
  $$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
* **Used in**: `semantic_retriever.py` and `embedding_model.py`.

### B. Sparse Lexical Retrieval (BM25)
* **Concept**: Scores documents based on exact keyword occurrences. It uses the **Okapi BM25** formula, which evaluates term frequency (TF), document length normalization, and inverse document frequency (IDF - prioritizing rare query words).
* **Metric (BM25 Score)**:
  $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  * $f(q_i, D)$: Term frequency of word $q_i$ inside document $D$.
  * $|D|$ and $\text{avgdl}$: Current document length and average document length.
  * $k_1$ and $b$: Tuning parameters ($k_1$ controls term frequency saturation; $b$ controls length normalization).
* **Used in**: `bm25_retriever.py`.

---