# Conceptual Guide: `1) key_concepts.md`

This document serves as a high-level guide to the core software engineering, database, and machine learning concepts implemented throughout this Multi-Tenant Hybrid RAG project.

---

## 1. Retrieval-Augmented Generation (RAG)

### Definition
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that combines search-based retrieval with generative language models. Large language models (LLMs) have static knowledge locked at their training cutoff date and are prone to "hallucinations" (generating confident but incorrect facts). RAG resolves this by first searching a private database for text passages relevant to a user's query, and then feeding those passages to the LLM as grounding context. This forces the LLM to construct answers backed by verified source materials.

### Architecture Diagram
```
    User Query ─────────► [ Multi-Tenant Hybrid Retrieval ] ─────────► [ Augmentation Stage ] ─────────► [ Generation & Streaming ]
                             │                                              │                                  │
                             ▼                                              ▼                                  ▼
                     Scan Vector DB (768D) &                        Inject retrieved                   Local Qwen 2.5 / Groq
                     BM25 Keyword Indexes with                      document text into                 streams tokens via SSE
                     tenant_id payload filters                      strict grounding prompt            with FinOps telemetry
```

---

## 2. Dense Semantic vs. Sparse Lexical Retrieval

Information retrieval in modern RAG systems uses two main paradigms:

### A. Dense Semantic Retrieval
* **Concept**: Converts queries and document chunks into dense floating-point vectors (embeddings) where conceptual meaning is represented by spatial coordinates. Search matches concepts rather than character patterns.
* **Model Used**: **`BAAI/bge-base-en-v1.5`** (768-dimensional embeddings), specifically optimized for dense retrieval and asymmetric semantic search.
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

### C. LangChain Wrapper vs. Raw `SentenceTransformer` Interface
* **Concept**: Raw `SentenceTransformer` provides only a single `.encode()` method for all inputs. However, modern RAG architecture relies on **Asymmetric Retrieval**, where search queries and document passages have different embedding requirements.
* **Interface Distinction**:
  - `embed_documents(texts)`: Embeds batch document passages as static corpus chunks.
  - `embed_query(text)`: Embeds user queries, applying instruction prefixes or specialized query pooling.
* **Why the Wrapper?** Using `HuggingFaceEmbeddings` provides a standardized interface expected by LangChain chunkers, retrievers, and vector stores, ensuring queries and documents are processed with their correct respective methods.

---

## 3. High-Fidelity Tabular PDF Ingestion & Geometric Bounding-Box Cropping

PDF documents present a major hazard for RAG pipelines: standard text extractors (`pypdf`, basic OCR) read glyphs across columns, smashing structured tables into unreadable prose and creating duplicate chunks.

```
┌────────────────────────────────────────────────────────────────────────┐
│ GEOMETRIC BOUNDING-BOX CROP (backend/injest_documents.py)             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Prose ABOVE Table      ──► page.crop((0, prev_bottom, w, top))      │
│ 2. TABLE REGION           ──► page.find_tables() converted to Markdown │
│ 3. Prose BELOW Table      ──► page.crop((0, bottom, w, h))             │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Geometric Bounding-Box Detection**: Using `pdfplumber.find_tables()`, the parser locates exact table boundaries `(x0, top, x1, bottom)`. Narrative prose is cropped strictly *outside* table boxes.
2. **In-Place Markdown Assembly**: Tables are converted to clean GitHub-flavored Markdown tables and placed at their authentic reading positions between surrounding paragraphs.
3. **Chunk-Level Metadata Accuracy**: Rather than inheriting coarse page-level booleans (which caused pure text chunks to falsely advertise `"has_table": true`), each chunk is dynamically inspected for Markdown table rows (`| ... |`), ensuring metadata filters remain 100% accurate.
4. **Pre-Table Chunk Isolation**: When chunking, if a table is encountered and the running chunk already holds $>500$ characters, the chunker splits immediately before the table, ensuring tables remain intact.

---

## 4. Semantic Chunking & Boundary Optimization

### Definition
**Semantic Chunking** is a text splitting technique that groups text based on thematic transitions rather than arbitrary character or token counts. It processes a document sentence-by-sentence, calculates the semantic distance between consecutive sentences, and inserts a breakpoint (starts a new chunk) when the semantic distance exceeds a dynamically calculated percentile threshold.

### Breakpoint Evaluation Flow
```
Sentence S1 ──► [Embedding Model] ──► Vector V1 ──┐
                                                 ├──► Cosine Distance ──┐
Sentence S2 ──► [Embedding Model] ──► Vector V2 ──┘                      │
                                                                         ▼
                                                          Does Distance > 75th Percentile?
                                                          ├── Yes ──► Split (Start New Chunk)
                                                          └── No  ──► Merge sentences
```

---

## 5. Score Fusion: Concept, Evolution, and Architecture

### What is Score Fusion?
In a hybrid search system, dense semantic retrieval and sparse BM25 retrieval output independent lists of candidates, each with its own scoring metric. **Score Fusion** is the mathematical technique used to combine these disparate scores into a single, unified ranking score. 

```
    [ Simple Addition ] ─────────► [ Weighted Linear Fusion ] ─────────► [ Reciprocal Rank Fusion (RRF) ]
   Sum normalized scores.         Introduces alpha/beta weights          Rank-based, score-independent.
   Treats all systems as          to prioritize systems (requires        Extremely robust against
   equally important.             hyperparameter tuning).                outliers and scaling biases.
```

#### 1. Simple Addition Fusion *(Chosen Stage-1 Baseline)*
* **How it works**: Raw scores are normalized to a $[0.0, 1.0]$ range using Min-Max scaling, and added together directly for duplicate document IDs:
  $$\text{Final Score} = S_{\text{normalized\_dense}} + S_{\text{normalized\_sparse}}$$
* **Limitation**: Outlier sensitivity and equal weighting bias.

#### 2. Compensating for Basic Fusion: The Cross-Encoder Safeguard
In this project, we implement a **Two-Stage Multi-Stage Retrieval Architecture**:
1. **Stage 1 (High Recall)**: Hybrid search retrieves 10 candidates using Min-Max addition fusion.
2. **Stage 2 (High Precision Safeguard)**: All 10 candidates are passed to a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The Cross-Encoder evaluates full token-to-token cross-attention between query and passage, correcting any ranking errors from Stage 1 and selecting the top 3 highest-quality chunks.

---

## 6. Multi-Tenancy & Role-Based Vector Access Control

Enterprise RAG applications require strict data isolation across multiple organizations and users.
* **The Vulnerability**: Simple vector databases query all points globally, creating cross-tenant data leakage risks.
* **Our Solution**: Every vector point stored in Qdrant contains payload metadata:
  ```json
  {
    "tenant_id": "tenant_default",
    "user_id": "usr_admin",
    "source": "machine_learning.pdf"
  }
  ```
* **Payload Pre-Filtering**: When querying Qdrant, we apply hard Boolean filters *before* semantic similarity search:
  ```python
  Filter(
      must=[
          FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
          FieldCondition(key="user_id", match=MatchValue(value=user_id))
      ]
  )
  ```
  This guarantees that vectors outside the user's authorization are completely invisible to the search engine.

---

## 7. Quantization, Local Inference & Multi-Provider Architecture

### Quantization & GGUF
* **Quantization**: Compresses 16-bit float weights (`FP16`) to 4-bit integers (`Q4_K_M`), shrinking the 7B parameter Qwen model from 14GB+ to ~4.7GB RAM footprint.
* **Llama.cpp**: C++ inference engine exposing an OpenAI-compatible `/v1` endpoint on port 8080 with GPU offloading (`-ngl`).

### Multi-Provider Flexibility
In `rag_pipeline.py`, we implement a dynamic provider switch:
* **`LLM_PROVIDER=local`**: Routes to local llama.cpp (`Qwen2.5-7B`) at `$0.00` API cost and 100% offline privacy.
* **`LLM_PROVIDER=groq`**: Routes to Groq cloud API (`llama-3.3-70b-versatile`) for ultra-low latency server execution.

---

## 8. Server-Sent Events (SSE) Streaming & FinOps Observability

### Why SSE Streaming Matters:
Traditional blocking REST endpoints force the user to stare at a blank screen for 4–6 seconds while the entire response is synthesized. 
* **Server-Sent Events (SSE)**: Streams text tokens chunk-by-chunk in real time (`/chat/stream`), reducing perceived **Time-To-First-Token (TTFT)** to under **350ms**.

### FinOps Telemetry Drawer
The frontend tracks and displays real-time operational telemetry for every response:
* **TTFT (ms)**: Milliseconds from request dispatch until the first token appears.
* **Token Breakdown**: `Tokens: Total [Q: prompt_tokens | R: completion_tokens]`.
* **Cost Estimation (USD)**: Calculated dynamically using model pricing matrices (e.g. \$0.00 for local Qwen; \$0.59 / \$0.79 per 1M tokens for Llama-3.3-70B).

---

## 9. Enterprise RAG Quality Evaluation: The Zero-Cost Local RAG Triad

Traditional evaluation relies on **LLM-as-a-Judge** (asking GPT-4 or Groq to score responses). In production, this causes massive problems:
* **Cloud API Throttles (HTTP 429)**: Quota exhaustion when evaluating batch test suites.
* **Non-Determinism**: Generative judge scores drift due to sampling temperature.
* **The "Ground Truth Illusion"**: Relying on synthetic ground truths penalizes correct answers due to harmless lexical wording differences.

```
                                    ┌──────────────────────┐
                                    │    1. USER QUERY     │
                                    └──────────┬───────────┘
                                               │
                                 Answer        │  Context
                                 Relevancy     │  Relevance
                                 (Directness)  │  (Retrieval Signal)
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
              ┌─────────────────────┐   Faithfulness      ┌─────────────────────┐
              │ 2. GENERATED ANSWER │◄────────────────────│ 3. RETRIEVED CONTEXT│
              └─────────────────────┘   (Anti-Hallucination)└─────────────────────┘
```

### The Solution: The Reference-Free Local RAG Triad (`evaluations/fast_eval.py`)
1. **Faithfulness (Anti-Hallucination)**: Uses 768D Bi-Encoder embeddings (`BAAI/bge-base-en-v1.5`) to verify that each proposition in the generated answer is mathematically entailed within the retrieved context chunks ($\text{sim} \ge 0.60$). Zero external API calls, zero hallucination in grading.
2. **Answer Relevancy**: Computes cosine similarity between user query and generated answer vectors.
3. **Context Relevance**: Measures the signal-to-noise ratio of retrieved chunks against the query.
4. **Context Recall**: (Optional offline check) Evaluates coverage against gold-standard references in `test_dataset.py`.
* **Permanent Text Reporting**: Automatically saves human-readable audit logs to `evaluations/evaluation_report.txt` with timestamps, metrics, and PASS/WARN status.

---

## 10. LangSmith Tracing & Observability

The pipeline is wrapped with the `@traceable` decorator from LangSmith:
* Tracks nested latency breakdowns across retrieval, reranking, and generation.
* Logs complete payload prompts, chunk IDs, and token usage for visual debugging in the LangSmith dashboard.
