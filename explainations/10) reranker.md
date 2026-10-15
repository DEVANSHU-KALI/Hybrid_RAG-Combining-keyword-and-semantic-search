# Script Explanation: `10) reranker.md`

## 1. Overview
The primary role of `backend/reranker.py` is executing **Stage-2 Cross-Encoder Precision Reranking**. 

In production RAG systems, relying solely on Stage-1 retrieval (dense vector embeddings and BM25 keyword matching) creates a fundamental vulnerability: **false positive retrieval**. 
* Vector search compares query and document embeddings **independently** (Bi-Encoder architecture).
* BM25 matches keyword frequency without understanding context.

`reranker.py` acts as a deep precision filter that:
1. Takes the top candidate chunks (e.g. 10 candidates) retrieved by `hybrid_search`.
2. Pairs each candidate chunk directly with the user's raw query: `[query, chunk_text]`.
3. Passes these pairs through a specialized **Cross-Encoder Transformer Model** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
4. Computes full **token-to-token multi-head cross-attention** between the query and the document text simultaneously.
5. Re-scores and re-sorts candidates based on their deep contextual relevance, returning strictly the **top 3 highest precision chunks** for the LLM prompt.

---

## 2. Line-by-Line & Block-by-Block Code Walkthrough

```python
1: from sentence_transformers import CrossEncoder
2: 
3: # Load Cross-Encoder Model
4: reranker_model = CrossEncoder(
5:     "cross-encoder/ms-marco-MiniLM-L-6-v2"
6: )
```
* **Line 1 (`from sentence_transformers import CrossEncoder`)**: Imports the `CrossEncoder` class from the `sentence_transformers` library. Unlike standard sentence embedding models (which output 768D vectors), `CrossEncoder` outputs a direct scalar relevance score for text pairs.
* **Lines 4–6 (`reranker_model = CrossEncoder(...)`)**: Initializes the Cross-Encoder model in memory.
  * `"cross-encoder/ms-marco-MiniLM-L-6-v2"`: A 6-layer MiniLM transformer fine-tuned specifically on Microsoft's MS-MARCO passage retrieval benchmark (over 500,000 human search queries).
  * Runs locally on CPU or GPU with near-zero latency (~15–30ms for 10 candidates).

```python
14: async def rerank_results(query: str, retrieved_chunks: list, top_k: int = 3) -> list[dict]:
15:     if not retrieved_chunks:
16:         return []
```
* **Line 14 (`async def rerank_results(...)`)**: 
  * Defined as an `async` function so it can be called seamlessly inside our asynchronous FastAPI pipeline (`await rerank_results(...)`).
  * `top_k: int = 3`: Configurable parameter defining how many final chunks to return (default: 3).
* **Lines 15–16 (Defensive Empty Guard)**: If `retrieved_chunks` is empty (e.g., no documents match the tenant filter in Qdrant), immediately returns `[]` to prevent indexing errors downstream.

```python
18:     # 1. Create Query-Chunk Pairs
19:     pairs = [
20:         [query, chunk["text"]]
21:         for chunk in retrieved_chunks
22:     ]
```
* **Lines 19–22 (Pair Construction)**:
  * Uses a clean list comprehension to build a list of 2-element sub-lists: `[[query, text_1], [query, text_2], ...]`.
  * **Why pairs?** Unlike Bi-Encoders which embed the query alone and the document alone, a Cross-Encoder must receive both texts together so their tokens can cross-attend to one another.

```python
24:     # 2. Batch Cross-Encoder Inference
25:     scores = reranker_model.predict(pairs)
```
* **Line 25 (`reranker_model.predict(pairs)`)**:
  * Passes all pairs through the transformer in a single forward batch pass.
  * Internally concatenates each pair: `[CLS] User Query [SEP] Document Text [SEP]`.
  * Computes multi-head attention and outputs a 1D NumPy array of raw logit scores (e.g., `[4.82, -1.95, 2.14, ...]`).

```python
27:     # 3. Attach Reranker Scores
28:     for i in range(len(retrieved_chunks)):
29:         retrieved_chunks[i]["reranker_score"] = float(scores[i])
```
* **Lines 28–29 (Score Attachment & Type Casting)**:
  * Loops through `retrieved_chunks` and adds a new key: `"reranker_score"`.
  * **Why `float(scores[i])`?** `reranker_model.predict()` outputs values as NumPy `np.float32`. NumPy types are not JSON-serializable by FastAPI or Pydantic. Casting with `float()` converts them to standard Python floats, preventing serialization crashes.

```python
31:     # 4. Sort & Slice Top Precision Results
32:     reranked_results = sorted(
33:         retrieved_chunks,
34:         key=lambda x: x["reranker_score"],
35:         reverse=True
36:     )
37: 
38:     return reranked_results[:top_k]
```
* **Lines 32–36 (Descending Score Sort)**:
  * `sorted(...)`: Sorts the chunks based on the newly attached `"reranker_score"` using a lambda function key.
  * `reverse=True`: Ensures the highest-scoring candidate appears first.
* **Line 38 (`return reranked_results[:top_k]`)**:
  * Slices the sorted candidate list and returns only the top `top_k` chunks (top 3).
  * These 3 elite chunks form the grounding context injected into the final LLM prompt.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
              Inputs: User Query & Top 10 Candidate Chunks
                                  │
                                  ▼
                     Empty Input Guard Check
                     ├── If empty ──► Return []
                     └── If chunks exist ──► Proceed
                                  │
                                  ▼
                     Construct [Query, Text] Pairs
                  [[Q, Text 1], [Q, Text 2], ... [Q, Text 10]]
                                  │
                                  ▼
                   Batch Cross-Encoder Inference
             Concatenates: [CLS] Q [SEP] Text [SEP]
             Multi-Head Token Cross-Attention Layers
                                  │
                                  ▼
                        Raw Logit Scores
                   [3.85, -2.10, 5.42, 0.15, ...]
                                  │
                                  ▼
                   Attach Scores to Chunk Dicts
              (Cast np.float32 ──► Python float)
                                  │
                                  ▼
                     Sort Chunks Descending
                     (Highest Score First)
                                  │
                                  ▼
                     Slice Top K ([:top_k])
                    Output: Top 3 Chunks
```

---

### Input and Output Specifications
* **Inputs**:
  * `query` (`str`): The raw user search prompt (e.g., `"What was Acme Corp's operating margin in Q3 2024?"`).
  * `retrieved_chunks` (`list[dict]`): A list of candidate chunks returned by Stage-1 `hybrid_search` (typically 10 candidates).
  * `top_k` (`int`, default: `3`): The number of final chunks to return.
* **Output**: A list containing up to `top_k` dictionary objects (`list[dict]`), sorted in descending order by `"reranker_score"`. Each dictionary contains:
  * `"text"`: The document chunk text.
  * `"source"`: Origin filename.
  * `"chunk_id"`: Database identifier.
  * `"tenant_id"` & `"user_id"`: Tenant and user security tags.
  * `"semantic_score"` & `"bm25_score"`: Stage-1 retrieval scores.
  * `"final_score"`: Stage-1 hybrid score.
  * `"reranker_score"`: Stage-2 Cross-Encoder logit score.

---

### Step-by-Step Variable Trace Walkthrough

Let's trace an execution with a real enterprise example:

#### 1. Input State
* **`query`**: `"What was Acme Corp's operating margin in Q3 2024?"`
* **Stage-1 Retrieved Chunks (Candidates from Hybrid Search)**:
  * **Chunk A (ID 12)**: `"In Q3 2024, Acme Corp achieved an operating margin of 18.4%, up 210 basis points."` (Hybrid Score: 1.85)
  * **Chunk B (ID 44)**: `"Acme Corp manufactures industrial widgets across European manufacturing plants."` (Hybrid Score: 1.62 — *Keyword match on "Acme Corp", but irrelevant to margin!*)
  * **Chunk C (ID 13)**: `"Total Q3 2024 operating income reached $42.5M, reflecting margin expansion."` (Hybrid Score: 1.45)
  * **Chunk D (ID 88)**: `"In Q3 2023, the previous operating margin was 16.3% before restructuring."` (Hybrid Score: 1.40 — *Mentions operating margin, but wrong year!*)

#### 2. Pair Construction
```python
pairs = [
    ["What was Acme Corp's operating margin in Q3 2024?", "In Q3 2024, Acme Corp achieved an operating margin of 18.4%..."],
    ["What was Acme Corp's operating margin in Q3 2024?", "Acme Corp manufactures industrial widgets across European..."],
    ["What was Acme Corp's operating margin in Q3 2024?", "Total Q3 2024 operating income reached $42.5M..."],
    ["What was Acme Corp's operating margin in Q3 2024?", "In Q3 2023, the previous operating margin was 16.3%..."]
]
```

#### 3. Cross-Encoder Model Inference (`predict(pairs)`)
The model evaluates token cross-attention:
* **Pair 1 (Chunk A)**: `"Q3 2024"` matches `"Q3 2024"`, and `"operating margin"` matches `"operating margin of 18.4%"`. Token attention aligns perfectly.  
  $\rightarrow$ **Logit Score: `+5.82`** (Extremely high relevance)
* **Pair 2 (Chunk B)**: Only `"Acme Corp"` matches; no tokens match `"operating margin"` or `"Q3 2024"`. The model severely penalizes it.  
  $\rightarrow$ **Logit Score: `-3.45`** (Irrelevant false positive eliminated!)
* **Pair 3 (Chunk C)**: `"Q3 2024"` and `"margin expansion"` align logically with operating margin.  
  $\rightarrow$ **Logit Score: `+3.10`** (Strong supporting context)
* **Pair 4 (Chunk D)**: Matches `"operating margin"`, but the model's self-attention recognizes `"Q3 2023"` conflicts with `"Q3 2024"`.  
  $\rightarrow$ **Logit Score: `+0.85`** (Weak relevance due to year mismatch)

#### 4. Updating Chunks & Sorting
Chunks are re-ranked based on `"reranker_score"`:
```python
[
    {"chunk_id": 12, "reranker_score": 5.82, "text": "In Q3 2024, Acme Corp achieved an operating margin of 18.4%..."},
    {"chunk_id": 13, "reranker_score": 3.10, "text": "Total Q3 2024 operating income reached $42.5M..."},
    {"chunk_id": 88, "reranker_score": 0.85, "text": "In Q3 2023, the previous operating margin was 16.3%..."},
    {"chunk_id": 44, "reranker_score": -3.45, "text": "Acme Corp manufactures industrial widgets..."} # Dropped!
]
```

#### 5. Slicing Top 3
`[:top_k]` returns Chunks 12, 13, and 88. Notice how **Chunk 44** (which scored high in Stage-1 due to repeated keyword matches) was successfully filtered out by the Cross-Encoder!

---

## 4. Deep Technical Concepts

### 1. Bi-Encoder vs. Cross-Encoder Architecture

```
                 BI-ENCODER (Stage 1: Qdrant)
 Query ──► [Transformer] ──► Vector Q ──┐
                                        ├── Cosine Angle (Dot Product)
 Doc   ──► [Transformer] ──► Vector D ──┘
 (Query and Document NEVER see each other during encoding)

                CROSS-ENCODER (Stage 2: Reranker)
 [CLS] Query [SEP] Document [SEP] ──► [Transformer with Cross-Attention] ──► Logit Score
 (Every Query token computes self-attention with every Document token)
```

| Metric | Bi-Encoder (Embedding Model) | Cross-Encoder (Reranker) |
| :--- | :--- | :--- |
| **Input** | Query and Document encoded **separately** | Query and Document concatenated **together** |
| **Token Interaction** | ❌ None (Vectors compared via dot product) | ✅ Full Multi-Head Self-Attention across all tokens |
| **Speed** | $O(\log N)$ via HNSW index (1–5ms for millions of vectors) | $O(N)$ transformer forward pass (~15–30ms for 10 pairs) |
| **Accuracy** | Good for broad recall; misses subtle logic/negations | State-of-the-art precision; understands exact phrasing and constraints |
| **Scalability** | Can search 10,000,000 documents | Cannot search millions; only viable over top 10–50 candidates |

---

### 2. Multi-Stage Retrieval Pattern (Funnel Architecture)

To achieve both **sub-50ms speed** and **human-level precision**, enterprise search uses a two-stage funnel:

```
  Stage 1: High Recall (Hybrid Search: 768D Vector + BM25)
  Database: 1,000,000 Chunks  ──►  Retrieves Top 10 Candidates (Latency: ~20ms)
                                           │
                                           ▼
  Stage 2: High Precision (Cross-Encoder Reranker)
  Filters: 10 Candidates     ──►  Selects Top 3 Elite Chunks   (Latency: ~25ms)
                                           │
                                           ▼
  LLM Generation Context: Strictly Top 3 Grounded Chunks!
```

---

### 3. Why Can Reranker Scores Be Negative?
Unlike Cosine Similarity (which ranges from `-1.0` to `+1.0`) or normalized BM25 (which ranges from `0.0` to `1.0`), Cross-Encoder outputs are **unnormalized classification logits**:
* **Positive Logits ($> 0.0$):** The model predicts the document is relevant to the query (e.g. `+3.8` to `+8.5`).
* **Negative Logits ($< 0.0$):** The model predicts the document is irrelevant or a false positive distractor (e.g. `-2.1` to `-6.0`).
* Because sorting only cares about relative ranking (`reverse=True`), raw logits do not need sigmoid conversion to preserve order.

---

## 5. Architectural Choices and Alternatives

### Why `cross-encoder/ms-marco-MiniLM-L-6-v2`?
* **Architecture:** 6-layer MiniLM transformer with 22.7 million parameters.
* **Footprint:** Requires only ~80MB of RAM/VRAM.
* **Inference Speed:** Evaluates 10 candidate pairs in **under 25 milliseconds** on standard laptop CPUs without requiring dedicated GPU hardware.

#### Model & Service Comparison Matrix

| Reranker Solution | Architecture | Typical Latency | Cost | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **MiniLM-L-6-v2** *(Chosen)* | 6-layer Cross-Encoder | **~15–25ms** (Local CPU) | **$0.00** (Free, Open-Source) | Production CPU deployments, edge devices, high-throughput pipelines. |
| **BAAI/bge-reranker-base** | 12-layer RoBERTa Cross-Encoder | ~50–80ms (GPU recommended) | **$0.00** (Free, Open-Source) | Complex technical documents requiring deeper transformer layers. |
| **Cohere Rerank v3 API** | Proprietary cloud model | ~150–350ms (Network API) | ~$1.00 / 1,000 queries | Cloud-first setups with zero on-prem compute. |
| **ColBERTv2 (Late Interaction)**| Token-level MaxSim | ~10–20ms | $0.00 (Self-hosted) | Large candidate pools (top 100) requiring vector-speed cross-interaction. |
| **LLM-as-a-Ranker (e.g. GPT-4)**| Autoregressive LLM Prompt | ~1,500–3,000ms | High API token cost | Research experiments; far too slow and expensive for live user search. |
