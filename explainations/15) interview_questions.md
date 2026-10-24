# Technical Interview Guide: `17) interview_questions.md`

This document contains high-impact technical interview questions, architecture trade-offs, and senior developer-level answers regarding this Multi-Tenant Hybrid RAG platform.

---

## Category 1: System Architecture & Data Flow

### Question 1: What is the overall architecture of your RAG application, and how does data flow from user query to final streaming response?
**Answer**: 
"My application is built on a **decoupled Client-Server Architecture** utilizing a Multi-Stage Retrieval-Augmented Generation (RAG) pipeline:
* **Frontend Layer**: A Streamlit interface (`app.py`) captures user queries and multi-tenant parameters (`tenant_id`, `user_id`), establishing an asynchronous Server-Sent Events (SSE) connection to the backend via `httpx`.
* **API Gateway**: A FastAPI backend (`main.py`) exposes `/chat/stream` and `/chat`, enforcing schema validation using Pydantic (`QueryRequest`).
* **Retrieval Stage**: The core orchestrator (`rag_pipeline.py`) coordinates a **Two-Stage Hybrid Search**:
  1. **Stage 1 (High Recall)**: Merges dense 768D semantic vectors (`BAAI/bge-base-en-v1.5` in Qdrant) with sparse lexical BM25 matching, applying multi-tenant payload pre-filters to fetch 10 candidate chunks.
  2. **Stage 2 (High Precision Safeguard)**: Passes all 10 candidates to a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) which uses full token-to-token cross-attention to output the top 3 highest-signal passages.
* **Generation & Streaming Stage**: The orchestrator injects the top 3 chunks into an enterprise prompt enforcing strict factual grounding and Markdown table formatting. It dispatches the prompt to a multi-provider LLM (local quantized `Qwen 2.5 7B` via llama.cpp on port 8080 or Groq Cloud), streaming tokens live back to the UI along with real-time FinOps telemetry (TTFT, token split, and USD cost)."

---

## Category 2: Ingestion, Layout Parsing & Tabular Data Integrity

### Question 2: How do you handle complex PDF layouts and tables without smashing inline text or splitting table rows across chunks?
**Answer**: 
"Traditional PDF text extractors (like `pypdf` or naive OCR) extract glyphs sequentially across coordinates, which smashes table rows into unformatted inline prose and creates duplicate chunks. We resolved this through a 4-pillar ingestion architecture in `backend/injest_documents.py`:
1. **Geometric Bounding-Box Detection**: We use `pdfplumber.find_tables()` to identify the exact bounding box `(x0, top, x1, bottom)` of every table on a page.
2. **Cropped Prose Extraction**: We use `page.crop()` to extract narrative prose strictly *outside* table regions. Prose above the table and prose below the table are extracted independently, completely preventing table text from leaking into narrative paragraphs.
3. **In-Place Markdown Assembly**: Tables are converted into clean GitHub-flavored Markdown tables and inserted at their true reading position between the cropped prose blocks.
4. **Dynamic Chunk-Level Metadata**: Instead of inheriting coarse page-level booleans (which caused pure text chunks to falsely claim `"has_table": true`), each chunk is dynamically inspected for Markdown table rows (`| ... |`). Furthermore, the chunker uses pre-table split isolation: if a table is encountered and the running chunk already holds $>500$ characters, it splits immediately before the table so the table remains intact in a fresh chunk."

---

## Category 3: Dense vs. Sparse Retrieval & Score Fusion

### Question 3: Why build a hybrid search engine instead of relying purely on vector embeddings?
**Answer**: 
"Dense vector embeddings excel at semantic conceptual understanding and synonyms (e.g. matching 'overfitting' with 'memorizing training data' even when keywords differ). However, dense vectors suffer from two major production flaws:
1. **Exact Lexical Failure**: They struggle with exact technical identifiers, product model numbers, acronyms, or specific error codes.
2. **Short Query Sparsity**: Short queries produce sparse semantic representations that can easily drift in high-dimensional vector space.

By pairing 768D dense retrieval with **sparse lexical retrieval (Okapi BM25)**, we achieve the best of both worlds: BM25 guarantees precision on exact keyword matches, while dense vectors capture conceptual context. Combining them provides higher retrieval recall than either system could achieve independently."

---

### Question 4: How does your score fusion mechanism work, and what are its mathematical limitations?
**Answer**: 
"Because dense Cosine similarity ($[-1.0, 1.0]$) and sparse BM25 scores ($[0.0, +\infty)$) operate on incompatible numerical scales, we cannot add them directly.
1. **Min-Max Normalization**: We scale each list independently to $[0.0, 1.0]$:
   $$S_{\text{norm}} = \frac{S_{\text{raw}} - S_{\text{min}}}{S_{\text{max}} - S_{\text{min}}}$$
2. **Simple Addition Fusion**: We sum normalized scores for duplicate chunk IDs:
   $$\text{Final Score} = S_{\text{dense\_norm}} + S_{\text{sparse\_norm}}$$

**Limitations & Compensating Safeguard**:
Min-Max normalization is sensitive to extreme score outliers, and simple addition assumes equal weighting. In a single-stage system, this would cause ranking noise. However, in our system, **Stage 1 is purely a high-recall filter (retrieving 10 candidates)**. The entire list is immediately passed to our **Cross-Encoder Reranker**, which completely re-scores candidate relevance using multi-head cross-attention. This gives us the simplicity and speed of basic fusion during database retrieval while guaranteeing high precision via reranking."

---

## Category 4: Multi-Stage Retrieval & Cross-Encoder Reranking

### Question 5: What is the computational and architectural difference between a Bi-Encoder and a Cross-Encoder?
**Answer**: 
"The fundamental difference is in how token attention is calculated:
* **Bi-Encoders** (e.g., our dense embedding model `BAAI/bge-base-en-v1.5`): Encode queries and document passages independently into separate 768-dimensional vectors. Because documents are pre-indexed in Qdrant, retrieval is instantaneous (sub-5ms) using vector dot products. However, because query tokens and document tokens never interact inside transformer layers, fine-grained semantic nuance is missed.
* **Cross-Encoders** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`): Concatenate query and document into a single sequence:
  $$\text{Input} = \text{[CLS]} \, \text{Query} \, \text{[SEP]} \, \text{Passage} \, \text{[SEP]}$$
  Every query token attends directly to every passage token across all transformer layers. This produces unmatched relevance scoring accuracy, but is computationally expensive ($O(L^2)$ complexity per pair).

**Why we use both**:
We use Bi-Encoders for fast approximate candidate retrieval across thousands of chunks, and use the Cross-Encoder as a second-stage filter on only the top 10 candidates to select the final top 3 context chunks."

---

## Category 5: Multi-Tenancy & Data Security

### Question 6: How does your RAG system guarantee strict Multi-Tenancy and prevent cross-organization data leakage?
**Answer**: 
"In enterprise RAG, data leakage across client boundaries is a critical compliance liability. We prevent this using **Database-Level Payload Pre-Filtering in Qdrant**:
1. During ingestion, every vector chunk is stored with structured metadata: `tenant_id` and `user_id`.
2. When a search request arrives, Qdrant executes a boolean payload filter *before* conducting the semantic vector search:
   ```python
   Filter(
       must=[
           FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
           FieldCondition(key="user_id", match=MatchValue(value=user_id))
       ]
   )
   ```
3. Because filtering occurs at the database index layer, vector points belonging to other organizations are completely invisible to the nearest-neighbor algorithm. This eliminates data leakage at the hardware query level rather than relying on weak prompt instructions."

---

## Category 6: Local LLM Execution, Quantization & Multi-Provider Architecture

### Question 7: How is your local model configured, and how does your architecture handle multi-provider fallback?
**Answer**: 
"We run the 7-billion parameter `Qwen 2.5 7B Instruct` locally using **llama.cpp** and **GGUF quantization**:
* **GGUF Q4_K_M Quantization**: Compresses FP16 16-bit float weights into 4-bit integers. This shrinks model memory from 14GB+ to ~4.7GB, allowing high-speed inference on consumer hardware.
* **GPU Offloading**: Using `llama-server.exe` with `-ngl 25`, we offload 25 transformer layers to GPU VRAM, while CPU handles the remaining layers.
* **Multi-Provider Toggle (`LLM_PROVIDER`)**: In `rag_pipeline.py`, we implement a flexible provider switch:
  - `LLM_PROVIDER=local`: Routes to `http://localhost:8080/v1` for 100% offline privacy and \$0.00 token cost.
  - `LLM_PROVIDER=groq`: Routes to Groq cloud API (`llama-3.3-70b-versatile`) when ultra-low latency cloud execution is required."

---

## Category 7: Streaming, UX & FinOps Observability

### Question 8: Why did you implement Server-Sent Events (SSE) streaming, and how do you track FinOps telemetry?
**Answer**: 
"Traditional blocking HTTP endpoints force users to wait 4–6 seconds for full generation, creating a sluggish user experience.
* **SSE Streaming (`/chat/stream`)**: We yield text chunks in real time using Server-Sent Events. This drops perceived **Time-To-First-Token (TTFT)** to under **350ms**.
* **3-Stage SSE Protocol**:
  1. `data: {"type": "metadata"}`: Transmits citations and chunk IDs immediately (<150ms) so the UI can render source badges before text starts generating.
  2. `data: {"type": "token"}`: Streams individual token deltas live into the Streamlit container.
  3. `data: {"type": "telemetry"}`: Transmits final performance statistics once generation finishes.
* **FinOps Telemetry Drawer**: Displays TTFT in ms, exact token breakdown (`Tokens: Total [Q: prompt_tokens | R: completion_tokens]`), and dynamically calculates estimated USD query cost using pricing matrices."

---

## Category 8: Enterprise Evaluation & Overcoming the "Ground Truth Illusion"

### Question 9: Why is traditional LLM-as-a-Judge problematic in production, and how did you design a zero-cost local evaluation suite?
**Answer**: 
"Traditional evaluation uses an LLM (like GPT-4) to grade answers. In production, this causes 4 fatal issues:
1. **Cloud Rate Limits (HTTP 429)**: Batch test suites quickly breach TPM/RPM quotas.
2. **Non-Determinism**: Generative judges drift due to temperature sampling.
3. **High Latency & Cost**: Evaluating 7 questions across 4 metrics requires 28 sequential LLM calls, taking 3–5 minutes and incurring recurring API bills.
4. **The Ground Truth Illusion**: When reference ground truths are written by an LLM, lexical phrasing mismatches cause the judge to penalize correct answers that simply used different words.

**Our Solution (`evaluations/fast_eval.py`)**:
We designed a **Zero-Cost ($0.00), Sub-Second Reference-Free RAG Triad**:
* **Faithfulness**: Uses 768D Bi-Encoder embeddings (`BAAI/bge-base-en-v1.5`) to compute a proposition-to-chunk containment matrix:
  $$\text{sim\_matrix} = \text{prop\_embs} \cdot \text{ctx\_embs}^T$$
  Checks whether every sentence in the answer is entailed by at least one retrieved chunk ($\text{sim} \ge 0.60$).
* **Answer Relevancy**: Computes cosine similarity between user query and generated answer vectors.
* **Context Relevance**: Measures the signal-to-noise ratio of retrieved chunks against the query.
* **Output**: Executes the entire 7-query benchmark in **under 2 seconds** on CPU, producing both Excel spreadsheets and permanent timestamped audit logs in `evaluations/evaluation_report.txt`."

---

### Question 10: What is the difference between Load Testing (`locust_load_test.py`) and Quality Evaluation (`fast_eval.py`)?
**Answer**: 
"They evaluate two completely orthogonal dimensions of enterprise readiness:
* **Quality Evaluation (`fast_eval.py` / `ragas_eval.py`)**: Tests **semantic accuracy, faithfulness, and retrieval precision**. It answers: *'Are the answers accurate, grounded in context, and free of hallucinations?'*
* **Load & Stress Testing (`locust_load_test.py`)**: Tests **system concurrency, throughput, and hardware capacity**. Using Locust, it simulates 50 to 200 concurrent users hitting the FastAPI `/chat/stream` endpoint to determine server requests-per-second (RPS), memory saturation, and crash limits under heavy traffic."

---

### Question 11: There is an architectural bottleneck in the BM25 search path in your code. What is it, and how would you optimize it at enterprise scale?
**Answer**: 
"In `bm25_retriever.py`, on every user query, the script calls `client.scroll()` to download document payloads from Qdrant, tokenizes them, and builds the `BM25Okapi` index in RAM on the fly.

**Why it is a bottleneck**:
At scale (with millions of chunks), downloading the entire corpus on every query creates an $O(N)$ data transfer bottleneck that will saturate network bandwidth, exhaust RAM, and degrade latency.

**Production Fixes**:
1. **Qdrant Native Sparse Vectors**: Use Qdrant's native sparse vector search (e.g. using SPLADE or BM25 embeddings). This allows dense and sparse searches to execute simultaneously inside Qdrant in a single network call.
2. **Dedicated Search Cluster**: Offload lexical indexing to a dedicated search cluster like Elasticsearch or OpenSearch.
3. **In-Memory Persistent Cache**: Build the BM25 index once during startup or ingestion, cache it in shared memory (e.g. Redis), and apply incremental update hooks when new documents are indexed."
