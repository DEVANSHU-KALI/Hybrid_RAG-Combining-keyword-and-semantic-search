# Technical Interview Prep: `17) interview_questions.md`

This document contains expected technical interview questions and comprehensive, developer-level answers regarding this Hybrid RAG project.

---

## Category 1: System Architecture and RAG Pipeline Flow

### Question 1: What is the overall architecture of your RAG application, and how does data flow through the system?
**Answer**: 
"My application is built on a **decoupled Client-Server Architecture** using a Multi-Stage Retrieval-Augmented Generation (RAG) pipeline.
* **Frontend**: A Streamlit interface (`app.py`) captures user queries and sends them asynchronously to the backend using `httpx`.
* **Backend Gateway**: A FastAPI server (`main.py`) exposes a `/chat` POST endpoint and validates the payload schema using a Pydantic model (`QueryRequest`).
* **Retrieval Stage**: The core orchestrator (`rag_pipeline.py`) performs a hybrid search (`hybrid_retriever.py`) that merges dense semantic search results from Qdrant (`semantic_retriever.py`) with sparse lexical results from a BM25 index (`bm25_retriever.py`).
* **Reranking Stage**: The hybrid search outputs the top 5 chunks, which are then re-scored using a Cross-Encoder model (`reranker.py`) to output the top 3 most relevant passages.
* **Generation Stage**: The orchestrator builds a strict prompt template enclosing the top 3 passages and submits it asynchronously to a local **llama.cpp** server (`llama-server.exe`) running a quantized `Qwen 2.5 7B` model, which generates the final grounded response."

---

## Category 2: Dense vs. Sparse Retrieval & Fusion

### Question 2: Why did you build a hybrid search engine instead of just using vector embeddings?
**Answer**: 
"Vector embeddings (dense retrieval) excel at capturing conceptual semantics and synonyms (e.g., matching 'overfitting' with 'memorizing training data' even if the exact word isn't present). However, they have weaknesses:
1. **Exact Matches**: They struggle with exact keyword lookups, product model numbers, specific user IDs, or code symbols.
2. **Short Queries**: Short queries have sparse semantic vectors that Cosine similarity can easily misalign.

By combining dense vectors with **BM25 (sparse retrieval)**, we capture both conceptual meaning and exact term matches. If a user queries a specific technical term like 'BM25Okapi', lexical search guarantees an exact term match, while dense search ensures conceptual coverage. This dual-path approach maximizes overall retrieval recall."

---

### Question 3: How does your score fusion mechanism work, and what are its mathematical limitations?
**Answer**: 
"Because dense similarity scores (e.g., Cosine similarity from `-1.0` to `1.0`) and sparse lexical scores (BM25 scores from `0.0` to $+\infty$) are on completely different mathematical scales, we cannot add them directly.
1. **Normalization**: We apply **Min-Max Normalization** to both sets of scores independently to scale them onto a comparable range between `0.0` and `1.0`:
   $$S_{\text{normalized}} = \frac{S_{\text{raw}} - S_{\text{min}}}{S_{\text{max}} - S_{\text{min}}}$$
2. **Fusion**: We aggregate scores by chunk ID using **Simple Addition Fusion**:
   $$\text{Final Score} = S_{\text{normalized\_dense}} + S_{\text{normalized\_sparse}}$$

**Mathematical Limitations**:
* **Outlier Sensitivity**: Min-Max normalization is highly sensitive to outliers. If one candidate has an extremely high BM25 score, it compresses the scores of all other candidates towards `0.0`, rendering the normalized differences negligible.
* **Equal Weight Bias**: Simple addition assumes both dense and sparse models are equally important. In reality, one model is often cleaner than the other depending on query types. In production, we would use **Weighted Linear Fusion** ($\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$) or a rank-based strategy like **Reciprocal Rank Fusion (RRF)**."

---

## Category 3: Code Reviews and Performance Tuning

### Question 4: There is a significant architectural bottleneck in the BM25 search path in your code. What is it, why is it bad, and how would you fix it in production?
**Answer**: 
"Yes. In `bm25_retriever.py`, on every user query, the script executes `client.scroll()` to download all document payloads (up to 1,000 items) from the Qdrant database, tokenizes them, and instantiates the `BM25Okapi` index in RAM *on the fly*.

**Why it is bad**:
This represents an $O(N)$ data transfer and index construction bottleneck. At scale (with millions of documents), downloading the entire database and rebuilding the index on every single API request will crash server memory, consume massive network bandwidth, and increase request latency to unacceptable levels.

**Production Fixes**:
1. **Qdrant Sparse Vectors**: Configure Qdrant's native sparse vector indices (e.g., using models like SPLADE). This allows both dense and sparse searches to run natively on Qdrant, returning fused results in a single network round-trip.
2. **Dedicated Search Engine**: Use a dedicated full-text search engine like Elasticsearch or OpenSearch.
3. **In-Memory Caching**: If using a local index is required, the BM25 index should be built *once* during document ingestion or startup, cached in memory, and updated incrementally using thread-safe write hooks."

---

## Category 4: Reranking and Context Optimization

### Question 5: What is the difference between a Bi-Encoder and a Cross-Encoder, and why do you use both in your pipeline?
**Answer**: 
"The difference lies in how they process queries and documents:
* **Bi-Encoders** (e.g., our dense embedding model): Encode the query and the documents independently into separate vectors. Because the documents are pre-embedded and indexed in our database, we can perform fast similarity calculations (like Cosine distance) in milliseconds. However, because the query and document do not interact during encoding, it is less precise at capturing fine-grained relationships.
* **Cross-Encoders** (e.g., our reranking model): Feed both the query and the document into the transformer network simultaneously. The self-attention layers compute token-to-token relationships between all query words and all document words. This yields highly accurate relevance scores but is computationally slow, making it impossible to query millions of documents in real-time.

**Why we use both**:
We implement a **Multi-Stage Retrieval** architecture. We use the Bi-Encoder (hybrid search) as a fast, high-recall filter to retrieve the top 5 candidates. Then, we pass only those 5 candidates to the Cross-Encoder Reranker. This gives us the speed of vector search combined with the high accuracy of Cross-Encoder attention."

---

### Question 6: Explain the concept of Semantic Chunking. How is it different from traditional recursive text splitting?
**Answer**: 
"Traditional text splitting (like `RecursiveCharacterTextSplitter`) cuts text at static character limits or token lengths (e.g., every 500 characters, trying to split at newlines or spaces). This runs the risk of cutting a paragraph mid-sentence or separating related concepts into different chunks.

**Semantic Chunking** uses vector embeddings to split text along logical semantic boundaries:
1. It splits the document into individual sentences.
2. It generates vector embeddings for each sentence using our embedding model.
3. It measures the Cosine distance between consecutive sentences.
4. It sets a threshold (e.g., the 75th percentile of all distance differences inside the document) and triggers a breakpoint (splits the text) only where the semantic difference between two adjacent sentences exceeds that threshold. 

This guarantees that each chunk contains a complete, semantically cohesive concept, raising our downstream retrieval quality."

---

## Category 5: Local LLM Execution & Quantization

### Question 7: How did you configure your LLM to run locally? Explain the role of Llama.cpp and quantization (GGUF).
**Answer**: 
"We run the Qwen 2.5 7B model locally on consumer hardware without cloud API dependencies.
* **Llama.cpp**: A high-performance inference engine compiled in C/C++ that executes LLMs locally. We run `llama-server.exe` on port `8080` using the OpenAI-compatible REST API endpoint `/v1`.
* **Quantization (GGUF)**: Running a 7B parameter model in FP16 precision requires over 14GB of VRAM. We use a **Q4_K_M quantized GGUF model**. Quantization maps model weights from 16-bit floating-point numbers to 4-bit integers. This reduces the file size to ~4.7GB, enabling the model to run smoothly on standard systems.
* **GPU Offloading**: We use the `-ngl 25` flag to offload 25 transformer layers to the GPU, while the CPU handles the remaining layers, optimizing system execution."

---
