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