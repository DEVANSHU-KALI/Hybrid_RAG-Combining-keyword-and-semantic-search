 Script Explanation: `11) rag_pipeline.md`

## 1. Overview
The primary role of the `rag_pipeline.py` script is to orchestrate the entire end-to-end **Retrieval-Augmented Generation (RAG)** process. When a user asks a question, this script coordinates the retrieval modules to find relevant data, formats that data into a strict prompt structure, sends the request to our locally running language model, and returns the final answer along with document citations and source contexts.

It implements:
* **Hybrid Retrieval**: Queries `hybrid_search()` to get fused semantic/keyword candidates.
* **Cross-Encoder Reranking**: Filters candidates down to the top 3 using `rerank_results()`.
* **Prompt Engineering**: Inserts candidate text into a strict prompt template that forces the model to answer only using the provided documents.
* **Local Inference Client**: Communicates via an OpenAI-compatible client with a locally hosted **llama.cpp** server running a quantized `Qwen 2.5 7B` model.
* **Observability**: Utilizes LangSmith's `@traceable` decorator to log and visualize execution paths.

---
