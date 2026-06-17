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

## 2. Code Walkthrough

### Imports and Client Configuration
```python
from openai import AsyncOpenAI

from .hybrid_retriever import hybrid_search
from .reranker import rerank_results

from langsmith import traceable

# OpenAI Client
client = AsyncOpenAI(base_url="http://localhost:8080/v1",api_key="dummy")
```
- **Lines 1–9**:
  - We import `AsyncOpenAI`, which is the standard SDK (Software Development Kit) used to call OpenAI services asynchronously.
  - We import our retrieval orchestrations (`hybrid_search`, `rerank_results`) and the `traceable` decorator from the `langsmith` library.
  - **Line 9 (The Local Client Connection)**: We initialize the `AsyncOpenAI` client pointing to a local address: `base_url="http://localhost:8080/v1"` with `api_key="dummy"`.
    - *What is an API?* An API (Application Programming Interface) is a standardized set of rules that allows different software systems to talk to each other.
    - *Why `api_key="dummy"`?* We are running a local model server (llama.cpp) on our own computer, which does not require cloud authentication or API keys. However, the OpenAI Python library is programmed to throw an error if the `api_key` argument is empty. Passing `"dummy"` satisfies the SDK's validation check while routing requests to our local server.

---

### Core Pipeline Orchestration
```python
# Main RAG Pipeline
@traceable
async def generate_answer(query: str):
```
- **Lines 12–13**: 
  - We attach the `@traceable` decorator. This links our function execution to **LangSmith** (an LLM application monitoring platform), logging performance metrics, token usage, and retrieval latency.
  - We define the asynchronous function `generate_answer(query: str)`.

```python
    # Hybrid Retrieval
    retrieved_chunks = await hybrid_search(query)

    # Reranking
    reranked_chunks = await rerank_results(
        query,
        retrieved_chunks
    )
```

- **Lines 15–22**:
  - We retrieve normalized semantic and BM25 candidate chunks by awaiting `hybrid_search(query)`.
  - We pass these candidates to `rerank_results(query, retrieved_chunks)` to get the top 3 most relevant context blocks.

```python
    # Print Retrieved Sources
    print("\n===== FINAL RETRIEVED CHUNKS =====\n")

    for chunk in reranked_chunks:
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Reranker Score: {chunk['reranker_score']}")
        print("\n")
```
- **Lines 25–31**: Prints metadata and scores of the top 3 documents to the console.

---

### Context and Citation Building
```python
    # Build Context
    context = "\n\n".join(
        [
            chunk["text"]
            for chunk in reranked_chunks
        ]
    )

    # Build Citations
    citations = list(
        set(
            [
                chunk["source"]
                for chunk in reranked_chunks
            ]
        )
    )
```
- **Lines 34–49**:
  - We compile our LLM context by joining the text of the top 3 reranked chunks using double newlines (`\n\n`).
  - We extract the source filenames of the chunks. We wrap the extraction inside `set()` to remove duplicates (e.g., if multiple chunks came from the same file) and convert it back to a standard Python `list()`.

---