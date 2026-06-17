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