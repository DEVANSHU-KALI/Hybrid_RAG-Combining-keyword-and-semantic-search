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

### Prompt Engineering and LLM Inference
```python
    # Final Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}
"""
```

- **Lines 51–66**: We construct a strict **system prompt template**. We embed the retrieved context and user question, instructing the model to answer *only* using the context and to output a specific fallback error message if the answer is missing. This prevents the LLM from hallucinating answers based on its generic training data.

```python
    # Generate LLM Response
    response = await client.chat.completions.create(
        model="raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract Final Answer
    answer = response.choices[0].message.content

    # Return Final Response
    return {
        "answer": answer,
        "citations": citations,
        "contexts": [
        chunk["text"]
        for chunk in reranked_chunks
        ]
    }
```

- **Lines 69–90**:
  - We call `client.chat.completions.create()` to submit the prompt.
  - We request the model `"raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF"`, which matches the Qwen 2.5 7B model running locally via our llama.cpp server.
  - We extract the string answer from the response object and return a structured dictionary containing the answer, deduplicated citation filenames, and raw context strings.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     Input Query: "What is overfitting?"
                                │
                                ▼
                       hybrid_search(query)
                      [Gets Top Candidates]
                                │
                                ▼
                     rerank_results(query, list)
                       [Filters to Top 3]
                                │
                                ▼
                   Deduplicate Source Citations
                     (using set -> list)
                                │
                                ▼
                    Build Prompt Context block
                                │
                                ▼
                    Build Instructions Prompt
                                │
                                ▼
                 Async API call to localhost:8080
                  (Sends prompt using "dummy" key)
                                │
                                ▼
                     llama.cpp Server Receives
                    (Inference on Qwen 2.5 Q4)
                                │
                                ▼
                    Extract Answer & Return
```

---

### Input and Output Specifications
* **Input**: `query` (Type: `str`) - The plain-text user query (e.g., `"what is overfitting?"`).
* **Output**: A dictionary object (Type: `dict`) containing:
  * `"answer"`: The text response generated by the local Qwen model.
  * `"citations"`: A deduplicated list of source filenames.
  * `"contexts"`: A list containing the raw text strings of the 3 retrieved document chunks.

---

### Step-by-Step Variable Trace Walkthrough
Let's trace a run for the query: `"what is overfitting?"`.

1. **Query Input**: `query = "what is overfitting?"`.
2. **Retrieval**: `hybrid_search` returns 5 candidates.
3. **Reranking**: `rerank_results` scores candidates and outputs top 3.
   - Chunk 1 text: `"Overfitting occurs when a model memorizes..."` source: `"concepts.txt"`
   - Chunk 2 text: `"If a model is overfit, it fails on new data."` source: `"concepts.txt"`
   - Chunk 3 text: `"Overfitting can be reduced with dropout."` source: `"techniques.txt"`
4. **Context Synthesis**: Chunks are joined into `context = "Overfitting occurs... \n\n If a model is overfit... \n\n Overfitting can be..."`.
5. **Citations Filtering**: Chunks contain sources `["concepts.txt", "concepts.txt", "techniques.txt"]`.
   - `set(["concepts.txt", "concepts.txt", "techniques.txt"])` evaluates to `{"concepts.txt", "techniques.txt"}`.
   - `list(...)` evaluates to `["concepts.txt", "techniques.txt"]`.
6. **Prompt Assembly**: The context and query are inserted into the prompt template.
7. **Local LLM Call**: The client submits the prompt to `http://localhost:8080/v1` targeting model `raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF`.
   - The local llama.cpp server processes the request.
   - Qwen 2.5 uses the context to formulate an answer: `"Overfitting is when a model memorizes training data. It can be reduced using dropout."`
8. **Extraction & Return**: The answer is parsed and returned along with the citations list `["concepts.txt", "techniques.txt"]`.

---

## 4. Deep Technical Concepts

### Why Use a Local Model for RAG?
Using a local, self-hosted LLM (rather than a cloud service like OpenAI's GPT-4) has multiple engineering advantages:
1. **Model Knowledge Restrictions**: RAG systems rely on the LLM extracting information *only* from the provided context. Large public models have vast general knowledge, which makes them prone to ignoring instructions and hallucinating answers based on their pre-training. Local models with smaller parameter sizes are often more obedient to direct system instructions.
2. **Data Privacy**: Documents are processed entirely on your local machine, ensuring no sensitive data is sent to external servers.
3. **Cost and Independence**: No API subscription or key management is required.

### Quantization & GGUF Format
Running a modern 7-billion parameter language model in 16-bit precision requires over 14GB of video RAM (VRAM) just to load, plus extra memory for processing text.
* **Quantization**: A compression technique that converts the model's weights from 16-bit floating-point numbers (`FP16`) to lower precision configurations, such as 4-bit integers (`Q4`). 
* **Q4_K_M Quantization**: Reduces the model file size from ~14GB to ~4.7GB, enabling it to run smoothly on standard consumer computers with 8GB or 16GB of RAM.
* **GGUF File Format**: A file format optimized for fast loading and running on consumer hardware using CPU, GPU, or split CPU/GPU configurations.

### Llama.cpp & Local Server Launch
Llama.cpp is an open-source C/C++ execution engine designed to run quantized GGUF models locally with high performance.
* **Setup Process**:
  1. Download the executable zip folder from the official releases page of the **llama.cpp** GitHub repository. (Choose the CUDA GPU version if your system has an NVIDIA graphics card; otherwise, download the CPU version).
  2. Unzip the folder and open the command prompt (`cmd`) in that directory.
  3. Run the following command to download the model from Hugging Face and start the server:
     ```cmd
     .\llama-server.exe -hf raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF -ngl 25
     ```
     * `-hf`: Instructs llama.cpp to fetch the model directly from Hugging Face's registry.
     * `-ngl 25`: Stands for "Number of GPU Layers". This offloads 25 layers of the network to the GPU (leaving the rest to be handled by the CPU), optimizing performance.
  4. The server runs on `http://localhost:8080`, exposing an OpenAI-compatible API endpoint on port `8080/v1` and a web chat dashboard interface.
  5. To change the model, just replace the model name "raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF" with any other model from hugging face, as the command is for -hf.

---

## 5. Architectural Choices and Alternatives

### Why llama.cpp via OpenAI SDK?
By directing the standard OpenAI SDK to our local llama.cpp endpoint, we keep our code modular. If we decide to swap our local model for a commercial cloud LLM (like GPT-4o-mini) in the future, we only need to change the `base_url` and `api_key` variables—the rest of our RAG pipeline logic remains identical.

#### Alternatives and Trade-offs

| Inference Server | Strategy | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **llama.cpp** *(Chosen)* | C++ compiled local server. | • High speed on consumer hardware.<br>• Minimal dependencies.<br>• Easy split CPU/GPU offloading. | • Less optimized for concurrent multi-user production load. |
| **Ollama** | Background service manager for local models. | • Extremely user-friendly interface.<br>• Automatic model downloads and updates. | • Higher abstraction (makes it harder to tune specific model loading parameters). |
| **OpenAI Cloud API** | Cloud-based SaaS model. | • Zero local hardware constraints.<br>• World-class intelligence. | • High costs.<br>• Network latency.<br>• Data privacy issues. |
| **vLLM** | Enterprise-grade local server. | • Extreme throughput and speed via PagedAttention. | • Requires enterprise Linux environment and powerful NVIDIA GPUs. |
