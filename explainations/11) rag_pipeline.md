# Script Explanation: `11) rag_pipeline.md`

## 1. Overview
The primary role of `backend/rag_pipeline.py` is orchestrating the complete end-to-end RAG workflow. It glues together the entire system:
1. **Hybrid Retrieval (`hybrid_search`)**: Fetches candidate chunks using both 768D semantic search and BM25 keyword matching with dual tenant and user filters.
2. **Cross-Encoder Reranking (`rerank_results`)**: Deeply scores query-chunk pairs with `bge-reranker-base` to distill down to the top 3 most relevant chunks.
3. **Prompt & Context Assembly**: Formats strict grounding instructions so the LLM answers strictly from retrieved documents.
4. **Server-Sent Events (SSE) Token Streaming (`stream_rag_answer`)**: Streams individual tokens in real-time as an asynchronous generator.
5. **Time-To-First-Token (TTFT) Telemetry**: Measures the exact millisecond latency until the user sees the first word.
6. **FinOps Token Cost Economics (`calculate_query_cost`)**: Calculates the financial cost per query based on prompt and completion token counts.

---

## 2. Deep Line-by-Line Code Walkthrough

---

### Part A: Client Setup & Provider Configuration (Lines 1–25)

```python
1: import os
2: import json
3: import time
4: from typing import AsyncGenerator
5: from openai import AsyncOpenAI
6: from langsmith import traceable
8: from .hybrid_retriever import hybrid_search
9: from .reranker import rerank_results
```
* **Lines 1–6**: Standard libraries for environment variables, JSON serialization, wall-clock timing, typing definitions, and the official `AsyncOpenAI` client SDK.
* **Line 6 (`@traceable`)**: LangSmith decorator that records traces for evaluation, debugging, and latency tracking.
* **Lines 8–9**: Imports hybrid retrieval and Cross-Encoder reranking helpers.

```python
16: LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
17: GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
18: LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1")
19: LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF")
20: GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
...
26: if LLM_PROVIDER == "local" or (not LLM_PROVIDER and (not GROQ_API_KEY or len(GROQ_API_KEY) <= 5)):
27:     llm_client = AsyncOpenAI(base_url=LOCAL_LLM_URL, api_key="dummy")
28:     DEFAULT_MODEL = LOCAL_LLM_MODEL
29: else:
30:     llm_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
31:     DEFAULT_MODEL = GROQ_MODEL
```
* **Lines 16–31 (Multi-Provider Selection Logic)**:
  * **Option 1: Force Local Model (`LLM_PROVIDER=local` in `.env`)**: Connects to the local OpenAI-compatible inference server (e.g. `llama.cpp`, LM Studio, Ollama) on `http://localhost:8080/v1` running quantized `raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF`. Even if a `GROQ_API_KEY` exists in `.env`, setting `LLM_PROVIDER=local` guarantees that zero tokens or queries leave your machine!
  * **Option 2: Force Cloud Model (`LLM_PROVIDER=groq` in `.env`)**: Connects to Groq's high-speed cloud LPU API running `llama-3.3-70b-versatile`.
  * **Fallback Default**: If `LLM_PROVIDER` is not specified, it uses Groq if a key is present, otherwise falls back to the local model.
  * Both local and cloud providers use the exact same `AsyncOpenAI` client interface (`/v1/chat/completions`) with full streaming support (`stream=True`).

---

### Part B: FinOps Pricing Dictionary & `calculate_query_cost` (Lines 28–38)

```python
28: TOKEN_PRICING = {
29:     "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
30:     "raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF": {"input": 0.0, "output": 0.0},
31:     "default": {"input": 0.50, "output": 0.50}
32: }
```
* **Lines 28–32**: Pricing catalog storing prices in USD **per 1,000,000 tokens**.
  * `llama-3.3-70b-versatile`: $0.59 per 1M input tokens, $0.79 per 1M output tokens.
  * Local Qwen model: $0.00 (100% free on local hardware).
  * `"default"`: $0.50 per 1M tokens fallback rate.

```python
35: def calculate_query_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
36:     pricing = TOKEN_PRICING.get(model_name, TOKEN_PRICING["default"])
37:     cost = (prompt_tokens / 1_000_000 * pricing["input"]) + (completion_tokens / 1_000_000 * pricing["output"])
38:     return round(cost, 6)
```

#### Detailed Breakdown of Line 36 (`TOKEN_PRICING.get`):
```python
pricing = TOKEN_PRICING.get(model_name, TOKEN_PRICING["default"])
```
* **Why not write `pricing = TOKEN_PRICING[model_name]`?**
  In Python dictionaries, if you look up a key that doesn't exist (e.g. `TOKEN_PRICING["mistral-large"]`), Python crashes with a fatal `KeyError: 'mistral-large'`, which aborts the user's streaming connection!
* **How `dict.get(key, fallback)` works:**
  1. Searches `TOKEN_PRICING` for `model_name`.
  2. If found, returns that model's dictionary (e.g. `{"input": 0.59, "output": 0.79}`).
  3. If **NOT found** (an unlisted or newly switched model), it safely returns the fallback value: `TOKEN_PRICING["default"]`.
* **Why have a `"default"` key at all?**
  In production engineering, this is called **defensive programming**. If a DevOps engineer changes `DEFAULT_MODEL` in `.env` to a new model not yet added to `TOKEN_PRICING`, the cost calculator uses the safe default rather than crashing the API server.
* **Cost Formula (Line 37)**:
  $$\text{Cost} = \left(\frac{\text{prompt\_tokens}}{1{,}000{,}000} \times \text{input\_rate}\right) + \left(\frac{\text{completion\_tokens}}{1{,}000{,}000} \times \text{output\_rate}\right)$$

---

### Part C: Hybrid Retrieval, Reranking & Context Assembly (Lines 47–73)

```python
47: async def stream_rag_answer(
48:     query: str, 
49:     tenant_id: str = "tenant_default", 
50:     user_id: str = "usr_admin"
51: ) -> AsyncGenerator[str, None]:
52:     start_time = time.time()
53:     ttft_time = 0.0
```
* **Lines 47–51**: Defined as an `AsyncGenerator[str, None]`. Instead of returning a single string once at the end (`return answer`), it uses `yield` to push SSE data chunks continuously down the HTTP connection as tokens are generated.
* **Lines 52–53**: Initializes high-precision timestamp to measure Time-To-First-Token (TTFT) and total latency.

```python
56:     retrieved_chunks = await hybrid_search(query, tenant_id=tenant_id, user_id=user_id, limit=10)
59:     reranked_chunks = await rerank_results(query, retrieved_chunks, top_k=3)
```
* **Line 56**: Calls Stage-1 Hybrid Search to gather 10 candidate chunks with dual tenant and user security scoping.
  * **Syntax Note (Positional vs. Keyword Arguments):** Notice `query` is passed directly as a value without `query=query`. In Python, the 1st mandatory parameter is conventionally passed **positionally**, while optional/configuration parameters (`tenant_id=tenant_id`, `user_id=user_id`, `limit=10`) are passed as **keyword arguments** (`key=value`) to explicitly name what we are overriding. Writing `hybrid_search(query=query, ...)` works 100% identically in Python.
* **Line 59**: Calls Stage-2 Cross-Encoder Reranker (`bge-reranker-base`) to score semantic interactions and isolate the top 3 chunks (`top_k=3`). Here, `query` and `retrieved_chunks` are the two positional inputs, and `top_k=3` is the explicit keyword argument.

```python
62:     context = "\n\n".join([chunk["text"] for chunk in reranked_chunks])
63:     citations = list(set([chunk["source"] for chunk in reranked_chunks]))
65:     prompt = build_rag_prompt(context, query)
```
* **Line 62**: Combines the 3 top chunks into one formatted context string separated by double newlines.
* **Line 63**: `set(...)` deduplicates source filenames (e.g. `["machine_learning.pdf"]`) for clean citations.
* **Line 65 (`build_rag_prompt`)**: Calls the enterprise prompt constructor with 4 core directives:
  1. **Grounding:** Strict zero-hallucination boundary (`"Answer ONLY from the provided context"`).
  2. **Comprehensive Synthesis:** Defaults to a thorough, complete explanation connecting all 3 retrieved chunks rather than lazy 4-line summaries.
  3. **Table & Structured Data Preservation:** Mandates that comparison tables and parameters in context are preserved and rendered as clean Markdown tables.
  4. **User Directives & Constraints:** Dynamically respects user overrides when the user requests specific lengths or formats (e.g., *"summarize"*, *"shorten"*, *"in 2 bullet points"*).

---

### Part D: Metadata Emission Event (Lines 76–83)

```python
76:     metadata_event = {
77:         "type": "metadata",
78:         "citations": citations,
79:         "contexts": [chunk["text"] for chunk in reranked_chunks],
80:         "chunk_ids": [chunk.get("chunk_id") for chunk in reranked_chunks],
81:         "tenant_id": tenant_id
82:     }
83:     yield f"data: {json.dumps(metadata_event)}\n\n"
```

#### Why Emit Metadata BEFORE Any LLM Tokens?
1. **Instant UI Rendering:** Retrieval and reranking finish in ~200ms. By pushing metadata first, the frontend (Streamlit / React) can immediately draw the citation chips, source badges, and expandable "View Retrieved Chunks" drawer while the LLM is still preparing its first token!
2. **User Transparency & Trust:** Enterprise users see which documents were retrieved right away.
3. **Why Include `chunk_ids`?**
   * Normal users don't inspect integer IDs, but **production telemetry systems rely on them**.
   * If a user clicks **Thumbs Down** ("Incorrect Answer") in the UI, the frontend sends back the `chunk_ids` (`[14, 29]`). This allows ML engineers to trace which exact database points caused a bad answer without exposing raw database tables.
4. **SSE Format (`data: ...\n\n`)**:
   * The W3C Server-Sent Events standard mandates that every message start with `data: ` and end with two newlines (`\n\n`). This tells the browser/client that the JSON packet is complete.

#### Does Yielding Metadata First Delay TTFT? (The 0.05ms Reality)
**No!** Emitting metadata first takes less than $0.05\text{ milliseconds}$ (just stringifying a small JSON dictionary).
Here is the exact online millisecond timeline:

```
T = 0 ms     ──► User sends query over HTTP
T = 180 ms   ──► Qdrant Hybrid Search finishes
T = 240 ms   ──► Cross-Encoder Reranking finishes
T = 240.1 ms ──► yield metadata_event (JSON serialize takes ~0.05 ms!)
T = 240.2 ms ──► LLM API request dispatched (await llm_client.chat.completions.create)
T = 420 ms   ──► First LLM text token arrives over network socket (TTFT = 420 ms)
```

* **If metadata is emitted FIRST:** TTFT is **~420 ms**. At 240 ms, the user already sees citation chips appear on screen. The system feels instantaneous!
* **If metadata is emitted LAST:** TTFT is **~419.95 ms** (a 0.05 ms difference, humanly imperceptible). But citations would remain hidden for 3 whole seconds until the full answer completes!

---

### Part E: LLM Call with Streaming Configuration (Lines 85–90)

```python
85:     stream = await llm_client.chat.completions.create(
86:         model=DEFAULT_MODEL,
87:         messages=[{"role": "user", "content": prompt}],
88:         stream=True,
89:         stream_options={"include_usage": True}
90:     )
```
* **Line 85 (`await llm_client.chat.completions.create(...)`)**: Standard OpenAI-compatible API call. Sends the prompt over HTTP to the model provider (Groq or local llama.cpp).
* **Line 88 (`stream=True`)**:
  * **What does this do?** Without `stream=True`, the server blocks for 4–6 seconds while the LLM writes all 300 words, then returns one huge JSON blob.
  * With `stream=True`, the LLM server opens an HTTP chunked stream and sends each token packet over the wire as soon as the model predicts it!
* **Line 89 (`stream_options={"include_usage": True}`)**:
  * **Why is this needed?** In streaming mode, individual token packets contain only text deltas, not token counts. Adding `stream_options={"include_usage": True}` instructs Groq/OpenAI to append a final summary packet at the very end containing `prompt_tokens` and `completion_tokens`. Without this, token counts would be unavailable for cost calculation.

---

### Part F: The `async for chunk in stream:` Loop (Lines 96–110)

```python
92:     prompt_tokens = 0
93:     completion_tokens = 0
94:     first_token_received = False
96:     async for chunk in stream:
```

#### 1. What is `async for`?
* A normal `for x in list:` iterates over items stored in local memory synchronously.
* `async for chunk in stream:` iterates over an **asynchronous network stream**. When the next token is still being computed by the GPU/cloud, `async for` pauses this function and yields CPU control back to the FastAPI event loop so other users aren't blocked. As soon as the next token packet arrives over the socket, the loop wakes up and executes with `chunk`.

#### 2. The One-Time Latch Pattern (Boolean Flag) for TTFT (Lines 97–99)
```python
97:         if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
98:             first_token_received = True
99:             ttft_time = round((time.time() - start_time) * 1000, 2)
```
* **Before loop:** `first_token_received = False`.
* **When Chunk 1 arrives:** `not first_token_received` is `not False` $\rightarrow$ **`True`**. The condition passes!
  * `first_token_received` flips to **`True`**.
  * `ttft_time` records the exact elapsed milliseconds from start.
* **When Chunks 2, 3, 4 ... N arrive:**  
  * `first_token_received` is now `True`, so `not first_token_received` evaluates to **`False`**.
  * The `and` condition immediately short-circuits. Lines 98–99 are **never executed again** for the rest of the stream!
  * This guarantees that TTFT records **strictly the first token's latency** and is never overwritten.

#### 3. Safe Usage Extraction with `hasattr` (Lines 102–105)
```python
102:         # Check for usage stats in final streaming chunk
103:         if hasattr(chunk, "usage") and chunk.usage:
104:             prompt_tokens = chunk.usage.prompt_tokens
105:             completion_tokens = chunk.usage.completion_tokens
```
* **What is `hasattr`?**
  * `hasattr(object, "attribute_name")` returns a boolean: **`True` or `False`**.
  * During streaming, the first 50 token chunks **do not possess** a `usage` attribute. If we wrote `if chunk.usage:`, Python would immediately crash with an unhandled `AttributeError: 'ChatCompletionChunk' object has no attribute 'usage'`.
  * `hasattr(chunk, "usage")` verifies whether the attribute exists before attempting to read it!
  * When the final summary packet arrives, `hasattr` returns `True`, and lines 104–105 safely extract `prompt_tokens` and `completion_tokens`.

#### 4. Emitting Text Deltas (Lines 106–109)
```python
106:         if chunk.choices and chunk.choices[0].delta.content:
107:             token_text = chunk.choices[0].delta.content
108:             token_event = {"type": "token", "content": token_text}
109:             yield f"data: {json.dumps(token_event)}\n\n"
```
* Extracts the newly generated text piece (e.g. `"According"`, `" to"`, `" the"`).
* Wraps it into an SSE JSON event and yields it down the network connection.

---

### Part G: Telemetry Emission & Cost Calculation (Lines 112–130)

```python
112:     if prompt_tokens == 0:
113:         prompt_tokens = len(prompt.split()) * 4 // 3
114:         completion_tokens = 150
116:     total_time = round(time.time() - start_time, 2)
117:     estimated_cost = calculate_query_cost(DEFAULT_MODEL, prompt_tokens, completion_tokens)
119:     telemetry_event = {
120:         "type": "telemetry",
121:         "ttft_ms": ttft_time,
122:         "total_latency_sec": total_time,
123:         "prompt_tokens": prompt_tokens,
124:         "completion_tokens": completion_tokens,
125:         "total_tokens": prompt_tokens + completion_tokens,
126:         "estimated_cost_usd": estimated_cost,
127:         "model_used": DEFAULT_MODEL
128:     }
129:     yield f"data: {json.dumps(telemetry_event)}\n\n"
```
* Emits the final `telemetry` event containing TTFT latency, total generation time, total tokens, and USD cost so the frontend telemetry dashboard can render live performance metrics.

#### In-Depth Telemetry Mechanics: Token Breakdown & TTFT Dynamics

##### 1. The Token Economy: Prompt Tokens (Q) vs. Completion Tokens (R)
A frequent point of confusion is why a short 1-sentence answer (e.g. *"I could not find the answer in the provided documents."*) can show **454 Total Tokens** in telemetry.

* **Total Tokens Formula:**
  $$\text{Total Tokens} = \underbrace{\text{Prompt Tokens } (Q)}_{\text{Input Context}} + \underbrace{\text{Completion Tokens } (R)}_{\text{Generated Output}}$$
* **Why Prompt Tokens Dominate ($85\%–95\%$):**  
  In a RAG architecture, the prompt sent to the LLM does not merely consist of the user's question. It bundles:
  1. The System Instructions (~80 tokens).
  2. The 3 retrieved and reranked context chunks (~300–700 tokens).
  3. The User Question (~10–30 tokens).
* **The Reality:** For an answer of only 12 tokens ($R=12$), the LLM had to ingest 442 tokens of context ($Q=442$).
  $$\text{Total Tokens} = 442 + 12 = 454 \text{ tokens}$$
* **FinOps Implication:** In commercial production, LLM providers charge different rates for input vs. output tokens (input tokens are cheaper, but account for 90% of the volume). Displaying $Q$ and $R$ separately in the UI gives complete visibility into prompt efficiency.

##### 2. TTFT (Time-To-First-Token) Deep Dive

* **Why TTFT is Kept in Milliseconds (`ms`) in Production:**
  * **Human Perception Standards:** HCI and UX research (Nielsen Norman Group) categorize interaction latency:
    * $<300\text{ ms}$: Perceived as instant.
    * $<1000\text{ ms}$: Keeps the user's train of thought uninterrupted.
    * $>1000\text{ ms}$: Causes noticeable waiting.
  * **Micro-benchmarking Precision:** In engineering audits, a jump from $250\text{ ms}$ to $450\text{ ms}$ represents an $80\%$ latency regression. If rounded to seconds (`0.3s` vs `0.5s`), critical performance regressions become invisible. Therefore, production observability tools (Datadog, LangSmith, vLLM, OpenAI) universally report TTFT in `ms`.

* **Why TTFT is NOT Constant and Varies Widely Across Queries:**
  In a RAG pipeline, TTFT does not simply measure the LLM; it measures the **entire end-to-end pre-generation pipeline**:
  $$\text{TTFT} = \text{Query Embedding} + \text{Hybrid Search} + \text{Cross-Encoder Reranker} + \text{LLM Prompt Prefill}$$
  
  Why it fluctuates (e.g., $2{,}701\text{ ms}$ vs. $5{,}465\text{ ms}$ vs. $11{,}563\text{ ms}$):
  1. **Cross-Encoder Reranker Workload:** The reranker processes 10 `(query, chunk)` candidate pairs through a 12-layer transformer. If chunks contain large Markdown tables (many tokens), the transformer forward pass takes significantly longer than when chunks contain short paragraphs.
  2. **LLM Prompt Prefill Computation:** Before generating token #1, the LLM must evaluate all tokens in the prompt. Processing a 1,200-token prompt takes noticeably more time than a 400-token prompt.
  3. **Out-of-Vocabulary / Mismatched Queries & Cache Invalidation:** When a query has zero keyword overlap (such as *"can you elaborate the response"*), the BM25 index scores the entire dictionary with near-zero weights, forcing wide semantic scans. On local GGUF models, dramatic topic shifts cannot reuse prior KV-cache memory, leading to RAM reallocation spikes that push TTFT to $11{,}563\text{ ms}$.

---

### Part H: Non-Streaming Architecture & `getattr` Safe Inspection (Lines 134–162)

```python
134: async def generate_answer(query: str, tenant_id: str = "tenant_default", user_id: str = "usr_admin"):
...
146:     answer = response.choices[0].message.content
147:     usage = getattr(response, "usage", None)
148:     prompt_tokens = usage.prompt_tokens if usage else 0
149:     completion_tokens = usage.completion_tokens if usage else 0
151:     return {
152:         "answer": answer,
153:         "citations": citations,
154:         "contexts": [c["text"] for c in reranked],
155:         "chunk_ids": [c.get("chunk_id") for c in reranked],
156:         "telemetry": { ... }
157:     }
```

#### Why Does the Non-Streaming Code Look Different From the Streaming Code?
* **Streaming (`stream_rag_answer`):**  
  Uses an `AsyncGenerator` yielding discrete events over time (`type: metadata` $\rightarrow$ `type: token` $\rightarrow$ `type: telemetry`) for live web UI animations.
* **Non-Streaming (`generate_answer`):**  
  Returns a **single dictionary all at once**. It is specifically designed for:
  1. Automated evaluation frameworks (like **Ragas** and **DeepEval**) which require a single return dictionary with `answer` and `contexts`.
  2. Batch unit tests and developer scripts that do not open an SSE event stream.

#### Deep Breakdown of Line 147 (`getattr`):
```python
usage = getattr(response, "usage", None)
```
`getattr(object, attribute_name, default_value)` safely retrieves an object's attribute with a fallback default.

The **3 items** inside `getattr`:
1. **`response`**: The object returned by `llm_client.chat.completions.create(...)`.
2. **`"usage"`**: The name of the property we want to read (`response.usage`).
3. **`None`**: The fallback value if `response` does not have `.usage`.

* If `response.usage` exists $\rightarrow$ `usage = response.usage`.
* If `response.usage` does not exist $\rightarrow$ `usage = None` (no crash!).

#### Comparison: `hasattr` vs `getattr`

| Tool | Syntax | Return Value | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`hasattr`** | `hasattr(chunk, "usage")` | `True` or `False` | Checking if an attribute exists before executing an `if` block. |
| **`getattr`** | `getattr(response, "usage", None)` | The attribute's value (or default) | Grabbing a value in a single line with safe fallback if missing. |

---

## 3. End-to-End Real World Execution Trace

Let's walk through an actual execution trace using a cloud model (`llama-3.3-70b-versatile` on Groq).

### Step 1: Input Query & Parameters
* **`query`**: `"What was Acme Corp's operating margin in Q3 2024?"`
* **`tenant_id`**: `"acme_corp"`
* **`user_id`**: `"usr_analyst"`

---

### Step 2: Hybrid Retrieval & Reranking Results
`hybrid_search` and `rerank_results` identify the top 3 chunks from Qdrant:
* **Chunk 12** (`acme_financials_2024.pdf`, Page 4):  
  `"In Q3 2024, Acme Corp achieved an operating margin of 18.4%, up 210 basis points year-over-year."`
* **Chunk 13** (`acme_financials_2024.pdf`, Page 4):  
  `"Total Q3 operating income was $42.5M compared to $35.1M in Q3 2023."`
* **Chunk 29** (`shareholder_letter_q3.pdf`, Page 1):  
  `"Management noted strong operating leverage across enterprise software subscriptions."`

---

### Step 3: SSE Event 1 Emitted — `metadata` (At $T = 210\text{ms}$)
Before any text is generated, the generator yields the metadata event:
```json
data: {
  "type": "metadata",
  "citations": ["acme_financials_2024.pdf", "shareholder_letter_q3.pdf"],
  "contexts": [
    "In Q3 2024, Acme Corp achieved an operating margin of 18.4%...",
    "Total Q3 operating income was $42.5M...",
    "Management noted strong operating leverage..."
  ],
  "chunk_ids": [12, 13, 29],
  "tenant_id": "acme_corp"
}
```
*Result on frontend:* The Streamlit UI immediately displays source badges and opens the context accordion.

---

### Step 4: SSE Events 2..N Emitted — `token` Stream (Starting at $T = 312\text{ms}$)
* At $T = 312\text{ms}$, the first token arrives:
  * `ttft_time = 312.45` ms recorded via the one-time latch.
  * Emitted: `data: {"type": "token", "content": "According"}\n\n`
* Subsequent tokens streamed sequentially:
  * `data: {"type": "token", "content": " to"}\n\n`
  * `data: {"type": "token", "content": " the"}\n\n`
  * `data: {"type": "token", "content": " provided"}\n\n`
  * `data: {"type": "token", "content": " reports,"}\n\n`
  * `data: {"type": "token", "content": " Acme"}\n\n`
  * `data: {"type": "token", "content": " Corp's"}\n\n`
  * `data: {"type": "token", "content": " operating"}\n\n`
  * `data: {"type": "token", "content": " margin"}\n\n`
  * `data: {"type": "token", "content": " in"}\n\n`
  * `data: {"type": "token", "content": " Q3"}\n\n`
  * `data: {"type": "token", "content": " 2024"}\n\n`
  * `data: {"type": "token", "content": " was"}\n\n`
  * `data: {"type": "token", "content": " 18.4%."}\n\n`

---

### Step 5: Final Chunk Usage & FinOps Cost Calculation
In the final stream packet, Groq reports:
* `prompt_tokens = 480`
* `completion_tokens = 32`

The cost calculator runs:
$$\text{Input Cost} = \frac{480}{1{,}000{,}000} \times 0.59 = \$0.0002832$$
$$\text{Output Cost} = \frac{32}{1{,}000{,}000} \times 0.79 = \$0.00002528$$
$$\text{Total Cost} = \$0.0002832 + \$0.00002528 = \mathbf{\$0.000308}$$

---

### Step 6: SSE Final Event Emitted — `telemetry` (At $T = 1.34\text{s}$)
```json
data: {
  "type": "telemetry",
  "ttft_ms": 312.45,
  "total_latency_sec": 1.34,
  "prompt_tokens": 480,
  "completion_tokens": 32,
  "total_tokens": 512,
  "estimated_cost_usd": 0.000308,
  "model_used": "llama-3.3-70b-versatile"
}
```
*Result on frontend:* UI displays metrics badges:
* **TTFT:** `312.45 ms`
* **Total Time:** `1.34 s`
* **Tokens:** `512`
* **Query Cost:** `$0.000308` (Less than 4 hundredths of a cent!)
