# The Emerging 3-Tier AI Architecture: LLM vs. Jev vs. CDM

This document provides a comprehensive technical guide explaining the modern industry shift from monolithic LLMs to specialized tiered model architectures (**Large Language Models**, TypeSafe AI's **Jev**, and ScaleDown's **Constrained Decoding Models / CDMs**), detailing their mechanics, agent economics, and structured output implementations.

---

## 1. The Modern Architectural Shift: Why One Model Can't Do Everything

For years, developers defaulted to sending every task to a single massive, general-purpose LLM (like GPT-4 or Claude):
* Translating text
* Deciding tool routing
* Extracting JSON entities
* Summarizing chat memory
* Generating final conversational answers

**The Production Failure:** Using a 70B–1.8T parameter autoregressive reasoning model for routine background tasks creates massive latency, explodes API bills, and introduces non-deterministic syntax errors.

Today, enterprise architectures divide work across **three specialized tiers**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 3-TIER MODEL HIERARCHY                                     │
├──────────────────────────┬───────────────────────────┬───────────────────────────────────────────┤
│ 1. LLM (Generative)      │ 2. JEV (Decision)         │ 3. CDM (Constrained Decoding)             │
├──────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ • Massive Autoregressive │ • Fast Discriminative     │ • Small, Task-Specific Bounded Models     │
│ • Full Natural Language  │ • Probabilities & Routing │ • Strictly Schema-Locked Outputs          │
│ • Creative & Deep Logic  │ • Zero Generated Text     │ • Extraction, Summarization & Compression │
│ • High Latency & Cost    │ • Sub-50ms / Ultra-Cheap  │ • Up to 200x Faster / 400x Cheaper        │
└──────────────────────────┴───────────────────────────┴───────────────────────────────────────────┘
```

---

## 2. What is a CDM (Constrained Decoding Model)?

Pioneered by **ScaleDown** (and integrated into LangChain via `langchain-scaledown`), **CDMs** are lightweight, task-specialized models trained for narrow context operations: *Summarize*, *Extract*, *Compress*, and *Classify*.

### How It Operates Under the Hood:
Unlike a standard LLM that generates text freely, a CDM's decoding mechanism is bound by an **"Answer Contract"**:
* It ingests long context text and runs it through a specialized constrained decoder.
* The model mathematically masks out token transitions that violate the target schema, preventing preamble filler (*"Certainly, here is what you requested..."*) or syntax errors.
* ScaleDown reports these models run up to **200x faster** and **467x cheaper** than frontier general models.

---

## 3. What Does "Structured Output" Actually Mean? (The 3 Levels)

There is significant confusion between "asking an LLM for JSON" and true "Constrained Structured Output". In software engineering, there are **three distinct levels**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  THE 3 LEVELS OF STRUCTURED OUTPUT                                     │
├──────────────────────────┬───────────────────────────────────────────┬─────────────────────────────────┤
│ Level 1: Prompt-Only     │ Level 2: `with_structured_output(schema)` │ Level 3: Grammar-Level Logit    │
│ (Naive Prompting)        │ (Function Calling / JSON Mode)            │ Masking (True Constrained CDM)  │
├──────────────────────────┼───────────────────────────────────────────┼─────────────────────────────────┤
│ • "Please return JSON"   │ • Pydantic schema passed via API          │ • Finite State Machine / CFG    │
│ • Model writes backticks │ • Provider enforces JSON syntax           │ • Invalid tokens masked to -inf │
│ • Fails ~15-25% of time  │ • Reliable, native in LangChain           │ • 100% mathematically locked    │
│ • Prone to hallucination │ • Greatly reduces hallucinations          │ • Zero hallucination in format  │
└──────────────────────────┴───────────────────────────────────────────┴─────────────────────────────────┘
```

### Level 1: Naive Prompting (High Hallucination Risk)
* Prompt: `"Extract customer name and phone. Return as valid JSON."`
* **Why it breaks:** The LLM's autoregressive decoder is unconstrained. It often adds markdown backticks (` ```json `), includes conversational filler, invents keys, or produces trailing commas that crash `json.loads()`.

### Level 2: `model.with_structured_output(schema)` (Standard LangChain Pattern)
* Code:
  ```python
  from pydantic import BaseModel
  
  class EntityExtraction(BaseModel):
      topic: str
      confidence: float

  structured_llm = model.with_structured_output(EntityExtraction)
  result = structured_llm.invoke("Text...")
  ```
* **Why it succeeds:** The schema is converted into an OpenAI/Anthropic/Groq tool definition. The API sampler forces the model into a constrained state machine that outputs valid JSON matching the Pydantic fields. This is why you experienced **zero hallucinations in format**.

### Level 3: True Constrained Decoding (Grammar / Logit Masking)
* **Under the Hood:** At every single step of generation, the decoding engine evaluates the active grammar.
* If the schema expects a boolean (`true`/`false`), the model sets the logits for all 32,000 vocabulary tokens to $-\infty$, leaving *only* the tokens `t` and `f`.
* The model **cannot physically output an invalid character**.

---

## 4. Agent Economics: Are CDMs Practical in Real-World AI Agents?

A common developer skepticism is: *If an agent has to call a cloud CDM on every step, doesn't it become too expensive and slow?*

The economic reality is the exact opposite when compared to frontier LLMs:

```
AGENT STEP BUDGET (20-Step Task):
1. Pure Frontier LLM Architecture (20 steps of GPT-4o):
   20 steps × $0.005/call = $0.100 per agent run (Very expensive at scale)

2. Tiered Architecture (2 steps GPT-4o + 18 steps CDM / Jev):
   2 reasoning steps (GPT-4o)  = $0.010
   18 extraction/routing (CDM) = $0.001
   TOTAL COST = $0.011 (90% Cost Reduction!)
```

### Why Local Small Models (<1B) are Ultra-Fast (<30ms):
* When developers think of running models locally, they think of **7B or 14B parameter models** (like Qwen2.5-7B), which take **15–30 seconds on CPU**.
* **Task-specific extraction/decider models are NOT 7B parameters!**
  * Models like `DeBERTa-v3-small` have **44 Million to 86 Million parameters** (80x to 160x smaller than a 7B model).
  * Models like `Qwen2.5-0.5B` have **500 Million parameters**.
* On a standard consumer CPU, an 86M parameter model completes a forward pass in **15 to 30 milliseconds**.
* Therefore, running local extraction models is **instantaneous**, private, and costs **$0.00**.

---

## 5. Why Low-Knowledge / Small Models Hallucinate Less in RAG

One of the most counter-intuitive facts in RAG engineering is that **massive frontier models often hallucinate MORE than smaller, focused models when answering strictly from context**:

1. **Internal Memory Interference (Parametric Knowledge Bleed):**
   * A 1.8-Trillion parameter model (like GPT-4) contains massive internalized knowledge from Wikipedia, Reddit, and books.
   * When asked to answer from a specific PDF, its vast internal memory creates an "irresistible impulse" to fill in gaps, add background context, or correct the text using its pre-training memory.
2. **Focused / Low-Knowledge Grounding:**
   * A smaller, instruction-tuned model (like Qwen2.5-7B at `temperature=0.0`) lacks vast contradictory pre-training noise.
   * When instructed: `"Answer ONLY from the provided context"`, it treats the context chunks as its entire universe and performs direct factual synthesis without inventing external details.

---

## 6. How CDMs & Structured Outputs Improve RAG Pipelines

```
                             HYBRID RAG WITH CONSTRAINED DECODING
                             
User Query ──► [ Local CDM / Structured Extractor ] ──► Clean Search Filters & Keywords
                         │
                         ▼
             [ Hybrid Retrieval (Qdrant + BM25) ] ──► Top 10 Chunks
                         │
                         ▼
             [ Cross-Encoder Reranker ] ──► Top 3 Chunks (3,000 Chars)
                         │
                         ▼
                 [ Context Compression CDM ]
                 • Extracts ONLY query-answering sentences (500 Chars)
                 • Strips irrelevant background filler
                         │
                         ▼
                 [ Generator LLM (Qwen / Groq) ]
                 • Reads clean, compressed context
                 • Generates final answer 4x faster with ZERO context distraction!
```

1. **Context Compression (Passage Pruning):** Strips the 80% irrelevant text from retrieved PDF chunks, reducing prompt token costs and TTFT.
2. **Dynamic Query Decomposition:** Extracts structured search metadata (`tenant_id`, date ranges, technical terms) before searching Qdrant.
3. **Ingestion Metadata Extraction:** Extracts clean Pydantic entities (authors, dates, tables) during offline document parsing.
