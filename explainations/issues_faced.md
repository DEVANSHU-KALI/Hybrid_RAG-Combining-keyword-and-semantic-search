# Enterprise Hybrid RAG: Issues & Architectural Edge Cases Faced

This document serves as an engineering post-mortem log of real-world bugs, performance edge cases, and data anomalies discovered and resolved during the development of this Multi-Tenant Hybrid RAG system.

---

## Issue 1: PDF Table Extraction Duplication & Inline Prose Smashing

* **Date Discovered:** October 19, 2026
* **Component Affected:** `backend/injest_documents.py` & `backend/text_chunker.py`
* **Discovery Method:** Deep visual inspection comparing raw PDF pages (`deep_learning_test_corpus.pdf`) against stored point payloads in the Qdrant vector database (`http://localhost:6333/dashboard`).

---

### 1. Symptoms & Observations in Database Points
While auditing the ingestion output in the Qdrant dashboard:
1. **Point 73 (Page 2) & Point 77 (Page 3):**  
   The structured comparison table was extracted as an unformatted, continuous stream of smashed text jammed directly into the middle of a narrative paragraph:
   > *"Layers can represent complex functions, but extra depth also increases computational cost... Component Role Example in an image classifier Input Receives numeric data Pixel values scaled to a known range Weights and bias Learn how inputs influence a neuron..."*
2. **Point 78 (Page 3):**  
   The exact same table appeared a second time, this time formatted as a Markdown table (`| Training term | Meaning | ... |`), but it was positioned at the **very end** of the page after the page footer (`"Deep Learning | Retrieval test corpus Page 3"`).

---

### 2. Root Cause Analysis (Under the Hood)
In the initial implementation of `extract_pdf_content()`:
```python
# FLAWED INITIAL CODE:
page_text = page.extract_text() or ""
tables = page.extract_tables()
...
full_page_text = page_text + "\n\n" + "\n\n".join(table_markdowns)
```
* **Blind Character Extraction:** `page.extract_text()` in `pdfplumber` extracts all text glyphs on the canvas. Because it has no awareness of table borders, it reads horizontally across table columns, merging table cells into the surrounding narrative paragraphs.
* **Displaced Table Appending:** The structured Markdown table was then simply concatenated to the very end of `page_text`.
* **Chunk Fragmentation:** When `chunk_markdown_sections()` split the page text into 1,000-character chunks:
  - Middle chunks received the garbled, unformatted table text from `page.extract_text()`.
  - The final chunk of the page received the Markdown table, completely separated from the preceding paragraphs that explained the table!

---

### 3. Downstream RAG Impact (Why This Breaks the Pipeline)
1. **Semantic Vector Distortion:** Smashed words with no column headers produce noisy 768D embedding vectors, degrading dense similarity search.
2. **Table Duplication:** Every table was stored twice in Qdrant, inflating index size and causing redundant retrieval chunks.
3. **Loss of Contextual Grounding:** In Point 78, the Markdown table had zero explanatory prose around it. If a user asked a conceptual question about the table, the retriever would either retrieve the table without the paragraph explaining it, or retrieve the paragraph with the unreadable smashed table text.

---

### 4. Architectural Fix: Geometric Bounding-Box Cropping
Instead of reading the entire page as a single string, the parser now leverages geometric bounding boxes detected by `page.find_tables()`:

```
┌────────────────────────────────────────────────────────┐
│  1. Prose ABOVE Table                                  │
│     --> page.crop((0, prev_bottom, page.width, top))   │
├────────────────────────────────────────────────────────┤
│  2. TABLE REGION (Excluded from prose extraction)      │
│     --> table.extract() formatted to Markdown          │
│     --> Internal '\n' replaced with ' ' per cell       │
├────────────────────────────────────────────────────────┤
│  3. Prose BELOW Table                                  │
│     --> page.crop((0, bottom, page.width, page.height))│
└────────────────────────────────────────────────────────┘
```

#### Key Implementation Changes in `backend/injest_documents.py`:
1. **`page.find_tables()`**: Detects bounding box coordinates `(x0, top, x1, bottom)` for every table on the page.
2. **`page.crop(...)`**: Only extracts narrative text strictly **outside** the table bounding boxes. Table cell words are **never** dumped as raw inline text.
3. **In-Place Assembly:** The Markdown table is inserted in its authentic reading position between the preceding and following prose.
4. **Markdown Integrity:** Replaced `\n` in cell strings with `" "` so multi-line PDF table cells do not break Markdown table syntax.

#### Chunker Improvements in `backend/text_chunker.py`:
* Added pre-table isolation logic: If a table is encountered and `current_chunk` already contains $>500$ characters, the chunker splits immediately before the table so the table starts cleanly at the beginning of a fresh chunk and remains intact.

---

### 5. Verification & Resolution
* Cleared the database with `python -m backend.reset_qdrant`.
* Re-ran ingestion with `python -m backend.injest_documents`.
* **Result:** Verified in Qdrant dashboard that every table is stored once, formatted in valid Markdown, and placed directly alongside its introducing narrative text.

---

## Issue 2: Page-Level vs. Chunk-Level Metadata Bleed (`has_table: true` on Text Chunks)

* **Date Discovered:** October 19, 2026
* **Component Affected:** `backend/injest_documents.py`
* **Discovery Method:** Auditing Point 71 in Qdrant dashboard: pure narrative text displayed `"has_table": true` despite containing no table.

---

### 1. Symptoms & Observations in Database Points
In Point 71 (from Page 2 of `deep_learning_test_corpus.pdf`), the text payload is purely narrative:
> *"1. Deep Learning: Foundations and Neural Networks Deep Learning is a branch of machine learning that uses neural networks with multiple layers to learn useful representations from data..."*

However, the payload metadata explicitly declared:
```json
"has_table": true
```
Even though the actual table on Page 2 was in a subsequent chunk!

---

### 2. Root Cause Analysis (Under the Hood)
In `backend/injest_documents.py`:
```python
# FLAWED METADATA ASSIGNMENT:
for page_data in pdf_pages:
    chunks = chunk_markdown_sections(page_data["text"])
    for chunk_text in chunks:
        metadatas.append({
            ...
            "has_table": page_data["has_table"]  # <-- Inherited from entire page!
        })
```
* `page_data["has_table"]` was a **page-level flag** (i.e. "Does this 11-inch PDF page contain a table anywhere?").
* When the page was chunked into 3 or 4 smaller chunks, the loop blindly copied the page-level boolean to **every single chunk**.
* Consequently, prose chunks before and after the table falsely advertised that they contained a table.

---

### 3. Downstream RAG Impact
* **Metadata Filter Pollution:** If an enterprise user or API agent runs a filtered search for tabular data:
  ```python
  Filter(must=[FieldCondition(key="has_table", match=MatchValue(value=True))])
  ```
  The database returns standard narrative prose chunks (like Point 71) that have zero table data, polluting results and breaking table-specific reasoning.

---

### 4. Architectural Fix: Dynamic Chunk-Level Evaluation
Instead of inheriting the coarse page-level boolean, the ingestion engine now evaluates each chunk individually by inspecting its text for Markdown table delimiters:

```python
# PRECISE CHUNK-LEVEL METADATA:
chunks = chunk_markdown_sections(page_text)
for chunk_text in chunks:
    # Dynamically verify if THIS specific chunk contains Markdown table rows
    chunk_has_table = any(
        line.strip().startswith("|") and line.strip().endswith("|")
        for line in chunk_text.split("\n")
    )
    
    metadatas.append({
        ...
        "has_table": chunk_has_table  # <-- True ONLY if table rows exist in this chunk!
    })
```

---

### 5. Verification & Resolution
* Re-run `python -m backend.reset_qdrant` and `python -m backend.injest_documents`.
* **Result:** Point 71 now correctly shows `"has_table": false`. Only the specific chunk that physically holds the Markdown table (Point 72) carries `"has_table": true`.

---

## Issue 3: Stateless Follow-Up Query Failure ("Can you elaborate the response") in Single-Turn RAG

* **Date Discovered:** October 20, 2026
* **Component Affected:** `backend/rag_pipeline.py` & Multi-Turn Conversation Flow
* **Discovery Method:** End-to-end user chat query in Streamlit: following up an answer with *"can you elaborate the response"* yielded *"I could not find the answer in the provided documents."*

---

### 1. Symptoms & Observations in Live Execution
1. A user submitted Turn 1: *"what are the Foundations and problem framing in machine learning"* and received a 4-line answer.
2. The user submitted Turn 2: *"can you elaborate the response"*.
3. The system returned:
   > *"I could not find the answer in the provided documents."*
4. In the telemetry drawer, retrieved chunks were from `ai_concepts.txt` (completely unrelated to Turn 1's machine learning discussion), TTFT spiked to **11,563.72 ms** (11.5s), and total tokens showed **454**.

---

### 2. Root Cause Analysis (Under the Hood)
* **Stateless Single-Turn Retrieval:** The current retrieval pipeline has no conversational memory. When Turn 2 was submitted, the retriever searched the database for the literal phrase: `"can you elaborate the response"`.
* **Zero Semantic Anchor:** Since the document corpus contains no text about "elaborating responses", BM25 returned near-zero scores, and Qdrant returned the closest mathematical noise vectors (which happened to be from `ai_concepts.txt`).
* **Guardrail Enforcement:** The LLM prompt instructed:
  > *"Answer ONLY from the provided context. If the answer is not present, say: 'I could not find the answer in the provided documents.'"*
  Because the retrieved context about AI concepts did not explain how to elaborate the previous answer, the LLM correctly and faithfully adhered to the prompt guardrail and refused to hallucinate!
* **High TTFT (11.5s):** Evaluating a query with zero keyword matches across the entire corpus forced dense vector scans and full 12-layer Cross-Encoder scoring over unrelated chunks, and caused local GGUF memory cache invalidation.

---

### 3. Downstream RAG Impact
* Users cannot ask natural follow-up questions (*"can you elaborate"*, *"shorten it"*, *"give more examples"*, *"what are its limitations"*) without restating the entire topic from scratch.

---

### 4. Architectural Fix & Roadmap Integration
1. **Conversational Query Rewriter (Documented in `future_implementations.md` Phase 1):**
   * Before retrieval, an internal fast LLM pass takes the chat history (`[Turn 1 Q & A]`) + follow-up prompt (`"can you elaborate"`) and rewrites it into a self-contained search query:
     > *"Provide a comprehensive and detailed explanation of foundations and problem framing in machine learning."*
2. **Conversation History Context Window:**
   * Pass previous conversation messages in the LLM `messages` array so the generation model knows what it previously answered and expands upon it.

---

### 5. Verification & Resolution
* Confirmed that the current guardrail behavior is functioning as intended (preventing hallucination on out-of-domain queries).
* Logged in `future_implementations.md` as the core priority for Phase 1.

---

## Issue 4: LLM-as-a-Judge Evaluation Fragility, Rate-Limit Throttling (HTTP 429), and Grading Drift

* **Date Discovered:** October 22, 2026
* **Component Affected:** `evaluations/ragas_eval.py` & Cloud LLM Evaluator (Groq / OpenAI)
* **Discovery Method:** Automated benchmark runs failing due to cloud API token/request limits, high latency (>3 minutes per test run), and inconsistent non-deterministic scores from the evaluator LLM.

---

### 1. Symptoms & Observations
1. **HTTP 429 API Throttling:** When running automated Ragas evaluation using cloud LLM judges (such as Groq's `llama-3.3-70b-versatile`), the benchmark script repeatedly crashed with:
   > `HTTP 429: Too Many Requests (Rate limit reached for requests per minute / tokens per minute)`
2. **Extreme Evaluation Latency:** Even after implementing an in-memory rate limiter (1 request every 5 seconds), evaluating a modest 7-question test suite across 4 metrics took over **3 to 5 minutes** to execute.
3. **Fallback to Mock Data:** Because the cloud API frequently timed out or choked on rate limits, `ragas_eval.py` was forced to catch exceptions and fall back to hardcoded dummy scores (`[0.92, 0.89, 0.95, 0.91]`), defeating the purpose of objective quality testing.
4. **Judge LLM Hallucinations & Lexical Mismatch:** The generative judge LLM exhibited temperature-induced scoring drift (giving different grades to the identical answer on consecutive runs) and penalized correct answers simply because their phrasing differed from synthetic "Ground Truths".

---

### 2. Root Cause Analysis (Under the Hood)
* **Autoregressive Token Generation Overhead:** An LLM-as-a-Judge evaluates text by generating conversational reasoning tokens word-by-word. For 7 questions $\times$ 4 metrics, the system must execute **28 separate multi-turn LLM generation loops**, requiring thousands of sequential forward passes through a 70-Billion parameter model.
* **Cloud Rate Limits (TPM / RPM):** Free and Tier-1 cloud API keys have strict rate limits (e.g., 6,000 tokens/min or 12 requests/min). Feeding entire context passages and generated answers for 4 metrics rapidly exhausts token quotas.
* **The "Ground Truth Illusion":** Using an LLM to generate ground truths, and then using another LLM to compare against those ground truths, creates compounding errors. If the RAG prompt instructs `"Answer ONLY from the provided context"`, but the ground truth contains external facts, the judge model unfairly punishes the system for following its grounding rules.

---

### 3. Downstream RAG Impact
* Automated CI/CD regression testing became completely unusable due to flaky API timeouts and rate-limit aborts.
* High recurring cloud API bills for developer test suites.
* Inability to trust whether a drop in benchmark scores was caused by an actual retrieval bug or the judge LLM hallucinating during grading.

---

### 4. Architectural Fix & Elimination
We resolved this by overhauling our evaluation architecture into two robust pathways:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             THE ZERO-COST LOCAL EVALUATION SOLUTION                              │
├─────────────────────────────────────────────────┬────────────────────────────────────────────────┤
│ Pathway 1: Fast Local RAG Triad (`fast_eval.py`)│ Pathway 2: Local Qwen2.5-7B Ragas Offline      │
├─────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ • 100% Non-generative discriminative models     │ • Connects Ragas directly to local llama.cpp   │
│ • Local Cross-Encoder for Faithfulness & NLI    │ • Uses Qwen2.5-7B on `localhost:8080/v1`       │
│ • Local 768D Bi-Encoder for Answer Relevancy    │ • Temperature = 0.0 for deterministic grading  │
│ • Latency: < 2 seconds for entire benchmark     │ • Zero cloud API calls, zero rate limits (429) │
│ • Cost: $0.00 (Runs 100% on local CPU)          │ • Cost: $0.00 (100% private and offline)       │
└─────────────────────────────────────────────────┴────────────────────────────────────────────────┘
```

1. **Created `evaluations/fast_eval.py` (Local Reference-Free RAG Triad):**
   * **Faithfulness (Anti-Hallucination):** Uses local Cross-Encoder cross-attention (`ms-marco-MiniLM-L-6-v2`) to mathematically verify that answer propositions are logically entailed by the retrieved context chunks.
   * **Answer Relevancy:** Uses local 768D embeddings (`BAAI/bge-base-en-v1.5`) to compute exact cosine similarity between query and answer.
   * **Context Relevance:** Evaluates retrieved chunk signal-to-noise ratio using cross-encoder relevance scores.
2. **Multi-Provider Local Fallback in `evaluations/ragas_eval.py`:**
   * Added `LLM_PROVIDER=local` toggle allowing Ragas to point to `http://localhost:8080/v1` (with `Qwen2.5-7B`), removing rate-limiting restrictions and eliminating cloud token costs.
3. **Reference-Free Grounding:**
   * Transitioned runtime quality testing away from fragile synthetic ground truths to reference-free entailment verification.

---

### 5. Verification & Resolution
* Ran `python -m evaluations.fast_eval`: Completed evaluation of all benchmark queries in **1.85 seconds** on local CPU.
* Verified **100% deterministic output** (identical floating-point scores across repeated runs).
* Confirmed clean export of benchmark tables to `evaluations/evaluation_results/fast_eval_results.xlsx` with zero cloud network calls and zero rate limits.

---

## Template for Future Issues

```markdown
## Issue [Number]: [Title of the Issue]

* **Date Discovered:** [Date]
* **Component Affected:** [Script name / path]
* **Discovery Method:** [How it was detected]

### 1. Symptoms & Observations
[Detailed description of the bug or unexpected behavior]

### 2. Root Cause Analysis
[Why it happened in the code or architecture]

### 3. Downstream RAG Impact
[How this affected retrieval, embeddings, latency, or generation quality]

### 4. Architectural Fix
[Code changes and design decisions made to resolve it]

### 5. Verification & Resolution
[How the fix was tested and verified]
```

