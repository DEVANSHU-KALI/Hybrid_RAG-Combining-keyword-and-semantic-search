# Conceptual Guide: `1) key_concepts.md`

This document serves as a high-level guide to the core software engineering, database, and machine learning concepts implemented throughout this Hybrid RAG project.

---

## 1. Retrieval-Augmented Generation (RAG)

### Definition
**Retrieval-Augmented Generation (RAG)** is an architectural pattern that combines search-based retrieval with generative language models. Large language models (LLMs) have static knowledge locked at their training cutoff date and are prone to "hallucinations" (generating confident but incorrect facts). RAG resolves this by first searching a private database for text passages relevant to a user's query, and then feeding those passages to the LLM as grounding context. This forces the LLM to construct answers backed by verified source materials.

### Architecture Diagram
```
    User Query ─────────► [ Retrieval Stage ] ─────────► [ Augmentation Stage ] ─────────► [ Generation Stage ]
                             │                              │                                 │
                             ▼                              ▼                                 ▼
                     Scan Vector DB &                Inject retrieved                 Local Qwen model
                     Keyword Indexes                 document text into               generates grounded
                     for matching chunks             prompt context template          response
```

---

## 2. Dense Semantic vs. Sparse Lexical Retrieval

Information retrieval in modern RAG systems uses two main paradigms:

### A. Dense Semantic Retrieval
* **Concept**: Converts queries and document chunks into dense floating-point vectors (embeddings) where conceptual meaning is represented by spatial coordinates. Search matches concepts rather than character patterns.
* **Metric (Cosine Similarity)**: Measures the directional overlap (angle) between the query vector ($\mathbf{A}$) and document vector ($\mathbf{B}$):
  $$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
* **Used in**: `semantic_retriever.py` and `embedding_model.py`.

### B. Sparse Lexical Retrieval (BM25)
* **Concept**: Scores documents based on exact keyword occurrences. It uses the **Okapi BM25** formula, which evaluates term frequency (TF), document length normalization, and inverse document frequency (IDF - prioritizing rare query words).
* **Metric (BM25 Score)**:
  $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  * $f(q_i, D)$: Term frequency of word $q_i$ inside document $D$.
  * $|D|$ and $\text{avgdl}$: Current document length and average document length.
  * $k_1$ and $b$: Tuning parameters ($k_1$ controls term frequency saturation; $b$ controls length normalization).
* **Used in**: `bm25_retriever.py`.

---

## 3. Semantic Chunking

### Definition
**Semantic Chunking** is a text splitting technique that groups text based on thematic transitions rather than arbitrary character or token counts. It processes a document sentence-by-sentence, calculates the semantic distance between consecutive sentences, and inserts a breakpoint (starts a new chunk) when the semantic distance exceeds a dynamically calculated percentile threshold.

### Breakpoint Evaluation Flow
```
Sentence S1 ──► [Embedding Model] ──► Vector V1 ──┐
                                                 ├──► Cosine Distance ──┐
Sentence S2 ──► [Embedding Model] ──► Vector V2 ──┘                      │
                                                                         ▼
                                                          Does Distance > 75th Percentile?
                                                          ├── Yes ──► Split (Start New Chunk)
                                                          └── No  ──► Merge sentences
```

---

## 4. Score Fusion: Concept, Evolution, and Architecture

### What is Score Fusion?
In a hybrid search system, dense semantic retrieval and sparse BM25 retrieval output independent lists of candidates, each with its own scoring metric. **Score Fusion** is the mathematical technique used to combine these disparate scores into a single, unified ranking score. 

The strategy used to fuse these scores dictates how the search engine balances keyword matching vs. conceptual similarity.

---

### The Evolution of Score Fusion Techniques

Each fusion technique in search engineering evolved to address the weaknesses of its predecessor:

```
    [ Simple Addition ] ─────────► [ Weighted Linear Fusion ] ─────────► [ Reciprocal Rank Fusion (RRF) ]
   Sum normalized scores.         Introduces alpha/beta weights          Rank-based, score-independent.
   Treats all systems as          to prioritize systems (requires        Extremely robust against
   equally important.             hyperparameter tuning).                outliers and scaling biases.
```

#### 1. Simple Addition Fusion *(Basic)*
* **How it works**: Raw scores are normalized to a $[0.0, 1.0]$ range, and then added together directly for duplicate document IDs:
  $$\text{Final Score} = S_{\text{normalized\_dense}} + S_{\text{normalized\_sparse}}$$
* **Limitation**: It treats both retrieval systems as equally important. It has no way of prioritizing one model over another. If one retrieval run has lower quality results, it introduces ranking noise that pushes irrelevant documents to the top.

#### 2. Weighted Linear Fusion *(Parametric)*
* **How it works**: Introduces weighting coefficients (hyperparameters) like $\alpha$ (alpha) and $\beta$ (beta) to scale the influence of each retriever:
  $$\text{Final Score} = \alpha \cdot S_{\text{normalized\_dense}} + \beta \cdot S_{\text{normalized\_sparse}}$$
  *(Usually constrained where $\alpha + \beta = 1.0$, simplifying to: $\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$)*
* **Limitation**: Finding the optimal values for $\alpha$ and $\beta$ requires extensive offline testing, grid searches, and domain-specific validation datasets. A weight balance that works well for short keyword queries might perform poorly for long, conversational questions.

#### 3. Reciprocal Rank Fusion (RRF) *(Rank-Based)*
* **How it works**: Ignores raw scores entirely. It merges document candidates based on their rank positions (1st, 2nd, 3rd, etc.) in the individual lists using a decay formula:
  $$RRF(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  *(where $r_m(d)$ is the rank of document $d$ in system $m$, and $k$ is a constant, typically $60$, that prevents low ranks from dominating)*
* **Advantage**: It is highly robust to outliers because it is score-independent. It does not require score normalization or complex parameter tuning.

---

### Compensating for Basic Fusion: The Cross-Encoder Safeguard
In this project, we implemented **Simple Addition Fusion** (the basic, non-parametric method). In a standalone retrieval system, this could risk delivering irrelevant context to the LLM due to score-scaling bias and ranking noise. 

However, we built a highly resilient system by implementing a **Multi-Stage Retrieval Architecture**:

1. **Stage 1 (High Recall / Low Precision)**: The hybrid search retrieves a candidate list (top 5). We use basic Min-Max normalization and Simple Addition fusion here because it is computationally cheap and simple to execute.
2. **Stage 2 (High Precision Safeguard)**: The candidate list is passed to a **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`). The Cross-Encoder performs token-level attention comparison, re-scoring each chunk's direct relevance to the query.

#### Why this is a powerful pattern:
The **Cross-Encoder Reranker** acts as a high-fidelity filter. It corrects any ranking errors, noise, or bias introduced during the simple addition fusion stage. Even if the basic fusion method places an irrelevant chunk at rank 1, the Cross-Encoder will identify the lack of direct query-document alignment, re-score it negatively, and push the truly relevant context to the top before the prompt is sent to the LLM. 

This design pattern gives us the simplicity and speed of basic addition fusion during database retrieval, while guaranteeing high-precision contexts via reranking.

---

## 5. Multi-Stage Retrieval & Cross-Encoder Reranking

RAG architectures balance search speed and scoring accuracy using a multi-stage approach:

### A. Stage 1: Retrieval (Bi-Encoder)
* Uses a **Bi-Encoder** configuration where documents and queries are encoded independently. 
* This allows documents to be indexed in a vector database beforehand. Searches are fast but lack direct interaction between query and document tokens.

### B. Stage 2: Reranking (Cross-Encoder)
* Uses a **Cross-Encoder** configuration that processes the query and document *together* through self-attention layers.
* Self-attention evaluates token-to-token interactions directly (e.g., how the word "it" in the document relates to "overfitting" in the query). This is highly accurate but computationally heavy, which is why it is used as a second-stage filter on only a small subset of candidate chunks (e.g., re-scoring top 5 down to top 3).

```
[ Database (1000s of chunks) ] ──► (Bi-Encoder Search) ──► Top 5 Chunks ──► (Cross-Encoder Rerank) ──► Top 3 Chunks to LLM
```

---

## 6. Quantization and GGUF Local Inference

### Quantization
**Quantization** is a model compression technique that converts weight values from high-precision floating-point formats (e.g., 16-bit float, `FP16`) to lower precision representations (e.g., 4-bit integer, `Q4`). 
* Quantizing a model reduces its RAM footprint and storage size by ~70%, enabling large language models (like the 7-billion parameter Qwen 2.5) to run locally on consumer computers.

### GGUF Format & Llama.cpp
* **GGUF** is a binary file format designed for fast model loading and efficient split execution across GPU and CPU.
* **Llama.cpp** is a C++ inference engine that compiles GGUF models locally. It uses split offloading (`-ngl` parameter) to load a subset of layers into GPU VRAM while managing the remaining layers in system CPU memory.

---

## 7. RAG Evaluation Frameworks: RAGAS vs. DeepEval

Evaluating RAG performance utilizes an **LLM-as-a-Judge** framework to score four key metrics:

```
                  ┌──────────────────────┐
                  │    User Question     │
                  └──────────┬───────────┘
            Answer Relevancy │ Context Precision
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────┐ Context   ┌──────────┐ Faithfulness ┌──────────┐
│ Generated│◄──────────│Retrieved │◄────────────►│  Ground  │
│  Answer  │  Recall   │ Context  │              │  Truth   │
└──────────┘           └──────────┘              └──────────┘
```

### The 4 Core RAGAS Metrics
1. **Faithfulness** (Hallucination checker): Measures if the generated answer is derived *only* from the retrieved context.
   $$\text{Faithfulness Score} = \frac{\text{Number of statements in answer supported by context}}{\text{Total number of statements in generated answer}}$$
2. **Answer Relevancy** (Response quality): Evaluates if the generated answer directly addresses the query. Calculated by prompting a judge LLM to formulate hypothetical questions from the answer and measuring their vector similarity to the original query.
3. **Context Precision** (Retrieval order accuracy): Checks if the retrieved chunks containing relevant information are ranked at the top of the search results.
4. **Context Recall** (Retrieval completeness): Evaluates if all the information required to answer the question is present in the retrieved chunks by comparing the chunks to the reference ground truth.

---

### DeepEval: Extending RAG Evaluation

While **RAGAS** provides standard mathematical formulations for RAG components, **DeepEval** is an open-source enterprise-grade LLM evaluation framework designed for regression testing and continuous integration (CI/CD). It extends RAGAS capabilities in several key ways:

```
                   ┌──────────────────────────────────────┐
                   │               DeepEval               │
                   └──────────────────┬───────────────────┘
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  Unit Testing & CI/CD          G-Eval Framework              Comprehensive
  Integrates with Pytest to     Enables custom evaluator      Guardrails
  run evaluations as test       criteria based on natural     Checks for toxicity,
  assertions.                   language guidelines.          bias, and compliance.
```

1. **Unit Testing Integration (Pytest support)**: DeepEval allows developers to write evaluations as Python unit tests using Pytest. A test fails if a score drops below a set threshold, enabling automated regression checks in production deployment pipelines:
   ```python
   # Example DeepEval assertion
   assert faithfulness_metric.score >= 0.8, "Answer contains hallucinations!"
   ```
2. **G-Eval (Custom Evaluators)**: DeepEval implements **G-Eval**, a framework that enables developers to define custom evaluation criteria using simple natural language instructions. The framework automatically converts these guidelines into scoring criteria executed by the LLM judge.
3. **Broader Guardrail Metrics**: Beyond standard retrieval metrics, DeepEval includes built-in guardrail metrics such as:
   - **Toxicity**: Evaluates if the LLM response contains offensive or harmful language.
   - **Bias**: Identifies gender, racial, or political bias in generations.
   - **Conversational Alignment**: Evaluates multi-turn conversation memory and consistency.
4. **Vast SDK Backends**: DeepEval supports a broader range of local and cloud LLM APIs natively, providing comprehensive dashboards for tracking historical experiment runs.

---
