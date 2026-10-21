# Future Implementations & Architectural Roadmap

This document serves as the official tracking registry and engineering roadmap for architectural upgrades, advanced features, and optimizations planned for future iterations of this Multi-Tenant Hybrid RAG system.

---

## Roadmap Overview: Current State vs. Future Horizon

```
CURRENT STATE (Production Foundation):
  Query ──► Hybrid Search (BM25 + 768D Dense) ──► Cross-Encoder Rerank ──► Streaming LLM (Local/Cloud)
  • In-Memory Streamlit Chat Sessions
  • PDF Text & Markdown Table Extraction
  • Static CLI Document Ingestion

FUTURE HORIZON (Autonomous Enterprise Intelligence):
  Phase 1: Persistent Chat Storage & Conversational Memory (PostgreSQL + Query Rewriting)
  Phase 2: Full Multimodal Ingestion (Images, Diagrams, Charts via VLMs & ColPali)
  Phase 3: Agentic Multi-Step RAG (LangGraph, Query Decomposition, Self-Correction)
  Phase 4: Dynamic Web UI Document Upload & Async Processing (Celery + Redis)
  Phase 5: Enterprise RBAC Authentication & Document-Level Permissions
  Phase 6: Automated Continuous Evaluation & Guardrail Safety Shields
```

---

## 1. Persistent Chat History & Conversational Memory

### The Problem in Current System:
* **Session Ephemerality:** Streamlit stores conversation threads in `st.session_state`. When the browser is refreshed or closed, all chat history is wiped.
* **Single-Turn Blindness:** The retrieval pipeline treats every question as a standalone query. If a user asks *"What is dropout?"* followed by *"What are its main limitations?"*, the retriever searches for the ambiguous pronoun *"its"* without knowing the previous topic was dropout.

### Planned Architecture:

#### A. Relational / Document Database Layer
Implement a persistent storage backend (PostgreSQL with SQLAlchemy or MongoDB) with the following relational schema:

```
┌──────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│     Tenants      │ 1   * │         Users          │ 1   * │        Chats         │
│  tenant_id (PK)  ├───────┤  user_id (PK)          ├───────┤  chat_id (UUID, PK)  │
│  org_name        │       │  tenant_id (FK)        │       │  user_id (FK)        │
└──────────────────┘       │  email, role           │       │  title, created_at   │
                           └────────────────────────┘       └──────────┬───────────┘
                                                                       │ 1
                                                                       │ *
                                                            ┌──────────┴───────────┐
                                                            │       Messages       │
                                                            │  message_id (PK)     │
                                                            │  chat_id (FK)        │
                                                            │  role (user/assistant│
                                                            │  content (text)      │
                                                            │  citations (JSON)    │
                                                            │  latency_ms, cost    │
                                                            │  created_at          │
                                                            └──────────────────────┘
```

#### B. Conversational Query Rewriting (Contextual RAG)
Before passing the user prompt to retrieval, inject a lightweight LLM pre-processing step that condenses conversation history into a standalone retrieval query:

```python
# Conceptual Workflow:
# History: User: "Explain convolutional neural networks." -> AI: [Explains CNNs]
# Follow-up: User: "What are their main failure modes in medical imaging?"
#
# Rewriter Prompt: "Given the conversation history, rewrite the follow-up question
# into a self-contained search query."
#
# Rewritten Query: "Failure modes of convolutional neural networks in medical imaging"
```
* **Result:** Search accuracy on follow-up questions jumps from ~40% to >90%.

---

## 2. Multimodal Document Ingestion (Images, Diagrams, Charts & Graphs)

### The Problem in Current System:
* The current pipeline parses digital text and structured grid tables (`pdfplumber` $\rightarrow$ Markdown).
* Complex architectural diagrams (e.g., CNN layer diagrams, transformer attention heads), loss curves, performance graphs, flowcharts, and scanned document images are discarded during ingestion.

### Planned Architecture:

#### A. Dual-Track Multimodal Extraction Pipeline
```
                          Incoming PDF Document
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Digital Text & Tables           Images, Figures & Charts
            (pdfplumber / Docling)          (PyMuPDF / pdf2image)
                    │                               │
                    │                               ▼
                    │                   Vision-Language Model (VLM)
                    │                   (Qwen2-VL / GPT-4o-mini)
                    │                   "Generate detailed technical
                    │                    summary of this diagram"
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                     Combined Structured Markdown
              (Text + Markdown Tables + Image Descriptions)
                                    │
                                    ▼
                         768D Vector Embedding
```

#### B. Image Storage & Visual Grounding in UI
* Store extracted image assets in an S3-compatible object store (MinIO or AWS S3).
* Save image URLs and bounding box coordinates in the Qdrant payload.
* When the assistant references a chart or diagram in its answer, render the **actual source diagram image inline in the Streamlit chat feed** alongside text citations!

#### C. Native Vision Embedding Alternatives
* Investigate **ColPali** (Vision Retriever based on PaliGemma) which embeds raw PDF page screenshot patches directly into multi-vector space, eliminating OCR entirely.

---

## 3. Agentic RAG & Multi-Step Iterative Reasoning

### The Problem in Current System:
* The current pipeline is strictly **linear**: `Query -> Hybrid Search -> Rerank -> Generate`.
* If a question requires comparing two different concepts across multiple documents, or if the first retrieval search returns irrelevant chunks, the system cannot recover.

### Planned Architecture:

Transition from linear RAG to an autonomous graph-based agent (built with **LangGraph** or custom state machines):

```
                       User Prompt
                            │
                            ▼
                  ┌───────────────────┐
                  │ Query Analyzer &  │
                  │ Decomposer        │
                  └─────────┬─────────┘
                            │
            Is it a compound or multi-step question?
            ┌───────────────┴───────────────┐
            ▼ YES                           ▼ NO
   Break into Sub-queries           Single Hybrid Retrieval
   (Query 1, Query 2, Query 3)              │
            │                               ▼
            └───────────────┬───────────────┘
                            ▼
                  Cross-Encoder Reranker
                            │
                            ▼
                  ┌───────────────────┐
                  │ Self-RAG Grader   │◄──────────────┐
                  │ (Context Check)   │               │
                  └─────────┬─────────┘               │
                            │                         │
               Does context answer question?          │
               ┌────────────┴────────────┐            │
               ▼ YES                     ▼ NO         │
       Generate Grounded        Query Reformulation   │
       Answer with Citations    & Web Search Fallback ─┘
```

#### Key Agentic Capabilities to Add:
1. **Self-Correction & Relevance Grading:** An internal LLM check grades whether retrieved chunks contain the answer. If relevance is low, the agent reformulates the query and performs a second retrieval pass.
2. **Query Decomposition:** Complex queries (e.g., *"Compare the computational complexity of Adam optimizer vs. SGD with momentum and explain which fits large-batch training"*) are split into two parallel sub-queries, retrieved separately, and synthesized.
3. **Tool Calling Integration:** Equip the agent with external tools:
   * `Python Sandbox Tool`: For calculating mathematical formulas or statistics.
   * `Web Search Tool (Tavily / DuckDuckGo)`: As an automated fallback when local documents do not contain the answer.

---

## 4. Dynamic Web UI Ingestion & Background Task Queues

### The Problem in Current System:
* Documents must be manually placed inside the `data/` folder and ingested via the CLI (`python -m backend.injest_documents`) or synchronous Streamlit button.
* Ingesting large 100-page PDFs blocks the application and can lead to HTTP timeouts.

### Planned Architecture:
* **Drag-and-Drop Ingestion Drawer:** A file upload widget in the Streamlit sidebar allowing users to upload `.pdf`, `.docx`, and `.txt` files directly from their browser.
* **Distributed Task Queue (Celery + Redis / ARQ):** Offload document parsing, chunking, and embedding generation to an asynchronous background worker process.
* **Real-Time Progress Indicators:** Stream parsing progress (e.g., *"Extracting tables from Page 12/45..."*, *"Generating 768D embeddings..."*) back to the user interface via WebSockets or polling.

---

## 5. Enterprise Role-Based Access Control (RBAC) & Authentication

### The Problem in Current System:
* The current system relies on manual text inputs for `tenant_id` and `user_id` without password authentication or cryptographic verification.

### Planned Architecture:
* **Authentication Provider:** Integrate OAuth2 / OIDC via **Keycloak**, **Auth0**, or **Clerk** (supporting username/password, Google SSO, and Enterprise SAML).
* **Cryptographic JWT Tokens:** FastAPI endpoints will extract and verify `tenant_id`, `user_id`, and `roles` from signed JWT bearer headers.
* **Granular Document-Level Security (ACLs):**
  Add permission arrays to Qdrant payload:
  ```json
  {
    "tenant_id": "tenant_enterprise",
    "user_id": "usr_102",
    "access_groups": ["finance_team", "executives"],
    "clearance_level": "confidential"
  }
  ```
  Every query automatically injects user group filters into the Qdrant retrieval step, guaranteeing that users cannot retrieve documents above their security clearance.

---

## 6. Continuous Evaluation, Observability & Guardrails

### Planned Upgrades:
1. **In-App User Feedback Loop:**
   * Add Thumbs Up ($\text{👍}$) / Thumbs Down ($\text{👎}$) buttons to every assistant message.
   * Store user feedback in the database to curate high-quality benchmark datasets for fine-tuning and evaluation.
2. **Automated Production Ragas Pipeline:**
   * Periodically sample production queries and evaluate them automatically against Ragas metrics (*Faithfulness*, *Answer Relevancy*, *Context Precision*).
   * Alert engineering teams if faithfulness falls below 0.85 (indicating hallucination risk).
3. **Safety & Guardrail Shields (NeMo Guardrails / Llama-Guard):**
   * Input filtering: Detect and block prompt-injection attacks.
   * Output filtering: Mask sensitive Personally Identifiable Information (PII) before returning answers to the user.

---

## Priority & Implementation Roadmap Table

| Feature / Upgrade | Target Phase | Complexity | Primary Benefit |
| :--- | :---: | :---: | :--- |
| **Persistent Chat Storage (PostgreSQL/SQLite)** | Phase 1 | Medium | Chat history survives browser reloads; cross-device access |
| **Conversational Context & Query Rewriting** | Phase 1 | Medium | Flawless multi-turn dialogue and follow-up question answering |
| **Multimodal VLM Diagram & Image Extraction** | Phase 2 | High | Ingestion of architectures, flowcharts, graphs, and images |
| **Inline Diagram Rendering in Chat** | Phase 2 | Low-Med | Assistant displays actual source charts alongside text |
| **Agentic RAG & Query Decomposition (LangGraph)** | Phase 3 | High | Complex multi-hop reasoning, self-correction, web fallback |
| **Dynamic Drag-and-Drop Upload (Celery + Redis)** | Phase 4 | Medium | Upload and index personal documents directly from web UI |
| **Enterprise JWT Auth & Document RBAC** | Phase 5 | Medium | Military-grade data privacy and multi-tenant isolation |
| **Automated Ragas CI/CD & Guardrails** | Phase 6 | Medium | Zero hallucination drift, prompt-injection defense |

---

## Notes for Future Implementation Sprints
When picking up any of the above items for development:
1. Create a dedicated branch or feature script.
2. Preserve existing backward-compatibility for local offline inference (`LLM_PROVIDER=local`).
3. Add any newly encountered technical issues or discoveries to [`explainations/issues_faced.md`](file:///d:/projects/hybrid_rag%20-%20Copy/explainations/issues_faced.md).
