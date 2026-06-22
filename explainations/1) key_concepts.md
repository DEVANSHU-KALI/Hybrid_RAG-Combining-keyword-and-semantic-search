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