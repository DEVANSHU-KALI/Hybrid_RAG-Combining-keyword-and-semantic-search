# Script Explanation: `6) injest_documents.md`

## 1. Overview
The primary role of `backend/injest_documents.py` is reading local documents (`.pdf` and `.txt`), extracting narrative text and structured tables, chunking text using `text_chunker.py`, embedding chunks into **768D dense vectors**, attaching multi-tenant metadata payloads (`tenant_id`, `user_id`), and asynchronously upserting points into the Qdrant vector database using **`AsyncQdrantClient`**.

---

## 2. Line-by-Line & Block-by-Block Code Walkthrough

### Part A: Imports & Async Qdrant Client Setup (Lines 1–19)

```python
1: import os
2: import asyncio
5: from qdrant_client import AsyncQdrantClient
6: from qdrant_client.models import PointStruct
9: from .embedding_model import embedding_model
12: from .text_chunker import text_splitter, chunk_markdown_sections
15: client = AsyncQdrantClient(host="localhost", port=6333)
18: COLLECTION_NAME = "rag_docs"
```
* **Line 1 & 2 (`import os`, `import asyncio`)**: Standard libraries for directory traversal and running async loops.
* **Line 5 & 6 (`from qdrant_client import AsyncQdrantClient`)**: **Production Upgrade!** Uses `AsyncQdrantClient` instead of synchronous `QdrantClient`. This ensures network calls to Qdrant never block the main Python Event Loop.
* **Line 9 (`from .embedding_model...`)**: Imports `embedding_model` (`BAAI/bge-base-en-v1.5`, 768D HuggingFace embeddings).
* **Line 12 (`from .text_chunker...`)**: Imports `text_splitter` (768D semantic chunker) and `chunk_markdown_sections` (markdown table preservation chunker).
* **Line 15 (`client = AsyncQdrantClient(...)`)**: Connects asynchronously to local Qdrant container on port `6333`.

---

### Part B: Multimodal PDF Table Extractor (`extract_pdf_content`, Lines 22–73)

```python
22: def extract_pdf_content(file_path: str) -> list[dict]:
```
* Accepts a `.pdf` file path and returns a list of dictionaries containing page number, page text (with extracted Markdown tables), and table status.

```python
32:         import pdfplumber
33:         with pdfplumber.open(file_path) as pdf:
34:             for page_num, page in enumerate(pdf.pages, start=1):
36:                 page_text = page.extract_text() or ""
39:                 tables = page.extract_tables()
```
* Uses `pdfplumber` to extract grid tables from PDF pages and format them as Markdown table strings (`| Header | Header |`).

---

### Part C: Multi-Tenant Async Ingestion Engine (`ingest_documents`, Lines 77–182)

```python
77: async def ingest_documents(
78:     folder_path: str,
79:     tenant_id: str = "tenant_default",
80:     user_id: str = "usr_admin"
81: ):
```
* Defined as an **`async def`** function so it can be called inside live API web endpoints without blocking concurrent queries from other users.

```python
101:         if filename.endswith(".pdf"):
102:             pdf_pages = extract_pdf_content(file_path)
103:             for page_data in pdf_pages:
104:                 chunks = chunk_markdown_sections(page_data["text"])
124:         elif filename.endswith(".txt"):
129:             chunks = text_splitter.create_documents([text])
```
* **PDFs:** Passes text to `chunk_markdown_sections()` to preserve tables.
* **TXTs:** Passes text to `text_splitter.create_documents()` for 768D semantic percentile chunking.

```python
149:     embeddings = embedding_model.embed_documents(documents)
```
* Computes 768D vector embeddings for all document chunks in a single GPU/CPU batch.

```python
177:     await client.upsert(
178:         collection_name=COLLECTION_NAME,
179:         points=points
180:     )
```
* **Line 177 (`await client.upsert(...)`)**: **Non-blocking DB Upsert!** Uploads points to Qdrant asynchronously. While network bytes are transferred, the asyncio event loop remains free to serve other incoming user search requests.

```python
184: if __name__ == "__main__":
185:     asyncio.run(ingest_documents("data/rag_concepts", tenant_id="tenant_default", user_id="usr_admin"))
```
* Uses `asyncio.run()` when executing the script directly from the CLI.

---

## 3. Code History: Before vs. After

### Before (Synchronous Ingestion Engine)
```python
client = QdrantClient(host="localhost", port=6333)

def ingest_documents(folder_path: str):
    ...
    client.upsert(collection_name=COLLECTION_NAME, points=points) # Sync blocking network call
```

### After (Enterprise Non-Blocking Async Ingestion Engine)
```python
client = AsyncQdrantClient(host="localhost", port=6333)

async def ingest_documents(folder_path: str, tenant_id: str = "tenant_default", user_id: str = "usr_admin"):
    ...
    await client.upsert(collection_name=COLLECTION_NAME, points=points) # Non-blocking async network call
```

---

## 4. Full Pipeline Execution Flow Diagram

```
                 Local Folder ("data/rag_concepts")
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
          PDF Document                    TXT Document
               │                               │
    extract_pdf_content()               file.read()
  (pdfplumber table extract)                   │
               │                               ▼
               ▼                     text_splitter.create_documents()
   chunk_markdown_sections()       (768D Semantic Percentile Chunker)
  (1000 char budget + table lock)              │
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                embedding_model.embed_documents()
                (Generates 768-dimensional vectors)
                               │
                               ▼
               PointStruct Payload Tagging
       (tenant_id, user_id, document_type, page_number)
                               │
                               ▼
                    await client.upsert()
        (Async Non-Blocking Qdrant Upsert: "rag_docs")
```

---

## 5. Worked Real-World Ingestion Trace Example

Below is a complete end-to-end trace showing how a real PDF document containing narrative text and a Markdown table gets parsed, chunked, embedded, and stored in Qdrant.

### 1. Raw PDF Page Content (`file.pdf`, Page 1)
```
Q3 Financial Performance Report
Summary of revenue and growth metrics for tenant_default.

| Quarter | Revenue | Growth |
| Q1 | $1.2M | +12% |
| Q2 | $1.5M | +25% |

All figures verified by internal audit.
```

### 2. Output of `extract_pdf_content("file.pdf")`
`pdfplumber` detects the grid table and converts rows to Markdown table syntax:
```python
[
    {
        "page_number": 1,
        "text": "Q3 Financial Performance Report\nSummary of revenue and growth metrics for tenant_default.\n\n| Quarter | Revenue | Growth |\n| --- | --- | --- |\n| Q1 | $1.2M | +12% |\n| Q2 | $1.5M | +25% |\n\nAll figures verified by internal audit.",
        "has_table": True
    }
]
```

### 3. Output of `chunk_markdown_sections(page_text)`
`chunk_markdown_sections` processes lines sequentially, keeping table boundaries locked together:
```python
chunks = [
    "Q3 Financial Performance Report\nSummary of revenue and growth metrics for tenant_default.\n\n| Quarter | Revenue | Growth |\n| --- | --- | --- |\n| Q1 | $1.2M | +12% |\n| Q2 | $1.5M | +25% |\n\nAll figures verified by internal audit."
]
```

### 4. Output of `embedding_model.embed_documents(chunks)`
Generates a 768-dimensional float vector:
```python
embeddings = [
    [0.0142, -0.0521, 0.0891, ..., -0.0112] # 768 float values
]
```

### 5. Final Qdrant `PointStruct` Payload Inserted into Database
```python
PointStruct(
    id=0,
    vector=[0.0142, -0.0521, 0.0891, ..., -0.0112],
    payload={
        "text": "Q3 Financial Performance Report\nSummary of revenue...\n| Quarter | Revenue | Growth |...",
        "source": "financial_report.pdf",
        "chunk_id": 0,
        "tenant_id": "tenant_default",
        "user_id": "usr_admin",
        "document_type": "pdf",
        "page_number": 1,
        "has_table": True
    }
)
```
