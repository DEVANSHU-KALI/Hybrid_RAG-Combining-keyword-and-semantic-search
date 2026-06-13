# Script Explanation: `7) semantic_retriever.md`

## 1. Overview
The primary role of the `semantic_retriever.py` script is to run the **semantic search** component of our retrieval pipeline. It connects asynchronously to our Qdrant vector database. When a user asks a question, this script converts the query string into a dense vector embedding using our preloaded embedding model, performs a vector search against the database, extracts the relevant fields (text, source file, chunk ID, and mathematical similarity score) from Qdrant's response, and returns them as a structured list.

---