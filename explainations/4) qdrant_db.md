# Script Explanation: `4) qdrant_db.md`

## 1. Overview
The primary role of the `qdrant_db.py` script is to initialize the vector database infrastructure. It connects to a local instance of the **Qdrant Vector Database** (running inside a Docker container) and checks if the required storage collection (named `"rag_docs"`) already exists. If the collection is missing, the script creates it, configuring it to store 384-dimensional vectors and use Cosine Distance to calculate vector similarity.

---

## 2. Code Walkthrough

### Imports
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
```
- **Lines 1–2**: We import the primary connection client class `QdrantClient`, along with the configuration helpers `VectorParams` and `Distance` from Qdrant's HTTP model package.

---