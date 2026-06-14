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

### Database Connection
```python
client = QdrantClient(
    host="localhost",
    port=6333
)
```
- **Lines 4–7**: We instantiate our connection client. We point it to `localhost` (our own machine) on port `6333` (the default port where Qdrant runs when launched via Docker).

---

### Fetching Collection Names
```python
collection_name = "rag_docs"

collections = client.get_collections().collections

existing_collections = [
    collection.name
    for collection in collections
]
```
- **Lines 9–16**:
  1. We set the name of our target collection: `"rag_docs"`.
  2. We call `client.get_collections().collections` to get a list of metadata objects for all collections currently running in our database instance.
  3. We use a **list comprehension** to extract only the text name of each collection from the metadata objects, storing them in `existing_collections`.

---