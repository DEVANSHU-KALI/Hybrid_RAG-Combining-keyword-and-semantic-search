# Script Explanation: `12) main.md`

## 1. Overview
The primary role of the `main.py` script is to serve as the **Web API Gateway** for our backend. It uses the **FastAPI** web framework to expose our RAG pipeline as a web service. 

Its core responsibilities include:
* **Bootstrapping the Server**: Initializes the FastAPI application configuration.
* **Startup Health Check**: Performs an automatic startup check to verify that our Qdrant vector database is running and reachable.
* **Request Validation**: Defines a Pydantic data schema (`QueryRequest`) to validate that incoming requests contain properly structured JSON data.
* **Exposing the Chat Endpoint**: Exposes a `/chat` POST endpoint that receives the user's prompt, triggers our asynchronous RAG pipeline, and returns the generated response.

---

## 2. Code Walkthrough

### Imports and Server Initialization
```python
from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient

from .rag_pipeline import generate_answer

app = FastAPI(title="Hybrid RAG API")
```
- **Lines 1–9**:
  - We import `FastAPI` to build our API routing and request-handling structure.
  - We import `BaseModel` from `pydantic` to handle data serialization and type verification.
  - We import `QdrantClient` to perform our startup database check.
  - We load our pipeline orchestrator `generate_answer` from our local directory.
  - We instantiate the web app as `app`, naming it `"Hybrid RAG API"`.

---

### Startup Lifecycle Check
```python
# Qdrant Startup Check
@app.on_event("startup")
async def startup_check():

    try:
        client = QdrantClient(
            host="localhost",
            port=6333
        )

        client.get_collections()

        print("\n✅ Qdrant connection successful.\n")

    except Exception:
        print("\n❌ Qdrant is not running.")
        print("Start your Qdrant Docker container first.\n")
```
- **Lines 14–31**:
  - We use the `@app.on_event("startup")` event hook decorator. This tells FastAPI to execute the nested function `startup_check()` during the server startup sequence before accepting any web requests.
  - **The Verification Try-Except Block**:
    - We initialize a synchronous `QdrantClient` pointing to `localhost:6333`.
    - We call `client.get_collections()` to send a test request to Qdrant.
    - If Qdrant is running, the call succeeds and we print a success message to the console.
    - If the server is offline or unreachable, an exception is caught. We print a warning to alert the developer to check their running Docker containers.

---