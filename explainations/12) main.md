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

### Request Validation Schema
```python
# Request Validation
class QueryRequest(BaseModel):
    prompt: str
```
- **Lines 37–38**:
  - We declare a class `QueryRequest` inheriting from Pydantic's `BaseModel`.
  - We define a single required attribute `prompt` of type `str`.
  - *What does this do?* This defines the data schema of requests our API accepts. If a user submits a request with a missing `prompt` field or an incorrect format (e.g., passing a list instead of a string), FastAPI will automatically reject the request with a `422 Unprocessable Entity` error code, protecting our internal functions from bad inputs.

---

### Endpoint Mapping
```python
# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):

    result = await generate_answer(request.prompt)

    return result
```
- **Lines 42–47**:
  - We define an asynchronous routing handler `chat_endpoint` using the `@app.post("/chat")` decorator. This binds HTTP POST requests sent to `/chat` to this function.
  - We declare `request: QueryRequest` as a parameter. FastAPI parses the incoming JSON body, validates it against our schema, and instantiates the `request` variable.
  - We extract the prompt value using `request.prompt` and call the asynchronous RAG pipeline using `await generate_answer()`.
  - The pipeline returns the result dictionary, which FastAPI automatically converts (serializes) into a standard JSON response string and sends back to the client.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
             HTTP POST Client Request: {"prompt": "text"}
                                │
                                ▼
                       FastAPI Route Match
                            (/chat)
                                │
                                ▼
                      Pydantic Schema Check
                         (QueryRequest)
                 ├── Invalid ──► HTTP 422 Error
                 └── Valid   ──► Instantiate request
                                │
                                ▼
                       Execute chat_endpoint
                                │
                                ▼
                     await generate_answer()
                      (Triggers RAG Pipeline)
                                │
                                ▼
                    Serialize Result Dict to JSON
                                │
                                ▼
               HTTP 200 OK Response back to Client
```

---