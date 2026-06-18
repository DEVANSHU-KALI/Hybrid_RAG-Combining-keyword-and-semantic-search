# Script Explanation: `12) main.md`

## 1. Overview
The primary role of the `main.py` script is to serve as the **Web API Gateway** for our backend. It uses the **FastAPI** web framework to expose our RAG pipeline as a web service. 

Its core responsibilities include:
* **Bootstrapping the Server**: Initializes the FastAPI application configuration.
* **Startup Health Check**: Performs an automatic startup check to verify that our Qdrant vector database is running and reachable.
* **Request Validation**: Defines a Pydantic data schema (`QueryRequest`) to validate that incoming requests contain properly structured JSON data.
* **Exposing the Chat Endpoint**: Exposes a `/chat` POST endpoint that receives the user's prompt, triggers our asynchronous RAG pipeline, and returns the generated response.

---
