from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient

from .rag_pipeline import generate_answer


app = FastAPI(title="Hybrid RAG API")

# -----------------------------
# Qdrant Startup Check
# -----------------------------
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

# -----------------------------


# Request Validation
class QueryRequest(BaseModel):
    prompt: str


# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):

    result = await generate_answer(request.prompt)

    return result