# Imports time standard library for tracking request timestamps in sliding-window rate limiting
import time

# Imports FastAPI framework, HTTP exceptions, and request objects
from fastapi import FastAPI, HTTPException, Request

# Imports StreamingResponse for streaming Server-Sent Events (SSE) token data over HTTP
from fastapi.responses import StreamingResponse

# Imports CORSMiddleware to allow frontend clients (like Streamlit or React) to make cross-origin requests
from fastapi.middleware.cors import CORSMiddleware

# Imports Pydantic BaseModel and Field for JSON schema validation
from pydantic import BaseModel, Field

# Imports AsyncQdrantClient for non-blocking startup health check
from qdrant_client import AsyncQdrantClient

# Imports RAG pipeline functions: non-streaming generate_answer and SSE streaming stream_rag_answer
from .rag_pipeline import generate_answer, stream_rag_answer

# Instantiate FastAPI application instance
app = FastAPI(title="Enterprise Hybrid RAG API")

# Enable CORS middleware allowing external frontend clients on any domain to communicate with API
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CONCEPT: Token-Bucket Sliding-Window Rate Limiter Middleware
# In production, malicious users or scripts can flood LLM endpoints, causing high cloud costs.
# Rate limiters check request timestamps in a 60-second sliding window per IP/user.
# If requests exceed quota (e.g. 20 req/min), it immediately rejects with HTTP 429 Too Many Requests.

RATE_LIMIT_REQUESTS = 20          # Max allowed requests per client IP per window
RATE_LIMIT_WINDOW_SEC = 60        # Sliding time window in seconds
user_request_history = {}          # Dictionary mapping client IP -> list of recent request float timestamps


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    # Intercept HTTP requests sent to LLM endpoints
    if request.url.path in ["/chat", "/chat/stream"]:
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()

        # Clean history: discard timestamps older than 60 seconds
        timestamps = user_request_history.get(client_ip, [])
        valid_timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW_SEC]

        # Reject request with HTTP 429 if client exceeds 20 requests in the current window
        if len(valid_timestamps) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests. Rate limit exceeded (20 requests/minute). Please wait before retrying."
            )

        # Record current request timestamp and update history
        valid_timestamps.append(now)
        user_request_history[client_ip] = valid_timestamps

    # Pass request to downstream route handler
    response = await call_next(request)
    return response


# FastAPI startup lifecycle hook checking Qdrant availability asynchronously
@app.on_event("startup")
async def startup_check():
    try:
        client = AsyncQdrantClient(host="localhost", port=6333)
        await client.get_collections()
        print("\n✅ Async Qdrant connection successful.\n")
    except Exception as e:
        print(f"\n❌ Async Qdrant is not running on localhost:6333: {e}\n")


# CONCEPT: Multi-Tenant Pydantic Request Schema
# Automatically validates incoming JSON request body structures
class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User query prompt string")
    tenant_id: str = Field("tenant_default", description="Multi-tenant organization identity tag")
    user_id: str = Field("usr_admin", description="Authenticated user identity tag")


# CONCEPT: Server-Sent Events (SSE) Streaming Endpoint
@app.post("/chat/stream")
async def stream_chat_endpoint(request: QueryRequest):
    """
    Production SSE Streaming Endpoint.
    Yields data: {"type": "metadata" | "token" | "telemetry"} tokens in real-time.
    """
    return StreamingResponse(
        stream_rag_answer(
            query=request.prompt,
            tenant_id=request.tenant_id,
            user_id=request.user_id
        ),
        media_type="text/event-stream"
    )


# Standard Synchronous/Blocking Endpoint for backward compatibility
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    result = await generate_answer(
        query=request.prompt,
        tenant_id=request.tenant_id,
        user_id=request.user_id
    )
    return result



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from fastapi import FastAPI
# from pydantic import BaseModel
# from qdrant_client import QdrantClient
# from .rag_pipeline import generate_answer
# 
# app = FastAPI(title="Hybrid RAG API")
# 
# @app.on_event("startup")
# async def startup_check():
#     try:
#         client = QdrantClient(host="localhost", port=6333)
#         client.get_collections()
#         print("\n✅ Qdrant connection successful.\n")
#     except Exception:
#         print("\n❌ Qdrant is not running.\n")
# 
# class QueryRequest(BaseModel):
#     prompt: str
# 
# @app.post("/chat")
# async def chat_endpoint(request: QueryRequest):
#     result = await generate_answer(request.prompt)
#     return result
# =====================================================================