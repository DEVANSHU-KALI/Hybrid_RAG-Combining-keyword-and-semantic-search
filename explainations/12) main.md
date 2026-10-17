# Script Explanation: `12) main.md`

## 1. Overview
The primary role of `backend/main.py` is acting as the **Production API Gateway & Web Server** using the FastAPI framework. It exposes the hybrid RAG pipeline to external client applications (Streamlit frontend, React/Vue dashboards, or third-party webhooks).

This script is responsible for:
1. **CORS Security Configuration**: Permits frontend applications running on different ports or domains to make API requests without browser cross-origin blocks.
2. **Sliding-Window Rate Limiting (`HTTP 429`)**: Protects the expensive LLM endpoints against spam, Denial-of-Wallet (DoW) attacks, and brute-force flooding.
3. **Async Startup Health Check**: Pings Qdrant asynchronously on server launch to guarantee the vector database is operational before accepting incoming web traffic.
4. **Pydantic Request Validation**: Parses incoming JSON payloads and enforces strict typing and multi-tenant security tags (`prompt`, `tenant_id`, `user_id`).
5. **Real-Time SSE Streaming Endpoint (`/chat/stream`)**: Serves an asynchronous Server-Sent Events stream using `StreamingResponse`.
6. **Standard Synchronous Endpoint (`/chat`)**: Provides a backward-compatible, single-shot JSON response endpoint for evaluation frameworks (Ragas, DeepEval) and batch test scripts.

---

## 2. System Architecture & Request Lifecycle Flow

### Complete Request Lifecycle Diagram
```
                     User / Frontend Web Client (Streamlit :8501)
                                         │
                                         ▼  HTTP POST /chat/stream
                     ┌────────────────────────────────────────┐
                     │         CORS Middleware Check          │
                     │ (Verifies Origin, Headers, & Methods)  │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │    Sliding-Window Rate Limiter         │
                     │  Is IP within 20 req / 60 sec window?  │
                     └───────────┬────────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        [ Exceeds 20 requests ]     [ Within Quota Limit ]
         HTTP 429 Too Many Requests            │
         (Immediate Rejection)                 ▼
                             ┌──────────────────────────────────┐
                             │    Pydantic Schema Validation    │
                             │ (Checks prompt, tenant_id, usr)  │
                             └─────────────────┬────────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                          [ Valid JSON ]             [ Invalid Body ]
                                 │                 HTTP 422 Unprocessable
                                 ▼
                     ┌────────────────────────────────────────┐
                     │       Route Handler Execution          │
                     │    Calls stream_rag_answer(...)        │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   StreamingResponse (text/event-stream)│
                     │  Yields real-time SSE tokens to client │
                     └────────────────────────────────────────┘
```

---

## 3. Deep Line-by-Line Code Walkthrough

---

### Part A: Imports & App Instantiation (Lines 1–24)

```python
1: import time
2: from fastapi import FastAPI, HTTPException, Request
3: from fastapi.responses import StreamingResponse
4: from fastapi.middleware.cors import CORSMiddleware
5: from pydantic import BaseModel, Field
6: from qdrant_client import AsyncQdrantClient
7: from .rag_pipeline import generate_answer, stream_rag_answer
```
* **Line 1 (`import time`)**: Python standard library used to capture Unix epoch timestamps (`time.time()`) for calculating sliding-window rate limits.
* **Line 2 (`FastAPI, HTTPException, Request`)**:
  * `FastAPI`: The core web framework class.
  * `HTTPException`: Used to trigger standard HTTP error responses (e.g. HTTP 429 for rate limiting, HTTP 422 for bad data).
  * `Request`: Represents the incoming raw HTTP request object, allowing inspection of client IP address, URL path, and headers.
* **Line 3 (`StreamingResponse`)**: Specialized FastAPI response class that keeps the HTTP connection open to stream chunks using Server-Sent Events (`text/event-stream`).
* **Line 4 (`CORSMiddleware`)**: ASGI middleware used to inject security headers allowing web browsers to communicate across different host ports.
* **Line 5 (`BaseModel, Field`)**: Pydantic classes for building data validation schemas.
* **Line 6 (`AsyncQdrantClient`)**: Non-blocking asynchronous client used in the startup lifecycle health check.
* **Line 7 (`generate_answer, stream_rag_answer`)**: Imports the dual core RAG orchestrators from `rag_pipeline.py`.

```python
23: app = FastAPI(title="Enterprise Hybrid RAG API")
```
* **Line 23 (`app = FastAPI(...)`)**: Creates the main ASGI application instance.

---

### Part B: CORS Middleware Configuration (Lines 26–32)

```python
26: app.add_middleware(
27:     CORSMiddleware,
28:     allow_origins=["*"],
29:     allow_credentials=True,
30:     allow_methods=["*"],
31:     allow_headers=["*"],
32: )
```
* **Why is CORS required?**
  Web browsers enforce a security rule called the **Same-Origin Policy**. If your Streamlit frontend runs on `http://localhost:8501` and tries to fetch data from FastAPI on `http://localhost:8000`, the browser will automatically block the request because the ports differ (`8501` vs `8000`).
* **`allow_origins=["*"]`**: Allows any client origin domain or port to access this API (can be restricted to `["http://localhost:8501"]` in strict enterprise networks).
* **`allow_methods=["*"]`**: Allows all HTTP verbs (`GET`, `POST`, `OPTIONS`).
* **`allow_headers=["*"]`**: Permits custom authentication headers (like `Authorization: Bearer <token>`).

---

### Part C: Sliding-Window Rate Limiter Middleware (Lines 40–69)

In production AI applications, calling LLM endpoints is computationally expensive and incurs cloud API costs. A malicious user or run-away loop script could send 500 requests in 10 seconds, driving up massive bills or crashing the server.

```python
40: RATE_LIMIT_REQUESTS = 20          # Max allowed requests per client IP per window
41: RATE_LIMIT_WINDOW_SEC = 60        # Sliding time window in seconds
42: user_request_history = {}          # Dictionary mapping client IP -> list of recent request float timestamps
```
* **Line 40 (`RATE_LIMIT_REQUESTS = 20`)**: Quota limit allowing a maximum of 20 requests per client IP address.
* **Line 41 (`RATE_LIMIT_WINDOW_SEC = 60`)**: The sliding duration window (60 seconds / 1 minute).
* **Line 42 (`user_request_history = {}`)**: In-memory database mapping each client IP address to a list of recent request timestamps:
  ```python
  user_request_history = {
      "192.168.1.15": [1729001000.12, 1729001005.45, 1729001012.88],
      "10.0.0.42": [1729001020.00]
  }
  ```

```python
45: @app.middleware("http")
46: async def rate_limiting_middleware(request: Request, call_next):
```
* **Line 45 (`@app.middleware("http")`)**: Decorator registering an ASGI HTTP middleware. Every single HTTP request that hits the server passes through this function **before** reaching any route handler!
* **Line 46 (`async def rate_limiting_middleware(...)`)**:
  * `request`: The incoming HTTP request.
  * `call_next`: A function pointer that forwards the request to the next handler in line.

```python
48:     if request.url.path in ["/chat", "/chat/stream"]:
```
* **Line 48 (`if request.url.path in [...]`)**:
  * **Targeted Rate Limiting:** We only enforce the rate limit on resource-heavy LLM endpoints (`/chat` and `/chat/stream`).
  * Lightweight endpoints (like health checks, docs at `/docs`, or OpenAPI specs) are exempted so monitoring systems are never blocked.

```python
49:         client_ip = request.client.host if request.client else "127.0.0.1"
50:         now = time.time()
```
* **Line 49 (`client_ip = ...`)**: Extracts the client's IP address (`request.client.host`). If running behind an unusual proxy or test harness where client is `None`, safely defaults to `"127.0.0.1"`.
* **Line 50 (`now = time.time()`)**: Records the current Unix epoch time in floating-point seconds (e.g., `1729188405.32`).

```python
53:         timestamps = user_request_history.get(client_ip, [])
54:         valid_timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW_SEC]
```
* **Line 53 (`timestamps = user_request_history.get(client_ip, [])`)**: Retrieves the list of previous request timestamps for this specific IP. If it is the user's first request, returns an empty list `[]`.
* **Line 54 (`valid_timestamps = [...]`)**: **The Sliding Window Pruning Engine!**
  * `now - ts`: Calculates how many seconds ago each previous request occurred.
  * `now - ts < 60`: Keeps **only** timestamps that occurred within the last 60 seconds!
  * Any request older than 60 seconds is automatically pruned and discarded from memory.

```python
57:         if len(valid_timestamps) >= RATE_LIMIT_REQUESTS:
58:             raise HTTPException(
59:                 status_code=429,
60:                 detail="Too Many Requests. Rate limit exceeded (20 requests/minute). Please wait before retrying."
61:             )
```
* **Lines 57–61 (Quota Enforcement & HTTP 429)**:
  * If the count of recent requests is 20 or more (`>= 20`), the user has exceeded their quota.
  * Immediately raises `HTTPException(status_code=429)`.
  * **Why 429?** `429 Too Many Requests` is the official HTTP standard code for rate limiting. The request is rejected instantly without wasting any LLM or database compute!

```python
64:         valid_timestamps.append(now)
65:         user_request_history[client_ip] = valid_timestamps
```
* **Lines 64–65 (Updating History)**: If the request is under the quota, appends the current timestamp `now` to `valid_timestamps` and saves the updated list back into `user_request_history`.

```python
68:     response = await call_next(request)
69:     return response
```
* **Lines 68–69 (`response = await call_next(request)`)**: Passes control forward to the actual route handler (`/chat` or `/chat/stream`), waits for it to execute, and returns the response back to the client.

---

### Step-by-Step Numerical Example of Rate Limiting in Action

Let's trace how the sliding window behaves for a client at IP `192.168.1.50`:

* **Initial State:** `user_request_history["192.168.1.50"] = []`.
* **At $T = 0$ seconds:** Client sends Request #1.  
  $\rightarrow$ `valid_timestamps = [0.0]`. Allowed!
* **From $T = 1$ to $T = 15$ seconds:** Client rapidly sends 19 more requests (Requests #2 through #20).  
  $\rightarrow$ `len(valid_timestamps) = 20`. All allowed!
* **At $T = 20$ seconds:** Client attempts Request #21.  
  $\rightarrow$ Evaluates: `now = 20.0`. All 20 previous timestamps occurred between $T=0$ and $T=15$.  
  $\rightarrow$ `now - ts < 60` is True for all 20 timestamps ($20 - 0 = 20 < 60$).  
  $\rightarrow$ `len(valid_timestamps) == 20 >= 20` $\longrightarrow$ **TRIGGER HTTP 429!** Request is blocked immediately with *"Too Many Requests"*.
* **At $T = 62$ seconds:** Client sends Request #22.  
  $\rightarrow$ Evaluates: `now = 62.0`.  
  $\rightarrow$ Request #1 was at $T=0$: $62 - 0 = 62 \ge 60$ $\rightarrow$ **Expired & pruned!**  
  $\rightarrow$ `valid_timestamps` count drops to 19.  
  $\rightarrow$ Request #22 is **ALLOWED**!

---

### Part D: Async Database Health Check Hook (Lines 72–80)

```python
72: @app.on_event("startup")
73: async def startup_check():
74:     try:
75:         client = AsyncQdrantClient(host="localhost", port=6333)
76:         await client.get_collections()
77:         print("\n✅ Async Qdrant connection successful.\n")
78:     except Exception as e:
79:         print(f"\n❌ Async Qdrant is not running on localhost:6333: {e}\n")
```
* **Line 72 (`@app.on_event("startup")`)**: Lifecycle hook that runs once automatically when FastAPI launches.
* **Lines 75–76**: Uses `AsyncQdrantClient` to ping Qdrant on port 6333 asynchronously (`await client.get_collections()`).
* **Why Async?** Ensures the server startup does not block the event loop thread while checking database availability.

---

### Part E: Multi-Tenant Pydantic Validation Schema (Lines 83–88)

```python
83: class QueryRequest(BaseModel):
84:     prompt: str = Field(..., description="User query prompt string")
85:     tenant_id: str = Field("tenant_default", description="Multi-tenant organization identity tag")
86:     user_id: str = Field("usr_admin", description="Authenticated user identity tag")
```
* **Line 83 (`class QueryRequest(BaseModel)`)**: Defines the expected JSON schema for incoming POST request bodies.
* **Line 84 (`prompt: str = Field(...)`)**:
  * The ellipsis (`...`) indicates that `prompt` is **mandatory**.
  * If a client sends an empty body or forgets the `prompt` key (`{}`), FastAPI automatically returns an **HTTP 422 Unprocessable Entity** error with exact details on what was missing.
* **Lines 85–86 (`tenant_id`, `user_id`)**:
  * Optional fields with safe defaults (`"tenant_default"`, `"usr_admin"`).
  * In production, these receive the authenticated tenant and user identities from frontend auth tokens to enforce data boundary isolation in Qdrant.

---

### Part F: Route Handlers (`/chat/stream` & `/chat`) (Lines 91–116)

```python
92: @app.post("/chat/stream")
93: async def stream_chat_endpoint(request: QueryRequest):
94:     return StreamingResponse(
95:         stream_rag_answer(
96:             query=request.prompt,
97:             tenant_id=request.tenant_id,
98:             user_id=request.user_id
99:         ),
100:        media_type="text/event-stream"
101:    )
```
* **Line 92 (`@app.post("/chat/stream")`)**: Registers the production streaming endpoint.
* **Line 94 (`StreamingResponse(...)`)**:
  * Takes the asynchronous generator `stream_rag_answer(...)`.
  * `media_type="text/event-stream"`: Sets the official W3C HTTP header for Server-Sent Events (SSE). The browser or frontend client reads data packets continuously without closing the connection until the stream ends.

```python
109: @app.post("/chat")
110: async def chat_endpoint(request: QueryRequest):
111:     result = await generate_answer(
112:         query=request.prompt,
113:         tenant_id=request.tenant_id,
114:         user_id=request.user_id
115:     )
116:     return result
```
* **Lines 109–116 (Non-Streaming Fallback)**:
  * Awaits `generate_answer(...)` and returns a single JSON object containing `answer`, `citations`, `contexts`, `chunk_ids`, and `telemetry`.
  * Designed for evaluation scripts (Ragas, DeepEval) and automated integration tests.

---

## 4. Deep Technical Concepts

### 1. Sliding Window vs. Fixed Window Rate Limiting

Why did we build a **Sliding Window** rate limiter instead of a simpler **Fixed Window** counter?

```
FIXED WINDOW VULNERABILITY (Reset at every clock minute):
[ Window 1: 00:00 - 01:00 ]          [ Window 2: 01:00 - 02:00 ]
             20 requests sent                      20 requests sent
             at 00:59                              at 01:01
                  └──────── 40 requests in 2 seconds! ────────┘
                 (Fixed counter resets at 01:00, allowing burst!)

SLIDING WINDOW PROTECTION (Our Implementation):
Evaluates actual rolling duration (now - 60 seconds).
At 01:01, requests from 00:59 are still inside the 60-second window!
Total requests = 40 >= 20 ──► HTTP 429 REJECTED!
```

* **Fixed Window:** Counts requests per clock minute (e.g., 12:00 to 12:01). An attacker can send 20 requests at 12:00:59 and another 20 at 12:01:01—resulting in **40 requests in 2 seconds**, defeating the purpose.
* **Sliding Window (Our Design):** Evaluates a true rolling 60-second window based on exact float timestamps (`now - ts < 60`). Bursts across clock boundaries are mathematically impossible!

---

### 2. How FastAPI Middleware Operates (ASGI Onion Model)

In FastAPI (based on Starlette and the ASGI specification), middleware acts like an "onion layer":

```
  Incoming HTTP Request
           │
           ▼
     [ Middleware: CORS ]
           │
           ▼
     [ Middleware: Rate Limiter ] ──► (Reject 429 if quota exceeded)
           │
           ▼
     [ Route Handler: /chat/stream ]
           │
           ▼
     [ StreamingResponse Generator ]
           │
           ▼
  Outgoing HTTP Response Stream
```

Every request enters through the outer layers (CORS $\rightarrow$ Rate Limiter) before reaching the route. If any middleware rejects the request, downstream code (database retrieval and LLM calls) is never touched!

---

## 5. Architectural Choices and Trade-offs

| Component | Our Approach | Alternative | Why Our Approach Was Chosen |
| :--- | :--- | :--- | :--- |
| **Rate Limiter** | Sliding Window (In-Memory per IP) | Redis-backed Token Bucket | Zero external dependencies for local/single-server setup; instant execution with zero network latency. |
| **API Protocol** | Server-Sent Events (`text/event-stream`) | WebSockets (`ws://`) | SSE is unidirectional, simpler to debug, fully compatible with HTTP/2, and auto-reconnects natively in browsers. |
| **Validation** | Pydantic v2 `BaseModel` | Manual Dict Parsing | Auto-generates interactive OpenAPI documentation (`/docs`) and validates types before code execution. |
| **Startup Check** | `AsyncQdrantClient` | Blocking `QdrantClient` | Prevents blocking the event loop on startup; allows graceful error logs if the Docker container is halted. |
