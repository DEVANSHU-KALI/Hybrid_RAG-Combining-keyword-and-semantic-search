import os
import json
import time
from typing import AsyncGenerator
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langsmith import traceable

from .hybrid_retriever import hybrid_search
from .reranker import rerank_results

# Load environment variables from .env
load_dotenv()

# Multi-Provider Client Setup (Local llama.cpp/LM Studio or Groq Cloud API)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Selection Logic:
# 1. If LLM_PROVIDER == 'local' -> Force Local LLM (even if GROQ_API_KEY is present)
# 2. If LLM_PROVIDER == 'groq'  -> Force Groq Cloud API
# 3. Default fallback: If LLM_PROVIDER is unset, prefer Groq if key exists, otherwise Local
if LLM_PROVIDER == "local" or (not LLM_PROVIDER and (not GROQ_API_KEY or len(GROQ_API_KEY) <= 5)):
    llm_client = AsyncOpenAI(
        base_url=LOCAL_LLM_URL,
        api_key="dummy"
    )
    DEFAULT_MODEL = LOCAL_LLM_MODEL
else:
    llm_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    DEFAULT_MODEL = GROQ_MODEL


# CONCEPT: FinOps Pricing Dictionary (Per 1 Million Tokens)
TOKEN_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.50, "output": 0.50}
}


def calculate_query_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = TOKEN_PRICING.get(model_name, TOKEN_PRICING["default"])
    cost = (prompt_tokens / 1_000_000 * pricing["input"]) + (completion_tokens / 1_000_000 * pricing["output"])
    return round(cost, 6)


# CONCEPT: Server-Sent Events (SSE) Streaming Async Generator
# Production systems yield tokens real-time using SSE (Server-Sent Events) rather than 
# waiting 4-6 seconds for full answer completion. This reduces Time-To-First-Token (TTFT) 
# to under 350ms, transforming user-perceived performance.

@traceable
async def stream_rag_answer(query: str, 
    tenant_id: str = "tenant_default", 
    user_id: str = "usr_admin"
) -> AsyncGenerator[str, None]:
    start_time = time.time()
    ttft_time = 0.0

    # 1. Parallel Hybrid Retrieval
    retrieved_chunks = await hybrid_search(query, tenant_id=tenant_id, user_id=user_id, limit=10)

    # 2. Cross-Encoder Reranking
    reranked_chunks = await rerank_results(query, retrieved_chunks, top_k=3)

    # 3. Context & Citations Assembly
    context = "\n\n".join([chunk["text"] for chunk in reranked_chunks])
    citations = list(set([chunk["source"] for chunk in reranked_chunks]))

    prompt = f"""You are a helpful enterprise AI assistant.
Answer ONLY from the provided context. If the answer is not present, say: "I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}
"""

    # Yield Metadata Event First (Citations, Contexts & Chunk IDs)
    metadata_event = {
        "type": "metadata",
        "citations": citations,
        "contexts": [chunk["text"] for chunk in reranked_chunks],
        "chunk_ids": [chunk.get("chunk_id") for chunk in reranked_chunks],
        "tenant_id": tenant_id
    }
    yield f"data: {json.dumps(metadata_event)}\n\n"

    # 4. Stream LLM Response Tokens
    stream = await llm_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}
    )

    prompt_tokens = 0
    completion_tokens = 0
    first_token_received = False

    async for chunk in stream:
        if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
            first_token_received = True
            ttft_time = round((time.time() - start_time) * 1000, 2)

        # Check for usage stats in final streaming chunk
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

        if chunk.choices and chunk.choices[0].delta.content:
            token_text = chunk.choices[0].delta.content
            token_event = {"type": "token", "content": token_text}
            yield f"data: {json.dumps(token_event)}\n\n"

    # Fallback token estimation if usage omitted by local model server
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split()) * 4 // 3
        completion_tokens = 150

    total_time = round(time.time() - start_time, 2)
    estimated_cost = calculate_query_cost(DEFAULT_MODEL, prompt_tokens, completion_tokens)

    # Yield Telemetry Final Event
    telemetry_event = {
        "type": "telemetry",
        "ttft_ms": ttft_time,
        "total_latency_sec": total_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": estimated_cost,
        "model_used": DEFAULT_MODEL
    }
    yield f"data: {json.dumps(telemetry_event)}\n\n"


# Non-streaming helper for backwards compatibility
async def generate_answer(query: str, tenant_id: str = "tenant_default", user_id: str = "usr_admin"):
    retrieved = await hybrid_search(query, tenant_id=tenant_id, user_id=user_id)
    reranked = await rerank_results(query, retrieved)
    context = "\n\n".join([c["text"] for c in reranked])
    citations = list(set([c["source"] for c in reranked]))

    response = await llm_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )

    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    return {
        "answer": answer,
        "citations": citations,
        "contexts": [c["text"] for c in reranked],
        "chunk_ids": [c.get("chunk_id") for c in reranked],
        "telemetry": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": calculate_query_cost(DEFAULT_MODEL, prompt_tokens, completion_tokens)
        }
    }


# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from openai import AsyncOpenAI
# from .hybrid_retriever import hybrid_search
# from .reranker import rerank_results
# from langsmith import traceable
# 
# client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
# 
# @traceable
# async def generate_answer(query: str):
#     retrieved_chunks = await hybrid_search(query)
#     reranked_chunks = await rerank_results(query, retrieved_chunks)
#     context = "\n\n".join([chunk["text"] for chunk in reranked_chunks])
#     citations = list(set([chunk["source"] for chunk in reranked_chunks]))
#     prompt = f"""
# You are a helpful AI assistant.
# Answer ONLY from the provided context.
# Context: {context}
# Question: {query}
# """
#     response = await client.chat.completions.create(
#         model="raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     answer = response.choices[0].message.content
#     return {
#         "answer": answer,
#         "citations": citations,
#         "contexts": [chunk["text"] for chunk in reranked_chunks]
#     }
# =====================================================================
