from openai import AsyncOpenAI

from .hybrid_retriever import hybrid_search
from .reranker import rerank_results

from langsmith import traceable

# OpenAI Client
client = AsyncOpenAI(base_url="http://localhost:8080/v1",api_key="dummy")

# Main RAG Pipeline
@traceable
async def generate_answer(query: str):

    # Hybrid Retrieval
    retrieved_chunks = await hybrid_search(query)

    # Reranking
    reranked_chunks = await rerank_results(
        query,
        retrieved_chunks
    )

    # Print Retrieved Sources
    print("\n===== FINAL RETRIEVED CHUNKS =====\n")

    for chunk in reranked_chunks:
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Reranker Score: {chunk['reranker_score']}")
        print("\n")

    # Build Context
    context = "\n\n".join(
        [
            chunk["text"]
            for chunk in reranked_chunks
        ]
    )

    # Build Citations
    citations = list(
        set(
            [
                chunk["source"]
                for chunk in reranked_chunks
            ]
        )
    )

    # Final Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}
"""

    # Generate LLM Response
    response = await client.chat.completions.create(
        model="raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract Final Answer
    answer = response.choices[0].message.content

    # Return Final Response
    return {
        "answer": answer,
        "citations": citations,
        "contexts": [
        chunk["text"]
        for chunk in reranked_chunks
        ]
    }