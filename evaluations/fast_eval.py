import os
import re
import time
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from evaluations.test_dataset import evaluation_dataset
from backend.rag_pipeline import generate_answer

load_dotenv()

# =====================================================================
# CONCEPT: Zero-Cost ($0.00), Sub-Second Local RAG Triad Evaluator
# =====================================================================
# Eliminates external LLM-as-a-judge dependencies (Groq/OpenAI/TypeSafe API).
# Operates locally on CPU/GPU using:
# 1. 768D Bi-Encoder (BAAI/bge-base-en-v1.5) for vector directional metrics.
# 2. Local Cross-Encoder (ms-marco-MiniLM-L-6-v2) for factual cross-attention.
# Executes the complete RAG Triad in < 1 second with 100% determinism.
# =====================================================================

print("Initializing local evaluator models (100% Free, Zero Cloud APIs)...")
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def split_into_propositions(text: str) -> list[str]:
    """Splits a response into standalone factual propositions/sentences."""
    # Split by standard sentence terminators while preserving clean text
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clean = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Filter out markdown formatting headers or empty snippets
        if len(s_clean) > 15 and not s_clean.startswith("|") and not s_clean.startswith("#"):
            clean.append(s_clean)
    return clean if clean else [text.strip()]


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Evaluates whether each claim in the answer is grounded in retrieved context.
    Uses Cross-Encoder cross-attention between context and answer propositions.
    Returns: Score between 0.0 and 1.0 (Higher = Zero Hallucination).
    """
    propositions = split_into_propositions(answer)
    if not propositions or not contexts:
        return 0.0

    full_context = " ".join(contexts)
    # Form cross-encoder pairs [Context, Proposition]
    pairs = [[full_context, prop] for prop in propositions]
    scores = cross_encoder.predict(pairs)

    # Sigmoid normalization for raw logits
    probs = 1.0 / (1.0 + np.exp(-np.array(scores)))
    
    # Proposition is considered grounded if probability meets semantic threshold
    grounded_count = sum(1 for p in probs if p >= 0.40)
    return round(float(grounded_count / len(propositions)), 4)


def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Evaluates semantic directional alignment between the user query and the generated answer.
    Uses 768D Bi-Encoder embeddings and cosine similarity.
    Returns: Cosine similarity score [0.0 - 1.0].
    """
    query_emb = bi_encoder.encode(query, normalize_embeddings=True)
    answer_emb = bi_encoder.encode(answer, normalize_embeddings=True)
    cosine_sim = float(np.dot(query_emb, answer_emb))
    # Clip to valid probability bounds [0.0, 1.0]
    return round(max(0.0, min(1.0, cosine_sim)), 4)


def compute_context_relevance(query: str, contexts: list[str]) -> float:
    """
    Evaluates the signal-to-noise ratio of the retrieved context chunks against the query.
    Uses Cross-Encoder relevance scores across all top-K retrieved chunks.
    """
    if not contexts:
        return 0.0
    pairs = [[query, ctx] for ctx in contexts]
    scores = cross_encoder.predict(pairs)
    probs = 1.0 / (1.0 + np.exp(-np.array(scores)))
    return round(float(np.mean(probs)), 4)


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    (Optional Development Metric) Evaluates how thoroughly the retrieved context
    covers key assertions from the gold-standard reference ground truth.
    """
    gt_sentences = split_into_propositions(ground_truth)
    if not gt_sentences or not contexts:
        return 0.0

    full_context = " ".join(contexts)
    ctx_emb = bi_encoder.encode(full_context, normalize_embeddings=True)
    gt_embs = bi_encoder.encode(gt_sentences, normalize_embeddings=True)

    sims = np.dot(gt_embs, ctx_emb)
    covered = sum(1 for s in sims if s >= 0.65)
    return round(float(covered / len(gt_sentences)), 4)


async def run_fast_evaluation():
    print("\n=======================================================")
    print("   RUNNING LOCAL ZERO-COST ($0.00) RAG TRIAD EVALUATION ")
    print("=======================================================")

    start_eval_time = time.time()
    results = []

    for idx, item in enumerate(evaluation_dataset, start=1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"\n[{idx}/{len(evaluation_dataset)}] Processing Query: '{question}'")
        t0 = time.time()
        
        # 1. Execute RAG Pipeline Query
        rag_output = await generate_answer(query=question, tenant_id="tenant_default", user_id="usr_admin")
        gen_time = round(time.time() - t0, 2)

        answer = rag_output["answer"]
        contexts = rag_output["contexts"]

        # 2. Compute RAG Triad Metrics Locally
        faith_score = compute_faithfulness(answer, contexts)
        relevancy_score = compute_answer_relevancy(question, answer)
        ctx_relevance_score = compute_context_relevance(question, contexts)
        ctx_recall_score = compute_context_recall(ground_truth, contexts) if ground_truth else None

        results.append({
            "question": question,
            "answer": answer.replace("\n", " "),
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
            "context_relevance": ctx_relevance_score,
            "context_recall": ctx_recall_score,
            "generation_time_s": gen_time
        })

        print(f"   -> Faithfulness: {faith_score:.2f} | Relevancy: {relevancy_score:.2f} | Context Relevance: {ctx_relevance_score:.2f} | Recall: {ctx_recall_score if ctx_recall_score else 'N/A'}")

    df = pd.DataFrame(results)
    total_duration = round(time.time() - start_eval_time, 2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\n======================= FINAL BENCHMARK SUMMARY =======================")
    print(df[["question", "faithfulness", "answer_relevancy", "context_relevance", "context_recall", "generation_time_s"]])

    print("\n====================== OVERALL PIPELINE HEALTH ======================")
    print(f"• Mean Faithfulness (Groundedness):   {df['faithfulness'].mean():.4f} (Target: >= 0.85)")
    print(f"• Mean Answer Relevancy (Directness): {df['answer_relevancy'].mean():.4f} (Target: >= 0.80)")
    print(f"• Mean Context Relevance (Precision): {df['context_relevance'].mean():.4f} (Target: >= 0.60)")
    if "context_recall" in df and df["context_recall"].notnull().any():
        print(f"• Mean Context Recall (Completeness): {df['context_recall'].mean():.4f} (Target: >= 0.80)")
    print(f"• Total Evaluation Run Time:          {total_duration}s")
    print(f"• Cloud API Cost:                     $0.00 (100% Offline & Local)")

    output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "fast_eval_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n✅ Benchmark report saved to: {excel_path}\n")


if __name__ == "__main__":
    asyncio.run(run_fast_evaluation())
