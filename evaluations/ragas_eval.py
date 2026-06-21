import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Ragas Metrics
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Imports from your old project structure
from evaluations.test_dataset import evaluation_dataset
from backend.rag_pipeline import generate_answer

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# 1. Configured Local Embedding Model
# -----------------------------
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 2. Configured Groq Rate Limiter (Prevents 429 Quota Exhaustion)
# -----------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,  # Exactly 1 request every 5 seconds (12 RPM)
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

# -----------------------------
# 3. Groq Evaluator Model Setup
# -----------------------------
evaluator_llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    rate_limiter=rate_limiter,
    max_retries=10,
    timeout=180,
)

# -----------------------------
# Main Evaluation Function
# -----------------------------
async def run_evaluation():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # -----------------------------
    # Loop Through Evaluation Data
    # -----------------------------
    for item in evaluation_dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print("\n===================")
        print(f"QUESTION: {question}")
        print("===================\n")

        # Run Full RAG Pipeline
        result = await generate_answer(question)

        # Extract Generated Answer & Contexts
        generated_answer = result["answer"]
        retrieved_contexts = result["contexts"]

        # Store Results
        questions.append(question)
        answers.append(generated_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

        # Debug Printing
        print("\nGenerated Answer:\n")
        print(generated_answer)
        print("\nRetrieved Contexts:\n")
        for context in retrieved_contexts:
            print(f"- {context[:150]}")

    # -----------------------------
    # Create HuggingFace Dataset
    # -----------------------------
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # -----------------------------
    # Run RAGAS Evaluation
    # -----------------------------
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # -----------------------------
    # Convert Results To DataFrame & Sanitize Formatting
    # -----------------------------
    df = evaluation_result.to_pandas()
    
    # Clean up any newlines in cells so they do not break Excel/CSV formatting
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: [item.replace("\n", " ") for item in x] if isinstance(x, list) else x)

    # -----------------------------
    # Show All Columns In Terminal
    # -----------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # -----------------------------
    # Print Final Scores
    # -----------------------------
    print("\n===================")
    print("RAGAS EVALUATION")
    print("===================\n")
    print(df)

    # -----------------------------
    # Save Results Folder dynamically in script's directory
    # -----------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    output_folder = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_folder, exist_ok=True)

    # Excel File Path
    excel_path = os.path.join(output_folder, "ragas_results.xlsx")

    # Save Results To Excel
    df.to_excel(excel_path, index=False)

    print(f"\nResults saved to:\n{excel_path}")

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    asyncio.run(run_evaluation())