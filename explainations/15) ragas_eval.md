# Script Explanation: `15) ragas_eval.md`

## 1. Overview
The primary role of the `ragas_eval.py` script is to run **automated quality evaluations** on our RAG chatbot. Manual inspections are time-consuming and subjective, so this script implements the **RAGAS (Retrieval Augmented Generation Assessment)** framework. 

It executes the following tasks:
* Loads environment configurations and Groq API credentials.
* Connects to the **Groq API endpoint** hosting a high-performance open-source model (`llama-3.3-70b-versatile`) to act as the "evaluator LLM judge."
* Configures an **In-Memory Rate Limiter** to throttle requests to the Groq API, preventing 429 quota exhaustion errors.
* Sets a custom client **Timeout** to ensure queued requests do not expire while waiting for rate-limiter slots.
* Imports our target evaluation list from `test_dataset.py` and loops through it, feeding the questions to our local `generate_answer()` RAG pipeline.
* Captures the generated answers, source citation files, and retrieved contexts.
* Formulates a Hugging Face `Dataset` structure from these outputs and passes it to the RAGAS evaluation runner.
* Calculates 4 core quality scores: **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.
* Formats the final scores into a clean Pandas table, sanitizes newline characters to prevent cell corruption, and exports the results to an Excel spreadsheet (`ragas_results.xlsx`) inside `evaluation_results/`.

---

## 2. Code Walkthrough

### Imports and Library Configuration
```python
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
```
- **Lines 1–11**:
  - We import `os` to load environment variables, `asyncio` to execute the async pipeline loop, and `pandas` (`pd`) to manipulate tables and write Excel files.
  - We load the environment configurator `load_dotenv` and Hugging Face's `Dataset` format.
  - We import `evaluate` from RAGAS.
  - We import `HuggingFaceEmbeddings` for embedding metrics and `ChatOpenAI` to initialize our client connection.
  - **`InMemoryRateLimiter`**: Imported from `langchain_core` to throttle client requests.

```python
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```
- **Lines 33–35**: We initialize our embedding model `all-MiniLM-L6-v2` using the LangChain Hugging Face wrapper. The evaluator uses this model to calculate mathematical similarity vector alignments during evaluation.

---

### Rate Limiter and Client Setup
```python
# 2. Configured Groq Rate Limiter (Prevents 429 Quota Exhaustion)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,  # Exactly 1 request every 5 seconds (12 RPM)
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)
```
- **Lines 40–44**:
  - We configure an `InMemoryRateLimiter` to restrict client calls to exactly `0.2` requests per second (which translates to 1 request every 5 seconds, or 12 RPM).
  - *Why this is necessary*: Groq's free tier has strict Requests Per Minute (RPM) limits. Throttling requests keeps our evaluation safely below the API's rate threshold.

```python
# 3. Groq Evaluator Model Setup
evaluator_llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    rate_limiter=rate_limiter,
    max_retries=10,
    timeout=180,
)
```
- **Lines 49–55**:
  - We initialize the evaluator LLM (`evaluator_llm`) using the `ChatOpenAI` client wrapper. We point it to the **Groq API endpoint** (`https://api.groq.com/openai/v1`), load our credentials `os.getenv("GROQ_API_KEY")`, and request the powerful `llama-3.3-70b-versatile` model to act as our evaluator judge.
  - **`rate_limiter=rate_limiter`**: Passes our rate limiter directly into the HTTP client, instructing LangChain to queue requests automatically.
  - **`timeout=180`**: Sets a 3-minute request timeout limit. Since RAGAS runs multiple metrics concurrently, requests at the back of the queue must wait for their turn. Increasing the client timeout prevents requests from expiring while waiting in the rate-limiter queue.

---

### Processing the Test Dataset
```python
async def run_evaluation():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Loop Through Evaluation Data
    for item in evaluation_dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print("\n===================")
        print(f"QUESTION: {question}")
        print("===================\n")

        # Run Full RAG Pipeline
        result = await generate_answer(question)

        # Extract Generated Answer and Retrieved Contexts
        generated_answer = result["answer"]
        retrieved_contexts = result["contexts"]

        # Store Results
        questions.append(question)
        answers.append(generated_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)
```
- **Lines 60–96**:
  - We initialize empty lists to accumulate evaluation variables.
  - We loop through each QA item inside `evaluation_dataset` (imported from `test_dataset.py`).
  - We pass the question to our local RAG pipeline: `await generate_answer(question)`.
  - We extract the generated answer string and the list of retrieved context chunk texts.
  - We append these values, along with the test question and reference ground truth, to our lists.

---