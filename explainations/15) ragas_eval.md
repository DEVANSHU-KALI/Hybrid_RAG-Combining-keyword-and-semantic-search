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

### Executing the RAGAS Benchmarking Run
```python
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run RAGAS Evaluation
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
```
- **Lines 100–120**:
  - RAGAS requires inputs to be structured in a Hugging Face `Dataset` table. We convert our lists into this format using `Dataset.from_dict()`.
  - We invoke the `evaluate()` function, passing our dataset, the 4 quality metrics, our Groq-hosted LLM judge, and the Hugging Face embedding model.

---

### Exporting and Saving Results
```python
    # Convert Results To DataFrame & Sanitize Formatting
    df = evaluation_result.to_pandas()
    
    # Clean up any newlines in cells so they do not break Excel/CSV formatting
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: [item.replace("\n", " ") for item in x] if isinstance(x, list) else x)
```
- **Lines 125–130**:
  - We convert the evaluation results object into a Pandas DataFrame using `to_pandas()`.
  - **Formatting Sanitization**: We loop through all DataFrame cells and replace newline characters (`\n`) with spaces. This prevents raw line breaks from breaking spreadsheet layouts when viewed in Excel or CSV readers.

```python
    # Save Results Folder dynamically in script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    output_folder = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_folder, exist_ok=True)

    # Excel File Path
    excel_path = os.path.join(output_folder, "ragas_results.xlsx")

    # Save Results To Excel
    df.to_excel(excel_path, index=False)
```
- **Lines 150–158**:
  - We check if `__file__` is available in `locals()`. Since `__file__` is a global module variable, this check returns `False` inside the function, falling back to `os.getcwd()` (our project root directory).
  - Creates the directory `evaluation_results/` in the root folder using `os.makedirs()`.
  - Writes the DataFrame to an Excel spreadsheet (`ragas_results.xlsx`) using `to_excel(..., index=False)`.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
              Load Environment Variables (.env)
                              │
                              ▼
                 Instantiate Groq API Client
             (llama-3.3-70b-versatile via ChatOpenAI)
                              │
                              ▼
                   Loop over validation items
                  in evaluations.test_dataset
                              │
                              ▼
                 Call generate_answer(query)
               (RAG pipeline retrieves context
                 and generates answer locally)
                              │
                              ▼
                Aggregate: questions[], answers[],
                     contexts[], ground_truths[]
                              │
                              ▼
                Convert to Hugging Face Dataset
                              │
                              ▼
                 Execute RAGAS evaluate() API
              (Rate-limited: 1 call / 5 seconds;
                queued requests timeout = 180s)
                              │
                              ▼
                Convert scores to Pandas DataFrame
                              │
                              ▼
                 Sanitize newline characters (\n)
                              │
                              ▼
                Export to Excel spreadsheet file
```

---

### Input and Output Specifications
* **Input**: 
  * Static evaluation scenarios in `test_dataset.py` (containing 3 queries).
  * Groq API key inside the `.env` file: `GROQ_API_KEY=your_key_here`.
* **Output**: Writes the detailed metric evaluation report to `evaluation_results/ragas_results.xlsx`. Prints the scoring table to the terminal.

---

### Step-by-Step Variable Trace Walkthrough
Let's trace a run of the evaluation script with our 3-question dataset:

1. **Load Environment**: `load_dotenv()` checks for the `.env` file and loads `GROQ_API_KEY`.
2. **Evaluator Init**: Instantiates `evaluator_llm` pointing to Groq's API endpoint with model `llama-3.3-70b-versatile`.
3. **Execution Loop**:
   - Loops through 3 items in `evaluation_dataset`:
     - *Index 0*: `"What is overfitting?"`
     - *Index 1*: `"How do Vector Embeddings and Cosine Similarity work together..."`
     - *Index 2*: `"How can Dropout Regularization help reduce..."`
   - For each query, `generate_answer()` fetches contexts from Qdrant, reranks them to a top 3 subset, and runs Qwen local generation to capture `answer` and `contexts`.
4. **Dataset Conversion**: Constructs a dataset table with these columns.
5. **RAGAS scoring (The Rate-Limited LLM-as-a-judge process)**:
   - RAGAS initiates 12 metric evaluation calls concurrently.
   - The `InMemoryRateLimiter` serializes the requests, allowing only 1 call every 5 seconds.
   - While requests wait in the queue, their client timeouts are set to `180s`, preventing them from expiring.
   - Groq evaluates the metrics:
     - **Faithfulness check**: Analyzes if statements in the answer are supported by contexts. (For Q1, it returns a low score of `0.28` because the retriever missed the Cosine Similarity context chunk).
     - **Context Recall check**: Verifies if the context contains all facts required to answer the question. (For Q1, it returns `0.6` because the Cosine Similarity definition is missing from the retrieved context).
6. **Save Stage**: Converts results to a Pandas DataFrame, replaces all instances of `\n` in cells with spaces to prevent CSV/Excel row-splitting layout corruption, and writes the data to `evaluation_results/ragas_results.xlsx`.

---
