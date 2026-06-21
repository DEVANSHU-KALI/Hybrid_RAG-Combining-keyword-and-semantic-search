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