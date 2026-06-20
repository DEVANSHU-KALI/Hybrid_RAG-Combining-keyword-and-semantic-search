# Script Explanation: `14) test_dataset.md`

## 1. Overview
The primary role of the `test_dataset.py` script is to declare our validation test suite. It stores a static Python list named `evaluation_dataset` containing test scenarios. Each test scenario is represented by a dictionary holding:
1. **`question`**: A target question designed to test our pipeline's retrieval and answer accuracy.
2. **`ground_truth`**: The reference, ideal answer (also known as the "gold-standard" answer) written by a human expert.

This dataset is imported and processed by our evaluation runner script (`ragas_eval.py`) to run RAGAS benchmarking metrics.

---

## 2. Code Walkthrough

### Dataset Definition
```python
evaluation_dataset = [

    {
        "question": "What is overfitting?",

        "ground_truth":
        "Overfitting occurs when a machine learning model memorizes training data instead of learning general patterns, causing poor performance on unseen data."
    }
]
```
- **Lines 1–9**:
  - We initialize a global list variable `evaluation_dataset`.
  - Inside the list, we define our first test case dictionary:
    - `"question"`: The text query `"What is overfitting?"`. This query is passed through our RAG system's retrieval and generation pipeline.
    - `"ground_truth"`: A concise, correct definition of overfitting. When evaluating our system, we compare the LLM's generated response to this text block to check if the generated answer is mathematically correct and covers all necessary concepts.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     RAGAS Evaluation Launch
                                │
                                ▼
                       Import test_dataset
                                │
                                ▼
                        Loop Over Items:
                     question & ground_truth
                                │
                                ▼
                 Run query through RAG pipeline
                 (generate_answer -> generated)
                                │
                                ▼
                    Bundle Evaluation Record:
                     {question, generated,
                      ground_truth, contexts}
                                │
                                ▼
                      RAGAS Metric Scoring
```

---