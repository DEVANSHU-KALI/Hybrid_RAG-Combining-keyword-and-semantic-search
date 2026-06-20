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

### Input and Output Specifications
* **Input**: None (static variable definition).
* **Output**: `evaluation_dataset` (Type: `list[dict]`) - A list containing structured validation cases.

---

### Step-by-Step Variable Trace Walkthrough
Let's trace the lifecycle of this data during evaluation:

1. **Import Stage**: The evaluation runner execution loads the list:
   `from evaluations.test_dataset import evaluation_dataset`.
2. **Loop Iteration**:
   - `item` is set to `{"question": "What is overfitting?", "ground_truth": "Overfitting occurs..."}`.
3. **Pipeline query**: The system extracts the question:
   - `question = item["question"]` -> `"What is overfitting?"`.
   - Passes `"What is overfitting?"` to `generate_answer()`.
4. **Answer Grounding Comparison**:
   - The pipeline returns `generated_answer`.
   - RAGAS loads `generated_answer` and the reference `ground_truth` (`"Overfitting occurs when..."`), comparing them using a judge LLM to evaluate faithfulness and correctness.

---

## 4. Deep Technical Concepts

### Evaluation Datasets (Gold Standards)
In generative AI development, manual testing is insufficient. Developers curate an **Evaluation Dataset** containing representative questions. For each question, they define a **Ground Truth** (the reference correct answer). These gold-standard QA pairs allow automatic evaluators to score the chatbot's performance before and after code changes.

---

## 5. Architectural Choices and Alternatives

### Why Static Inline Python Lists?
Using a static Python list is simple and fast. It requires zero filesystem parsing at runtime (no JSON or CSV parsing logic is written). The file acts as a module that Python scripts can import directly.

#### Alternatives and Trade-offs

| Storage Structure | Strategy | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Static Python Variable** *(Chosen)* | Declared inline in a `.py` file. | • Instant importing.<br>• Simplest local structure. | • Mixes raw test data with code execution space.<br>• Not easily readable by non-python software. |
| **JSON File** | `dataset.json` storing structured QA arrays. | • Language independent (can be read by Node.js or Python). | • Requires writing filesystem read and JSON parse code. |
| **CSV / Excel Sheet** | Standard database sheets. | • Highly readable for business analysts and testers. | • Requires installing external libraries (like `pandas` or `openpyxl`). |
| **Hugging Face Hub** | Hosted dataset repository. | • Version controlled and hosted on the cloud.<br>• Easily shared across teams. | • Requires internet connectivity and dependency libraries. |
