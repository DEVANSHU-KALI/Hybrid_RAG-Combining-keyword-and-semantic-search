 Script Explanation: `10) reranker.md`

## 1. Overview
The primary role of the `reranker.py` script is to run the **Cross-Encoder reranking** stage of our retrieval pipeline. While the hybrid search retriever is extremely fast and effective at finding the top candidate text chunks from the database, it evaluates documents independently or via simple term matches. 

This script refines those results by:
* Pairing each retrieved candidate chunk's text directly with the user's query.
* Feeding these query-document pairs into a highly accurate **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) that evaluates direct textual interaction between them.
* Re-scoring and re-sorting the documents based on their output attention correlation.
* Selecting and returning only the top 3 most relevant context chunks to build the final prompt for the LLM.

---

## 2. Code Walkthrough

### Imports and Model Loading
```python
from sentence_transformers import CrossEncoder

# Load Cross-Encoder Model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```
- **Lines 1–6**:
  - We import the `CrossEncoder` class from the `sentence_transformers` library.
  - We initialize our model using `"cross-encoder/ms-marco-MiniLM-L-6-v2"`. This loads the weights of a MiniLM transformer model that was pre-trained specifically on the MS-MARCO passage retrieval dataset. This model runs locally in CPU or GPU memory.

---

### Reranking Core Function
```python
async def rerank_results(query: str, retrieved_chunks: list):

    # Create Query-Chunk Pairs
    pairs = []

    for chunk in retrieved_chunks:
        pairs.append(
            [
                query,
                chunk["text"]
            ]
        )
```

- **Lines 9–20**:
  - We define the asynchronous function `rerank_results`.
  - We initialize an empty list `pairs`.
  - We loop through each document chunk retrieved by the hybrid search. For each candidate, we create a sub-list containing `[query, chunk["text"]]` and append it to `pairs`.
    - *Why?* Cross-encoders do not process queries and documents in isolation. They require both texts to be concatenated together so they can analyze their intersection.

```python
    # Generate Reranker Scores
    scores = reranker_model.predict(pairs)
```
- **Line 23**:
  - We run `reranker_model.predict(pairs)`. This passes the query-document pairs through the transformer's multi-head attention layers in a single batch, outputting a list of floating-point similarity scores (logits).

```python
    # Attach Reranker Scores
    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]["reranker_score"] = float(scores[i])
```
- **Lines 26–27**:
  - We iterate through our original list of candidate chunks.
  - For each chunk, we add a new key `"reranker_score"`. We extract the corresponding prediction value from `scores` and cast it from a NumPy float32 type to a standard Python `float` using `float()`.

```python
    # Sort By Reranker Score
    reranked_results = sorted(
        retrieved_chunks,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # Return Top 3 Chunks
    return reranked_results[:3]
```
- **Lines 30–37**:
  - We sort the chunks in descending order (highest score first) using their new `"reranker_score"` values.
  - We return the top 3 highest-scoring results using the list slice syntax `[:3]`. These top 3 documents will serve as the grounding context for the LLM.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
             Inputs: Query & List of Chunks
                           │
                           ▼
                Construct [Query, Text] Pairs
             [[Q, Text1], [Q, Text2], [Q, Text3]]
                           │
                           ▼
                  Batch Cross-Encoder
                       Inference
                   (MiniLM Attention)
                           │
                           ▼
                    Generate Logits
                [Score1, Score2, Score3]
                           │
                           ▼
              Attach Scores to original Dicts
               (cast numpy float to float)
                           │
                           ▼
                Sort Chunks Descending
                 (Highest score first)
                           │
                           ▼
                Slice & Return Top 3 Chunks
```

---

### Input and Output Specifications
* **Input**:
  * `query` (Type: `str`): The user question (e.g., `"what is overfitting?"`).
  * `retrieved_chunks` (Type: `list[dict]`): A list containing up to 5 document chunks returned by the hybrid search stage.
* **Output**: A list containing the top 3 most relevant dictionary objects (Type: `list[dict]`), sorted in descending order by `"reranker_score"`.

---

### Step-by-Step Variable Trace Walkthrough
Let's trace what happens when `await rerank_results("what is overfitting?", retrieved_chunks)` is executed:
