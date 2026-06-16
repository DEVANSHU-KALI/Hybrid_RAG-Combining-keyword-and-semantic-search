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