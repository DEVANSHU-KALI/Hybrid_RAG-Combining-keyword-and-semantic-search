# Script Explanation: `8) bm25_retriever.md`

## 1. Overview
The primary role of `bm25_retriever.py` is executing **Sparse Lexical Search** using the **Okapi BM25 ranking algorithm**. Sparse search evaluates exact keyword occurrences, alphanumeric part numbers, abbreviations, and exact jargon that dense vector models can miss due to semantic drift.

---

## 2. Deep Mechanics: How BM25 Keyword Search Works & Statistics Calculated

### Statistics Calculated by Okapi BM25
BM25 (Best Match 25) calculates 4 statistical properties across document chunks:
1. **Term Frequency ($f(q_i, D)$):** How many times a query word $q_i$ appears in document chunk $D$.
2. **Inverse Document Frequency ($\text{IDF}(q_i)$):** Evaluates how rare or common a word is across all documents:
   $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
   * $N$: Total number of document chunks in the tenant collection.
   * $n(q_i)$: Number of document chunks containing term $q_i$.
   * *Result:* Common words (`"is"`, `"the"`) get IDF near `0.0`. Rare words (`"overfitting"`) get high IDF scores.
3. **Document Length Normalization ($|D| / \text{avgdl}$):**
   * $|D|$: Character/word count of current document $D$.
   * $\text{avgdl}$: Average document length across all tenant chunks.
   * *Result:* Short, focused documents containing a keyword score higher than long, verbose documents containing the same keyword.
4. **Term Frequency Saturation ($k_1$ parameter):**
   * Applies logarithmic scaling to term frequency. Repeating a word 10 times gives diminishing returns compared to matching it 2 times.

---

## 3. RAM Lifecycle of BM25 Index in our Code

### Where is the BM25 Index stored and how long does it live?
* **Storage Location:** In Python RAM (Memory).
* **Lifecycle in our Code (`bm25_retriever.py`):**
  1. When a user sends a query, `bm25_search()` is invoked.
  2. `client.scroll()` fetches tenant document text payloads into Python RAM.
  3. `bm25 = BM25Okapi(tokenized_docs)` builds term matrices in Python RAM.
  4. `bm25.get_scores(tokenized_query)` calculates scores.
  5. As soon as `bm25_search()` finishes and returns results, Python's Garbage Collector frees the BM25 RAM memory.
  6. **On Next Query:** The process activates again, building a fresh index for the requested tenant.

---

## 4. Line-by-Line Code Walkthrough

```python
async def bm25_search(
    query: str,                              # User prompt string to score against keywords
    tenant_id: str = "tenant_default",       # Tenant ID filter for multi-tenant isolation
    limit: int = 10                          # Top K candidate chunks to return
) -> list[dict]:
```

* **Pre-Filtering (`FieldCondition`):** Only fetches chunks belonging to `tenant_id`.
* **Document Scroll (`limit=1000`):** Downloads text payloads for up to 1,000 tenant chunks into RAM.
* **Payload Extraction (`.get("chunk_id", point.id)`):** Extracts payload keys. If payload `"chunk_id"` is missing, safely falls back to `point.id`.
* **Tokenization (`.lower().split()`):** Converts text and query into lowercase token lists.
* **Scoring & Sorting:** Calculates BM25 float scores, sorts descending by score, and slices `return results[:limit]`.
