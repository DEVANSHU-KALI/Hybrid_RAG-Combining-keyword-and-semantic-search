# Script Explanation: `3) text_chunker.md`

## 1. Overview
The primary role of the `text_chunker.py` script is to configure the text splitting strategy for incoming raw documents. Instead of splitting text at arbitrary character or token lengths (which can cut off sentences mid-thought and destroy semantic context), this script uses a **Semantic Chunker**. 

It works by:
* Breaking the document into individual sentences.
* Embedding each sentence into a vector representation using our `embedding_model`.
* Measuring the semantic distance (difference in meaning) between consecutive sentences.
* Setting a threshold (in this case, the 75th percentile of distance differences) to identify "breakpoints" where a new paragraph/concept starts, and splitting the document at those points.

---

## 2. Code Walkthrough

### Imports
```python
from langchain_experimental.text_splitter import SemanticChunker
```
- **Line 1**: We import the `SemanticChunker` class from the `langchain_experimental` package. This experimental utility is designed to leverage vector embeddings to split text along semantic boundary lines.

```python
from .embedding_model import embedding_model
```
- **Line 3**: We import the pre-configured `embedding_model` (`all-MiniLM-L6-v2`) from our local directory, which the chunker will use to embed sentences.

---

### Instantiating the Semantic Chunker
```python
text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75
)
```
- **Lines 5–9**: We initialize the semantic splitter with specific hyperparameters:
  - `embedding_model`: The model used to convert sentences into vector space.
  - `breakpoint_threshold_type="percentile"`: We tell the splitter to compute the semantic differences between all adjacent sentences and determine splits based on percentile distributions.
  - `breakpoint_threshold_amount=75`: This specifies that any semantic distance difference between two adjacent sentences that falls in the top 25% (exceeding the 75th percentile) of all measured differences in the document will trigger a breakpoint (a document split).

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     Input Raw Document Text
                                │
                                ▼
                   Sentence Splitting (Regex)
               [Sentence 1, Sentence 2, Sentence 3]
                                │
                                ▼
                 Embed Sentences (all-MiniLM)
             [Vector 1, Vector 2, Vector 3, ...]
                                │
                                ▼
                 Compute Consecutive Distances
                [Distance(1,2), Distance(2,3)]
                                │
                                ▼
               Calculate Percentile Threshold (75%)
                                │
                                ▼
                  Evaluate Split Boundaries:
                Is Distance > 75th Percentile?
                 ├── Yes ──► Create Breakpoint
                 └── No  ──► Keep in Current Chunk
                                │
                                ▼
                     Output: Document Chunks
```

---

### Input and Output Specifications
* **Input**: A list containing a single long string of document text (e.g., `["First concept text. Second concept text..."]`).
* **Output**: A list of LangChain `Document` objects (Type: `list[Document]`). Each document contains:
  * `page_content`: The text chunk string.
  * `metadata`: Dictionary containing metadata (which is empty by default).

---

### Step-by-Step Variable Trace Walkthrough
Let's trace what happens when we call `text_splitter.create_documents([text])` on a raw document containing 4 sentences:

#### Step 1: Sentence Splitting
The chunker parses the text into individual sentences:
* $S_1$: `"Vector databases store data as high-dimensional coordinates."`
* $S_2$: `"This makes semantic search extremely fast and accurate."`
* $S_3$: `"In other news, Python is a popular programming language."`
* $S_4$: `"It is widely used in data science and web development."`

#### Step 2: Vector Embedding Generation
Each sentence is embedded using `embedding_model`:
* $V_1 = \text{embed}(S_1)$ (384 float vector)
* $V_2 = \text{embed}(S_2)$ (384 float vector)
* $V_3 = \text{embed}(S_3)$ (384 float vector)
* $V_4 = \text{embed}(S_4)$ (384 float vector)

#### Step 3: Compute Adjacent Cosine Distances
The similarity distance between consecutive sentences is calculated:
* $\text{Dist}(S_1, S_2) = 0.15$ *(Low distance: sentences are semantically similar)*
* $\text{Dist}(S_2, S_3) = 0.85$ *(High distance: sentences change topics completely)*
* $\text{Dist}(S_3, S_4) = 0.20$ *(Low distance: sentences are semantically similar)*

#### Step 4: Calculate Percentile Threshold
The algorithm computes the threshold based on all distances:
* Distance list: `[0.15, 0.85, 0.20]`
* Sorting distances: `[0.15, 0.20, 0.85]`
* The 75th percentile threshold of these distances is calculated (mathematically falls between `0.20` and `0.85`, e.g., `0.525`).

#### Step 5: Boundary Evaluation and Splitting
We check each adjacent distance against the 75th percentile threshold (`0.525`):
1. $\text{Dist}(S_1, S_2) = 0.15 < 0.525 \rightarrow$ Keep $S_1$ and $S_2$ together.
2. $\text{Dist}(S_2, S_3) = 0.85 > 0.525 \rightarrow$ **Breakpoint triggered!** Split document here.
3. $\text{Dist}(S_3, S_4) = 0.20 < 0.525 \rightarrow$ Keep $S_3$ and $S_4$ together.

#### Output Chunks
The function returns two separate document objects:
* **Chunk 1**: `"Vector databases store data as high-dimensional coordinates. This makes semantic search extremely fast and accurate."`
* **Chunk 2**: `"In other news, Python is a popular programming language. It is widely used in data science and web development."`

---

## 4. Deep Technical Concepts

### Semantic Chunking
Traditional text splitting splits text at fixed intervals (e.g., every 500 characters). This often slices sentences in half or separates highly related sentences, lowering retrieval accuracy. **Semantic Chunking** groups sentences dynamically based on their conceptual similarity, ensuring each chunk contains a single, complete topic.

### Cosine Distance
To measure the difference between two sentence vectors, the algorithm uses **Cosine Distance** (defined as 1 minus Cosine Similarity). 
$$\text{Cosine Distance} = 1 - \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
* A distance close to `0.0` means the sentences have highly similar vector directions (close in meaning).
* A distance close to `1.0` (or greater) means the sentences are directionally orthogonal or opposite (different in meaning).

### Percentile Thresholding
Percentile thresholding is a statistical filtering technique where split boundaries are calculated dynamically relative to the variance of the document. Rather than hardcoding a static distance limit (e.g., split if distance $> 0.5$), the percentile method computes the mathematical distribution of distances *for each specific document* and splits at the most dramatic gaps (the top 25% of changes).

---

## 5. Architectural Choices and Alternatives

### Why Semantic Chunking with Percentiles?
By using semantic boundaries, we maximize the likelihood that our RAG retrieval pulls complete context blocks. The **percentile** thresholding strategy is highly adaptive because it calculates breakpoints relative to the current document's natural flow rather than applying a rigid distance metric.

#### Alternatives and Trade-offs

| Splitting Method | How it Works | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Semantic Chunker** *(Chosen)* | Splits based on cosine distance variance between sentence embeddings. | • Keeps semantic concepts whole.<br>• Highly adaptive to varying document lengths/styles. | • Slowest method (requires calling an embedding model for every sentence).<br>• High computational overhead. |
| **Recursive Character Splitter** | Splits text using a hierarchy of characters (e.g., double newlines, single newlines, spaces) until chunks are under a target size. | • Extremely fast (simple regex splits).<br>• Respects paragraphs and sentences when possible. | • Arbitrary limits can still split a conceptual topic across boundaries. |
| **Fixed-Size Token Splitter** | Counts tokens (e.g., using TikToken) and splits at a rigid threshold (e.g., every 256 tokens). | • Direct control over LLM token usage.<br>• Prevents out-of-token errors during API calls. | • Zero awareness of formatting or sentence meaning. |
