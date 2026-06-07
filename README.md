# Hybrid RAG Chatbot: Dense Semantic & Sparse Lexical Retrieval Pipeline

This repository implements a local **Hybrid Retrieval-Augmented Generation (RAG)** chatbot. It combines vector similarity search and keyword search to retrieve the most relevant contexts, refines them using a Cross-Encoder model, and generates answers using a locally running, quantized large language model.

---

## 🚀 Key Features

* **Hybrid Search (Stage 1)**: Fuses dense semantic embeddings (via Qdrant and Cosine Distance) with sparse lexical keywords (via BM25Okapi) using Min-Max Normalization.
* **Cross-Encoder Reranking (Stage 2)**: Re-scores candidate context passages using `ms-marco-MiniLM-L-6-v2` attention predictions to maximize final prompt context precision.
* **Local Quantized Inference**: Leverages `llama.cpp` and GGUF (`Qwen 2.5 7B Q4`) to run models locally on standard consumer computers (no OpenAI API charges, complete data privacy).
* **Robust Observability**: Integrated with LangSmith tracing to visually debug async latency, prompts, and tokens.
* **Automated Evaluation**: Evaluates the pipeline's Faithfulness, Answer Relevancy, Context Precision, and Context Recall using the RAGAS framework.
* **Comprehensive Explanations**: A complete, step-by-step code walkthrough and conceptual guide is available inside the `explainations/` directory.

---

## 📂 Project Structure

```
├── backend/
│   ├── embedding_model.py     # SentenceTransformer embeddings config
│   ├── text_chunker.py        # LangChain Semantic Chunker splitter
│   ├── qdrant_db.py           # Qdrant collection initialization
│   ├── reset_qdrant.py        # Database wipes and drops helper
│   ├── injest_documents.py    # Reads, chunks, embeds, and uploads texts
│   ├── semantic_retriever.py  # Vector nearest-neighbor queries
│   ├── bm25_retriever.py      # Keyword token queries
│   ├── hybrid_retriever.py    # Merges and fuses semantic & BM25 scores
│   ├── reranker.py            # Cross-Encoder candidate re-scoring
│   ├── rag_pipeline.py        # Orchestrates retrieval and generation
│   └── main.py                # FastAPI web API gateway service
│
├── frontend/
│   └── app.py                 # Streamlit UI interface chat dashboard
│
├── evaluations/
│   ├── test_dataset.py        # Validation QA datasets definition
│   └── ragas_eval.py          # RAGAS metrics evaluation runner
│
├── data/
│   └── rag_concepts/          # Raw source documents for ingestion
│
├── explainations/             # Detailed step-by-step markdown documentation
│   ├── 1) key_concepts.md     # Mathematical and AI core concepts guide
│   ├── 2) to 15) ...          # Script-by-script technical breakdowns
│   ├── 16) project_flow.md    # Mermaid pipeline charts & variable traces
│   ├── 17) interview_questions.md # Interview prep questions & answers
│   └── 18) master_prompt.md   # Reusable code walkthrough prompt template
│
├── .gitignore                 # Files excluded from GitHub
├── requirements.txt           # Python package dependencies
└── README.md                  # Project overview and run guide
```

---

## 🛠️ Step-by-Step Setup Guide

### 1. Initialize Virtual Environment & Install Dependencies
Open your terminal in the project directory, initialize a Python virtual environment, and install all required packages:
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```
> [!NOTE]
> Ensure all dependencies in `requirements.txt` install successfully. Key packages installed include `fastapi` and `uvicorn` for hosting the web service, `ragas` and `datasets` for automated validation, and `langsmith` for transaction tracing.

### 2. Run Qdrant Vector Database
Make sure you have Docker running on your system, and start a Qdrant container:
```cmd
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Run Llama.cpp Local Model Server
1. Download the executable zip folder from the official releases section on the [llama.cpp Github page](https://github.com/ggerganov/llama.cpp/releases) (choose CUDA version if you have an NVIDIA GPU, otherwise CPU).
2. Unzip it and open `cmd` inside that folder.
3. Start the server using Qwen 2.5 (this will download the model from Hugging Face on the first run and host it on port 8080):
```cmd
.\llama-server.exe -hf raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF -ngl 25
```
*(Adjust `-ngl 25` to change the number of layers offloaded to your GPU. Set to 0 if running on CPU-only).*

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory (this file is ignored by git to keep your credentials safe). Add the following template and fill in your values:

```text
# --- RAGAS Evaluation Config ---
# OpenRouter API Key (Used to run evaluations with ChatOpenAI)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# --- LangSmith Observability & Tracing Config ---
# Enable/Disable tracing (set to true to enable)
LANGCHAIN_TRACING_V2=true

# Your LangSmith API Key (Generated from your LangSmith settings dashboard)
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# The name of the project to trace under inside LangSmith
LANGCHAIN_PROJECT=hybrid-rag-chatbot
```

---

## 🔍 How to Enable and View LangSmith Tracing

We use **LangSmith** to monitor execution flows, view call logs, track latency bottlenecks, and audit prompt payloads.

### 1. Get an API Key
1. Go to [LangSmith (smith.langchain.com)](https://smith.langchain.com/) and sign up for a free developer account.
2. Navigate to your **Settings** (bottom left profile icon) and click on **API Keys**.
3. Generate a new API Key and copy it.

### 2. Set Up Tracing
Ensure your `.env` contains:
```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_copied_api_key
LANGCHAIN_PROJECT=hybrid-rag-chatbot
```
The `@traceable` decorator in our RAG pipeline (`rag_pipeline.py`) will automatically intercept all asynchronous database queries, reranking predictions, and model completions, routing execution logs to the cloud.

### 3. View Your Traces
1. Go to the [LangSmith Projects Dashboard](https://smith.langchain.com/projects).
2. Select your project **`hybrid-rag-chatbot`**.
3. Click on any query run to view a nested execution chart showing:
   * **Latency breakdowns**: Time spent on hybrid search vs. Cross-Encoder reranking vs. LLM generation.
   * **Payload details**: The exact text segments retrieved and the prompt sent to `llama.cpp`.

---

## 💻 Running the Application

### Step 1: Initialize Database & Ingest Documents
Run the startup checker to initialize the collection, and execute the ingestion script to semantic-chunk, embed, and upload source files from `data/rag_concepts/`:
```cmd
# Initialize collection
python -m backend.qdrant_db

# Ingest raw documents
python -m backend.injest_documents
```

### Step 2: Start the FastAPI Backend
Start the backend server on port 8000:
```cmd
uvicorn backend.main:app --reload
```

### Step 3: Start the Streamlit Frontend
Launch the user interface in your browser:
```cmd
streamlit run frontend/app.py
```
You can now ask questions on the Streamlit page at `http://localhost:8501`.

---

## 📊 Running Evaluations
To run automated RAGAS metrics benchmarking against the pipeline and export results to an Excel worksheet, run:
```cmd
python -m evaluations.ragas_eval
```
The results sheet will be saved to `evaluations/evaluation_results/ragas_results.xlsx`.
