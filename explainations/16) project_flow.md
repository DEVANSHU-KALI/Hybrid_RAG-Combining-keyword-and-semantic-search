# Script Explanation: `16) project_flow.md`

## 1. Pipeline Flowchart
This diagram illustrates the files, components, and variables through which query data is transformed from user input to the final UI response.

```mermaid
flowchart TD
    subgraph Frontend [Streamlit UI: app.py]
        A[User Input String: query] -->│httpx POST payload│ B[API Payload: payload]
    end

    subgraph API [FastAPI Web Server: main.py]
        B -->│Request Validation│ C[Pydantic Object: request]
    end

    subgraph Pipeline [RAG Orchestration: rag_pipeline.py]
        C -->│Call generate_answer│ D[Orchestration Engine]
        
        subgraph Retrieval [Hybrid Search: hybrid_retriever.py]
            D -->│1. Call hybrid_search│ E[Search Coordinator]
            
            E -->│2a. retrieve_chunks│ F[semantic_retriever.py]
            F -->│Encode Query│ G[embedding_model.py]
            G -->│Cosine Search│ H[(Qdrant Database)]
            H -->│semantic_results│ I[Score Normalizer]
            
            E -->│2b. bm25_search│ J[bm25_retriever.py]
            J -->│Download Document Texts│ H
            J -->│Build Index & Score│ K[BM25Okapi Index]
            K -->│bm25_results│ I
            
            I -->│Min-Max Normalization│ L[Deduplication & Sum Fusion]
            L -->│final_results top 5│ M[retrieved_chunks]
        end
        
        subgraph Re-Scoring [Reranker: reranker.py]
            M -->│3. Call rerank_results│ N[Cross-Encoder Reranker]
            N -->│ms-marco-MiniLM model prediction│ O[reranked_chunks top 3]
        end
        
        O -->│4. Compile prompt & context│ P[Prompt String]
        
        subgraph Inference [Local Model Server: llama.cpp]
            P -->│5. Async API call using dummy API key│ Q[llama-server.exe]
            Q -->│Inference on Qwen 2.5 7B Q4 model│ R[Raw Answer Content]
        end
        
        R -->│6. Parse answer & citations dict│ S[Pipeline Output Result]
    end

    S -->│HTTP 200 JSON Response│ T[Streamlit Render Engine]
    T -->│Render UI elements│ U[Display Answer & Sources list]
```

---

## 2. Sequence Diagram
This chronological diagram illustrates the timeline lifecycle of a single user request and the interactions between subsystems.

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant App as frontend/app.py (Streamlit)
    participant Main as backend/main.py (FastAPI)
    participant Pipe as backend/rag_pipeline.py (Orchestration)
    participant Hybrid as backend/hybrid_retriever.py (Score Fusion)
    participant Semantic as backend/semantic_retriever.py
    participant BM25 as backend/bm25_retriever.py
    participant Qdrant as Qdrant Vector DB (Port 6333)
    participant Rerank as backend/reranker.py (Cross-Encoder)
    participant LLM as llama-server.exe (Port 8080)

    User->>App: Input text question & press Enter
    Note over App: Start page reload & display st.spinner loading indicator
    App->>Main: HTTP POST /chat {"prompt": "query"}
    Note over Main: Validate request body against QueryRequest schema
    Main->>Pipe: Call generate_answer(query)
    
    rect rgb(230, 240, 255)
        Note over Pipe, Hybrid: Stage 1: Retrieval (Hybrid Search)
        Pipe->>Hybrid: Call hybrid_search(query)
        par Semantic search path
            Hybrid->>Semantic: Call retrieve_chunks(query)
            Note over Semantic: Embed query using embedding_model.py (all-MiniLM-L6-v2)
            Semantic->>Qdrant: query_points (Cosine similarity lookup, limit=10)
            Qdrant-->>Semantic: Return top 10 vectors & payloads
            Semantic-->>Hybrid: Return semantic_results list
        and BM25 search path
            Hybrid->>BM25: Call bm25_search(query)
            BM25->>Qdrant: scroll (Download up to 1000 document text payloads)
            Qdrant-->>BM25: Return document text payloads
            Note over BM25: Tokenize documents and build BM25Okapi index in memory
            Note over BM25: Calculate BM25 scores for query terms
            BM25-->>Hybrid: Return bm25_results list
        end
        Note over Hybrid: Run normalize_scores() to scale metrics between 0.0 and 1.0
        Note over Hybrid: Merge duplicate chunk IDs and sum scores (Simple Addition Fusion)
        Hybrid-->>Pipe: Return top 5 combined retrieved_chunks
    end

    rect rgb(240, 255, 240)
        Note over Pipe, Rerank: Stage 2: Reranking
        Pipe->>Rerank: Call rerank_results(query, retrieved_chunks)
        Note over Rerank: Construct query-document string pairs
        Note over Rerank: Score pairs using Cross-Encoder (ms-marco-MiniLM-L-6-v2)
        Note over Rerank: Sort candidates by reranker_score in descending order
        Rerank-->>Pipe: Return top 3 reranked_chunks
    end

    rect rgb(255, 245, 230)
        Note over Pipe, LLM: Stage 3: Contextual Generation
        Note over Pipe: Compile context text and build citations list
        Note over Pipe: Assemble prompt specifying strict context answering rules
        Pipe->>LLM: Async completion call (raaedk/Qwen2.5-7B-Instruct-Q4_K_M-GGUF, API key="dummy")
        Note over LLM: Compute response text from context parameters
        LLM-->>Pipe: Return completions response JSON
    end

    Pipe-->>Main: Return dictionary containing answer, citations, and contexts
    Main-->>App: HTTP 200 JSON payload
    Note over App: Remove loading spinner and render response on page
    App-->>User: Display markdown answer and bulleted sources
```

---