from langchain_huggingface import HuggingFaceEmbeddings

# CONCEPT: 768D Enterprise Dense Embedding Model (BAAI/bge-base-en-v1.5)
# In enterprise RAG, 768-dimensional models like bge-base-en-v1.5 provide higher semantic 
# resolution for technical documents, tables, and domain jargon compared to 384D models,
# while maintaining fast CPU inference speed and low memory footprint (~440MB).

# CONCEPT: LangChain Wrapper vs Raw SentenceTransformer Interface Rationale
# Raw SentenceTransformer only exposes a single generic `.encode()` method for all inputs.
# In contrast, the `HuggingFaceEmbeddings` wrapper separates `.embed_documents()` (for batch passage 
# vectorization) from `.embed_query()` (for single search query vectorization). 
# This distinction is critical for asymmetric retrieval models (like BGE/E5), where queries 
# require instruction prefixes or different pooling than passive document chunks.

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from langchain_huggingface import HuggingFaceEmbeddings
# 
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# =====================================================================
