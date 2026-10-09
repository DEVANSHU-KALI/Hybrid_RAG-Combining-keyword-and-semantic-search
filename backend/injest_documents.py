# Imports standard os library for inspecting file directories and paths
import os
import asyncio

# Imports AsyncQdrantClient for non-blocking vector database ingestion
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

# Imports pre-configured 768D HuggingFace embedding model (BAAI/bge-base-en-v1.5)
from .embedding_model import embedding_model

# Imports semantic chunker and markdown table section chunker helpers
from .text_chunker import text_splitter, chunk_markdown_sections

# Initialize asynchronous Qdrant client pointing to local Docker container on port 6333
client = AsyncQdrantClient(host="localhost", port=6333)

# Qdrant collection target name where vectors and payloads are stored
COLLECTION_NAME = "rag_docs"


# CONCEPT: Multimodal PDF & Markdown Table Parsing Utility
def extract_pdf_content(file_path: str) -> list[dict]:
    """
    Extracts narrative text and structured tables from PDF files.
    Converts raw tables into Markdown table strings (| Header | Header |) 
    so column headers and tabular mathematical relationships remain intact inside vector chunks.
    """
    pages_content = []
    
    try:
        # Import pdfplumber library for precise PDF layout and table extraction
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. Extract regular narrative text from PDF page
                page_text = page.extract_text() or ""
                
                # 2. Extract tables from PDF page and format as Markdown
                tables = page.extract_tables()
                table_markdowns = []
                for table in tables:
                    if not table:
                        continue
                    # Convert row lists into Markdown table strings (| Header 1 | Header 2 |)
                    headers = [str(cell or "").strip() for cell in table[0]]
                    md_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                    for row in table[1:]:
                        clean_row = [str(cell or "").strip() for cell in row]
                        md_rows.append("| " + " | ".join(clean_row) + " |")
                    table_markdowns.append("\n".join(md_rows))
                
                # Combine page text with Markdown tables
                full_page_text = page_text + "\n\n" + "\n\n".join(table_markdowns)
                pages_content.append({
                    "page_number": page_num,
                    "text": full_page_text.strip(),
                    "has_table": len(tables) > 0
                })
    except ImportError:
        # Fallback to PyPDF or basic text reader if pdfplumber is not installed
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                pages_content.append({
                    "page_number": page_num,
                    "text": page.extract_text() or "",
                    "has_table": False
                })
        except Exception as e:
            print(f"⚠️ PDF extraction error for {file_path}: {e}")
            
    return pages_content


# CONCEPT: Multi-Tenant Async Ingestion Engine
async def ingest_documents(
    folder_path: str,                        # Local directory path containing documents to index
    tenant_id: str = "tenant_default",       # Target tenant identity tag for multi-tenant data isolation
    user_id: str = "usr_admin"               # Target user identity tag for security auditing
):
    """
    Asynchronously ingests text and PDF files into Qdrant with 768D embeddings 
    and multi-tenant metadata payload tags (tenant_id, user_id).
    Using AsyncQdrantClient ensures non-blocking document uploads in live API web services.
    """
    documents = []      # Parallel list storing raw chunk text strings
    ids = []            # Parallel list storing unique integer chunk IDs
    sources = []        # Parallel list storing source filenames for citations
    metadatas = []      # Parallel list storing multi-tenant payload dictionaries
    counter = 0         # Incremental ID counter for point primary keys

    if not os.path.exists(folder_path):
        print(f"⚠️ Path '{folder_path}' does not exist.")
        return

    # Loop through all files in folder directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # --- SECTION A: PDF DOCUMENT INGESTION ---
        if filename.endswith(".pdf"):
            pdf_pages = extract_pdf_content(file_path)
            for page_data in pdf_pages:
                page_text = page_data["text"]
                if not page_text:
                    continue
                
                # Chunk page text using Markdown table-aware section chunker
                chunks = chunk_markdown_sections(page_text)
                for chunk_text in chunks:
                    documents.append(chunk_text)
                    ids.append(counter)
                    sources.append(filename)
                    metadatas.append({
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "document_type": "pdf",
                        "page_number": page_data["page_number"],
                        "has_table": page_data["has_table"]
                    })
                    counter += 1
                    
        # --- SECTION B: TXT DOCUMENT INGESTION ---
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Chunk using 768D semantic chunker (percentile distance boundaries)
            chunks = text_splitter.create_documents([text])
            for chunk in chunks:
                documents.append(chunk.page_content)
                ids.append(counter)
                sources.append(filename)
                metadatas.append({
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "document_type": "txt",
                    "page_number": 1,
                    "has_table": False
                })
                counter += 1

    if not documents:
        print("⚠️ No documents found or chunked for ingestion.")
        return

    # --- SECTION C: BATCH EMBEDDING VECTORIZATION ---
    # Calculates 768-dimensional dense vectors for all document chunks in a single GPU/CPU batch
    print(f"\n======= GENERATING 768D EMBEDDINGS FOR {len(documents)} CHUNKS =======\n")
    embeddings = embedding_model.embed_documents(documents)
    print("Embedding vector dimensions:", len(embeddings[0]))

    # --- SECTION D: POINTSTRUCT PAYLOAD ASSEMBLY ---
    # Assembles Qdrant PointStruct records containing ID, vector array, and metadata payload
    points = []
    for i in range(len(documents)):
        payload = {
            "text": documents[i],
            "source": sources[i],
            "chunk_id": ids[i],
            "tenant_id": metadatas[i]["tenant_id"],
            "user_id": metadatas[i]["user_id"],
            "document_type": metadatas[i]["document_type"],
            "page_number": metadatas[i]["page_number"],
            "has_table": metadatas[i]["has_table"]
        }
        points.append(
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload=payload
            )
        )

    # --- SECTION E: NON-BLOCKING ASYNC DATABASE UPSERT ---
    # Uploads (upserts) points to Qdrant asynchronously without blocking event loop thread
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"\n✅ {len(points)} Chunks successfully ingested asynchronously with multi-tenant metadata into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    asyncio.run(ingest_documents("data/rag_concepts", tenant_id="tenant_default", user_id="usr_admin"))



# =====================================================================
# --- PREVIOUS SYNCHRONOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct
# from .embedding_model import embedding_model
# from .text_chunker import text_splitter, chunk_markdown_sections
# 
# client = QdrantClient(host="localhost", port=6333)
# COLLECTION_NAME = "rag_docs"
# 
# def ingest_documents(folder_path: str, tenant_id: str = "tenant_default", user_id: str = "usr_admin"):
#     ...
#     client.upsert(collection_name=COLLECTION_NAME, points=points)
# =====================================================================