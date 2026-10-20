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


# CONCEPT: Multimodal PDF & Bounding-Box Markdown Table Parsing Utility
def extract_pdf_content(file_path: str) -> list[dict]:
    """
    Extracts narrative text and structured tables from PDF files in true reading order.
    Uses bounding-box table detection (page.find_tables()) to extract narrative text
    strictly outside the tables and replaces the table regions with clean Markdown tables.
    This prevents duplicate table text and avoids unformatted run-on table text inside prose chunks.
    """
    pages_content = []
    
    try:
        # Import pdfplumber library for layout inspection and table extraction
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.find_tables()
                    
                    # If page has no tables, extract regular narrative text
                    if not tables:
                        page_text = page.extract_text() or ""
                        pages_content.append({
                            "page_number": page_num,
                            "text": page_text.strip(),
                            "has_table": False
                        })
                        continue

                    # Sort tables vertically from top to bottom by their vertical coordinate (bbox[1])
                    tables = sorted(tables, key=lambda t: t.bbox[1])
                    
                    page_elements = []
                    prev_bottom = 0.0

                    for table_obj in tables:
                        t_bbox = table_obj.bbox  # (x0, top, x1, bottom)
                        top = t_bbox[1]
                        bottom = t_bbox[3]

                        # 1. Extract narrative text strictly ABOVE this table
                        if top > prev_bottom:
                            text_above = page.crop((0, prev_bottom, page.width, top)).extract_text() or ""
                            if text_above.strip():
                                page_elements.append(text_above.strip())

                        # 2. Extract structured table cells and convert to Markdown
                        raw_table = table_obj.extract()
                        if raw_table and len(raw_table) > 0:
                            # Clean cell text and replace internal newlines with space to preserve Markdown row integrity
                            headers = [str(cell or "").strip().replace("\n", " ") for cell in raw_table[0]]
                            md_rows = [
                                "| " + " | ".join(headers) + " |",
                                "| " + " | ".join(["---"] * len(headers)) + " |"
                            ]
                            for row in raw_table[1:]:
                                clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                                md_rows.append("| " + " | ".join(clean_row) + " |")
                            
                            page_elements.append("\n".join(md_rows))

                        # Update boundary to avoid re-extracting table text
                        prev_bottom = max(prev_bottom, bottom)

                    # 3. Extract narrative text strictly BELOW the last table
                    if prev_bottom < page.height:
                        text_below = page.crop((0, prev_bottom, page.width, page.height)).extract_text() or ""
                        if text_below.strip():
                            page_elements.append(text_below.strip())

                    # Assemble page text in authentic document flow: Text -> Table (MD) -> Text
                    full_page_text = "\n\n".join(page_elements)
                    pages_content.append({
                        "page_number": page_num,
                        "text": full_page_text.strip(),
                        "has_table": True
                    })
                except Exception as page_err:
                    # Fallback to standard text extraction if bounding-box cropping encounters an anomaly
                    page_text = page.extract_text() or ""
                    pages_content.append({
                        "page_number": page_num,
                        "text": page_text.strip(),
                        "has_table": False
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
                    # Detect if THIS specific chunk contains a Markdown table
                    chunk_has_table = any(
                        line.strip().startswith("|") and line.strip().endswith("|")
                        for line in chunk_text.split("\n")
                    )

                    documents.append(chunk_text)
                    ids.append(counter)
                    sources.append(filename)
                    metadatas.append({
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "document_type": "pdf",
                        "page_number": page_data["page_number"],
                        "has_table": chunk_has_table
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
    asyncio.run(ingest_documents("data", tenant_id="tenant_default", user_id="usr_admin"))



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