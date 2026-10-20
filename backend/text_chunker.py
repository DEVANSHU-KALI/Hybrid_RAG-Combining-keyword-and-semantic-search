from langchain_experimental.text_splitter import SemanticChunker
from .embedding_model import embedding_model

# CONCEPT: 768D Semantic Chunker & Dynamic Boundary Detection
# In production RAG, semantic chunking embeds each sentence with our 768D model 
# and splits text at natural semantic transitions (percentile thresholding) 
# rather than cutting sentences mid-thought at arbitrary character limits.

text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75
)

# CONCEPT: Structure & Table-Aware Section Chunker Helper
def chunk_markdown_sections(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Helper utility for structured documents (PDFs with extracted Markdown tables).
    Ensures that Markdown tables (| Header | Value |) remain intact inside a single chunk 
    and are not severed across vector chunks.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0
    in_table = False

    for line in lines:
        is_table_line = line.strip().startswith("|") and line.strip().endswith("|")
        line_len = len(line) + 1

        # Check if we should split before adding this line:
        # 1. We've exceeded max_chunk_size AND we are not currently inside a table
        # 2. Or a table is starting and current_chunk is already substantial (> 500 chars)
        should_split = False
        if not in_table and current_chunk:
            if current_length + line_len > max_chunk_size:
                should_split = True
            elif is_table_line and current_length > (max_chunk_size // 2):
                should_split = True

        if should_split:
            chunk_str = "\n".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            current_chunk = []
            current_length = 0

        # Update in_table status for subsequent lines
        in_table = is_table_line
        current_chunk.append(line)
        current_length += line_len

    if current_chunk:
        chunk_str = "\n".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks


# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from langchain_experimental.text_splitter import SemanticChunker
# 
# from .embedding_model import embedding_model
# 
# text_splitter = SemanticChunker(
#     embedding_model,
#     breakpoint_threshold_type="percentile",
#     breakpoint_threshold_amount=75
# )
# =====================================================================
