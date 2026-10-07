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
        
        # If transitioning into or out of a Markdown table
        if is_table_line:
            in_table = True
        elif in_table and not is_table_line:
            in_table = False

        line_len = len(line) + 1
        # Avoid breaking a table in half even if it reaches max_chunk_size
        if current_length + line_len > max_chunk_size and not in_table:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

        current_chunk.append(line)
        current_length += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

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
