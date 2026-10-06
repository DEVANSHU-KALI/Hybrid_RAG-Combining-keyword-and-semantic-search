# Imports QdrantClient for database initialization and HTTP model classes for vector & payload indexing schemas
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType

# Initialize synchronous Qdrant client pointing to local Docker container on port 6333
client = QdrantClient(host="localhost", port=6333)

# Target collection name in Qdrant database
collection_name = "rag_docs"

# Fetch existing collections metadata from Qdrant database engine
collections = client.get_collections().collections

# Extract collection text names into a list of strings
existing_collections = [collection.name for collection in collections]


# CONCEPT: 768D Vector Collection Schema & Multi-Tenant Payload Indexing
# 1. 768 Vector Size matches enterprise embedding model (BAAI/bge-base-en-v1.5).
# 2. Payload Indexing: Creating keyword payload indexes on tenant_id and user_id 
#    allows Qdrant to execute sub-millisecond pre-filtering, physically isolating 
#    multi-tenant data and preventing data leakage.

if collection_name not in existing_collections:

    # Create collection with 768-dimensional vectors and Cosine distance metric
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,                          # Vector coordinate dimensions matching BAAI/bge-base-en-v1.5
            distance=Distance.COSINE           # Distance metric for calculating directional angle similarity
        ),
    )

    # CONCEPT: Multi-Tenant Payload Indexes (Pre-Filtering Acceleration)
    # What does create_payload_index do?
    # It creates an inverted index on disk/RAM specifically for the "tenant_id" and "user_id" payload fields.
    # Without this index, Qdrant would have to scan every single vector point in memory (slow full scan).
    # With this index, Qdrant instantly jumps to the exact subset of vectors belonging to that tenant!
    client.create_payload_index(
        collection_name=collection_name,
        field_name="tenant_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print(f"Collection '{collection_name}' created with 768D vectors and Multi-Tenant Payload Indexes.")

else:
    print(f"Collection '{collection_name}' already exists.")



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, Distance
# 
# client = QdrantClient(
#     host="localhost",
#     port=6333
# )
# 
# collection_name = "rag_docs"
# 
# collections = client.get_collections().collections
# 
# existing_collections = [
#     collection.name
#     for collection in collections
# ]
# 
# if collection_name not in existing_collections:
# 
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(
#             size=384,
#             distance=Distance.COSINE
#         ),
#     )
# 
#     print(f"Collection '{collection_name}' created.")
# 
# else:
#     print(f"Collection '{collection_name}' already exists.")
# =====================================================================
