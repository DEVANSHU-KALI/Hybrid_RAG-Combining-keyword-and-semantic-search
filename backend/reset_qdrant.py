from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType

# Connect To Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

# Collection Name
COLLECTION_NAME = "rag_docs"

# CONCEPT: Safe Reset & Automated Multi-Tenant Collection Re-creation
# In development & testing environments, resetting wiping old vectors and 
# immediately recreating the 768D schema + payload indexes guarantees test cleanliness.

try:
    client.delete_collection(
        collection_name=COLLECTION_NAME
    )
    print(f"\n✅ Deleted collection: '{COLLECTION_NAME}'")
except Exception as error:
    print(f"\n⚠️ Collection '{COLLECTION_NAME}' does not exist or deletion failed: {error}")

# Re-create 768D Collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    ),
)

# Re-create Multi-Tenant Payload Indexes
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="tenant_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="user_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

print(f"✅ Reset complete: Collection '{COLLECTION_NAME}' recreated with 768D vectors & payload indexes.\n")


# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# from qdrant_client import QdrantClient
# 
# # Connect To Qdrant
# client = QdrantClient(
#     host="localhost",
#     port=6333
# )
# 
# # Collection Name
# COLLECTION_NAME = "rag_docs"
# 
# # Delete Collection
# try:
#     client.delete_collection(
#         collection_name=COLLECTION_NAME
#     )
#     print(
#         f"\nDeleted collection: "
#         f"{COLLECTION_NAME}"
#     )
# 
# except Exception as error:
#     print(
#         f"\nError deleting collection: "
#         f"{error}"
#     )
# 
# # Reset Complete
# print(
#     "\nQdrant reset complete."
# )
# =====================================================================
