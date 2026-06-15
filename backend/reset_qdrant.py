from qdrant_client import QdrantClient

# Connect To Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

# Collection Name
COLLECTION_NAME = "rag_docs"

# Delete Collection
try:
    client.delete_collection(
        collection_name=COLLECTION_NAME
    )
    print(
        f"\nDeleted collection: "
        f"{COLLECTION_NAME}"
    )

except Exception as error:
    print(
        f"\nError deleting collection: "
        f"{error}"
    )

# Reset Complete
print(
    "\nQdrant reset complete."
)