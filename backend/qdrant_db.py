from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "rag_docs"

collections = client.get_collections().collections

existing_collections = [
    collection.name
    for collection in collections
]

if collection_name not in existing_collections:

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )

    print(f"Collection '{collection_name}' created.")

else:
    print(f"Collection '{collection_name}' already exists.")