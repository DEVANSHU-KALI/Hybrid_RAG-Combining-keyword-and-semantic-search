import os

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from .embedding_model import embedding_model
from .text_chunker import text_splitter


# -----------------------------
# Connect to Qdrant
# -----------------------------
client = QdrantClient(
    host="localhost",
    port=6333
)

# -----------------------------
# Collection Name
# -----------------------------
COLLECTION_NAME = "rag_docs"

# -----------------------------
# Document Ingestion Function
# -----------------------------
def ingest_documents(folder_path: str):
    documents = []
    ids = []
    sources = []
    counter = 0

    # ----------------------------- 
    # Read TXT Files
    # -----------------------------
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(
            folder_path,
            filename
        )
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()


        # -----------------------------
        # Semantic Chunking
        # -----------------------------
        chunks = text_splitter.create_documents([text])

        # -----------------------------
        # Store Chunk Data
        # -----------------------------
        for chunk in chunks:
            documents.append(chunk.page_content)
            ids.append(counter)
            sources.append(filename)
            counter += 1


    # -----------------------------
    # Print Chunks
    # -----------------------------
    print("\n======= DOCUMENT CHUNKS =======\n")
    for doc in documents:
        print(doc)
        print("-------------")

    # -----------------------------
    # Generate Embeddings
    # -----------------------------
    embeddings = embedding_model.embed_documents(documents)
    print(
        "\nEmbedding vector size:",
        len(embeddings[0])
    )

    # -----------------------------
    # Create Qdrant Points
    # -----------------------------
    points = []

    for i in range(len(documents)):
        points.append(
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "text": documents[i],
                    "source": sources[i],
                    "chunk_id": ids[i]
                }
            )
        )

    # -----------------------------
    # Upload Points to Qdrant
    # -----------------------------
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(
        f"\n✅ Documents successfully ingested into '{COLLECTION_NAME}'"
    )


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    ingest_documents(
        "data/rag_concepts"
    )