from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "knowledge_base"

existing = [
    c.name
    for c in client.get_collections().collections
]

if collection_name not in existing:

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Collection Created")

else:

    print("Collection Already Exists")