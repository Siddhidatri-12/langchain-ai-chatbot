from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

info = client.get_collection(
    "knowledge_base"
)

print(info)