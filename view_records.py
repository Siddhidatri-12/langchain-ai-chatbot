from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

results = client.scroll(
    collection_name="knowledge_base",
    limit=5,
    with_payload=True
)

print(results)