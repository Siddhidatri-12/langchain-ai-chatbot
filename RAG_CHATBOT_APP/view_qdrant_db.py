from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

records, _ = client.scroll(
    collection_name="knowledge_base",
    limit=1000,
    with_payload=True
)

sources = set()

for point in records:

    if point.payload:

        source = point.payload.get("source")

        if source:
            sources.add(source)

print("\nFILES FOUND IN QDRANT:\n")

for source in sorted(sources):
    print(source)

print("\nTotal Files:", len(sources))