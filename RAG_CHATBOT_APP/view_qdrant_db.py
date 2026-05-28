from qdrant_client import QdrantClient

from pprint import pprint

client = QdrantClient(
    host="localhost",
    port=6333
)

records = client.scroll(
    collection_name="knowledge_base",
    limit=10,
    with_payload=True
)

for point in records[0]:

    print("\n====================\n")

    pprint(point.payload)