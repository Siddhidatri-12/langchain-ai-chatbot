from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from qdrant_client.models import Distance

client = QdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "knowledge_base"

# -------------------------
# DELETE OLD COLLECTION
# -------------------------

try:

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

    print("Collection Deleted")

except:

    print("No Existing Collection Found")

# -------------------------
# CREATE NEW COLLECTION
# -------------------------

client.create_collection(
    collection_name=COLLECTION_NAME,

    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection Created Successfully")