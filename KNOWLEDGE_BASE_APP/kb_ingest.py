import uuid

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import PointStruct

from kb_config import *


# -------------------------
# QDRANT
# -------------------------

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

# -------------------------
# EMBEDDING MODEL
# -------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# -------------------------
# TEXT SPLITTER
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


# -------------------------
# INGEST FUNCTION
# -------------------------

def ingest_documents(files, tag):

    total_chunks = 0

    for file in files:

        pdf_reader = PdfReader(file.name)

        full_text = ""

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:

                full_text += text

        chunks = text_splitter.split_text(
            full_text
        )

        points = []

        for chunk in chunks:

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),

                vector=embedding,

                payload={
                    "text": chunk,
                    "tag": tag,
                    "source": file.name
                }
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        total_chunks += len(points)

    return f"{total_chunks} chunks inserted successfully"