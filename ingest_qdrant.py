import os
import json
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer


# ----------------------------
# LOAD TAG CONFIGURATION
# ----------------------------

with open(
    "tags.json",
    "r",
    encoding="utf-8"
) as f:

    tags_config = json.load(f)


# ----------------------------
# CONNECT TO QDRANT
# ----------------------------

client = QdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "knowledge_base"


# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


# ----------------------------
# TEXT SPLITTER
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# ----------------------------
# PROCESS PDFs
# ----------------------------

folder = "knowledge_base"

points = []

for tag, files in tags_config.items():

    for pdf_file in files:

        pdf_path = os.path.join(
            folder,
            pdf_file
        )

        if not os.path.exists(pdf_path):

            print(
                f"File not found: {pdf_file}"
            )

            continue

        print(
            f"Processing: {pdf_file}"
        )

        loader = PyPDFLoader(
            pdf_path
        )

        documents = loader.load()

        chunks = splitter.split_documents(
            documents
        )

        for chunk in chunks:

            text = chunk.page_content

            embedding = embedding_model.encode(
                text
            ).tolist()

            page = chunk.metadata.get(
                "page",
                0
            )

            points.append(

                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={

                        "text": text,

                        "tag": tag,

                        "source": pdf_file,

                        "page": page

                    }
                )

            )


# ----------------------------
# INSERT INTO QDRANT
# ----------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(
    f"{len(points)} chunks inserted successfully"
)