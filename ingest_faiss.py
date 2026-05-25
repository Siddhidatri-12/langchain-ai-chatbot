import os
import json
import numpy as np
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ==================================================
# Create vector_db folder if not exists
# ==================================================

os.makedirs("vector_db", exist_ok=True)

print("\nLoading PDF files...\n")

# ==================================================
# Load PDFs
# ==================================================

documents = []

folder = "knowledge_base"

for file in os.listdir(folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder, file)

        print(f"Reading: {file}")

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)

print(f"\nTotal Pages Loaded: {len(documents)}")


# ==================================================
# Split into chunks
# ==================================================

print("\nSplitting documents into chunks...\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")


# ==================================================
# Extract text and metadata
# ==================================================

texts = []
metadata = []

for i, chunk in enumerate(chunks):

    texts.append(chunk.page_content)

    metadata.append({
        "chunk_id": i,
        "source": chunk.metadata.get("source"),
        "page": chunk.metadata.get("page")
    })


# ==================================================
# Save chunks.json
# ==================================================

print("\nSaving chunks.json...")

with open(
    "vector_db/chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        texts,
        f,
        indent=4,
        ensure_ascii=False
    )

print("chunks.json saved successfully")


# ==================================================
# Save metadata.json
# ==================================================

print("\nSaving metadata.json...")

with open(
    "vector_db/metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print("metadata.json saved successfully")


# ==================================================
# Generate embeddings
# ==================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating embeddings...\n")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print(
    f"Embeddings shape: {embeddings.shape}"
)


# ==================================================
# Save embeddings.npy
# ==================================================

print("\nSaving embeddings.npy...")

np.save(
    "vector_db/embeddings.npy",
    embeddings
)

print("embeddings.npy saved successfully")


# ==================================================
# Create FAISS Index
# ==================================================

print("\nCreating FAISS index...\n")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings.astype("float32")
)

print(
    f"Vectors stored in FAISS: {index.ntotal}"
)


# ==================================================
# Save FAISS Index
# ==================================================

print("\nSaving faiss_index.bin...")

faiss.write_index(
    index,
    "vector_db/faiss_index.bin"
)

print("faiss_index.bin saved successfully")


# ==================================================
# Finished
# ==================================================

print("\n===================================")
print("FAISS VECTOR DATABASE CREATED")
print("===================================")

print("\nFiles Created:")

print("vector_db/chunks.json")
print("vector_db/metadata.json")
print("vector_db/embeddings.npy")
print("vector_db/faiss_index.bin")

print("\nDone!\n")