import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load PDFs
documents = []

folder = "knowledge_base"

for file in os.listdir(folder):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            os.path.join(folder, file)
        )

        documents.extend(loader.load())

print("Documents Loaded:", len(documents))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks Created:", len(chunks))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Model Loaded")

# Create Vector Database
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vector_db"
)

print("Vector Database Created Successfully")