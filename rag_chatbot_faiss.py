import os
import json
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index(
    "vector_db/faiss_index.bin"
)

# Load chunks
with open(
    "vector_db/chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

# Load metadata
with open(
    "vector_db/metadata.json",
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

# OpenRouter LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)


def ask_question(question):

    # Convert question to embedding
    query_embedding = embedding_model.encode(
        [question]
    ).astype("float32")

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        k=3
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):

            retrieved_chunks.append(
                chunks[idx]
            )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
Answer ONLY using the provided context.

If the answer is not available in the context,
say:

I don't know based on the provided documents.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content