import os

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI

from constants import (
    AGENT_PROMPT,
    RAG_PROMPT
)

from config import *

load_dotenv()

# ------------------------
# QDRANT
# ------------------------

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

# ------------------------
# EMBEDDING MODEL
# ------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# ------------------------
# LLM
# ------------------------

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)


def ask_question(question):

    # ------------------------
    # QUESTION EMBEDDING
    # ------------------------

    query_vector = embedding_model.encode(
        question
    ).tolist()

    # ------------------------
    # SEARCH QDRANT
    # ------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RESULTS
    ).points

    print("\n====================")
    print("RESULTS FOUND:", len(results))
    print("====================")

    for result in results:
        print(result.payload)

    # ------------------------
    # BUILD CONTEXT
    # ------------------------

    context_chunks = []

    for result in results:

        if result.payload and "text" in result.payload:

            context_chunks.append(
                result.payload["text"]
            )

    context = "\n\n".join(
        context_chunks
    )

    print("\n====================")
    print("CONTEXT LENGTH:", len(context))
    print("====================")
    print(context[:1000])

    # ------------------------
    # NO CONTEXT FOUND
    # ------------------------

    if not context.strip():

        return "I don't know based on the knowledge base."

    # ------------------------
    # PROMPT
    # ------------------------

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    final_prompt = f"""
{AGENT_PROMPT}

{prompt}
"""

    # ------------------------
    # LLM RESPONSE
    # ------------------------

    response = llm.invoke(
        final_prompt
    )

    return response.content