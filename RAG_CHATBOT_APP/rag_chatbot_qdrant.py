import os

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI

from query_classifier import identify_tag

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
    # STEP 1
    # DETECT TAG
    # ------------------------

    tag = identify_tag(question)

    print(f"Detected Tag: {tag}")

    # ------------------------
    # STEP 2
    # QUESTION EMBEDDING
    # ------------------------

    query_vector = embedding_model.encode(
        question
    ).tolist()

    # ------------------------
    # STEP 3
    # FILTER BY TAG
    # ------------------------

    search_filter = Filter(
        must=[
            FieldCondition(
                key="tag",
                match=MatchValue(
                    value=tag
                )
            )
        ]
    )

    # ------------------------
    # STEP 4
    # SEARCH QDRANT
    # ------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=TOP_K_RESULTS
    ).points

    print(f"Results Found: {len(results)}")

    for result in results:
        print(result.payload)

    # ------------------------
    # STEP 5
    # BUILD CONTEXT
    # ------------------------

    context_chunks = []

    for result in results:

        context_chunks.append(
            result.payload["text"]
        )

    context = "\n\n".join(
        context_chunks
    )

    # ------------------------
    # NO CONTEXT FOUND
    # ------------------------

    if not context.strip():

        return "I don't know based on the knowledge base."

    # ------------------------
    # STEP 6
    # FINAL PROMPT
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
    # STEP 7
    # GENERATE RESPONSE
    # ------------------------

    response = llm.invoke(
        final_prompt
    )

    return response.content
