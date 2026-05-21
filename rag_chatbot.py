import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
print("API Key:", os.getenv("OPENROUTER_API_KEY"))

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# OpenRouter LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not present in the context,
reply:

I don't know based on the provided documents.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content