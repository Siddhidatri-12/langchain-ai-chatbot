import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)

AVAILABLE_TAGS = [
    "ai",
    "machine_learning",
    "deep_learning"
]


def identify_tag(question):

    prompt = f"""
You are a tag classifier.

Available tags:

ai
machine_learning
deep_learning

Question:
{question}

Return ONLY one tag.
Do not explain.
"""

    response = llm.invoke(prompt)

    return response.content.strip()