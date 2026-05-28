import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from constants import CLASSIFIER_PROMPT

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)


def identify_tag(question):

    prompt = CLASSIFIER_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    return response.content.strip()