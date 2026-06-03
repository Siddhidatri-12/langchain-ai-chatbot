import os
import json

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from constants import CLASSIFIER_PROMPT

load_dotenv()

# -------------------------
# LOAD AVAILABLE TAGS
# -------------------------

with open(
    "tags.json",
    "r",
    encoding="utf-8"
) as f:

    AVAILABLE_TAGS = json.load(f)["tags"]

# -------------------------
# LLM
# -------------------------

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-super-120b-a12b:free"
)

# -------------------------
# IDENTIFY TAG
# -------------------------

def identify_tag(question):

    with open(
        "tags.json",
        "r",
        encoding="utf-8"
    ) as f:

        available_tags = json.load(f)["tags"]

    tags_text = "\n".join(
        available_tags
    )

    prompt = CLASSIFIER_PROMPT.format(
        available_tags=tags_text,
        question=question
    )

    response = llm.invoke(
        prompt
    )

    return response.content.strip().lower()