from fastapi import FastAPI
from pydantic import BaseModel

from rag_chatbot_qdrant import ask_question

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.get("/llm/chat")
def chatbot_status():

    return {
        "application": "RAG Chatbot",
        "status": "Running",
        "vector_database": "Qdrant",
        "llm": "NVIDIA Nemotron-3 Super 120B"
    }


@app.post("/llm/chat")
def chatbot(request: QueryRequest):

    answer = ask_question(
        request.query
    )

    return {
        "query": request.query,
        "answer": answer
    }