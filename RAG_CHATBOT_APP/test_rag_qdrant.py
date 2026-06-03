from rag_chatbot_qdrant import ask_question

question = input(
    "Ask Question: "
)

answer = ask_question(
    question
)

print("\nAnswer:\n")

print(answer)