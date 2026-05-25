from rag_chatbot_faiss import ask_question

question = input("Ask a question: ")

answer = ask_question(question)

print("\nAnswer:")
print(answer)