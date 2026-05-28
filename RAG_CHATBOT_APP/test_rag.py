from rag_chatbot import ask_question

question = input("Ask a question: ")

answer = ask_question(question)

print("\nAnswer:")
print(answer)