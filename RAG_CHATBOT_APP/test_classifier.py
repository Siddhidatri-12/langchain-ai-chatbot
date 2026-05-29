from RAG_CHATBOT_APP.query_classifier import identify_tag

question = input("Ask Question: ")

tag = identify_tag(question)

print("Detected Tag:", tag)
