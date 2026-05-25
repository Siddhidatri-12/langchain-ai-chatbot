import gradio as gr

from rag_chatbot_faiss import ask_question


def respond(message, history):

    answer = ask_question(message)

    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="📚 AI Knowledge Base Assistant",
    description="Answers questions from locally stored PDF documents using FAISS Vector DB."
)

demo.launch()