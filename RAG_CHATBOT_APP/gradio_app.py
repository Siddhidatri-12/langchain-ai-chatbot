import gradio as gr

from RAG_CHATBOT_APP.rag_chatbot_qdrant import ask_question


def respond(message, history):

    return ask_question(message)


demo = gr.ChatInterface(
    fn=respond,
    title="AI Knowledge Base Assistant",
    description="Powered by Qdrant + OpenRouter"
)

demo.launch()