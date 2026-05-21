import gradio as gr

from rag_chatbot import ask_question


def chatbot_response(question):

    answer = ask_question(question)

    return answer


interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question from your knowledge base..."
    ),
    outputs="text",
    title="AI Knowledge Base Chatbot",
    description="Ask questions from uploaded PDF documents."
)

interface.launch()