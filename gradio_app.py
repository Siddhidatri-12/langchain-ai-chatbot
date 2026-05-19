import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize LangChain model
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Chat function
def chatbot_response(user_input):

    # Generate response
    response = llm.invoke(user_input)

    return response.content

# Gradio interface
interface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="AI Chatbot using LangChain",
    description="Ask any question to the AI chatbot"
)

# Launch app
interface.launch()