import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("LANGCHAIN_API_KEY"))

# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"

print(os.getenv("LANGCHAIN_TRACING_V2"))
print(os.getenv("LANGCHAIN_PROJECT"))


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user questions."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key = api_key
    llm=ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer=chain.invoke({"question": question})
    return answer

# Title of the app
st.title("Q&A Chatbot")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Drop down for model selection
llm = st.sidebar.selectbox("Select a model", ["gpt-4o", "gpt-4o-mini"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150, step=100)

# Main interface for user input
st.write("Enter your question below")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write("Bot: ", response)
else:
    st.write("Please enter a question")