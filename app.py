# app.py

import streamlit as st
from utils.load_docs import clean_text
from utils.build_index import load_faiss_index
from utils.retriever import get_relevant_chunks
from utils.generator import generate_answer
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="🍳 Cooking Chatbot", layout="wide")
st.title("🍳 Ask your Cooking Assistant!")

@st.cache_resource
def setup():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = load_faiss_index(embedding_model, load_path="recipe_faiss_index")
    retriever = vectorstore.as_retriever()
    return retriever

retriever = setup()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a cooking-related question...")

if st.button("Reset Chat"):
    st.session_state.chat_history = []

if user_query:
    # Retrieve relevant documents
    relevant_chunks = get_relevant_chunks(user_query, retriever)

    if not relevant_chunks:
        answer = "Sorry, I couldn't find any relevant recipes!"
    else:
        answer = generate_answer(relevant_chunks, user_query)

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("bot", answer))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

    # Show context snippets only for bot responses
    if role == "bot" and 'relevant_chunks' in locals():
        with st.expander("🔍 See Retrieved Context"):
            for idx, chunk in enumerate(relevant_chunks, start=1):
                st.markdown(f"**Snippet {idx}:** {chunk.page_content}")
