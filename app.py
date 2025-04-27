import streamlit as st
from utils.load_docs import load_documents, clean_text, split_text
from utils.build_index import load_faiss_index
from utils.retriever import get_relevant_chunks
from utils.generator import generate_answer
from langchain.embeddings import GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="Cooking Chatbot 🍳", layout="wide")
st.title("🍳 Cooking Assistant Chatbot")


@st.cache_resource
def setup():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = load_faiss_index(embedding_model)
    retriever = vectorstore.as_retriever()
    return retriever


retriever = setup()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a cooking question...")

if st.button("Reset Chat"):
    st.session_state.chat_history = []

if user_query:
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

    if role == "bot":
        with st.expander("See retrieved context"):
            for chunk in relevant_chunks:
                st.write(chunk.page_content)
