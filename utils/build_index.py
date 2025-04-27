import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_faiss_index(documents, embedding_model, save_path="recipe_faiss_index"):
    vectorstore = FAISS.from_documents(documents, embedding_model)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

def load_faiss_index(embedding_model, load_path="recipe_faiss_index"):
    return FAISS.load_local(load_path, embedding_model)

