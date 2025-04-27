from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import re

def load_documents(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def clean_text(text):
    text = re.sub(r'\n+', '\n', text) # Remove extra newlines
    text = re.sub(r'[^\w\s.,]', '', text) # Remove weird symbols
    text = text.lower()
    return text

def split_text(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

def semantic_split(docs, embedding_model):
    chunker = SemanticChunker(embedding_model)
    return chunker.split_documents(docs)
