import pytest
from utils.build_index import load_faiss_index
from utils.retriever import get_relevant_chunks
from langchain.embeddings import GoogleGenerativeAIEmbeddings

@pytest.fixture
def retriever():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = load_faiss_index(embedding_model)
    return vectorstore.as_retriever()

def test_basic_retrieval(retriever):
    query = "How to make chicken curry?"
    chunks = get_relevant_chunks(query, retriever)
    assert any("chicken" in c.page_content and "curry" in c.page_content for c in chunks)

def test_threshold_filtering(retriever):
    query = "What is the capital of France?"
    chunks = get_relevant_chunks(query, retriever)
    assert len(chunks) == 0

def test_edge_case(retriever):
    query = "Vegetarian gluten-free dessert with less than 5 ingredients."
    chunks = get_relevant_chunks(query, retriever)
    assert any("vegetarian" in c.page_content or "gluten-free" in c.page_content for c in chunks)
