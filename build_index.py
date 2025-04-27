import os
from utils.load_docs import load_documents, clean_text, split_text
from utils.build_index import build_faiss_index
from langchain.embeddings import GoogleGenerativeAIEmbeddings

def main():
    # 1. Set the path to your recipe PDF
    data_dir = "data"
    recipe_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.pdf', '.txt'))]

    if not recipe_files:
        print("No recipe documents found!")
        return

    # 2. Initialize Gemini Embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    all_documents = []

    # 3. Load and preprocess all documents
    for file_path in recipe_files:
        print(f"Processing {file_path}...")
        docs = load_documents(file_path)
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        chunks = split_text(docs, chunk_size=500, chunk_overlap=50)
        all_documents.extend(chunks)

    print(f"Total chunks created: {len(all_documents)}")

    # 4. Build and save FAISS index
    build_faiss_index(all_documents, embedding_model, save_path="recipe_faiss_index")
    print("FAISS index built and saved successfully!")

if __name__ == "__main__":
    main()
