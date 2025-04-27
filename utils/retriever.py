def get_relevant_chunks(query, retriever, top_k=3, similarity_threshold=0.6):
    results = retriever.similarity_search_with_score(query, k=top_k)
    filtered = [doc for doc, score in results if score >= similarity_threshold]
    return filtered
