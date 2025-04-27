import google.generativeai as genai

def initialize_gemini():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

def generate_answer(context_chunks, user_query):
    initialize_gemini()
    context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])

    prompt = f"""
You are a cooking assistant. Using the provided context, answer the user query.
Structure the answer clearly into ingredients and steps if possible.

Context:
{context_text}

User Query:
{user_query}
"""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text
