# llm.py

from groq import Groq
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set or empty in environment variables.")

client = Groq(api_key=api_key)

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
