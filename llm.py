from groq import Groq
import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)


@traceable(name="LLM Answer Generation")
def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are a precise document assistant.

RULES:
- Use ONLY the context below
- If answer is missing say: "Not found in document"
- Be concise and accurate

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content