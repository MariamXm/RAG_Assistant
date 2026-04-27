# 📄 RAG PDF Chatbot (LangChain + Groq + ChromaDB + FastAPI + Streamlit)

A full-stack **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask questions about their content using AI.

Built with **FastAPI (backend)**, **Streamlit (frontend)**, **LangChain**, **Langsmith**, **ChromaDB**, and **Groq LLM (Llama 3)**.

---

# 🚀 Features

- 📤 Upload and process PDF files
- 🔍 Semantic search over document chunks
- 🤖 AI-powered answers using Groq (Llama 3)
- 🧠 Vector database using ChromaDB
- ⚡ FastAPI backend for API handling
- 🎨 Streamlit-based chat UI
- 💾 Persistent vector storage (ChromaDB)

---

# 🧰 Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API (Llama 3)
- PyPDF

---

# How It Works

1. Upload PDF
2. Split into chunks
3. Convert chunks into embeddings
4. Store in ChromaDB
5. Retrieve relevant chunks on query
6. Send context to Groq LLM
7. Return final answer
---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/MariamXm/RAG_Assistant.git
cd RAG_Assistant
````

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

# ▶️ Running the Project

## 1️⃣ Start FastAPI Backend

```bash
uvicorn api:app --reload --port 8000
```

API Docs:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2️⃣ Start Streamlit Frontend

```bash
streamlit run app.py
```

App URL:
[http://localhost:8501](http://localhost:8501)


