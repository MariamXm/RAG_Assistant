# 📄 RAG PDF Chatbot (LangChain + Groq + ChromaDB + FastAPI + Streamlit)

A full-stack **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask natural language questions about their content.

The project uses a **local FastAPI backend** for RAG processing and a **deployed Streamlit frontend** for user interaction.

---

# 🚀 System Architecture

```text
Streamlit Frontend (Deployed on Streamlit Cloud)
                ↓ HTTP Requests
FastAPI Backend (Running Locally)
                ↓
RAG Pipeline (LangChain + ChromaDB)
                ↓
Groq LLM (Llama 3)
                ↓
Final Answer Returned to UI
```
---

# ⚙️ Features
  - 📤 Upload PDF documents
  -  🧠 Automatic chunking + embedding generation
  - 🔍 Semantic search using ChromaDB
  - 🤖 AI-powered answers using Groq (Llama 3)
  - ⚡ FastAPI backend for RAG processing
  - 🎨 Streamlit cloud-based chat interface
  - 💾 Persistent vector storage (local)
---

# 🧰 Tech Stack

Python, FastAPI, Streamlit, LangChain, ChromaDB, HuggingFace Embeddings, Groq API (Llama 3), PyPDF

---

# 🚀 Setup Instructions
## 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/RAG_Assistant.git
cd RAG_Assistant
```
## 2️⃣ Backend Setup (Run Locally)
Create virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run FastAPI backend:
```bash
uvicorn api:app --reload --port 8000
```
Backend will run at:
```bash
http://127.0.0.1:8000
```
## 3️⃣ Frontend Deployed on Streamlit Cloud
Live App: https://ragassistant-apvdzttjd7f65ptfskybuh.streamlit.app/

Upload a PDF and start asking questions.

---
# 🚀 Future Improvements

- Deploy FastAPI backend on cloud (Render / Railway)
- Add authentication (JWT)
- Multi-document support
- Streaming responses
- Better retrieval with reranking
