from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
import tempfile

from vector_store import add_to_vector_db, query_vector_db
from llm import generate_answer
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter


# app init
app = FastAPI(title="RAG PDF Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# request model
class QuestionRequest(BaseModel):
    question: str


# upload endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    text = ""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    metadatas = [{"source": "uploaded_pdf"} for _ in chunks]

    add_to_vector_db(chunks, source="uploaded_pdf")

    return {
        "message": "PDF uploaded and indexed",
        "chunks": len(chunks)
    }


# ask endpoint
@app.post("/ask")
def ask_question(req: QuestionRequest):
    retrieved = query_vector_db(req.question)

    chunks = retrieved.get("documents", [])

    if not chunks:
        return {"answer": "No relevant content found"}

    context = "\n\n".join(chunks)

    answer = generate_answer(context, req.question)

    return {
        "question": req.question,
        "answer": answer,
        "context": chunks
    }