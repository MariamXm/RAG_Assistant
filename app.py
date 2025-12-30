# app.py

import streamlit as st
from pypdf import PdfReader
from vector_store import add_to_vector_db, query_vector_db
from llm import generate_answer

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Chunking
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    add_to_vector_db(chunks)
    st.success("PDF indexed successfully!")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        retrieved_chunks = query_vector_db(query)

        if retrieved_chunks:
            context = "\n\n".join(retrieved_chunks)

            answer = generate_answer(context, query)

            st.subheader(" Answer")
            st.write(answer)

            st.subheader(" Retrieved Context")
            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"**Chunk {i+1}:** {chunk}")
        else:
            st.write("No relevant content found.")
