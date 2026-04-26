import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")

st.title("📄 RAG PDF Chatbot")
st.caption("Upload a PDF and chat with it using AI (Groq + LangChain)")

# session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# pdf upload section
st.header("📤 Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload & Process"):
        with st.spinner("Uploading and processing PDF..."):

            files = {"file": uploaded_file}

            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )

            if response.status_code == 200:
                data = response.json()
                st.success(f"PDF processed successfully! Chunks: {data['chunks']}")
            else:
                st.error("Upload failed. Check backend.")

# chat interface
st.header(" Chat with PDF")

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# user input
user_query = st.chat_input("Ask something about the PDF...")

if user_query:

    # store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    # call backend
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": user_query}
        )

        if response.status_code == 200:
            answer = response.json()["answer"]
        else:
            answer = "Error: Could not get response from API."

    # store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)


if st.button(" Clear Chat"):
    st.session_state.messages = []
    st.rerun()