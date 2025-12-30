# vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from typing import List

# LangChain used ONLY for embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Native Chroma client (auto-persistent)
client = chromadb.Client(
    Settings(persist_directory="chroma_db")
)

collection = client.get_or_create_collection(name="pdf_docs")


def add_to_vector_db(texts: List[str]):
    embeddings = embedding_model.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )


def query_vector_db(query: str, k: int = 3):
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]
