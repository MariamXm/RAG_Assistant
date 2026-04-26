from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from typing import List

# embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# persistent chromadb
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="pdf_docs"
)


# add documents
def add_to_vector_db(texts: List[str], metadatas: List[dict] = None):
    embeddings = embedding_model.embed_documents(texts)

    ids = [f"doc_{i}" for i in range(len(texts))]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas if metadatas else [{} for _ in texts]
    )

# query vectordb
def query_vector_db(query: str, k: int = 3):
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results.get("metadatas", [[]])[0]
    }