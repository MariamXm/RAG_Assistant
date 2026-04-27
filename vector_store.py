# vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from typing import List
import uuid
from langsmith import traceable

# load embed model once
print("Loading embedding model (first run takes ~1-2 minutes)...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# persistent chromadb setup
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="pdf_docs")

# add documents to vectordb
@traceable(name="Vector Indexing")
def add_to_vector_db(texts: List[str], source: str = "uploaded_pdf"):
    if not texts:
        return

    print(f"📥 Indexing {len(texts)} chunks...")

    embeddings = embedding_model.embed_documents(texts)

    ids = [str(uuid.uuid4()) for _ in texts]

    metadatas = [
        {"source": source, "chunk_id": i}
        for i in range(len(texts))
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print("Indexing complete!")


# query vectordb
@traceable(name="Vector Retrieval")
def query_vector_db(query: str, k: int = 3):
    print(f"🔍 Searching vector DB for: {query}")

    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    print(f"Retrieved {len(documents)} chunks")

    return {
        "documents": documents,
        "metadatas": metadatas
    }