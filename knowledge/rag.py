import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent DB
client = chromadb.Client()
collection = client.get_or_create_collection(name="jarvis_knowledge")


def read_file(path):
    if path.endswith(".pdf"):
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def index_knowledge(folder="knowledge"):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        text = read_file(path)

        if not text.strip():
            continue

        embedding = embedder.encode(text).tolist()

        collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[file]
        )

    print("Knowledge indexed.")


def search_knowledge(query, k=2):
    embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    if results["documents"]:
        return "\n".join(results["documents"][0])
    return ""
