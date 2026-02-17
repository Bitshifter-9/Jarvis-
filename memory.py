import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client=chromadb.Client()
collection = client.get_or_create_collection(name="jarvis_memory")

def store_memory(text):
    embedding = embedder.encode(text).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[text],
        ids=[str(hash(text))]
    )
def recall_memory(query, k=3):
    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )
    return results["documents"][0] if results["documents"] else []
