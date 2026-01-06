from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTORSTORE_DIR = "faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"  # MUST match what you used when saving

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vs = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

print("Number of vectors in index:", vs.index.ntotal)
