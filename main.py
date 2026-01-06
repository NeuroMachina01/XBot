import os
import time
import streamlit as st

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# -------------------------------
# CONFIG
# -------------------------------
VECTORSTORE_DIR = "faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"  # best for embeddings
LLM_MODEL = "llama3"

# -------------------------------
# INIT MODELS
# -------------------------------
llm = OllamaLLM(model=LLM_MODEL)

def get_embeddings():
    """Create a fresh embeddings object (required for FAISS load/save)."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="XBot: News Research Tool", layout="wide")
st.title("📈 XBot: News Research Tool")

st.sidebar.header("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

status_box = st.empty()

# -------------------------------
# URL PROCESSING
# -------------------------------
if process_url_clicked:
    urls = [u for u in urls if u.strip()]

    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        try:
            status_box.info("🔄 Loading articles...")
            loader = WebBaseLoader(urls)
            documents = loader.load()

            status_box.info("✂️ Splitting text...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", ","]
            )
            docs = splitter.split_documents(documents)

            status_box.info("🧠 Creating embeddings...")
            embeddings = get_embeddings()

            vectorstore = FAISS.from_documents(docs, embeddings)

            status_box.info("💾 Saving vector store...")
            vectorstore.save_local(VECTORSTORE_DIR)

            status_box.success("✅ URLs processed successfully!")

        except Exception as e:
            st.error(f"Error processing URLs: {e}")

# -------------------------------
# QUESTION ANSWERING
# -------------------------------
st.markdown("---")
query = st.text_input("Ask a question based on the articles:")

if query:
    if not os.path.exists(VECTORSTORE_DIR):
        st.warning("Please process URLs first.")
    else:
        try:
            embeddings = get_embeddings()

            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
            )

            with st.spinner("🤖 Thinking..."):
                result = chain.invoke(
                    {"question": query},
                    return_only_outputs=True
                )

            st.subheader("🧠 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                for src in sources.split("\n"):
                    if src.strip():
                        st.write(src)

        except Exception as e:
            st.error(f"Error processing query: {e}")
