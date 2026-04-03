import os
import streamlit as st
from dotenv import load_dotenv
import requests

load_dotenv()

st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")
st.title("🤖 Advanced RAG Chatbot (Ollama + RAG)")

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

process_btn = st.sidebar.button("📄 Process Documents")

# ================== SESSION STATE ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

# ================== OLLAMA FUNCTION ==================
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# ================== PROCESS DOCUMENTS ==================
if process_btn and uploaded_files:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    docs = []

    for file in uploaded_files:
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("vectorstore")

    st.session_state.db = db
    st.session_state.all_docs = docs

    st.success("✅ Documents processed!")

# ================== LOAD VECTORSTORE ==================
if st.session_state.db is None and os.path.exists("vectorstore"):
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    st.session_state.db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ================== DISPLAY CHAT ==================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== USER INPUT ==================
query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.db is None:
        st.warning("⚠️ Upload and process documents first.")
    else:
        retriever = st.session_state.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])

        # 🔥 Smart Prompt
        prompt = f"""
You are an AI assistant.

Rules:
- Answer ONLY from context
- If not found, say "Not mentioned in document"
- Be clear and professional

Context:
{context}

Question:
{query}

Answer:
"""

        response = ask_llm(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

        # 🔥 Show Sources
        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content[:300])
                st.write("------")

# ================== RESUME COMPARISON ==================
st.sidebar.markdown("---")
if st.sidebar.button("📊 Compare Resumes"):
    if st.session_state.all_docs:
        combined = "\n\n".join([doc.page_content for doc in st.session_state.all_docs])

        prompt = f"""
Compare the resumes and provide:

1. Key differences
2. Stronger candidate
3. Skills comparison
4. Final recommendation

Resumes:
{combined}
"""

        result = ask_llm(prompt)

        st.subheader("📊 Resume Comparison Result")
        st.write(result)
    else:
        st.warning("Upload resumes first.")