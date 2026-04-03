🤖 Advanced RAG Chatbot with Ollama (Llama3)

An AI-powered chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions from documents, analyze resumes, and compare candidates — powered by a **local LLM (Llama3 via Ollama)**.

---

## 🚀 Features

- 📄 Multi-PDF Upload & Processing  
- 🔍 Context-aware Question Answering (RAG)  
- 🤖 Local LLM using Ollama (Llama3)  
- 📊 Resume Comparison  
- 🧠 Smart Retrieval (FAISS + MMR)  
- 💬 ChatGPT-like UI (Streamlit)  
- 🔒 No API Cost (Fully Local)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Llama3 (Ollama)  
- **Embeddings:** HuggingFace (MiniLM)  
- **Vector DB:** FAISS  
- **Framework:** LangChain  
- **Language:** Python  

---

## 📂 Project Structure


rag-chatbot/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore


---

## ⚙️ Installation

### 1. Clone the Repository

bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-ollama.git
cd rag-chatbot-ollama
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Install Ollama

Download and install:
👉 https://ollama.com/download

Run:

ollama run llama3
▶️ Run the App
streamlit run app.py
