# RAG Website Chatbot

A personal project exploring **Retrieval-Augmented Generation (RAG)** using FastAPI and LangChain.  
The chatbot answers questions **strictly from uploaded documents**, with transparent source citations and document-level control.

This project is built for learning, experimentation, and incremental improvement in modern LLM system design.

---

## ✨ Key Features

- 📄 **Document-grounded answers**  
  Responses are generated only from uploaded documents (no background knowledge).

- 🔍 **Source transparency**  
  Each answer includes the source document and page numbers.

- 🗂️ **Document management**  
  Upload, enable/disable, or remove documents used for retrieval.

- 🎯 **Scoped retrieval**  
  Questions can be restricted to selected documents only.

- 🛡️ **Hallucination-aware design**  
  If information is not found in the documents, the system explicitly says so.

- 🌐 **Simple web UI**  
  Minimal HTML/CSS/JS interface for interacting with the chatbot.

---

## 🧠 Architecture Overview

```
Frontend (HTML / CSS / JS)
        ↓
FastAPI (API, uploads, document control)
        ↓
LangChain (retrieval + prompt orchestration)
        ↓
Chroma Vector Store + OpenAI LLM
```

- **FastAPI** handles routing, uploads, and document lifecycle.
- **LangChain** manages chunking, retrieval, and controlled generation.
- **ChromaDB** stores vector embeddings for semantic search.
- **OpenAI models** generate answers strictly from retrieved context.

---

## 🧰 Tech Stack

- Python 3.11
- FastAPI
- LangChain
- ChromaDB
- OpenAI API
- HTML / CSS / JavaScript

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Daivik2605/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run the application
```bash
python -m uvicorn backend.main:app --reload
```

Open in browser:
```
http://127.0.0.1:8000
```

---

## 📌 Current Limitations

- No authentication or multi-user isolation
- No persistent metadata store outside the vector database
- UI is intentionally minimal to keep focus on RAG behavior

These tradeoffs are intentional for learning clarity.

---

## 🗓️ Next Week Tasks

- Add configurable retriever parameters (`k`, `score_threshold`) to control retrieval strictness  
- Implement optional query rewriting to improve recall for vague user queries  
- Refactor RAG chain to support dynamic model selection via configuration  
- Add type hints and docstrings across ingestion, retrieval, and API layers  
- Introduce basic retrieval logging (matched documents, scores, and filters applied)

---

## 📚 Learning Goals

- Understand RAG architecture and retrieval behavior deeply  
- Explore hallucination control strategies  
- Practice clean LangChain integration patterns  
- Build production-minded AI systems incrementally

---

## ⚠️ Disclaimer

This is a **personal learning project**, not a production SaaS.  
No guarantees are made regarding accuracy or suitability for real-world decision-making.
