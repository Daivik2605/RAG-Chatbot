from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.ingest import ingest_pdf
from backend.rag import get_rag_chain
from backend.config import CHROMA_DIR


app = FastAPI(title="RAG Website Chatbot")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")


# ---------- Document APIs ----------

@app.get("/documents")
def list_documents():
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings()
    )

    docs = {}
    for d in store.get()["metadatas"]:
        docs[d["doc_id"]] = d["doc_name"]

    return [{"id": k, "name": v} for k, v in docs.items()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings()
    )

    store.delete(where={"doc_id": doc_id})
    store.persist()
    return {"status": "deleted"}


# ---------- Upload ----------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    path = f"data/uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    return ingest_pdf(path)


# ---------- Chat ----------

class ChatRequest(BaseModel):
    query: str
    document_ids: list[str] | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    chain = get_rag_chain(req.document_ids)
    result = chain.invoke({"input": req.query})

    context = result.get("context", [])
    answer = result.get("answer", "")

    if not context:
        return {
            "answer": "I could not find this information in the provided documents.",
            "sources": []
        }

    sources = {}
    for doc in context:
        src = doc.metadata["source"]
        page = doc.metadata.get("page")
        sources.setdefault(src, set())
        if page is not None:
            sources[src].add(page)

    return {
        "answer": answer,
        "sources": [
            {"source": s, "pages": sorted(p)}
            for s, p in sources.items()
        ]
    }
