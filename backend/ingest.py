import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import CHROMA_DIR


def ingest_pdf(file_path: str):
    doc_id = str(uuid.uuid4())
    filename = file_path.split("/")[-1]

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No content found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    for doc in docs:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["doc_name"] = filename
        doc.metadata["source"] = filename
        doc.metadata["page"] = doc.metadata.get("page")

    embeddings = OpenAIEmbeddings()
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    store.add_documents(docs)
    store.persist()

    return {"id": doc_id, "name": filename}
