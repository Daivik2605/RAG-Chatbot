from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from backend.config import CHROMA_DIR


def get_rag_chain(document_ids=None):
    embeddings = OpenAIEmbeddings()
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    search_kwargs = {"k": 3}

    if document_ids:
        search_kwargs["filter"] = {
            "doc_id": {"$in": document_ids}
        }

    retriever = store.as_retriever(search_kwargs=search_kwargs)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not explicitly stated, say so."
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}")
        ]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, doc_chain)