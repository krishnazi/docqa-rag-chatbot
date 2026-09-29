"""Core RAG pipeline: load PDFs -> chunk -> embed -> store in ChromaDB -> retrieve -> generate."""

from __future__ import annotations

import os
import uuid
from typing import BinaryIO, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about the user's documents. "
            "Use ONLY the context below. If the answer is not in the context, say you could "
            "not find it in the documents. Be concise and mention page numbers when useful.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def load_pdf(file: str | BinaryIO, source_name: str) -> list[Document]:
    """Read a PDF (path or file-like object) into one Document per page."""
    reader = PdfReader(file)
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": source_name, "page": page_number}))
    return docs


def split_documents(docs: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Split pages into overlapping chunks so retrieval returns focused passages."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(list(docs))


def get_embeddings():
    """Sentence-Transformers embeddings (runs locally, downloads ~90 MB on first use)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document], embeddings, persist_directory: str | None = None) -> Chroma:
    """Embed chunks and store them in ChromaDB (in-memory unless persist_directory is given)."""
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded documents.")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"docs_{uuid.uuid4().hex[:8]}",
        persist_directory=persist_directory,
    )


def get_llm():
    """Return a Gemini chat model if GOOGLE_API_KEY is set, otherwise None (retrieval-only mode)."""
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')} | page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )


def answer_question(question: str, store: Chroma, llm=None, k: int = 4) -> dict:
    """Retrieve the top-k chunks and generate a grounded answer.

    Returns {"answer": str, "sources": list[Document]}. With no LLM, returns the retrieved
    passages themselves so the app still works without an API key.
    """
    sources = store.similarity_search(question, k=k)
    if llm is None:
        return {"answer": "_No LLM configured - showing the most relevant passages instead._", "sources": sources}
    messages = PROMPT.format_messages(context=format_context(sources), question=question)
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": sources}
