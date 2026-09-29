from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.pipeline import answer_question, build_vector_store, load_pdf, split_documents

EMB = DeterministicFakeEmbedding(size=64)


def test_split_creates_overlapping_chunks():
    doc = Document(page_content="word " * 1000, metadata={"source": "a.pdf", "page": 1})
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)


def test_sample_pdf_loads_with_page_metadata():
    docs = load_pdf("data/sample.pdf", "sample.pdf")
    assert docs and docs[0].metadata == {"source": "sample.pdf", "page": 1}


def test_answer_with_llm_returns_answer_and_k_sources():
    chunks = split_documents(load_pdf("data/sample.pdf", "sample.pdf"), 300, 50)
    store = build_vector_store(chunks, EMB)
    result = answer_question("What is RAG?", store, FakeListChatModel(responses=["stub answer"]), k=3)
    assert result["answer"] == "stub answer"
    assert len(result["sources"]) == 3


def test_retrieval_only_mode_without_llm():
    chunks = split_documents(load_pdf("data/sample.pdf", "sample.pdf"), 300, 50)
    store = build_vector_store(chunks, EMB)
    result = answer_question("What is RAG?", store, llm=None, k=2)
    assert len(result["sources"]) == 2
