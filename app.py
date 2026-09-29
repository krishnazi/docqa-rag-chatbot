"""Streamlit UI for the Document Q&A chatbot (RAG)."""

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import (
    answer_question,
    build_vector_store,
    get_embeddings,
    get_llm,
    load_pdf,
    split_documents,
)

load_dotenv()
st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Q&A Chatbot")
st.caption("Upload PDFs and ask questions. Answers are grounded in your documents via RAG.")


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings():
    return get_embeddings()


with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, 4)
    if st.button("Process documents", type="primary", disabled=not files):
        with st.spinner("Reading, chunking and embedding..."):
            pages = [d for f in files for d in load_pdf(f, f.name)]
            chunks = split_documents(pages, chunk_size, min(chunk_overlap, chunk_size - 1))
            try:
                st.session_state.store = build_vector_store(chunks, cached_embeddings())
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")
            except ValueError as e:
                st.error(str(e))

llm = get_llm()
if llm is None:
    st.info("No GOOGLE_API_KEY found - running in retrieval-only mode (shows matching passages).")

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about your documents"):
    if "store" not in st.session_state:
        st.warning("Upload and process at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(question, st.session_state.store, llm, k=top_k)
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for d in result["sources"]:
                    st.markdown(f"**{d.metadata['source']} - page {d.metadata['page']}**")
                    st.text(d.page_content[:500])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
