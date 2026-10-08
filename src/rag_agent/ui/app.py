"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Layout:
  - Left sidebar: Document ingestion and corpus stats
  - Left column: Document viewer
  - Right column: RAG question-and-answer chat

How a question is answered (Retrieval-Augmented Generation):
  1. The question is embedded and the closest chunks are retrieved
     from ChromaDB (VectorStoreManager.query).
  2. If nothing is similar enough, we say so instead of guessing
     (the hallucination guard).
  3. Otherwise the retrieved chunks are pasted into the prompt as
     context and the Groq LLM writes an answer grounded in them.
  4. The answer is shown with its source citations.

Run with: uv run streamlit run src/rag_agent/ui/app.py

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# st.cache_resource keeps these objects alive across Streamlit reruns, so
# ChromaDB, the embedding model and the LLM are only created once.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager (ChromaDB + embeddings)."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """Return the cached chat model (Groq, as configured in .env)."""
    return LLMFactory().create()


@st.cache_resource
def get_graph():
    """
    Return the compiled LangGraph agent, or None if it cannot be built.

    The chat below calls the retriever and LLM directly, so the app
    still works even if the optional graph is not available.
    """
    try:
        from rag_agent.agent.graph import get_compiled_graph

        return get_compiled_graph()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Streamlit reruns the whole script on every click, so anything that
    must survive between clicks (like chat history) lives in session_state.
    """
    defaults = {
        "chat_history": [],  # list of {"role", "content", "sources", ...}
        "selected_document": None,
        "last_ingestion_result": None,
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Lets the user upload one or more Markdown files, chunks them,
    stores them in ChromaDB and reports how many chunks were added
    or skipped as duplicates.
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown notes (.md)",
        type=["md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
        type="primary",
    )

    if ingest_clicked and uploaded_files:
        all_chunks = []
        errors = []

        with st.spinner("Chunking and embedding documents..."):
            # Save uploads to a temporary folder, keeping their original
            # names so the filename (e.g. lstm_intermediate.md) can be
            # used for topic/difficulty and for citations.
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded in uploaded_files:
                    file_path = Path(tmp_dir) / uploaded.name
                    file_path.write_bytes(uploaded.getvalue())
                    try:
                        all_chunks.extend(chunker.chunk_file(file_path))
                    except Exception as exc:
                        errors.append(f"{uploaded.name}: {exc}")

            result = store.ingest(all_chunks)

        result.errors.extend(errors)
        st.session_state.last_ingestion_result = result

    # Show the result of the most recent ingestion
    result = st.session_state.last_ingestion_result
    if result is not None:
        message = (
            f"{result.ingested} chunks added, "
            f"{result.skipped} duplicates skipped"
        )
        if result.errors:
            st.sidebar.error(message + "\n\n" + "\n".join(result.errors))
        elif result.ingested > 0:
            st.sidebar.success(message)
        else:
            st.sidebar.warning(message)

    # List what is already stored
    documents = store.list_documents()
    if documents:
        st.sidebar.subheader("Ingested documents")
        for doc in documents:
            col_name, col_btn = st.sidebar.columns([4, 1])
            col_name.caption(
                f"**{doc['source']}** — {doc['topic']} ({doc['chunk_count']} chunks)"
            )
            if col_btn.button("🗑", key=f"delete_{doc['source']}", help="Remove"):
                store.delete_document(doc["source"])
                st.session_state.last_ingestion_result = None
                st.rerun()
    else:
        st.sidebar.info("Upload .md files and click Ingest Documents.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """Render a compact summary of what is in the vector store."""
    stats = store.get_collection_stats()
    st.sidebar.divider()
    st.sidebar.metric("ChromaDB chunk count", stats["total_chunks"])
    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))


# ---------------------------------------------------------------------------
# Document Viewer Panel
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """Show the chunks of one ingested document."""
    st.subheader("📄 Document Viewer")

    documents = store.list_documents()
    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in documents]
    selected = st.selectbox("Select document", options=sources)
    st.session_state.selected_document = selected

    chunks = store.get_document_chunks(selected)
    with st.container(height=450):
        for chunk in chunks:
            meta = chunk.metadata
            st.caption(f"{meta.topic} | {meta.difficulty} | {meta.type}")
            st.markdown(chunk.chunk_text)
            st.divider()
    st.caption(f"{len(chunks)} chunks in {selected}")


# ---------------------------------------------------------------------------
# RAG question answering
# ---------------------------------------------------------------------------


def answer_question(
    store: VectorStoreManager,
    query: str,
    topic_filter: str | None,
    difficulty_filter: str | None,
) -> dict:
    """
    Retrieve relevant chunks and ask the LLM to answer from them.

    Returns a chat message dict with the answer, the source citations
    and a no_context_found flag.
    """
    # 1. Retrieve the most relevant chunks
    chunks = store.query(
        query,
        topic_filter=topic_filter,
        difficulty_filter=difficulty_filter,
    )

    # 2. Hallucination guard: no relevant context → don't guess
    if not chunks:
        return {
            "role": "assistant",
            "content": (
                "I couldn't find anything relevant to that question in the "
                "ingested notes, so I won't guess. Try rephrasing, or ingest "
                "a note that covers this topic."
            ),
            "sources": [],
            "no_context_found": True,
        }

    # 3. Build the context block from the retrieved chunks
    context = "\n\n---\n\n".join(
        f"[SOURCE: {c.metadata.topic} | {c.metadata.source}] "
        f"(similarity {c.score:.2f})\n{c.chunk_text}"
        for c in chunks
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                "Answer the question using only the context above, "
                "and cite the sources you used."
            )
        ),
    ]

    # 4. Ask the LLM
    response = get_llm().invoke(messages)
    answer = response.content
    if isinstance(answer, list):  # some models return a list of parts
        answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in answer
        )

    # 5. Return the answer together with its citations
    sources = [f"{c.to_citation()} — similarity {c.score:.2f}" for c in chunks]
    return {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "no_context_found": False,
    }


# ---------------------------------------------------------------------------
# Chat Interface Panel
# ---------------------------------------------------------------------------


def render_chat_interface(store: VectorStoreManager) -> None:
    """
    Render the RAG question-and-answer chat.

    Every answer is shown with an expandable list of the source
    chunks it was built from.
    """
    st.subheader("💬 RAG Question & Answer")

    # Optional filters
    stats = store.get_collection_stats()
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic = st.selectbox("Topic filter", ["All"] + stats["topics"])
        st.session_state.topic_filter = None if topic == "All" else topic
    with col_diff:
        difficulty = st.selectbox(
            "Difficulty filter", ["All", "beginner", "intermediate", "advanced"]
        )
        st.session_state.difficulty_filter = (
            None if difficulty == "All" else difficulty
        )

    # Chat history
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources", expanded=True):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask a question about your notes...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    try:
                        reply = answer_question(
                            store,
                            query,
                            st.session_state.topic_filter,
                            st.session_state.difficulty_filter,
                        )
                    except Exception as exc:
                        reply = {
                            "role": "assistant",
                            "content": f"❌ Error while answering: {exc}",
                            "sources": [],
                            "no_context_found": False,
                        }
        st.session_state.chat_history.append(reply)
        st.rerun()

    if st.session_state.chat_history and st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered Q&A — built with LangChain, ChromaDB and Groq "
        f"(model: {settings.groq_model})"
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    get_graph()  # optional: builds the LangGraph agent if available

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")
    with viewer_col:
        render_document_viewer(store)
    with chat_col:
        render_chat_interface(store)


if __name__ == "__main__":
    main()
