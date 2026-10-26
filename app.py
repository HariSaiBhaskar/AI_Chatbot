import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
import io
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Chat — RAG Q&A", page_icon="📄", layout="wide")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"

AVAILABLE_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def extract_pages(pdf_file) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], filename: str) -> list[dict]:
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "page": page_num, "file": filename})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(chunks: list[dict], embedder: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, embedder, bm25=None, k=TOP_K):
    # FAISS semantic
    q_emb = embedder.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, idx = index.search(q_emb, k * 2)
    semantic = set(int(i) for i in idx[0] if i < len(chunks))
    # BM25 lexical
    lexical = set()
    if bm25 is not None:
        tokens = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: k * 2]
        lexical = set(top)
    combined = list(semantic | lexical)
    # rerank by combined presence
    combined.sort(key=lambda i: (i in semantic) + (i in lexical), reverse=True)
    return [chunks[i] for i in combined[:k]]


def make_llm(api_key, model, temperature):
    return ChatGroq(api_key=api_key, model=model, temperature=temperature)


st.title("📄 PDF Chat — Next Level")
st.caption("Multi-PDF RAG with citations, streaming answers, and chat memory.")

with st.sidebar:
    st.header("Groq")
    groq_key = st.text_input("API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    md = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
    st.download_button("Export chat (.md)", md, file_name="chat.md", mime="text/markdown")
    if st.button("Clear index"):
        for k in ("chunks", "index", "pdf_names", "bm25"):
            st.session_state.pop(k, None)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Features:** multi-PDF · page citations · streaming · chat memory")

if not groq_key:
    st.warning("Enter your Groq API key in the sidebar.")
    st.stop()

if uploaded:
    if "pdf_names" not in st.session_state or st.session_state.get("pdf_names") != [f.name for f in uploaded]:
        with st.spinner("Indexing PDFs..."):
            embedder = load_embedder()
            all_chunks = []
            for f in uploaded:
                pages = extract_pages(io.BytesIO(f.read()))
                all_chunks.extend(chunk_pages(pages, f.name))
            if not all_chunks:
                st.error("No extractable text found in the uploaded PDF(s).")
                st.stop()
            index = build_index(all_chunks, embedder)
            tokenized = [re.findall(r"\w+", c["text"].lower()) for c in all_chunks]
            bm25 = BM25Okapi(tokenized)
            st.session_state.chunks = all_chunks
            st.session_state.index = index
            st.session_state.bm25 = bm25
            st.session_state.pdf_names = [f.name for f in uploaded]
            st.session_state.messages = []
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded)} file(s).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.caption(f"{s['file']} - Page {s['page']}: {s['text'][:200]}...")

if "chunks" not in st.session_state:
    st.info("Upload one or more PDFs in the sidebar to begin.")
    st.stop()

llm = make_llm(groq_key, model, temperature)

if question := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        embedder = load_embedder()
        relevant = retrieve(question, st.session_state.index, st.session_state.chunks, embedder, st.session_state.get("bm25"))
        context = "\n\n---\n\n".join(f"[{c['file']} - Page {c['page']}] {c['text']}" for c in relevant)
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Use the provided document context when relevant to answer. "
                "If the answer is not in the context, you may answer from your general knowledge. "
                "Be concise and cite page numbers when using document context.\n\n"
                f"CONTEXT:\n{context}"
            )),
            HumanMessage(content=question),
        ]

        def stream():
            for chunk in llm.stream(messages):
                yield chunk.content

        reply = st.write_stream(stream())
        sources = [{"file": c["file"], "page": c["page"], "text": c["text"]} for c in relevant]
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
        with st.expander("📎 Sources"):
            for s in sources:
                st.caption(f"Page {s['page']}: {s['text'][:200]}...")
