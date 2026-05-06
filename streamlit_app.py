"""Streamlit RAG chat UI for the Vector Podcast knowledge base."""
import streamlit as st
from sentence_transformers import SentenceTransformer
from src.opensearch_client import get_client, knn_search, hybrid_search, INDEX_NAME
from src.rag import SYSTEM_PROMPT, build_context, DEFAULT_MODEL, OLLAMA_HOST
from ollama import Client
import os

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vector Podcast RAG",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Vector Podcast — Ask Anything")
st.caption(
    "RAG-powered Q&A over 33 episodes of the Vector Podcast. "
    "Backed by OpenSearch k-NN + Ollama."
)

# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_opensearch():
    client = get_client()
    count = client.count(index=INDEX_NAME)["count"]
    return client, count

@st.cache_resource
def load_ollama():
    return Client(host=OLLAMA_HOST)

model = load_model()
os_client, doc_count = load_opensearch()
ollama_client = load_ollama()

# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Search Settings")
    search_mode = st.radio("Search mode", ["hybrid", "semantic"], index=0)
    top_k = st.slider("Retrieve top-k chunks", 1, 15, 5)
    st.divider()

    llm_model = st.text_input("Ollama model", value=DEFAULT_MODEL)

    st.divider()
    st.metric("Indexed chunks", f"{doc_count:,}")
    st.caption("Embedding: `all-MiniLM-L6-v2` (384 dim)")
    st.caption(f"LLM: Ollama `{llm_model}` (local)")

# ── Chat history ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about vector search..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    query_vec = model.encode(prompt, normalize_embeddings=True).tolist()
    if search_mode == "hybrid":
        hits = hybrid_search(os_client, prompt, query_vec, k=top_k)
    else:
        hits = knn_search(os_client, query_vec, k=top_k)

    # Show sources in an expander
    with st.chat_message("assistant"):
        with st.expander(f"📚 Retrieved {len(hits)} sources", expanded=False):
            for i, hit in enumerate(hits, 1):
                st.markdown(
                    f"**[{i}]** {hit['episode_title']} "
                    f"(score: {hit['score']:.4f})"
                )
                st.caption(hit["chunk_text"][:300] + "...")

        # Generate with streaming via Ollama
        context = build_context(hits)
        user_message = (
            f"Context from Vector Podcast transcripts:\n\n{context}\n\n"
            f"---\nQuestion: {prompt}"
        )

        with st.spinner("Thinking..."):
            stream = ollama_client.chat(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )

            def token_generator():
                for chunk in stream:
                    yield chunk.message.content

            response_text = st.write_stream(token_generator())

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
