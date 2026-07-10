"""
RAGDAIRY — Vocalkisan Dairy AI Document Assistant
Toy RAG demo for viksitdairy2047.in
Powered by Groq API
"""

import os
import shutil
import tempfile
import uuid
import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Vocalkisan Dairy AI Assistant", page_icon="🐄", layout="wide")
st.title("🐄 Vocalkisan Dairy — Document Q&A (RAG Demo)")
st.caption(
    "Ask questions about dairy project documents (PDF / Excel). "
    "Answers are grounded only in the uploaded/preloaded documents — the assistant "
    "will say so if it can't find the answer."
)

# --------------------------------------------------------------------------------
# Session state: give every visitor an isolated, in-memory vector collection
# so concurrent users on the free hosted demo don't mix each other's uploads.
# --------------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

DATA_DIR = "data"  # preloaded sample corpus (NDSP EFA Annex, breed-improvement model, etc.)

# --------------------------------------------------------------------------------
# Sidebar — document upload & management
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("📄 Add documents")
    uploaded_files = st.file_uploader(
        "Upload dairy project PDF or Excel files",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    use_preloaded = st.checkbox("Also use preloaded sample documents", value=True)
    build_clicked = st.button("🔄 Build / Rebuild Index", type="primary")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------------
# Embeddings — free, CPU-friendly
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = get_embed_model()
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

# --------------------------------------------------------------------------------
# LLM binding — Groq Configuration
# --------------------------------------------------------------------------------
def configure_llm():
    from llama_index.llms.groq import Groq
    
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error(
            "No GROQ_API_KEY found. Add it under Streamlit Cloud → "
            "App settings → Secrets (see .streamlit/secrets.toml.example) "
            "or set it as an environment variable."
        )
        st.stop()
    
    # Using a fast, high-context Groq model
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

# --------------------------------------------------------------------------------
# Build / rebuild the index for this session
# --------------------------------------------------------------------------------
def build_index(files, include_preloaded: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy in preloaded sample docs
        if include_preloaded and os.path.isdir(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                src = os.path.join(DATA_DIR, f)
                if os.path.isfile(src):
                    shutil.copy(src, tmp_dir)
                    
        # Save this user's uploaded files into the same temp folder
        for uf in files or []:
            with open(os.path.join(tmp_dir, uf.name), "wb") as out:
                out.write(uf.getbuffer())
                
        if not os.listdir(tmp_dir):
            st.warning("No documents to index yet — upload a file or enable the preloaded sample set.")
            return None
            
        documents = SimpleDirectoryReader(input_dir=tmp_dir).load_data()
        
        # Ephemeral, per-session, in-memory vector store — no disk persistence needed
        chroma_client = chromadb.EphemeralClient()
        collection = chroma_client.get_or_create_collection(f"dairy_{st.session_state.session_id}")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_documents(documents, vector_store=vector_store)

if build_clicked:
    configure_llm()
    with st.spinner("Reading documents and building the index..."):
        st.session_state.index = build_index(uploaded_files, use_preloaded)
    if st.session_state.index is not None:
        st.success("Index ready — ask a question below.")

# --------------------------------------------------------------------------------
# Chat interface
# --------------------------------------------------------------------------------
GUARDRAIL_PROMPT = (
    "You are a dairy-sector planning assistant. Answer ONLY using the provided "
    "document context. If the answer is not clearly contained in the context, "
    "reply exactly: \"I don't have this information in the uploaded documents.\" "
    "Do not guess or use outside knowledge. Keep answers concise and, where "
    "relevant, mention which document section the information came from."
)

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

question = st.chat_input("Ask about the dairy project documents...")

if question:
    if st.session_state.index is None:
        st.warning("Click '🔄 Build / Rebuild Index' in the sidebar first.")
    else:
        configure_llm()
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant document sections..."):
                query_engine = st.session_state.index.as_query_engine(
                    similarity_top_k=4,
                    system_prompt=GUARDRAIL_PROMPT,
                )
                try:
                    response = query_engine.query(question)
                    answer_text = str(response)
                    st.write(answer_text)
                    
                    # Show which source chunks/files were used
                    sources = getattr(response, "source_nodes", [])
                    if sources:
                        with st.expander("📎 Sources used for this answer"):
                            for i, node in enumerate(sources, start=1):
                                fname = node.node.metadata.get("file_name", "unknown file")
                                score = getattr(node, "score", None)
                                score_txt = f" (relevance: {score:.2f})" if score is not None else ""
                                st.markdown(f"**{i}. {fname}**{score_txt}")
                                st.caption(node.node.get_text()[:300] + "...")
                except Exception as e:
                    answer_text = f"⚠️ Something went wrong: {e}"
                    st.error(answer_text)
                    
        st.session_state.chat_history.append(("assistant", answer_text))

# --------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------
st.divider()
st.caption(
    "Vocalkisan Dairy AI — proof-of-concept RAG demo. "
    "Documents are processed only for this browser session and are not permanently stored."
)
