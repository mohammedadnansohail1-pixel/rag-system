"""Professional RAG System UI with Streamlit."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline_production import ProductionRAGPipeline
from src.guardrails.config import GuardrailsConfig

# Page config
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with proper contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Override Streamlit dark text issues */
    .stMarkdown, .stText, p, span, label {
        color: #1e293b !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMetricValue,
    [data-testid="stSidebar"] .stMetricLabel {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem;
        text-align: center;
        margin: 0;
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .user-message p {
        color: #ffffff !important;
        margin: 0;
        font-weight: 500;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 4px 16px 16px 16px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .assistant-message p {
        color: #334155 !important;
        margin: 0;
        line-height: 1.7;
    }
    
    /* Confidence Badges */
    .badge-high {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .badge-medium {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    .badge-low {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    
    /* Score Badge */
    .score-badge {
        display: inline-block;
        background: #3b82f6;
        color: #ffffff !important;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Source Card */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .source-card p {
        color: #475569 !important;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0.5rem 0 0 0;
    }
    
    /* Stats Card */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        flex: 1;
    }
    
    .stat-value {
        color: #3b82f6 !important;
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #64748b !important;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        color: #1e293b !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e293b !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Example Question Buttons */
    .example-btn {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        color: #475569 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    .example-btn:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
        background: #f0f9ff !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(59, 130, 246, 0.05);
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 8px !important;
        color: #475569 !important;
    }
    
    /* Metrics Override */
    [data-testid="stMetricValue"] {
        color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_pipeline():
    """Initialize pipeline (cached)."""
    try:
        embeddings = OllamaEmbeddings(
            host="http://localhost:11434",
            model="nomic-embed-text",
            dimensions=768
        )
        
        vectorstore = QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="enterprise_rag",
            dimensions=768
        )
        
        retriever = DenseRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            top_k=10
        )
        
        llm = OllamaLLM(
            host="http://localhost:11434",
            model="llama3.2:latest",
            temperature=0.1
        )
        
        guardrails_config = GuardrailsConfig(
            score_threshold=0.4,
            min_sources=2,
            min_avg_score=0.5,
        )
        
        pipeline = ProductionRAGPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            chunker_config={
                "strategy": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            guardrails_config=guardrails_config,
        )
        
        return pipeline, None
    except Exception as e:
        return None, str(e)


def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">🔍 Enterprise RAG System</h1>
        <p class="header-subtitle">AI-powered document intelligence with production-grade guardrails</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
    pipeline, error = init_pipeline()
    
    if error:
        st.error(f"⚠️ Connection Error: {error}")
        st.info("Please ensure Ollama and Qdrant are running.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("📥 Process Files", use_container_width=True):
                with st.spinner("Processing..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        temp_path = Path(f"/tmp/{uploaded_file.name}")
                        temp_path.write_bytes(uploaded_file.read())
                        try:
                            chunks = pipeline.ingest_file(str(temp_path))
                            total_chunks += chunks
                            st.success(f"✅ {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}")
                        finally:
                            temp_path.unlink(missing_ok=True)
                    
                    if total_chunks > 0:
                        st.session_state.chunks_indexed = st.session_state.get('chunks_indexed', 0) + total_chunks
        
        st.markdown("---")
        
        # Stats
        st.markdown("## 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.get('chunks_indexed', 0))
        with col2:
            st.metric("Queries", st.session_state.get('query_count', 0))
        
        st.markdown("---")
        
        # Status
        st.markdown("## ⚡ Status")
        health = pipeline.health_check()
        for component, status in health.items():
            icon = "🟢" if status else "🔴"
            st.markdown(f"{icon} {component.title()}")
        
        st.markdown("---")
        
        # Guardrails info
        st.markdown("## 🛡️ Guardrails")
        st.markdown("Score: `≥0.4` | Sources: `≥2` | Avg: `≥0.5`")
        
        st.markdown("---")
        
        # Sample data
        if st.button("📚 Load Samples", use_container_width=True):
            with st.spinner("Loading..."):
                try:
                    chunks = pipeline.ingest_directory("data/sample", file_types=[".txt", ".md"])
                    st.session_state.chunks_indexed = st.session_state.get('chunks_indexed', 0) + chunks
                    st.success(f"✅ {chunks} chunks loaded!")
                except Exception as e:
                    st.error(str(e))
    
    # Main chat area
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <p>💬 {msg['content']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = msg.get('confidence', 'medium')
            st.markdown(f"""
            <div class="badge-{confidence}">{confidence.upper()} CONFIDENCE</div>
            <div class="assistant-message">
                <p>{msg['content']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources expander
            if msg.get('sources'):
                with st.expander(f"📎 {len(msg['sources'])} Sources"):
                    for src in msg['sources']:
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="score-badge">{src['score']:.0%} match</span>
                            <p>{src['content'][:250]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Input area
    st.markdown('<p class="section-header">💡 Ask a Question</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input(
            "Question",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
    with col2:
        ask = st.button("Ask", use_container_width=True)
    
    if ask and query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.spinner("🔍 Searching..."):
            response = pipeline.query(query, top_k=10)
            st.session_state.query_count = st.session_state.get('query_count', 0) + 1
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "confidence": response.confidence,
                "sources": response.sources,
            })
        st.rerun()
    
    # Example questions
    if not st.session_state.messages:
        st.markdown('<p class="section-header">🎯 Try These</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        examples = [
            "What is this document about?",
            "What are the key points?",
            "Summarize the main findings"
        ]
        
        for col, q in zip([col1, col2, col3], examples):
            with col:
                if st.button(q, use_container_width=True, key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Clear button
    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
