"""Enhanced RAG System UI with Multi-Document Support."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings import OllamaEmbeddings, CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval import HybridRetriever
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig

# Page config
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stMarkdown, .stText, p, span, label {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    .header-card {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1rem;
        text-align: center;
        margin: 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 1rem 0;
    }
    
    .user-message p {
        color: #ffffff !important;
        margin: 0;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 4px 16px 16px 16px;
        margin: 1rem 0;
    }
    
    .assistant-message p {
        color: #334155 !important;
        margin: 0;
        line-height: 1.7;
    }
    
    .badge-high {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .badge-medium {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .badge-low {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .company-tag {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca !important;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    
    .topic-tag {
        display: inline-block;
        background: #dcfce7;
        color: #166534 !important;
        padding: 0.2rem 0.5rem;
        border-radius: 8px;
        font-size: 0.7rem;
        margin-right: 0.25rem;
    }
    
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .score-badge {
        display: inline-block;
        background: #3b82f6;
        color: #ffffff !important;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .stat-card {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        color: #3b82f6 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #64748b !important;
        font-size: 0.75rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div {
        background: #ffffff !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_pipeline():
    """Initialize enhanced pipeline (cached)."""
    try:
        base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = CachedEmbeddings(base_embeddings, enabled=True)
        
        vectorstore = QdrantHybridStore(
            collection_name="enterprise_rag_v2",
            dense_dimensions=768,
        )
        
        base_retriever = HybridRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            sparse_encoder="fastembed",
        )
        
        llm = OllamaLLM(model="llama3.2")
        
        config = EnhancedRAGConfig.from_dict({
            "chunking": {"strategy": "structure_aware", "chunk_size": 1500},
            "enrichment": {"enabled": True, "mode": "fast"},
            "summarization": {"enabled": False},
            "parent_child": {"enabled": False},
        })
        
        pipeline = MultiDocumentPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=base_retriever,
            llm=llm,
            config=config,
            registry_path=".cache/ui_doc_registry.json",
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
        <p class="header-subtitle">Multi-document AI with cross-company analysis</p>
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
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("📥 Process Files", use_container_width=True):
                with st.spinner("Processing..."):
                    for uploaded_file in uploaded_files:
                        temp_path = Path(f"/tmp/{uploaded_file.name}")
                        temp_path.write_bytes(uploaded_file.read())
                        try:
                            stats = pipeline.ingest_file(str(temp_path))
                            if stats.get("skipped"):
                                st.info(f"⏭️ {uploaded_file.name} (already indexed)")
                            else:
                                st.success(f"✅ {uploaded_file.name} ({stats.get('chunks', 0)} chunks)")
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: {e}")
                        finally:
                            temp_path.unlink(missing_ok=True)
        
        # Directory ingestion
        st.markdown("---")
        dir_path = st.text_input("📂 Directory Path", placeholder="data/sec-filings/")
        if dir_path and st.button("📥 Ingest Directory", use_container_width=True):
            with st.spinner("Ingesting directory..."):
                try:
                    stats = pipeline.ingest_directory(dir_path, skip_existing=True)
                    st.success(f"✅ {stats['files_processed']} files, {stats['total_chunks']} chunks")
                    if stats['companies']:
                        st.info(f"Companies: {', '.join(stats['companies'])}")
                except Exception as e:
                    st.error(str(e))
        
        st.markdown("---")
        
        # Registry stats
        st.markdown("## 📊 Registry")
        reg_stats = pipeline.registry_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", reg_stats.get("total_documents", 0))
        with col2:
            st.metric("Chunks", reg_stats.get("total_chunks", 0))
        
        # Companies
        companies = pipeline.companies
        if companies:
            st.markdown("### 🏢 Companies")
            for company in companies:
                st.markdown(f"• {company}")
        
        st.markdown("---")
        
        # Filters
        st.markdown("## 🔍 Query Filters")
        
        filter_company = st.multiselect(
            "Filter by Company",
            options=companies if companies else [],
            default=[],
        )
        
        filter_type = st.multiselect(
            "Filter by Filing Type",
            options=reg_stats.get("filing_types", []),
            default=[],
        )
        
        st.markdown("---")
        
        # Status
        st.markdown("## ⚡ Status")
        health = pipeline.health_check()
        for component, status in health.items():
            icon = "🟢" if status else "🔴"
            st.markdown(f"{icon} {component.title()}")
        
        st.markdown("---")
        
        # Load SEC samples
        if st.button("📚 Load SEC Samples", use_container_width=True):
            with st.spinner("Loading SEC filings..."):
                try:
                    stats = pipeline.ingest_directory(
                        "data/test_adaptive/sec-edgar-filings/",
                        recursive=True,
                        skip_existing=True,
                    )
                    st.success(f"✅ {stats['files_processed']} filings loaded")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    # Main area - tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Compare Companies", "📋 Registry"])
    
    # TAB 1: Chat
    with tab1:
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
                companies_cited = msg.get('companies_cited', [])
                topics = msg.get('topics', [])
                
                # Badges
                badges_html = f'<span class="badge-{confidence}">{confidence.upper()}</span>'
                for company in companies_cited[:3]:
                    short_name = company.split(',')[0].split(' ')[0]
                    badges_html += f'<span class="company-tag">{short_name}</span>'
                
                st.markdown(f"""
                <div>{badges_html}</div>
                <div class="assistant-message">
                    <p>{msg['content']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Topics
                if topics:
                    topics_html = ''.join([f'<span class="topic-tag">{t}</span>' for t in topics[:5]])
                    st.markdown(f"<div>{topics_html}</div>", unsafe_allow_html=True)
                
                # Sources
                if msg.get('sources'):
                    with st.expander(f"📎 {len(msg['sources'])} Sources"):
                        for src in msg['sources']:
                            company = src.get('metadata', {}).get('company_name', 'Unknown')
                            score = src.get('score', 0)
                            content = src.get('content', '')[:200]
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="score-badge">{score:.0%}</span>
                                <span class="company-tag">{company.split(',')[0]}</span>
                                <p>{content}...</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Input
        col1, col2 = st.columns([6, 1])
        with col1:
            query = st.text_input(
                "Question",
                placeholder="Ask about your documents...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col2:
            ask = st.button("Ask", use_container_width=True, key="ask_btn")
        
        if ask and query:
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.spinner("🔍 Searching..."):
                response = pipeline.query(
                    query,
                    top_k=10,
                    filter_companies=filter_company if filter_company else None,
                    filter_filing_types=filter_type if filter_type else None,
                )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "confidence": response.confidence,
                    "sources": response.sources,
                    "companies_cited": response.companies_cited,
                    "topics": response.topics_found,
                })
            st.rerun()
        
        # Example questions
        if not st.session_state.messages:
            st.markdown("### 🎯 Try These")
            examples = [
                "What are the main risk factors?",
                "What was the company's revenue?",
                "What is the cybersecurity approach?",
            ]
            cols = st.columns(3)
            for col, q in zip(cols, examples):
                with col:
                    if st.button(q, use_container_width=True, key=f"ex_{q}"):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()
        
        # Clear
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    # TAB 2: Compare Companies
    with tab2:
        st.markdown("### 📊 Cross-Company Comparison")
        
        if len(companies) < 2:
            st.info("Ingest documents from multiple companies to enable comparison.")
        else:
            selected_companies = st.multiselect(
                "Select Companies to Compare",
                options=companies,
                default=companies[:3] if len(companies) >= 3 else companies,
            )
            
            comparison_query = st.text_input(
                "Comparison Question",
                placeholder="Compare revenue growth across companies...",
                key="compare_input"
            )
            
            if st.button("🔍 Compare", use_container_width=True) and comparison_query and selected_companies:
                with st.spinner("Analyzing companies..."):
                    response = pipeline.compare_companies(
                        comparison_query,
                        companies=selected_companies,
                        top_k_per_company=3,
                    )
                    
                    st.markdown(f"""
                    <div class="assistant-message">
                        <p>{response.answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"📎 {len(response.sources)} Sources"):
                        for src in response.sources:
                            company = src.get('metadata', {}).get('company_name', 'Unknown')
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="company-tag">{company.split(',')[0]}</span>
                                <p>{src.get('content', '')[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    # TAB 3: Registry
    with tab3:
        st.markdown("### 📋 Document Registry")
        
        docs = pipeline.registry.all_documents
        
        if not docs:
            st.info("No documents ingested yet.")
        else:
            for doc in docs:
                with st.expander(f"📄 {doc.company_name or 'Unknown'} - {doc.filing_type or 'Document'}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks", doc.chunk_count)
                    with col2:
                        st.metric("Characters", f"{doc.total_chars:,}")
                    with col3:
                        st.metric("Sections", len(doc.sections))
                    
                    st.markdown(f"**Filing Date:** {doc.filing_date or 'N/A'}")
                    st.markdown(f"**Source:** `{doc.source_path}`")
                    
                    if doc.sections:
                        st.markdown(f"**Sections:** {', '.join(doc.sections[:5])}")


if __name__ == "__main__":
    main()
