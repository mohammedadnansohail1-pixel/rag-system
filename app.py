"""Streamlit UI for RAG System - Business Demo."""

import streamlit as st
import time
from pathlib import Path
import os

# Must be first Streamlit command
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = 0


def initialize_pipeline():
    """Initialize RAG pipeline."""
    with st.spinner("🚀 Starting AI engine..."):
        try:
            from src.embeddings.ollama_embeddings import OllamaEmbeddings
            from src.vectorstores.qdrant_store import QdrantVectorStore
            from src.retrieval.dense_retriever import DenseRetriever
            from src.generation.ollama_llm import OllamaLLM
            from src.pipeline.rag_pipeline_production import ProductionRAGPipeline
            from src.guardrails.config import GuardrailsConfig
            from qdrant_client import QdrantClient
            
            # Clean up existing collection
            client = QdrantClient(host="localhost", port=6333)
            try:
                client.delete_collection("streamlit_demo")
            except:
                pass
            
            embeddings = OllamaEmbeddings(
                host="http://localhost:11434",
                model="nomic-embed-text",
                dimensions=768
            )
            
            vectorstore = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="streamlit_demo",
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
            
            return pipeline
            
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return None


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory."""
    save_dir = Path("data/uploads")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


# ============== UI ==============

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .confidence-high {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .confidence-medium {
        background-color: #F59E0B;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .confidence-low {
        background-color: #EF4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .source-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6366F1;
    }
    .chat-user {
        background-color: #EEF2FF;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .chat-assistant {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🔍 AI Document Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents and get instant, accurate answers with source citations</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Start AI Engine", use_container_width=True):
            pipeline = initialize_pipeline()
            if pipeline:
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.success("✅ AI Engine Ready!")
                st.rerun()
    else:
        st.success("✅ AI Engine Running")
    
    st.markdown("---")
    
    # File upload
    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=not st.session_state.initialized
    )
    
    if uploaded_files and st.session_state.initialized:
        if st.button("📥 Process Documents", use_container_width=True):
            progress_bar = st.progress(0)
            total_chunks = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.text(f"Processing {uploaded_file.name}...")
                file_path = save_uploaded_file(uploaded_file)
                
                try:
                    chunks = st.session_state.pipeline.ingest_file(file_path)
                    total_chunks += chunks
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.documents_loaded += total_chunks
            st.success(f"✅ Indexed {total_chunks} chunks!")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # Stats
    st.markdown("## 📊 Stats")
    st.metric("Documents Indexed", f"{st.session_state.documents_loaded} chunks")
    st.metric("Questions Asked", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
    
    st.markdown("---")
    
    # Sample documents
    st.markdown("## 📚 Sample Data")
    if st.button("Load Sample Documents", use_container_width=True, disabled=not st.session_state.initialized):
        with st.spinner("Loading samples..."):
            try:
                # Check if sample data exists
                if Path("data/sample").exists():
                    chunks = st.session_state.pipeline.ingest_directory(
                        "data/sample", 
                        file_types=[".txt"]
                    )
                    st.session_state.documents_loaded += chunks
                    st.success(f"✅ Loaded {chunks} sample chunks!")
                    st.rerun()
                else:
                    st.warning("No sample data found in data/sample/")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Main content area
if not st.session_state.initialized:
    # Landing page
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📄 Upload Any Document
        - PDF reports & manuals
        - Text files & notes
        - Markdown documentation
        """)
    
    with col2:
        st.markdown("""
        ### 💬 Ask Questions
        - Natural language queries
        - Get instant answers
        - Source citations included
        """)
    
    with col3:
        st.markdown("""
        ### ✅ Trust the Results
        - Confidence indicators
        - Anti-hallucination guardrails
        - Relevance scoring
        """)
    
    st.markdown("---")
    
    st.info("👈 Click **Start AI Engine** in the sidebar to begin")
    
    # Demo video or screenshot placeholder
    st.markdown("### 🎬 How It Works")
    st.markdown("""
    1. **Start the AI Engine** - Initialize the system
    2. **Upload Documents** - PDFs, text files, or markdown
    3. **Ask Questions** - Get accurate answers with sources
    4. **Review Confidence** - Green = high confidence, Yellow = medium, Red = low
    """)

else:
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <strong>🧑 You:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message with confidence
            confidence = message.get("confidence", "medium")
            confidence_class = f"confidence-{confidence}"
            emoji = message.get("emoji", "🟡")
            
            st.markdown(f"""
            <div class="chat-assistant">
                <strong>🤖 Assistant:</strong> 
                <span class="{confidence_class}">{emoji} {confidence.upper()}</span>
                <br><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources in expander
            if message.get("sources"):
                with st.expander(f"📚 View {len(message['sources'])} Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}</strong> (Relevance: {src['score']:.0%})<br>
                            <small>{src['content'][:300]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Query input
    st.markdown("---")
    
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic of the document?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")
    
    if ask_button and query:
        if st.session_state.documents_loaded == 0:
            st.warning("⚠️ Please upload and process some documents first!")
        else:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })
            
            # Get response
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    response = st.session_state.pipeline.query(query, top_k=10)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "confidence": response.confidence,
                        "emoji": response.confidence_emoji,
                        "sources": response.sources,
                        "avg_score": response.avg_score
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}",
                        "confidence": "low",
                        "emoji": "🔴",
                        "sources": []
                    })
            
            st.rerun()
    
    # Quick questions
    if st.session_state.documents_loaded > 0:
        st.markdown("### 💡 Try these questions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("What is this document about?"):
                st.session_state.chat_history.append({"role": "user", "content": "What is this document about?"})
                response = st.session_state.pipeline.query("What is this document about?", top_k=10)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": response.answer,
                    "confidence": response.confidence, "emoji": response.confidence_emoji,
                    "sources": response.sources
                })
                st.rerun()
        
        with col2:
            if st.button("What are the key points?"):
                st.session_state.chat_history.append({"role": "user", "content": "What are the key points?"})
                response = st.session_state.pipeline.query("What are the key points?", top_k=10)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": response.answer,
                    "confidence": response.confidence, "emoji": response.confidence_emoji,
                    "sources": response.sources
                })
                st.rerun()
        
        with col3:
            if st.button("Summarize the main findings"):
                st.session_state.chat_history.append({"role": "user", "content": "Summarize the main findings"})
                response = st.session_state.pipeline.query("Summarize the main findings", top_k=10)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": response.answer,
                    "confidence": response.confidence, "emoji": response.confidence_emoji,
                    "sources": response.sources
                })
                st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    Built with ❤️ | Production-Ready RAG System with Guardrails
</div>
""", unsafe_allow_html=True)
