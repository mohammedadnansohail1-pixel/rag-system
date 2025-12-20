"""Streamlit UI for RAG System - Multi-Collection Support."""

import streamlit as st
import time
import json
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "collection_info" not in st.session_state:
    st.session_state.collection_info = {}

# Collection configs
COLLECTIONS = {
    "medical": {
        "name": "medical_final",
        "title": "🏥 Medical RAG",
        "description": "Medical textbooks (Harrison's, Robbins, Katzung, First Aid)",
        "registry": ".cache/medical_final_registry.json",
        "cache": ".cache/medical_embeddings",
        "sample_queries": [
            "What are the symptoms of diabetes mellitus?",
            "Explain the mechanism of beta blockers",
            "What causes iron deficiency anemia?",
        ]
    },
    "nhtsa": {
        "name": "nhtsa_complaints",
        "title": "🚗 NHTSA Vehicle Complaints",
        "description": "7,000 vehicle safety complaints (Tesla, Honda, Fisker)",
        "registry": ".cache/nhtsa_registry.json",
        "cache": ".cache/nhtsa_embeddings",
        "sample_queries": [
            "What are the most common steering issues?",
            "Show brake failure complaints with crashes",
            "Compare Honda vs Tesla safety complaints",
        ]
    },
    "custom": {
        "name": "custom_collection",
        "title": "📁 Custom Collection",
        "description": "Upload your own documents or JSON data",
        "registry": ".cache/custom_registry.json",
        "cache": ".cache/custom_embeddings",
        "sample_queries": []
    }
}


@st.cache_resource
def get_base_components():
    """Initialize base components (cached)."""
    from src.embeddings.ollama_embeddings import OllamaEmbeddings
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.generation.ollama_llm import OllamaLLM
    
    base_emb = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
    
    return base_emb, reranker, llm


def initialize_collection(mode: str):
    """Initialize pipeline for specific collection."""
    config = COLLECTIONS[mode]
    
    try:
        from src.embeddings.cached_embeddings import CachedEmbeddings
        from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.documents import MultiDocumentPipeline
        from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig
        from qdrant_client import QdrantClient
        
        base_emb, reranker, llm = get_base_components()
        
        embeddings = CachedEmbeddings(base_emb, enabled=True, cache_dir=config["cache"])
        vectorstore = QdrantHybridStore(
            host="localhost", port=6333,
            collection_name=config["name"],
            dense_dimensions=768
        )
        retriever = HybridRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            sparse_encoder="fastembed",
            top_k=20,
            reranker=reranker
        )
        cfg = EnhancedRAGConfig(
            chunking_strategy="structure_aware",
            chunk_size=1500,
            enable_enrichment=True,
            top_k=20
        )
        pipeline = MultiDocumentPipeline(
            embeddings=embeddings,
            vectorstore=vectorstore,
            retriever=retriever,
            llm=llm,
            config=cfg,
            registry_path=config["registry"]
        )
        
        # Get collection info
        client = QdrantClient(host="localhost", port=6333)
        try:
            info = client.get_collection(config["name"])
            vector_count = info.points_count
        except:
            vector_count = 0
        
        return pipeline, retriever, embeddings, vectorstore, vector_count
    
    except Exception as e:
        st.error(f"Failed to initialize {mode}: {e}")
        return None, None, None, None, 0


def format_sources(sources, mode: str):
    """Format sources based on collection mode."""
    formatted = []
    for src in sources:
        meta = src.metadata if hasattr(src, 'metadata') else {}
        content = src.content[:300] if hasattr(src, 'content') else str(src)[:300]
        score = src.score if hasattr(src, 'score') else 0
        
        if mode == "medical":
            source_file = meta.get('source', 'Unknown').split('/')[-1]
            formatted.append({
                "title": source_file,
                "subtitle": f"Section: {meta.get('section', 'N/A')}",
                "content": content,
                "score": score
            })
        elif mode == "nhtsa":
            vehicle = meta.get('vehicle', 'Unknown')
            component = meta.get('component', 'Unknown')
            crash = meta.get('crash', 'N/A')
            fire = meta.get('fire', 'N/A')
            year = meta.get('year', 'N/A')
            formatted.append({
                "title": f"{year} {vehicle}",
                "subtitle": f"Component: {component} | Crash: {crash} | Fire: {fire}",
                "content": content,
                "score": score
            })
        else:
            formatted.append({
                "title": meta.get('source', 'Document'),
                "subtitle": "",
                "content": content,
                "score": score
            })
    
    return formatted


def get_collection_stats(mode: str):
    """Get detailed stats for NHTSA collection."""
    if mode != "nhtsa":
        return None
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        results, _ = client.scroll(collection_name="nhtsa_complaints", limit=7000, with_payload=True)
        
        vehicles = {}
        components = {}
        crashes = 0
        
        for r in results:
            p = r.payload
            veh = p.get('vehicle', 'Unknown')
            vehicles[veh] = vehicles.get(veh, 0) + 1
            
            comp = p.get('component', 'Unknown')
            components[comp] = components.get(comp, 0) + 1
            
            if p.get('crash') == 'Yes':
                crashes += 1
        
        return {
            "vehicles": dict(sorted(vehicles.items(), key=lambda x: -x[1])[:5]),
            "components": dict(sorted(components.items(), key=lambda x: -x[1])[:5]),
            "crashes": crashes
        }
    except:
        return None


# ============== Header ==============
st.title("🔍 RAG Document Assistant")

if st.session_state.current_mode:
    st.caption(COLLECTIONS[st.session_state.current_mode]["description"])


# ============== Sidebar ==============
with st.sidebar:
    st.header("🎯 Select Mode")
    
    mode = st.radio(
        "Choose collection:",
        options=list(COLLECTIONS.keys()),
        format_func=lambda x: COLLECTIONS[x]["title"],
        index=0 if st.session_state.current_mode is None else list(COLLECTIONS.keys()).index(st.session_state.current_mode) if st.session_state.current_mode in COLLECTIONS else 0,
        key="mode_selector"
    )
    
    # Check if mode changed
    if mode != st.session_state.current_mode:
        st.session_state.current_mode = mode
        st.session_state.initialized = False
        st.session_state.pipeline = None
        st.session_state.chat_history = []
    
    st.info(COLLECTIONS[mode]['description'])
    
    st.divider()
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Load Collection", use_container_width=True, type="primary"):
            with st.spinner(f"Loading {COLLECTIONS[mode]['title']}..."):
                pipeline, retriever, embeddings, vectorstore, vector_count = initialize_collection(mode)
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.session_state.retriever = retriever
                    st.session_state.embeddings = embeddings
                    st.session_state.vectorstore = vectorstore
                    st.session_state.collection_info = {"vectors": vector_count}
                    st.session_state.initialized = True
                    st.success(f"✅ Loaded {vector_count:,} vectors!")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.success(f"✅ {COLLECTIONS[mode]['title']} Ready")
        st.metric("Vectors", f"{st.session_state.collection_info.get('vectors', 0):,}")
        
        # Show NHTSA stats
        if mode == "nhtsa":
            with st.expander("📊 Data Breakdown"):
                stats = get_collection_stats(mode)
                if stats:
                    st.write("**Top Vehicles:**")
                    for v, c in stats["vehicles"].items():
                        st.caption(f"• {v}: {c:,}")
                    st.write(f"**Crashes:** {stats['crashes']:,}")
    
    st.divider()
    
    # Custom mode - file upload
    if mode == "custom" and st.session_state.initialized:
        st.subheader("📤 Upload Data")
        
        upload_type = st.radio("Upload type:", ["Text/PDF", "JSON"])
        
        if upload_type == "Text/PDF":
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("📥 Process Files", use_container_width=True):
                progress = st.progress(0)
                total_chunks = 0
                
                for i, f in enumerate(uploaded_files):
                    save_dir = Path("data/uploads")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    file_path = save_dir / f.name
                    with open(file_path, "wb") as out:
                        out.write(f.getbuffer())
                    
                    try:
                        stats = st.session_state.pipeline.ingest_file(str(file_path))
                        total_chunks += stats.get('chunks', 0)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    
                    progress.progress((i + 1) / len(uploaded_files))
                
                st.success(f"✅ Added {total_chunks} chunks!")
                st.rerun()
        
        else:  # JSON upload
            uploaded_json = st.file_uploader("Upload JSON", type=["json"])
            
            if uploaded_json:
                try:
                    data = json.load(uploaded_json)
                    uploaded_json.seek(0)
                    
                    if isinstance(data, list) and len(data) > 0:
                        st.write(f"**Found {len(data)} records**")
                        st.write("**Fields:**", list(data[0].keys()))
                        
                        content_field = st.selectbox(
                            "Content field (main text):",
                            options=list(data[0].keys())
                        )
                        
                        meta_fields = st.multiselect(
                            "Metadata fields:",
                            options=[f for f in data[0].keys() if f != content_field],
                            default=[f for f in data[0].keys() if f != content_field][:5]
                        )
                        
                        if st.button("📥 Ingest JSON", use_container_width=True):
                            with st.spinner("Processing JSON..."):
                                from src.loaders.json_loader import JSONLoader, JSONFieldConfig
                                from src.retrieval.fastembed_sparse_encoder import FastEmbedSparseEncoder
                                
                                save_path = Path("data/uploads") / uploaded_json.name
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(save_path, "w") as f:
                                    json.dump(data, f)
                                
                                config = JSONFieldConfig(
                                    content_fields=[content_field],
                                    metadata_fields=meta_fields
                                )
                                loader = JSONLoader(config)
                                documents = loader.load(save_path)
                                
                                sparse_encoder = FastEmbedSparseEncoder()
                                batch_size = 100
                                
                                progress = st.progress(0)
                                for i in range(0, len(documents), batch_size):
                                    batch = documents[i:i+batch_size]
                                    texts = [d.content for d in batch]
                                    metadatas = [d.metadata for d in batch]
                                    
                                    dense_embs = st.session_state.embeddings.embed_batch(texts)
                                    sparse_vecs = sparse_encoder.encode_batch(texts)
                                    
                                    st.session_state.vectorstore.add_hybrid(
                                        texts=texts,
                                        dense_embeddings=dense_embs,
                                        sparse_vectors=sparse_vecs,
                                        metadatas=metadatas
                                    )
                                    progress.progress(min(i + batch_size, len(documents)) / len(documents))
                                
                                st.success(f"✅ Added {len(documents)} documents!")
                                st.rerun()
                
                except Exception as e:
                    st.error(f"Error reading JSON: {e}")
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Stats
    st.subheader("📊 Session Stats")
    st.metric("Questions Asked", len([m for m in st.session_state.chat_history if m["role"] == "user"]))


# ============== Main Content ==============

st.divider()

if not st.session_state.initialized:
    # Landing page
    st.subheader("👈 Select a mode and click **Load Collection** to begin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏥 Medical RAG")
        st.markdown("""
        - 37,000+ chunks from medical textbooks
        - Harrison's, Robbins, Katzung
        - Clinical questions & drug info
        """)
    
    with col2:
        st.markdown("### 🚗 NHTSA Complaints")
        st.markdown("""
        - 7,000 vehicle safety complaints
        - Tesla, Honda, Fisker
        - Crash reports & safety issues
        """)
    
    with col3:
        st.markdown("### 📁 Custom Collection")
        st.markdown("""
        - Upload your own documents
        - PDF, TXT, MD, JSON
        - Build custom knowledge base
        """)
    
    st.divider()
    
    st.success("""
    **Pipeline Features:**
    ✅ Hybrid Search (Dense + Sparse vectors) | 
    ✅ Cross-Encoder Reranking | 
    ✅ Confidence Scoring & Guardrails | 
    ✅ Structure-Aware Chunking | 
    ✅ FastEmbed Sparse Encoder
    """)

else:
    # Chat interface using st.chat_message
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                # Confidence badge
                confidence = message.get("confidence", "medium").lower()
                emoji_map = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                emoji = emoji_map.get(confidence, "🟡")
                
                st.markdown(f"**Confidence:** {emoji} {confidence.upper()}")
                st.write(message["content"])
                
                # Show sources
                if message.get("sources"):
                    with st.expander(f"📚 View {len(message['sources'])} Sources"):
                        for i, src in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {src['title']}**")
                            if src['subtitle']:
                                st.caption(src['subtitle'])
                            st.text(src['content'][:200] + "...")
                            st.divider()
    
    # Query input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("🔍 Searching..."):
            try:
                result = st.session_state.pipeline.query(query)
                
                formatted_sources = format_sources(
                    result.sources[:5] if hasattr(result, 'sources') else [],
                    st.session_state.current_mode
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result.answer,
                    "confidence": result.confidence if hasattr(result, 'confidence') else "medium",
                    "sources": formatted_sources
                })
            
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "confidence": "low",
                    "sources": []
                })
        
        st.rerun()
    
    # Sample queries
    sample_queries = COLLECTIONS[st.session_state.current_mode]["sample_queries"]
    if sample_queries:
        st.subheader("💡 Try these:")
        cols = st.columns(len(sample_queries))
        for i, (col, q) in enumerate(zip(cols, sample_queries)):
            with col:
                if st.button(q[:35] + "..." if len(q) > 35 else q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    
                    with st.spinner("🔍 Searching..."):
                        result = st.session_state.pipeline.query(q)
                        formatted_sources = format_sources(
                            result.sources[:5] if hasattr(result, 'sources') else [],
                            st.session_state.current_mode
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.answer,
                            "confidence": result.confidence if hasattr(result, 'confidence') else "medium",
                            "sources": formatted_sources
                        })
                    st.rerun()


# Footer
st.divider()
st.caption("RAG System | Hybrid Search + Cross-Encoder Reranking + Guardrails")
