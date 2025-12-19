"""
Medical RAG - Enhanced Ingestion

Ingests medical textbooks with ALL advanced features:
- Structure-aware chunking
- Metadata enrichment
- Parent-child chunks
"""

import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")

from src.core.config import Config
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline, EnhancedRAGConfig


def main():
    print("=" * 70)
    print("MEDICAL RAG - ENHANCED INGESTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Data path
    data_dir = Path("data/medical/subset")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    files = list(data_dir.glob("*.txt"))
    print(f"📁 Found {len(files)} files to ingest:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"   • {f.name} ({size_kb:.0f} KB)")
    
    # Load config
    print("\n[1/4] Loading configuration...")
    config = Config.load("config/rag.yaml")
    enhanced_config = EnhancedRAGConfig.from_config(config)
    
    print("\n  📋 Enhanced Features:")
    print(f"     • Chunking strategy:   {enhanced_config.chunking_strategy}")
    print(f"     • Chunk size:          {enhanced_config.chunk_size}")
    print(f"     • Chunk overlap:       {enhanced_config.chunk_overlap}")
    print(f"     • Parent chunks:       {enhanced_config.generate_parent_chunks}")
    print(f"     • Parent chunk size:   {enhanced_config.parent_chunk_size}")
    print(f"     • Metadata enrichment: {enhanced_config.enable_enrichment}")
    print(f"     • Enrichment mode:     {enhanced_config.enrichment_mode}")
    print(f"     • Parent-child retrieval: {enhanced_config.enable_parent_child}")
    
    # Initialize components
    print("\n[2/4] Initializing components...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text (768d)")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.1:8b")
    
    # Create NEW collection for enhanced data
    collection_name = "medical_enhanced"
    print(f"\n[3/4] Creating collection: {collection_name}")
    
    vectorstore = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name=collection_name,
        dimensions=768
    )
    
    # Delete existing if present
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            print(f"  ⚠ Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"  Note: {e}")
    
    # Recreate vectorstore after delete
    vectorstore = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name=collection_name,
        dimensions=768
    )
    print(f"  ✓ Collection ready: {collection_name}")
    
    # Create retriever
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=enhanced_config.top_k
    )
    
    # Create enhanced pipeline
    pipeline = EnhancedRAGPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        config=enhanced_config
    )
    print("  ✓ Enhanced pipeline ready")
    
    # Ingest files
    print("\n[4/4] Ingesting with enhanced features...")
    print("=" * 70)
    
    total_stats = {
        "documents": 0,
        "chunks": 0,
        "parent_chunks": 0,
        "files": 0
    }
    
    start_time = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n📄 [{i}/{len(files)}] Processing: {file_path.name}")
        file_start = time.time()
        
        try:
            stats = pipeline.ingest_file(str(file_path))
            
            total_stats["documents"] += stats.get("documents", 0)
            total_stats["chunks"] += stats.get("chunks", 0)
            total_stats["parent_chunks"] += stats.get("parent_chunks", 0)
            total_stats["files"] += 1
            
            file_time = time.time() - file_start
            print(f"   ✓ Chunks: {stats.get('chunks', 0)} | Parents: {stats.get('parent_chunks', 0)} | Time: {file_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"\n  ✓ Files processed:    {total_stats['files']}/{len(files)}")
    print(f"  ✓ Total documents:    {total_stats['documents']}")
    print(f"  ✓ Total chunks:       {total_stats['chunks']}")
    print(f"  ✓ Parent chunks:      {total_stats['parent_chunks']}")
    print(f"  ✓ Total time:         {total_time:.1f}s")
    
    # Verify collection
    try:
        info = client.get_collection(collection_name)
        print(f"\n  📦 Collection '{collection_name}':")
        print(f"     • Vectors stored: {info.points_count}")
        print(f"     • Vector size:    {info.config.params.vectors.size}")
    except Exception as e:
        print(f"  ⚠ Could not verify: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Enhanced ingestion complete!")
    print(f"  Run evaluation: python scripts/eval_medical_best.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
