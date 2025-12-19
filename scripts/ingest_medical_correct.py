"""
Medical RAG - Using the ACTUAL system correctly

Uses:
- MultiDocumentPipeline (the proper pipeline)
- CachedEmbeddings (436x speedup)
- QdrantHybridStore (hybrid search)
- HybridRetriever (dense + BM25)
- CrossEncoderReranker
- Document registry
"""

import sys
from pathlib import Path
sys.path.insert(0, ".")

from qdrant_client import QdrantClient

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig


COLLECTION_NAME = "medical_correct"


def main():
    print("=" * 70)
    print("MEDICAL RAG - USING THE SYSTEM CORRECTLY")
    print("=" * 70)
    
    # Data
    data_dir = Path("data/medical/subset")
    files = list(data_dir.glob("*.txt"))
    print(f"\n📁 Files to ingest: {len(files)}")
    for f in files:
        print(f"   • {f.name}")
    
    # Delete existing collection
    print(f"\n[1/4] Setting up collection: {COLLECTION_NAME}")
    client = QdrantClient(host="localhost", port=6333)
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            print(f"  Deleting existing: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"  Note: {e}")
    
    # Initialize with proper components
    print("\n[2/4] Initializing components...")
    
    # Cached embeddings (436x speedup on repeated content)
    base_embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    embeddings = CachedEmbeddings(base_embeddings, enabled=True)
    print("  ✓ CachedEmbeddings: nomic-embed-text")
    
    # Hybrid vectorstore
    vectorstore = QdrantHybridStore(
        host="localhost",
        port=6333,
        collection_name=COLLECTION_NAME,
        dense_dimensions=768
    )
    print(f"  ✓ QdrantHybridStore: {COLLECTION_NAME}")
    
    # Reranker
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  ✓ CrossEncoderReranker: ms-marco-MiniLM-L-6-v2")
    
    # Hybrid retriever
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
        top_k=20,
        reranker=reranker
    )
    print("  ✓ HybridRetriever: Dense + BM25 + Reranking")
    
    # LLM
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    print("  ✓ OllamaLLM: llama3.1:8b")
    
    # Enhanced config
    config = EnhancedRAGConfig(
        chunking_strategy="structure_aware",
        chunk_size=1500,
        chunk_overlap=150,
        enable_enrichment=True,
        enable_parent_child=True,
        top_k=20
    )
    print("  ✓ EnhancedRAGConfig: structure_aware + enrichment")
    
    # MultiDocumentPipeline
    print("\n[3/4] Creating MultiDocumentPipeline...")
    pipeline = MultiDocumentPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        config=config,
        registry_path=".cache/medical_registry.json",
        clear_registry=True
    )
    print("  ✓ Pipeline ready")
    
    # Ingest
    print("\n[4/4] Ingesting documents...")
    print("=" * 70)
    
    stats = pipeline.ingest_directory(
        str(data_dir),
        recursive=False,
        file_types=[".txt"],
        skip_existing=False,
        progress_callback=lambda i, t, f: print(f"  [{i}/{t}] {Path(f).name}")
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"  ✓ Files processed:  {stats.get('files_processed', 0)}")
    print(f"  ✓ Files skipped:    {stats.get('files_skipped', 0)}")
    print(f"  ✓ Total chunks:     {stats.get('total_chunks', 0)}")
    print(f"  ✓ Total summaries:  {stats.get('total_summaries', 0)}")
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n  📦 Collection: {info.points_count} vectors")
    
    # Check registry
    print(f"\n  📋 Registry: {len(pipeline.registry.all_documents)} documents")
    for doc in pipeline.registry.all_documents:
        print(f"     • {doc.source_path.split('/')[-1]}: {doc.chunk_count} chunks")
    
    # Test query
    print("\n" + "=" * 70)
    print("🧪 TEST QUERY")
    print("=" * 70)
    
    response = pipeline.query("What are the symptoms of myocardial infarction?")
    print(f"\nQ: What are the symptoms of myocardial infarction?")
    print(f"\nA: {response.answer[:500]}...")
    print(f"\n📊 Confidence: {response.confidence}")
    print(f"📊 Sources: {len(response.sources)}")
    
    print("\n" + "=" * 70)
    print("✓ Complete! Run: python scripts/eval_medical_correct.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
