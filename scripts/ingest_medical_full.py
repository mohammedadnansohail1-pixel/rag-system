"""
Medical RAG - FULL FEATURED Ingestion with Batching

Uses ALL features:
- QdrantHybridStore (dense + sparse)
- HybridRetriever (BM25 + dense)
- CrossEncoderReranker
- Structure-aware chunking
- Batched ingestion (prevents timeouts)
"""

import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")

from qdrant_client import QdrantClient

from src.core.config import Config
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.loaders.factory import LoaderFactory
from src.chunkers import StructureAwareChunker


COLLECTION_NAME = "medical_full"
BATCH_SIZE = 50  # Small batches to prevent timeout


def main():
    print("=" * 70)
    print("MEDICAL RAG - FULL FEATURED INGESTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Data path
    data_dir = Path("data/medical/subset")
    files = list(data_dir.glob("*.txt"))
    print(f"📁 Found {len(files)} files:")
    total_size = 0
    for f in files:
        size = f.stat().st_size / 1024
        total_size += size
        print(f"   • {f.name} ({size:.0f} KB)")
    print(f"   Total: {total_size/1024:.1f} MB")
    
    # Load config
    print("\n[1/5] Loading configuration...")
    config = Config.load("config/rag.yaml")
    
    retrieval_cfg = config.get_section("retrieval") or {}
    enhanced_cfg = config.get_section("enhanced") or {}
    chunking_cfg = enhanced_cfg.get("chunking", {})
    reranking_cfg = retrieval_cfg.get("reranking", {})
    
    print("\n  📋 Full Feature Set:")
    print(f"     • Hybrid search:       Dense + BM25")
    print(f"     • Reranking:           cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"     • Chunking:            structure_aware ({chunking_cfg.get('chunk_size', 1500)})")
    print(f"     • Batch size:          {BATCH_SIZE}")
    print(f"     • Retrieval top_k:     {retrieval_cfg.get('retrieval_top_k', 20)}")
    print(f"     • Rerank top_n:        {reranking_cfg.get('top_n', 5)}")
    
    # Initialize components
    print("\n[2/5] Initializing components...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text (768d)")
    
    # Delete existing collection
    print(f"\n[3/5] Setting up hybrid collection: {COLLECTION_NAME}")
    client = QdrantClient(host="localhost", port=6333)
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            print(f"  ⚠ Deleting existing: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"  Note: {e}")
    
    # Create hybrid vectorstore
    vectorstore = QdrantHybridStore(
        host="localhost",
        port=6333,
        collection_name=COLLECTION_NAME,
        dense_dimensions=768,
        distance_metric="cosine"
    )
    print(f"  ✓ Hybrid vectorstore: {COLLECTION_NAME}")
    
    # Create reranker
    print("\n[4/5] Setting up retrieval pipeline...")
    reranker = CrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("  ✓ Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Create hybrid retriever
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
        top_k=20,
        reranker=reranker
    )
    print("  ✓ Hybrid retriever: Dense + BM25 + Reranking")
    
    # Create chunker
    chunker = StructureAwareChunker(
        chunk_size=chunking_cfg.get("chunk_size", 1500),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 150)
    )
    print(f"  ✓ Chunker: structure_aware")
    
    # Ingest files
    print("\n[5/5] Ingesting with batching...")
    print("=" * 70)
    
    total_chunks = 0
    total_docs = 0
    start_time = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n📄 [{i}/{len(files)}] {file_path.name}")
        file_start = time.time()
        
        try:
            # Load document
            documents = LoaderFactory.load(str(file_path))
            total_docs += len(documents)
            
            # Chunk
            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)
            
            # Filter garbage
            valid_chunks = []
            for chunk in all_chunks:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if len(content) > 50 and not content.startswith("data:"):
                    valid_chunks.append(chunk)
            
            print(f"   • Chunks: {len(valid_chunks)} (filtered {len(all_chunks) - len(valid_chunks)})")
            
            # Prepare texts and metadata
            texts = []
            metadatas = []
            for chunk in valid_chunks:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
                metadata["source"] = file_path.name
                texts.append(content)
                metadatas.append(metadata)
            
            # Add in batches to prevent timeout
            num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(texts))
                
                batch_texts = texts[start_idx:end_idx]
                batch_metas = metadatas[start_idx:end_idx]
                
                retriever.add_documents(batch_texts, batch_metas)
                total_chunks += len(batch_texts)
                
                # Progress
                pct = (batch_idx + 1) / num_batches * 100
                print(f"   • Batch {batch_idx+1}/{num_batches} ({pct:.0f}%)", end="\r")
            
            file_time = time.time() - file_start
            print(f"   ✓ Indexed: {len(texts)} chunks in {file_time:.1f}s          ")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Verify
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"\n  ✓ Collection:       {COLLECTION_NAME}")
        print(f"  ✓ Vectors stored:   {info.points_count}")
        print(f"  ✓ Documents:        {total_docs}")
        print(f"  ✓ Total chunks:     {total_chunks}")
        print(f"  ✓ Time:             {total_time:.1f}s")
        print(f"  ✓ Speed:            {total_chunks/total_time:.1f} chunks/sec")
    except Exception as e:
        print(f"  ⚠ Verify error: {e}")
    
    print("\n  📋 Features Active:")
    print("     ✓ Hybrid vectorstore (dense + sparse)")
    print("     ✓ BM25 sparse encoding")
    print("     ✓ Structure-aware chunking")
    print("     ✓ Cross-encoder reranking")
    
    print("\n" + "=" * 70)
    print("✓ Full-featured ingestion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
