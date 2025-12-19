"""Add Harrison's Internal Medicine to medical_full collection"""

import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

from qdrant_client import QdrantClient
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.loaders.factory import LoaderFactory
from src.chunkers import StructureAwareChunker

COLLECTION_NAME = "medical_full"
BATCH_SIZE = 50

def main():
    print("=" * 70)
    print("ADDING HARRISON'S INTERNAL MEDICINE")
    print("=" * 70)
    
    file_path = Path("data/medical/textbooks/InternalMed_Harrison.txt")
    print(f"\n📄 File: {file_path.name}")
    print(f"   Size: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Check current collection
    client = QdrantClient(host="localhost", port=6333)
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n📦 Current collection: {info.points_count} vectors")
    
    # Initialize components
    print("\n[1/3] Initializing...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    
    vectorstore = QdrantHybridStore(
        host="localhost",
        port=6333,
        collection_name=COLLECTION_NAME,
        dense_dimensions=768
    )
    
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
        top_k=20,
        reranker=reranker
    )
    
    chunker = StructureAwareChunker(chunk_size=1500, chunk_overlap=150)
    print("  ✓ Components ready")
    
    # Load and chunk
    print("\n[2/3] Loading and chunking...")
    documents = LoaderFactory.load(str(file_path))
    
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
    
    print(f"  ✓ Chunks: {len(valid_chunks)} (filtered {len(all_chunks) - len(valid_chunks)})")
    
    # Prepare data
    texts = []
    metadatas = []
    for chunk in valid_chunks:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
        metadata["source"] = file_path.name
        texts.append(content)
        metadatas.append(metadata)
    
    # Ingest in batches
    print(f"\n[3/3] Indexing {len(texts)} chunks...")
    start_time = time.time()
    
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(texts))
        
        batch_texts = texts[start_idx:end_idx]
        batch_metas = metadatas[start_idx:end_idx]
        
        retriever.add_documents(batch_texts, batch_metas)
        
        pct = (batch_idx + 1) / num_batches * 100
        print(f"  Batch {batch_idx+1}/{num_batches} ({pct:.0f}%)", end="\r")
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Indexed in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n📦 Updated collection: {info.points_count} vectors")
    print(f"   Added: {info.points_count - 14299} new vectors")
    
    print("\n" + "=" * 70)
    print("✓ Harrison's added! Run: python scripts/eval_medical_full.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
