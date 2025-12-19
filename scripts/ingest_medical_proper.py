"""
Medical RAG - PROPER Full-Featured Ingestion

Actually uses ALL features correctly:
- Structure-aware chunking with section metadata
- Metadata enrichment (entities, topics, keywords)
- Hybrid vectorstore (dense + BM25)
- All chunk attributes preserved
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
from src.enrichment import EnrichmentPipeline, EnrichmentConfig


COLLECTION_NAME = "medical_proper"
BATCH_SIZE = 50


def extract_full_metadata(chunk, source_file: str, enrichment_result=None):
    """Extract ALL metadata from chunk + enrichment."""
    metadata = {}
    
    # Basic metadata from chunk.metadata
    if hasattr(chunk, 'metadata') and chunk.metadata:
        metadata.update(chunk.metadata)
    
    # Chunk object attributes
    if hasattr(chunk, 'section') and chunk.section:
        metadata['section'] = chunk.section
    if hasattr(chunk, 'section_hierarchy') and chunk.section_hierarchy:
        metadata['section_hierarchy'] = chunk.section_hierarchy
    if hasattr(chunk, 'chunk_type') and chunk.chunk_type:
        metadata['chunk_type'] = chunk.chunk_type
    if hasattr(chunk, 'parent_id') and chunk.parent_id:
        metadata['parent_id'] = chunk.parent_id
    if hasattr(chunk, 'chunk_id') and chunk.chunk_id:
        metadata['chunk_id'] = chunk.chunk_id
    
    # Source info
    metadata['source'] = source_file
    
    # Enrichment results
    if enrichment_result:
        if hasattr(enrichment_result, 'entities') and enrichment_result.entities:
            metadata['entities'] = enrichment_result.entities
        if hasattr(enrichment_result, 'topics') and enrichment_result.topics:
            metadata['topics'] = enrichment_result.topics
        if hasattr(enrichment_result, 'keywords') and enrichment_result.keywords:
            metadata['keywords'] = enrichment_result.keywords
        if hasattr(enrichment_result, 'summary') and enrichment_result.summary:
            metadata['summary'] = enrichment_result.summary
    
    return metadata


def main():
    print("=" * 70)
    print("MEDICAL RAG - PROPER FULL-FEATURED INGESTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Data path - use subset for faster testing
    data_dir = Path("data/medical/subset")
    files = list(data_dir.glob("*.txt"))
    print(f"📁 Found {len(files)} files:")
    for f in files:
        print(f"   • {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    
    # Load config
    print("\n[1/6] Loading configuration...")
    config = Config.load("config/rag.yaml")
    enhanced_cfg = config.get_section("enhanced") or {}
    chunking_cfg = enhanced_cfg.get("chunking", {})
    
    print("\n  📋 Features to Apply:")
    print(f"     • Structure-aware chunking (size={chunking_cfg.get('chunk_size', 1500)})")
    print(f"     • Section/hierarchy extraction")
    print(f"     • Entity extraction (money, dates, orgs)")
    print(f"     • Topic/keyword extraction")
    print(f"     • Hybrid search (Dense + BM25)")
    print(f"     • Cross-encoder reranking")
    
    # Initialize components
    print("\n[2/6] Initializing components...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text")
    
    # Delete and recreate collection
    print(f"\n[3/6] Setting up collection: {COLLECTION_NAME}")
    client = QdrantClient(host="localhost", port=6333)
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            print(f"  ⚠ Deleting existing: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"  Note: {e}")
    
    vectorstore = QdrantHybridStore(
        host="localhost",
        port=6333,
        collection_name=COLLECTION_NAME,
        dense_dimensions=768
    )
    print(f"  ✓ Hybrid vectorstore: {COLLECTION_NAME}")
    
    # Retriever with reranker
    print("\n[4/6] Setting up retrieval...")
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
        top_k=20,
        reranker=reranker
    )
    print("  ✓ Hybrid retriever + reranker")
    
    # Chunker
    print("\n[5/6] Setting up processing pipeline...")
    chunker = StructureAwareChunker(
        chunk_size=chunking_cfg.get("chunk_size", 1500),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 150)
    )
    print("  ✓ Structure-aware chunker")
    
    # Enrichment
    enrichment = EnrichmentPipeline(config=EnrichmentConfig.fast())
    print("  ✓ Enrichment pipeline (fast mode)")
    
    # Ingest
    print("\n[6/6] Ingesting with full metadata...")
    print("=" * 70)
    
    total_chunks = 0
    total_enriched = 0
    total_with_section = 0
    start_time = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n📄 [{i}/{len(files)}] {file_path.name}")
        file_start = time.time()
        
        try:
            # Load
            documents = LoaderFactory.load(str(file_path))
            
            # Chunk with structure awareness
            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)
            
            print(f"   • Chunks: {len(all_chunks)}")
            
            # Count sections
            sections_found = sum(1 for c in all_chunks if hasattr(c, 'section') and c.section)
            if sections_found > 0:
                print(f"   • With sections: {sections_found}")
                total_with_section += sections_found
            
            # Prepare with enrichment
            texts = []
            metadatas = []
            enriched_count = 0
            
            for chunk in all_chunks:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                
                # Skip garbage
                if len(content) < 50 or content.startswith("data:"):
                    continue
                
                # Enrich
                enrichment_result = None
                try:
                    enrichment_result = enrichment.enrich(content)
                    enriched_count += 1
                except Exception:
                    pass  # Continue without enrichment on error
                
                # Extract ALL metadata
                metadata = extract_full_metadata(chunk, file_path.name, enrichment_result)
                
                texts.append(content)
                metadatas.append(metadata)
            
            print(f"   • Enriched: {enriched_count}")
            total_enriched += enriched_count
            
            # Index in batches
            num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(texts))
                
                batch_texts = texts[start_idx:end_idx]
                batch_metas = metadatas[start_idx:end_idx]
                
                retriever.add_documents(batch_texts, batch_metas)
                total_chunks += len(batch_texts)
            
            file_time = time.time() - file_start
            print(f"   ✓ Indexed: {len(texts)} in {file_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Verify with sample
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n  ✓ Collection:       {COLLECTION_NAME}")
    print(f"  ✓ Vectors stored:   {info.points_count}")
    print(f"  ✓ Chunks enriched:  {total_enriched}")
    print(f"  ✓ With sections:    {total_with_section}")
    print(f"  ✓ Time:             {total_time:.1f}s")
    
    # Sample metadata
    print("\n  📋 Sample Metadata:")
    result = client.scroll(collection_name=COLLECTION_NAME, limit=3, with_payload=True)
    for p in result[0][:2]:
        print(f"\n  ID: {str(p.id)[:8]}...")
        print(f"  Keys: {list(p.payload.keys())}")
        for k, v in p.payload.items():
            if k not in ['content', 'text']:
                val_str = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                print(f"    • {k}: {val_str}")
    
    print("\n" + "=" * 70)
    print("✓ Proper ingestion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
