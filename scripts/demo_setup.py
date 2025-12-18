#!/usr/bin/env python3
"""
Demo Setup Script
=================
Run this BEFORE recording to ensure everything is warmed up
and results are consistent.
"""

import time
from src.embeddings import OllamaEmbeddings, CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval import HybridRetriever
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline

def main():
    print("=" * 60)
    print("🎬 DEMO SETUP - Warming up for recording")
    print("=" * 60)
    
    # Initialize
    print("\n1️⃣  Initializing components...")
    embeddings = CachedEmbeddings(
        OllamaEmbeddings(model="nomic-embed-text"), 
        enabled=True
    )
    vectorstore = QdrantHybridStore(
        collection_name="demo_video",
        dense_dimensions=768,
        recreate_collection=True,
    )
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="fastembed",
    )
    llm = OllamaLLM(model="llama3.2")
    
    pipeline = MultiDocumentPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        registry_path=".cache/demo_video.json",
    )
    print("   ✅ Components ready")
    
    # Ingest
    print("\n2️⃣  Ingesting SEC filings...")
    start = time.time()
    stats = pipeline.ingest_directory(
        "data/test_adaptive/sec-edgar-filings/",
        recursive=True,
    )
    ingest_time = time.time() - start
    print(f"   ✅ {len(stats['companies'])} companies, {stats['total_chunks']} chunks")
    print(f"   ⏱️  Time: {ingest_time:.1f}s")
    
    # Warm up queries (these will be cached)
    print("\n3️⃣  Warming up demo queries...")
    
    demo_queries = [
        ("What factors affect Meta's advertising revenue?", ["Meta"]),
        ("What are Tesla's manufacturing and supply chain risks?", ["Tesla"]),
        ("What are NVIDIA's key products and growth drivers?", ["NVIDIA"]),
        ("What cybersecurity risks do these companies face?", None),
    ]
    
    for query, filter_co in demo_queries:
        start = time.time()
        response = pipeline.query(query, top_k=5, filter_companies=filter_co)
        qtime = time.time() - start
        
        company_str = filter_co[0] if filter_co else "All"
        print(f"   [{company_str}] {response.confidence_emoji} {qtime:.2f}s - {query[:40]}...")
    
    # Warm up comparison
    print("\n4️⃣  Warming up comparison query...")
    start = time.time()
    comparison = pipeline.compare_companies(
        "What is the company's AI and machine learning strategy?",
        companies=["Meta", "Tesla", "NVIDIA"],
        top_k_per_company=2,
    )
    ctime = time.time() - start
    print(f"   ✅ Comparison ready: {ctime:.2f}s")
    
    # Stats
    print("\n" + "=" * 60)
    print("✅ DEMO READY!")
    print("=" * 60)
    print(f"""
Cache stats: {embeddings.stats}

Expected results during demo:
┌─────────────────────────────────────────────────────────────┐
│ Ingestion:      ~2.3 seconds (531 chunks)                   │
│ Meta query:     ~2-3 seconds, HIGH confidence               │
│ Tesla query:    ~2-3 seconds, MEDIUM confidence             │
│ NVIDIA query:   ~2-3 seconds, HIGH confidence               │
│ Comparison:     ~3-4 seconds                                │
└─────────────────────────────────────────────────────────────┘

Now open: streamlit run src/ui/app.py
Collection to use: demo_video
""")


if __name__ == "__main__":
    main()
