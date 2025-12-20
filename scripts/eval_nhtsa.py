"""NHTSA Complaints RAG - Evaluation using full pipeline."""

import sys
import time
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig

COLLECTION = "nhtsa_complaints"

# Test queries
QUERIES = [
    "Tesla sudden unintended acceleration crash",
    "brake pedal failure no stopping",
    "battery fire while parked",
    "steering wheel locked while driving",
    "autopilot failed to stop collision",
]

print("=" * 70)
print("NHTSA COMPLAINTS RAG - EVALUATION")
print("=" * 70)

# Initialize pipeline
print("\nInitializing pipeline...")
embeddings = CachedEmbeddings(
    OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768),
    enabled=True, cache_dir=".cache/nhtsa_embeddings"
)
vectorstore = QdrantHybridStore(host="localhost", port=6333, collection_name=COLLECTION, dense_dimensions=768)
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = HybridRetriever(
    embeddings=embeddings,
    vectorstore=vectorstore,
    sparse_encoder="fastembed",
    top_k=20,
    reranker=reranker
)
llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
cfg = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, top_k=20)

pipeline = MultiDocumentPipeline(
    embeddings=embeddings, vectorstore=vectorstore, retriever=retriever, llm=llm, config=cfg,
    registry_path=".cache/nhtsa_registry.json"
)
print("✓ Pipeline ready\n")

# Run queries
results_summary = []

for i, q in enumerate(QUERIES, 1):
    print(f"{'─'*70}")
    print(f"[{i}/{len(QUERIES)}] Q: {q}")
    print("─"*70)
    
    t0 = time.time()
    result = pipeline.query(q)
    elapsed = time.time() - t0
    
    conf = result.confidence.upper() if hasattr(result, 'confidence') else 'N/A'
    results_summary.append({"q": q, "conf": conf, "sources": len(result.sources), "time": elapsed})
    
    print(f"\n📊 Confidence: {conf} | Sources: {len(result.sources)} | Time: {elapsed:.2f}s")
    print(f"\n💬 Answer:\n{result.answer[:400]}..." if len(result.answer) > 400 else f"\n💬 Answer:\n{result.answer}")
    
    # Show top sources with metadata
    if result.sources:
        print(f"\n📄 Top Sources:")
        for j, src in enumerate(result.sources[:3], 1):
            # Handle different source formats
            if hasattr(src, 'metadata'):
                meta = src.metadata
            elif isinstance(src, dict):
                meta = src.get('metadata', src)
            else:
                meta = {}
            
            vehicle = meta.get('vehicle', 'N/A')
            component = meta.get('component', 'N/A')
            crash = meta.get('crash', 'N/A')
            content = src.content[:100] if hasattr(src, 'content') else str(src)[:100]
            print(f"  [{j}] {vehicle} | {component} | Crash: {crash}")
            print(f"      {content}...")

# Summary
print(f"\n{'='*70}")
print("📈 SUMMARY")
print("="*70)
print(f"\n{'Query':<45} {'Conf':<8} {'Sources':<8} {'Time'}")
print("-"*70)
for r in results_summary:
    print(f"{r['q'][:44]:<45} {r['conf']:<8} {r['sources']:<8} {r['time']:.2f}s")

high = sum(1 for r in results_summary if r['conf'] == 'HIGH')
med = sum(1 for r in results_summary if r['conf'] == 'MEDIUM')
low = sum(1 for r in results_summary if r['conf'] == 'LOW')
avg_time = sum(r['time'] for r in results_summary) / len(results_summary)

print(f"\n✓ HIGH: {high}/{len(QUERIES)} | MEDIUM: {med} | LOW: {low}")
print(f"✓ Avg Time: {avg_time:.2f}s")
print("="*70)
