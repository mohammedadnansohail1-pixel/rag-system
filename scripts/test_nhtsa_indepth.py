"""NHTSA RAG - In-depth Testing."""

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
from qdrant_client import QdrantClient

COLLECTION = "nhtsa_complaints"

print("=" * 70)
print("NHTSA RAG - IN-DEPTH TESTING")
print("=" * 70)

# Initialize
print("\n[1/5] Initializing pipeline...")
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
print("✓ Pipeline ready")

# Collection stats
print("\n[2/5] Collection Statistics...")
client = QdrantClient(host="localhost", port=6333)
info = client.get_collection(COLLECTION)
print(f"      Vectors: {info.points_count}")

# Sample data for stats
results, _ = client.scroll(collection_name=COLLECTION, limit=7000, with_payload=True)

components = {}
crashes, fires, injuries, deaths = 0, 0, 0, 0
years = {}
vehicles = {}

for r in results:
    p = r.payload
    comp = p.get('component', 'Unknown')
    components[comp] = components.get(comp, 0) + 1
    
    yr = p.get('year', 'Unknown')
    years[yr] = years.get(yr, 0) + 1
    
    veh = p.get('vehicle', 'Unknown')
    vehicles[veh] = vehicles.get(veh, 0) + 1
    
    if p.get('crash') == 'Yes': crashes += 1
    if p.get('fire') == 'Yes': fires += 1
    if p.get('injuries', '0') not in ['0', '', None]: injuries += int(p.get('injuries', 0) or 0)
    if p.get('deaths', '0') not in ['0', '', None]: deaths += int(p.get('deaths', 0) or 0)

print(f"\n      📊 Summary:")
print(f"         Crashes: {crashes}")
print(f"         Fires: {fires}")
print(f"         Total Injuries: {injuries}")
print(f"         Total Deaths: {deaths}")

print(f"\n      📊 Top Components:")
for comp, count in sorted(components.items(), key=lambda x: -x[1])[:7]:
    print(f"         {count:>4}: {comp[:50]}")

print(f"\n      📊 By Year:")
for yr, count in sorted(years.items()):
    print(f"         {yr}: {count}")

print(f"\n      📊 Top Vehicles:")
for veh, count in sorted(vehicles.items(), key=lambda x: -x[1])[:5]:
    print(f"         {count:>4}: {veh}")

# Test retrieval quality
print("\n[3/5] Retrieval Quality Test...")
test_cases = [
    {"query": "sudden acceleration", "expected_component": "VEHICLE SPEED CONTROL"},
    {"query": "steering locked", "expected_component": "STEERING"},
    {"query": "brake failure", "expected_component": "SERVICE BRAKES"},
    {"query": "battery fire explosion", "expected_component": "FUEL/PROPULSION"},
    {"query": "autopilot crash", "expected_component": "FORWARD COLLISION"},
    {"query": "touchscreen black", "expected_component": "ELECTRICAL"},
    {"query": "suspension noise", "expected_component": "SUSPENSION"},
]

retrieval_scores = []
for tc in test_cases:
    results = retriever.retrieve(tc["query"])
    
    # Check if expected component in top 5
    found = False
    for r in results[:5]:
        meta = r.metadata if hasattr(r, 'metadata') else {}
        comp = meta.get('component', '')
        if tc["expected_component"].lower() in comp.lower():
            found = True
            break
    
    retrieval_scores.append(1 if found else 0)
    status = "✓" if found else "✗"
    print(f"      {status} '{tc['query']}' → {tc['expected_component']}")

retrieval_accuracy = sum(retrieval_scores) / len(retrieval_scores) * 100
print(f"\n      Retrieval Accuracy: {retrieval_accuracy:.0f}%")

# Test LLM generation quality
print("\n[4/5] Generation Quality Test...")
gen_queries = [
    "What are the most common steering issues reported?",
    "How many crashes involved sudden acceleration?",
    "What happens when the touchscreen goes black while driving?",
    "Are there any reports of fires in Tesla vehicles?",
    "What problems are reported with autopilot?",
]

gen_results = []
for q in gen_queries:
    t0 = time.time()
    result = pipeline.query(q)
    elapsed = time.time() - t0
    
    conf = result.confidence.upper() if hasattr(result, 'confidence') else 'N/A'
    gen_results.append({"q": q, "conf": conf, "sources": len(result.sources), "time": elapsed})
    
    status = "✓" if conf in ['HIGH', 'MEDIUM'] else "✗"
    print(f"      {status} [{conf}] {q[:45]}... ({elapsed:.1f}s)")

# Test edge cases
print("\n[5/5] Edge Case Testing...")
edge_cases = [
    ("Empty-ish query", "car problem"),
    ("Very specific", "Model Y 2021 steering wheel locked highway 70mph"),
    ("Misspelled", "teslla accelaration"),
    ("Non-existent", "flying car malfunction"),
    ("Multi-issue", "brake failure and steering locked same time"),
]

for name, query in edge_cases:
    t0 = time.time()
    result = pipeline.query(query)
    elapsed = time.time() - t0
    conf = result.confidence.upper() if hasattr(result, 'confidence') else 'N/A'
    sources = len(result.sources)
    print(f"      [{conf}] {name}: {sources} sources ({elapsed:.1f}s)")

# Final summary
print(f"\n{'='*70}")
print("📈 FINAL SUMMARY")
print("="*70)
print(f"\n   Collection: {info.points_count} vectors")
print(f"   Crashes in data: {crashes}")
print(f"   Retrieval accuracy: {retrieval_accuracy:.0f}%")

high = sum(1 for r in gen_results if r['conf'] == 'HIGH')
med = sum(1 for r in gen_results if r['conf'] == 'MEDIUM')
low = sum(1 for r in gen_results if r['conf'] == 'LOW')
avg_time = sum(r['time'] for r in gen_results) / len(gen_results)

print(f"   Generation: HIGH={high} MEDIUM={med} LOW={low}")
print(f"   Avg response time: {avg_time:.2f}s")
print(f"\n   Status: {'✅ READY' if retrieval_accuracy >= 70 and high >= 3 else '⚠️ NEEDS WORK'}")
print("="*70)
