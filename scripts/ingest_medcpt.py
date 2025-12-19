"""Ingest medical corpus with MedCPT embeddings - SEPARATE collection."""

import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

from qdrant_client import QdrantClient
from src.embeddings.medcpt_embeddings import MedCPTEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig

# NEW collection - does not affect medical_final
COLLECTION = "medical_medcpt"

# Use subset for faster testing
FILES = [
    "data/medical/subset/First_Aid_Step1.txt",
    "data/medical/subset/First_Aid_Step2.txt",
    "data/medical/subset/Pathology_Robbins.txt",
    "data/medical/subset/Pharmacology_Katzung.txt",
    "data/medical/subset/Physiology_Levy.txt",
    "data/medical/textbooks/InternalMed_Harrison.txt",
]

print("="*60)
print("MEDCPT INGESTION - SEPARATE COLLECTION")
print("="*60)
print(f"Collection: {COLLECTION} (medical_final unchanged)")

# Delete collection if exists
client = QdrantClient(host="localhost", port=6333)
if COLLECTION in [c.name for c in client.get_collections().collections]:
    client.delete_collection(COLLECTION)
    print(f"Deleted existing: {COLLECTION}")

# Initialize MedCPT
print("\nLoading MedCPT embeddings...")
medcpt = MedCPTEmbeddings()
embeddings = CachedEmbeddings(medcpt, enabled=True, cache_dir=".cache/medcpt_embeddings")
print("✓ MedCPT loaded")

# Setup pipeline
vs = QdrantHybridStore(host="localhost", port=6333, collection_name=COLLECTION, dense_dimensions=768)
rr = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
ret = HybridRetriever(embeddings=embeddings, vectorstore=vs, sparse_encoder="fastembed", top_k=20, reranker=rr)
llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
cfg = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, enable_enrichment=True, top_k=20)

pipe = MultiDocumentPipeline(
    embeddings=embeddings, vectorstore=vs, retriever=ret, llm=llm, config=cfg,
    registry_path=".cache/medcpt_registry.json", clear_registry=True
)
print("✓ Pipeline ready")

# Ingest
print(f"\nIngesting {len(FILES)} files...")
start = time.time()
total = 0

for i, f in enumerate(FILES, 1):
    p = Path(f)
    print(f"[{i}/{len(FILES)}] {p.name}...", end=" ", flush=True)
    t0 = time.time()
    stats = pipe.ingest_file(f, skip_if_exists=False)
    chunks = stats.get('chunks', 0)
    total += chunks
    print(f"{chunks} chunks ({time.time()-t0:.1f}s)")

elapsed = time.time() - start
info = client.get_collection(COLLECTION)

print(f"\n{'='*60}")
print(f"✓ Collection: {COLLECTION}")
print(f"✓ Vectors: {info.points_count}")
print(f"✓ Time: {elapsed:.1f}s")
print(f"✓ medical_final: UNCHANGED")
print("="*60)
