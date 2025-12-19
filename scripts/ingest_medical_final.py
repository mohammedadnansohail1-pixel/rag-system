"""Medical RAG - Final ingestion with fastembed (no fitting needed)"""
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

COLLECTION = "medical_final"

# All textbooks
files = [
    "data/medical/subset/First_Aid_Step1.txt",
    "data/medical/subset/First_Aid_Step2.txt", 
    "data/medical/subset/Pathology_Robbins.txt",
    "data/medical/subset/Pharmacology_Katzung.txt",
    "data/medical/subset/Physiology_Levy.txt",
    "data/medical/textbooks/InternalMed_Harrison.txt",
]

print("=" * 70)
print("MEDICAL RAG - FINAL INGESTION (FASTEMBED)")
print("=" * 70)

# Delete collection
client = QdrantClient(host="localhost", port=6333)
try:
    if COLLECTION in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION)
        print(f"Deleted: {COLLECTION}")
except: pass

# Initialize with fastembed (stateless, no fitting)
print("\nInitializing with FastEmbed sparse encoder...")
base_emb = OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768)
embeddings = CachedEmbeddings(base_emb, enabled=True)
vectorstore = QdrantHybridStore(host="localhost", port=6333, collection_name=COLLECTION, dense_dimensions=768)
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Use fastembed instead of bm25
retriever = HybridRetriever(
    embeddings=embeddings, 
    vectorstore=vectorstore, 
    sparse_encoder="fastembed",  # Hash-based, no fitting needed
    top_k=20, 
    reranker=reranker
)
llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", temperature=0.1)
config = EnhancedRAGConfig(chunking_strategy="structure_aware", chunk_size=1500, enable_enrichment=True, top_k=20)

pipeline = MultiDocumentPipeline(
    embeddings=embeddings, vectorstore=vectorstore, retriever=retriever, llm=llm, config=config,
    registry_path=".cache/medical_final_registry.json", clear_registry=True
)

print("✓ Pipeline ready (FastEmbed sparse encoder)\n")

# Ingest
total = 0
for i, f in enumerate(files, 1):
    p = Path(f)
    print(f"[{i}/6] {p.name} ({p.stat().st_size/1024/1024:.1f}MB)")
    stats = pipeline.ingest_file(f, skip_if_exists=False)
    chunks = stats.get('chunks', 0)
    total += chunks
    print(f"      → {chunks} chunks")

print(f"\n{'=' * 70}")
print(f"✓ Total: {total} chunks")
info = client.get_collection(COLLECTION)
print(f"✓ Vectors: {info.points_count}")
print("=" * 70)
