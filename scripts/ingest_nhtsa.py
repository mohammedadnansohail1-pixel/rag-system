"""NHTSA Complaints Ingestion - 1 complaint = 1 vector."""

import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.embeddings.cached_embeddings import CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.fastembed_sparse_encoder import FastEmbedSparseEncoder
from src.reranking.cross_encoder import CrossEncoderReranker
from src.loaders.json_loader import JSONLoader, JSONFieldConfig

COLLECTION = "nhtsa_complaints"
JSON_FILE = "data/nhtsa_complaints_merged.json"

print("=" * 70)
print("NHTSA COMPLAINTS - INGESTION")
print("=" * 70)

# NHTSA field config
config = JSONFieldConfig(
    content_fields=["summary"],
    metadata_fields=["nhtsaId", "year", "vehicle", "component", 
                     "consumerLocation", "crash", "fire", "injuries", "deaths"],
    content_template="""Vehicle: {year} {vehicle}
Component: {component}
Location: {consumerLocation}
Crash: {crash} | Fire: {fire} | Injuries: {injuries} | Deaths: {deaths}

{summary}""",
    id_field="nhtsaId"
)

# Load JSON
print("\n[1/4] Loading JSON...")
loader = JSONLoader(config)
documents = loader.load(Path(JSON_FILE))
print(f"      Loaded {len(documents)} complaints")

# Delete old collection
print("\n[2/4] Preparing collection...")
client = QdrantClient(host="localhost", port=6333)
if COLLECTION in [c.name for c in client.get_collections().collections]:
    client.delete_collection(COLLECTION)
    print(f"      Deleted existing: {COLLECTION}")

# Initialize components
print("\n[3/4] Initializing...")
embeddings = CachedEmbeddings(
    OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768),
    enabled=True, cache_dir=".cache/nhtsa_embeddings"
)
vectorstore = QdrantHybridStore(host="localhost", port=6333, collection_name=COLLECTION, dense_dimensions=768)
sparse_encoder = FastEmbedSparseEncoder()

print("      ✓ Embeddings: nomic-embed-text")
print("      ✓ Sparse: FastEmbed")

# Ingest: 1 complaint = 1 vector
print("\n[4/4] Ingesting (1 complaint = 1 vector)...")
start = time.time()
batch_size = 100
total = len(documents)

for i in range(0, total, batch_size):
    batch = documents[i:i+batch_size]
    texts = [d.content for d in batch]
    metadatas = [d.metadata for d in batch]
    
    # Get dense embeddings
    dense_embeddings = embeddings.embed_batch(texts)
    
    # Get sparse embeddings
    sparse_vectors = sparse_encoder.encode_batch(texts)
    
    # Add to vectorstore
    vectorstore.add_hybrid(
        texts=texts,
        dense_embeddings=dense_embeddings,
        sparse_vectors=sparse_vectors,
        metadatas=metadatas
    )
    
    pct = min(i + batch_size, total) / total * 100
    print(f"      Progress: {min(i + batch_size, total)}/{total} ({pct:.0f}%)", end="\r")

elapsed = time.time() - start
info = client.get_collection(COLLECTION)

print(f"\n\n{'='*70}")
print(f"✓ Collection: {COLLECTION}")
print(f"✓ Vectors: {info.points_count}")
print(f"✓ Time: {elapsed:.1f}s")
print("="*70)
