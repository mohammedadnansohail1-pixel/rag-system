"""Test MedCPT embeddings vs nomic-embed-text on a subset."""

import sys
import time
sys.path.insert(0, ".")

from src.embeddings.medcpt_embeddings import MedCPTEmbeddings
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.ollama_llm import OllamaLLM
from src.documents import MultiDocumentPipeline
from src.pipeline.enhanced_rag_pipeline import EnhancedRAGConfig
from qdrant_client import QdrantClient

# Test questions
QUESTIONS = [
    ("What are the classic symptoms of myocardial infarction?", "cardiology"),
    ("What is the mechanism of action of ACE inhibitors?", "pharmacology"),
    ("What is the Frank-Starling mechanism?", "physiology"),
    ("What are the phases of wound healing?", "pathology"),
]

def test_retrieval(embeddings, name, collection):
    """Test retrieval quality."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Collection: {collection}")
    print(f"{'='*60}")
    
    vs = QdrantHybridStore(host="localhost", port=6333, collection_name=collection, dense_dimensions=768)
    rr = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ret = HybridRetriever(embeddings=embeddings, vectorstore=vs, sparse_encoder="fastembed", top_k=10, reranker=rr)
    
    for q, cat in QUESTIONS:
        print(f"\n[{cat.upper()}] {q[:50]}...")
        start = time.time()
        results = ret.retrieve(q)
        elapsed = time.time() - start
        
        if results:
            top = results[0]
            print(f"  Top score: {top.score:.3f} | Time: {elapsed:.2f}s")
            print(f"  Content: {top.content[:100]}...")
        else:
            print(f"  No results!")

def main():
    print("="*60)
    print("MEDCPT vs NOMIC-EMBED-TEXT COMPARISON")
    print("="*60)
    
    # Test 1: Current setup (nomic-embed-text)
    print("\n[1/2] Loading nomic-embed-text...")
    nomic = OllamaEmbeddings(host="http://localhost:11434", model="nomic-embed-text", dimensions=768)
    test_retrieval(nomic, "nomic-embed-text", "medical_final")
    
    # Test 2: MedCPT
    print("\n[2/2] Loading MedCPT...")
    try:
        medcpt = MedCPTEmbeddings()
        
        # Check if medcpt collection exists
        client = QdrantClient(host="localhost", port=6333)
        collections = [c.name for c in client.get_collections().collections]
        
        if "medical_medcpt" not in collections:
            print("\n⚠️  Collection 'medical_medcpt' not found.")
            print("   Run: python scripts/ingest_medcpt.py to create it")
            print("   (This will NOT affect medical_final)")
        else:
            test_retrieval(medcpt, "MedCPT", "medical_medcpt")
            
    except Exception as e:
        print(f"MedCPT error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
