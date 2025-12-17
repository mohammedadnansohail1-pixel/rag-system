"""
Full evaluation on Netflix 10-K dataset.

Compares Dense vs Hybrid (BM25) vs Hybrid (TFIDF) on real financial data.
Uses stable MD5 hashes for consistent IDs.
"""
import sys
import json
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import SECLoader
from src.chunkers.factory import ChunkerFactory
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.base import RetrievalResult
from src.evaluation import EvaluationDataset, RetrievalEvaluator


def stable_hash(text: str) -> str:
    """Generate stable hash that persists across Python sessions."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


class DenseRetriever:
    """Dense-only retriever."""
    def __init__(self, embeddings, vectorstore):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
    
    def retrieve(self, query, top_k=5):
        query_emb = self.embeddings.embed_text(query)
        results = self.vectorstore.search(query_emb, top_k=top_k)
        return [RetrievalResult(content=r.content, metadata=r.metadata, score=r.score) for r in results]
    
    def add_documents(self, texts, metadatas=None):
        emb_list = self.embeddings.embed_batch(texts)
        return self.vectorstore.add(texts=texts, embeddings=emb_list, metadatas=metadatas)
    
    def health_check(self):
        return True


def main():
    print("=" * 70)
    print("NETFLIX 10-K RETRIEVAL EVALUATION")
    print("=" * 70)
    
    # Load and chunk document FIRST (to ensure consistent hashes)
    print("\n[1/5] Loading Netflix 10-K...")
    loader = SECLoader(download_dir='data/test_docs')
    docs = loader.load(
        'data/test_docs/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt'
    )
    
    chunker = ChunkerFactory.from_config({
        'strategy': 'recursive',
        'chunk_size': 1000,
        'chunk_overlap': 200,
    })
    chunks = chunker.chunk(docs[0])
    
    # Clean chunks
    clean_texts = []
    for chunk in chunks:
        content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
        if len(content) > 100:
            clean_texts.append(content)
    
    print(f"  Chunks: {len(clean_texts)}")
    
    # Rebuild dataset with same chunks
    print("\n[2/5] Building evaluation dataset...")
    chunk_ids = {stable_hash(c): c for c in clean_texts}
    
    # Load existing dataset structure
    dataset_path = Path("data/eval_datasets/netflix_10k_comprehensive.json")
    dataset = EvaluationDataset.load(str(dataset_path))
    print(f"  Loaded {len(dataset)} test cases")
    
    # Initialize embeddings
    print("\n[3/5] Initializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    # Create retrievers
    print("\n[4/5] Setting up retrievers...")
    
    # Dense
    print("  Creating Dense retriever...")
    dense_store = QdrantVectorStore(collection_name="eval_netflix_dense", dimensions=768)
    dense_retriever = DenseRetriever(embeddings, dense_store)
    
    start = time.perf_counter()
    dense_retriever.add_documents(clean_texts)
    dense_time = time.perf_counter() - start
    print(f"    Indexed in {dense_time:.1f}s")
    
    # Hybrid BM25
    print("  Creating Hybrid (BM25) retriever...")
    hybrid_bm25_store = QdrantHybridStore(
        collection_name="eval_netflix_bm25",
        dense_dimensions=768,
        recreate_collection=True,
    )
    hybrid_bm25 = HybridRetriever(
        embeddings=embeddings,
        vectorstore=hybrid_bm25_store,
        sparse_encoder="bm25",
    )
    
    start = time.perf_counter()
    hybrid_bm25.add_documents(clean_texts)
    bm25_time = time.perf_counter() - start
    print(f"    Indexed in {bm25_time:.1f}s")
    
    # Evaluate
    print("\n[5/5] Running evaluation...")
    
    retrievers = {
        "Dense Only": dense_retriever,
        "Hybrid (BM25)": hybrid_bm25,
    }
    
    results = {}
    for name, retriever in retrievers.items():
        print(f"\n  Evaluating: {name}")
        evaluator = RetrievalEvaluator(
            retriever,
            id_extractor=lambda r: stable_hash(r.content),
        )
        results[name] = evaluator.evaluate(dataset, k=5, verbose=True)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n" + RetrievalEvaluator.comparison_table(results))
    
    # Best performer details
    best_name = max(results.keys(), key=lambda k: results[k].mean_mrr)
    print(f"\n" + "=" * 70)
    print(f"BEST PERFORMER: {best_name}")
    print("=" * 70)
    print(results[best_name].summary())
    
    # Show some example results
    print("\n" + "-" * 70)
    print("SAMPLE QUERY RESULTS")
    print("-" * 70)
    
    best_result = results[best_name]
    for qr in best_result.query_results[:5]:
        status = "✓" if qr.hit else "✗"
        print(f"\n  {status} Query: {qr.query[:60]}...")
        print(f"    MRR: {qr.mrr_score:.3f}, P@5: {qr.precision:.3f}, R@5: {qr.recall:.3f}")
        if qr.retrieved_contents:
            print(f"    Top result: {qr.retrieved_contents[0][:70]}...")
    
    # Save results
    results_path = Path("data/eval_datasets/netflix_10k_results.json")
    results_data = {name: r.to_dict() for name, r in results.items()}
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_path}")
    
    # Cleanup
    print("\nCleaning up...")
    dense_store._client.delete_collection("eval_netflix_dense")
    hybrid_bm25_store._client.delete_collection("eval_netflix_bm25")
    
    print("Done!")


if __name__ == "__main__":
    main()
