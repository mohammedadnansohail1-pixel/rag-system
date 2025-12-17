"""
Research and test retrieval improvements.

Tests:
1. Reranking impact
2. Query expansion
3. Different retrieval depths
"""
import sys
import json
import hashlib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import SECLoader
from src.chunkers.factory import ChunkerFactory
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.reranking import RerankerFactory
from src.retrieval.base import RetrievalResult
from src.evaluation import EvaluationDataset, RetrievalEvaluator


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


class DenseRetrieverWithReranking:
    """Dense retriever with cross-encoder reranking."""
    
    def __init__(self, embeddings, vectorstore, reranker, retrieve_k=20):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.retrieve_k = retrieve_k  # Retrieve more, then rerank
    
    def retrieve(self, query, top_k=5):
        # Over-retrieve
        query_emb = self.embeddings.embed_text(query)
        results = self.vectorstore.search(query_emb, top_k=self.retrieve_k)
        
        if not results:
            return []
        
        # Rerank
        texts = [r.content for r in results]
        reranked = self.reranker.rerank(query, texts, top_n=top_k)
        
        return [
            RetrievalResult(content=r.content, metadata={"id": stable_hash(r.content)}, score=r.score)
            for r in reranked
        ]
    
    def health_check(self):
        return True


class DenseRetriever:
    """Basic dense retriever."""
    
    def __init__(self, embeddings, vectorstore):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
    
    def retrieve(self, query, top_k=5):
        query_emb = self.embeddings.embed_text(query)
        results = self.vectorstore.search(query_emb, top_k=top_k)
        return [
            RetrievalResult(content=r.content, metadata=r.metadata, score=r.score)
            for r in results
        ]
    
    def add_documents(self, texts, metadatas=None):
        emb_list = self.embeddings.embed_batch(texts)
        return self.vectorstore.add(texts=texts, embeddings=emb_list, metadatas=metadatas)
    
    def health_check(self):
        return True


def main():
    print("=" * 70)
    print("RETRIEVAL IMPROVEMENT RESEARCH")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
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
    
    clean_texts = []
    for chunk in chunks:
        content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
        if len(content) > 100:
            clean_texts.append(content)
    
    print(f"  Chunks: {len(clean_texts)}")
    
    # Load dataset
    dataset = EvaluationDataset.load("data/eval_datasets/netflix_10k_comprehensive.json")
    print(f"  Test cases: {len(dataset)}")
    
    # Initialize components
    print("\n[2/4] Initializing components...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    reranker = RerankerFactory.create("cross_encoder")
    
    # Create vector store
    vectorstore = QdrantVectorStore(collection_name="research_test", dimensions=768)
    
    # Index
    print("\n[3/4] Indexing...")
    emb_list = embeddings.embed_batch(clean_texts)
    vectorstore.add(texts=clean_texts, embeddings=emb_list)
    print(f"  Indexed {len(clean_texts)} chunks")
    
    # Test different configurations
    print("\n[4/4] Testing configurations...")
    
    configs = {
        "Dense (k=5)": DenseRetriever(embeddings, vectorstore),
        "Dense (k=10)": DenseRetriever(embeddings, vectorstore),
        "Dense + Rerank (20→5)": DenseRetrieverWithReranking(
            embeddings, vectorstore, reranker, retrieve_k=20
        ),
        "Dense + Rerank (50→5)": DenseRetrieverWithReranking(
            embeddings, vectorstore, reranker, retrieve_k=50
        ),
    }
    
    results = {}
    
    for name, retriever in configs.items():
        print(f"\n  Testing: {name}")
        
        # Determine k based on config name
        k = 10 if "k=10" in name else 5
        
        evaluator = RetrievalEvaluator(
            retriever,
            id_extractor=lambda r: stable_hash(r.content),
        )
        
        start = time.perf_counter()
        result = evaluator.evaluate(dataset, k=k if "k=10" in name else 5, verbose=False)
        elapsed = time.perf_counter() - start
        
        results[name] = result
        print(f"    MRR: {result.mean_mrr:.3f}, Hit Rate: {result.hit_rate:.1%}, Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(RetrievalEvaluator.comparison_table(results))
    
    # Improvement analysis
    baseline_mrr = results["Dense (k=5)"].mean_mrr
    best_name = max(results.keys(), key=lambda k: results[k].mean_mrr)
    best_mrr = results[best_name].mean_mrr
    
    print(f"\nBaseline MRR: {baseline_mrr:.3f}")
    print(f"Best MRR: {best_mrr:.3f} ({best_name})")
    print(f"Improvement: {((best_mrr - baseline_mrr) / baseline_mrr * 100):.1f}%")
    
    # Detailed analysis of reranking impact
    print("\n" + "-" * 70)
    print("RERANKING IMPACT ANALYSIS")
    print("-" * 70)
    
    baseline = results["Dense (k=5)"]
    reranked = results["Dense + Rerank (20→5)"]
    
    print(f"\n{'Metric':<20} {'Baseline':<12} {'+ Rerank':<12} {'Delta':<12}")
    print("-" * 56)
    print(f"{'MRR':<20} {baseline.mean_mrr:<12.3f} {reranked.mean_mrr:<12.3f} {reranked.mean_mrr - baseline.mean_mrr:+.3f}")
    print(f"{'Precision@5':<20} {baseline.mean_precision:<12.3f} {reranked.mean_precision:<12.3f} {reranked.mean_precision - baseline.mean_precision:+.3f}")
    print(f"{'Recall@5':<20} {baseline.mean_recall:<12.3f} {reranked.mean_recall:<12.3f} {reranked.mean_recall - baseline.mean_recall:+.3f}")
    print(f"{'Hit Rate':<20} {baseline.hit_rate:<12.3f} {reranked.hit_rate:<12.3f} {reranked.hit_rate - baseline.hit_rate:+.3f}")
    print(f"{'NDCG@5':<20} {baseline.mean_ndcg:<12.3f} {reranked.mean_ndcg:<12.3f} {reranked.mean_ndcg - baseline.mean_ndcg:+.3f}")
    print(f"{'Latency (ms)':<20} {baseline.latency_mean_ms:<12.1f} {reranked.latency_mean_ms:<12.1f} {reranked.latency_mean_ms - baseline.latency_mean_ms:+.1f}")
    
    # Cleanup
    print("\nCleaning up...")
    vectorstore._client.delete_collection("research_test")
    print("Done!")


if __name__ == "__main__":
    main()
