"""
Benchmark rerankers against each other.

Compares:
- Cross-encoder (ms-marco-MiniLM)
- BGE reranker (base)

Metrics:
- Latency (ms per query)
- Ranking quality (relevant docs in top-k)
- Score distribution
"""
import time
import sys
from pathlib import Path
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reranking import RerankerFactory


# Test dataset - queries with known relevant documents
TEST_CASES = [
    {
        "query": "how does kafka handle message ordering?",
        "documents": [
            "Redis is an in-memory data store used for caching.",
            "Kafka guarantees message ordering within a partition using offset sequences.",
            "PostgreSQL is a relational database with ACID compliance.",
            "Kafka consumers read messages in order from partitions they are assigned to.",
            "Docker containers package applications with their dependencies.",
        ],
        "relevant_indices": [1, 3],  # 0-indexed positions of relevant docs
    },
    {
        "query": "what is vector similarity search?",
        "documents": [
            "Vectors represent data as points in high-dimensional space.",
            "SQL databases use B-tree indexes for fast lookups.",
            "Cosine similarity measures the angle between two vectors.",
            "REST APIs use HTTP methods like GET and POST.",
            "Vector databases find nearest neighbors using distance metrics.",
        ],
        "relevant_indices": [0, 2, 4],
    },
    {
        "query": "how to deploy machine learning models?",
        "documents": [
            "CSS styles web pages with colors and layouts.",
            "MLflow tracks experiments and packages models for deployment.",
            "JavaScript runs in the browser.",
            "Docker containers package ML models with dependencies for deployment.",
            "FastAPI can serve model predictions via REST endpoints.",
        ],
        "relevant_indices": [1, 3, 4],
    },
    {
        "query": "explain database transactions and ACID",
        "documents": [
            "ACID stands for Atomicity, Consistency, Isolation, Durability.",
            "HTML structures web page content.",
            "Transactions ensure database operations complete fully or not at all.",
            "Git tracks changes in source code.",
            "PostgreSQL supports serializable transaction isolation levels.",
        ],
        "relevant_indices": [0, 2, 4],
    },
]


def calculate_precision_at_k(results, relevant_indices, k):
    """Calculate precision@k - what fraction of top-k are relevant."""
    top_k_original_indices = []
    for r in results[:k]:
        # original_rank is 1-indexed, convert to 0-indexed
        top_k_original_indices.append(r.original_rank - 1)
    
    relevant_in_top_k = sum(1 for idx in top_k_original_indices if idx in relevant_indices)
    return relevant_in_top_k / k


def calculate_mrr(results, relevant_indices):
    """Calculate Mean Reciprocal Rank - how early is first relevant doc."""
    for rank, r in enumerate(results, 1):
        original_idx = r.original_rank - 1
        if original_idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def benchmark_reranker(reranker, name):
    """Run benchmark on a reranker."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    latencies = []
    precisions_at_3 = []
    mrrs = []
    
    # Warm-up run (triggers model loading)
    print("Warming up (loading model)...")
    _ = reranker.rerank("warmup query", ["warmup doc"])
    
    for i, test in enumerate(TEST_CASES):
        query = test["query"]
        documents = test["documents"]
        relevant_indices = test["relevant_indices"]
        
        # Time the reranking
        start = time.perf_counter()
        results = reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        latencies.append(elapsed_ms)
        precisions_at_3.append(calculate_precision_at_k(results, relevant_indices, k=3))
        mrrs.append(calculate_mrr(results, relevant_indices))
        
        print(f"\nTest {i+1}: {query[:50]}...")
        print(f"  Latency: {elapsed_ms:.1f}ms")
        print(f"  P@3: {precisions_at_3[-1]:.2f}")
        print(f"  MRR: {mrrs[-1]:.2f}")
        print(f"  Top 3 reranked:")
        for r in results[:3]:
            marker = "✓" if (r.original_rank - 1) in relevant_indices else "✗"
            print(f"    {marker} [{r.original_rank}→{r.new_rank}] score={r.score:.2f}: {r.content[:50]}...")
    
    # Summary
    print(f"\n{'-'*40}")
    print(f"SUMMARY: {name}")
    print(f"{'-'*40}")
    print(f"  Avg Latency: {mean(latencies):.1f}ms (±{stdev(latencies):.1f})")
    print(f"  Avg P@3:     {mean(precisions_at_3):.3f}")
    print(f"  Avg MRR:     {mean(mrrs):.3f}")
    
    return {
        "name": name,
        "avg_latency_ms": mean(latencies),
        "std_latency_ms": stdev(latencies),
        "avg_precision_at_3": mean(precisions_at_3),
        "avg_mrr": mean(mrrs),
    }


def main():
    print("="*60)
    print("RERANKER BENCHMARK")
    print("="*60)
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Documents per query: {len(TEST_CASES[0]['documents'])}")
    
    results = []
    
    # Benchmark Cross-Encoder
    cross_encoder = RerankerFactory.create("cross_encoder")
    results.append(benchmark_reranker(cross_encoder, "Cross-Encoder (ms-marco-MiniLM)"))
    
    # Benchmark BGE
    bge = RerankerFactory.create("bge", model="base")
    results.append(benchmark_reranker(bge, "BGE Reranker (base)"))
    
    # Comparison table
    print("\n")
    print("="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Reranker':<35} {'Latency':<12} {'P@3':<8} {'MRR':<8}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<35} {r['avg_latency_ms']:.1f}ms{'':<6} {r['avg_precision_at_3']:.3f}{'':<4} {r['avg_mrr']:.3f}")
    
    # Winner
    print("\n" + "-"*60)
    best_quality = max(results, key=lambda x: x['avg_mrr'])
    best_speed = min(results, key=lambda x: x['avg_latency_ms'])
    print(f"Best Quality (MRR): {best_quality['name']}")
    print(f"Best Speed:         {best_speed['name']}")


if __name__ == "__main__":
    main()
