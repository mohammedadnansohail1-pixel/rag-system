"""
Benchmark dense vs hybrid retrieval.

Compares:
- Dense retrieval (embedding similarity only)
- Sparse retrieval (SPLADE only)
- Hybrid retrieval (dense + sparse with RRF)

Metrics:
- Latency
- Precision@K (relevant docs in top-k)
- MRR (Mean Reciprocal Rank)
"""
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_encoder import SpladeEncoder


# Test documents - mix of topics
TEST_DOCUMENTS = [
    # Kafka docs (indices 0-4)
    {"text": "Apache Kafka is a distributed event streaming platform capable of handling trillions of events a day.", "topic": "kafka"},
    {"text": "Kafka guarantees message ordering within a partition using sequential offset numbers.", "topic": "kafka"},
    {"text": "Kafka consumers can be grouped into consumer groups for parallel processing.", "topic": "kafka"},
    {"text": "Kafka uses ZooKeeper for cluster coordination and leader election.", "topic": "kafka"},
    {"text": "Kafka Connect provides connectors for integrating with external systems like databases.", "topic": "kafka"},
    
    # Vector DB docs (indices 5-9)
    {"text": "Vector databases store high-dimensional embeddings for similarity search.", "topic": "vector_db"},
    {"text": "Qdrant is a vector similarity search engine with filtering support.", "topic": "vector_db"},
    {"text": "Cosine similarity measures the angle between two vectors in high-dimensional space.", "topic": "vector_db"},
    {"text": "HNSW is an approximate nearest neighbor algorithm used by many vector databases.", "topic": "vector_db"},
    {"text": "Sparse vectors represent text as weighted term frequencies for keyword matching.", "topic": "vector_db"},
    
    # ML/AI docs (indices 10-14)
    {"text": "Transformers use self-attention mechanisms to process sequential data.", "topic": "ml"},
    {"text": "BERT is a bidirectional encoder trained on masked language modeling.", "topic": "ml"},
    {"text": "Fine-tuning adapts pre-trained models to specific downstream tasks.", "topic": "ml"},
    {"text": "RAG combines retrieval with generation to reduce hallucinations in LLMs.", "topic": "ml"},
    {"text": "Embeddings are dense vector representations that capture semantic meaning.", "topic": "ml"},
    
    # Database docs (indices 15-19)
    {"text": "PostgreSQL is a relational database with strong ACID compliance.", "topic": "database"},
    {"text": "Redis is an in-memory data store used for caching and session management.", "topic": "database"},
    {"text": "Database indexes speed up queries by creating sorted data structures.", "topic": "database"},
    {"text": "Transactions ensure database operations are atomic and consistent.", "topic": "database"},
    {"text": "SQL joins combine rows from multiple tables based on related columns.", "topic": "database"},
]

# Test queries with known relevant documents
TEST_QUERIES = [
    {
        "query": "how does kafka guarantee message ordering?",
        "relevant_topics": ["kafka"],
        "expected_keywords": ["kafka", "ordering", "partition", "message"],
    },
    {
        "query": "vector similarity search cosine",
        "relevant_topics": ["vector_db"],
        "expected_keywords": ["vector", "similarity", "cosine"],
    },
    {
        "query": "what is BERT and how does it work?",
        "relevant_topics": ["ml"],
        "expected_keywords": ["bert", "transformer", "attention"],
    },
    {
        "query": "database transaction ACID properties",
        "relevant_topics": ["database"],
        "expected_keywords": ["transaction", "acid", "database"],
    },
    {
        "query": "Qdrant HNSW approximate nearest neighbor",
        "relevant_topics": ["vector_db"],
        "expected_keywords": ["qdrant", "hnsw", "vector", "nearest"],
    },
]


def calculate_precision(results, relevant_topics, k):
    """Calculate precision@k."""
    hits = 0
    for r in results[:k]:
        doc_topic = r.metadata.get("topic", "")
        if doc_topic in relevant_topics:
            hits += 1
    return hits / k


def calculate_mrr(results, relevant_topics):
    """Calculate Mean Reciprocal Rank."""
    for rank, r in enumerate(results, 1):
        doc_topic = r.metadata.get("topic", "")
        if doc_topic in relevant_topics:
            return 1.0 / rank
    return 0.0


def run_benchmark():
    print("=" * 60)
    print("HYBRID RETRIEVAL BENCHMARK")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/4] Initializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    print("[2/4] Initializing sparse encoder (SPLADE)...")
    sparse_encoder = SpladeEncoder()
    
    print("[3/4] Creating hybrid vector store...")
    vectorstore = QdrantHybridStore(
        collection_name="retrieval_benchmark",
        dense_dimensions=768,
        recreate_collection=True,  # Fresh start
    )
    
    print("[4/4] Creating hybrid retriever...")
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder=sparse_encoder,
        top_k=5,
    )
    
    # Index documents
    print(f"\nIndexing {len(TEST_DOCUMENTS)} documents...")
    texts = [d["text"] for d in TEST_DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in TEST_DOCUMENTS]
    
    start = time.perf_counter()
    retriever.add_documents(texts, metadatas)
    index_time = time.perf_counter() - start
    print(f"Indexing completed in {index_time:.2f}s")
    
    # Benchmark each mode
    modes = ["dense", "sparse", "hybrid"]
    results_by_mode = {mode: {"latencies": [], "precisions": [], "mrrs": []} for mode in modes}
    
    print(f"\n{'=' * 60}")
    print("Running queries...")
    print("=" * 60)
    
    for query_data in TEST_QUERIES:
        query = query_data["query"]
        relevant_topics = query_data["relevant_topics"]
        
        print(f"\nQuery: {query[:50]}...")
        
        for mode in modes:
            start = time.perf_counter()
            results = retriever.retrieve(query, top_k=5, mode=mode)
            latency = (time.perf_counter() - start) * 1000
            
            precision = calculate_precision(results, relevant_topics, k=3)
            mrr = calculate_mrr(results, relevant_topics)
            
            results_by_mode[mode]["latencies"].append(latency)
            results_by_mode[mode]["precisions"].append(precision)
            results_by_mode[mode]["mrrs"].append(mrr)
            
            # Show top result
            top_match = results[0] if results else None
            top_preview = top_match.content[:40] + "..." if top_match else "N/A"
            marker = "✓" if top_match and top_match.metadata.get("topic") in relevant_topics else "✗"
            
            print(f"  {mode:8s}: {latency:6.1f}ms | P@3={precision:.2f} | MRR={mrr:.2f} | {marker} {top_preview}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<10} {'Latency':<15} {'P@3':<10} {'MRR':<10}")
    print("-" * 45)
    
    for mode in modes:
        data = results_by_mode[mode]
        avg_lat = mean(data["latencies"])
        std_lat = stdev(data["latencies"]) if len(data["latencies"]) > 1 else 0
        avg_p = mean(data["precisions"])
        avg_mrr = mean(data["mrrs"])
        
        print(f"{mode:<10} {avg_lat:>6.1f}ms ±{std_lat:<5.1f} {avg_p:<10.3f} {avg_mrr:<10.3f}")
    
    # Winner
    print(f"\n{'-' * 45}")
    best_quality = max(modes, key=lambda m: mean(results_by_mode[m]["mrrs"]))
    best_speed = min(modes, key=lambda m: mean(results_by_mode[m]["latencies"]))
    
    print(f"Best Quality (MRR): {best_quality}")
    print(f"Best Speed:         {best_speed}")
    
    # Cleanup
    print(f"\nCleaning up benchmark collection...")
    vectorstore._client.delete_collection("retrieval_benchmark")
    print("Done!")


if __name__ == "__main__":
    run_benchmark()
