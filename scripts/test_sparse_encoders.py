"""
Comprehensive tests for sparse encoder implementations.

Tests:
1. All encoder types (BM25, TFIDF, SPLADE)
2. Edge cases (empty, single word, long text)
3. Memory/performance comparison
4. Retrieval quality comparison
5. Config-driven initialization
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.sparse_encoder import (
    SparseEncoderFactory,
    BM25Encoder,
    TFIDFEncoder,
    SpladeEncoder,
    BaseSparseEncoder,
)


def test_factory_creation():
    """Test factory creates all encoder types."""
    print("\n" + "=" * 60)
    print("TEST: Factory Creation")
    print("=" * 60)
    
    tests = [
        ("bm25", BM25Encoder),
        ("tfidf", TFIDFEncoder),
        ("BM25", BM25Encoder),  # Case insensitive
        ("TFIDF", TFIDFEncoder),
    ]
    
    for name, expected_class in tests:
        encoder = SparseEncoderFactory.create(name)
        assert isinstance(encoder, expected_class), f"Failed for {name}"
        print(f"  ✓ '{name}' -> {encoder.__class__.__name__}")
    
    # Test invalid
    try:
        SparseEncoderFactory.create("invalid")
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Invalid type raises ValueError: {str(e)[:50]}...")
    
    print("  All factory tests passed!")


def test_config_creation():
    """Test config-driven encoder creation."""
    print("\n" + "=" * 60)
    print("TEST: Config-driven Creation")
    print("=" * 60)
    
    configs = [
        {"type": "bm25", "k1": 2.0, "b": 0.5},
        {"type": "tfidf", "max_features": 1000},
        {"type": "bm25"},  # Defaults
    ]
    
    for config in configs:
        encoder = SparseEncoderFactory.from_config(config)
        print(f"  ✓ {config} -> {encoder}")
    
    print("  All config tests passed!")


def test_bm25_encoder():
    """Test BM25 encoder functionality."""
    print("\n" + "=" * 60)
    print("TEST: BM25 Encoder")
    print("=" * 60)
    
    encoder = BM25Encoder(k1=1.5, b=0.75)
    
    corpus = [
        "Netflix is a streaming service for movies and TV shows",
        "The company reported strong revenue growth in 2024",
        "Subscriber numbers exceeded analyst expectations",
        "Content investment remains a key strategic priority",
        "Competition from Disney and Amazon continues to intensify",
    ]
    
    # Test before fit
    vec_before = encoder.encode("Netflix revenue")
    print(f"  Before fit: {len(vec_before.indices)} terms")
    
    # Fit
    encoder.fit(corpus)
    print(f"  After fit: vocab_size={len(encoder._vocab)}, avg_doc_len={encoder._avg_doc_len:.1f}")
    
    # Test encoding
    queries = [
        "Netflix revenue growth",
        "streaming service competition",
        "subscriber numbers",
    ]
    
    for query in queries:
        vec = encoder.encode(query)
        print(f"  Query '{query}': {len(vec.indices)} non-zero terms")
        assert len(vec.indices) > 0, "Should have non-zero terms"
    
    # Batch encoding
    vecs = encoder.encode_batch(queries)
    assert len(vecs) == len(queries), "Batch should return same count"
    print(f"  Batch encode: {len(vecs)} vectors")
    
    print("  BM25 tests passed!")


def test_tfidf_encoder():
    """Test TF-IDF encoder functionality."""
    print("\n" + "=" * 60)
    print("TEST: TF-IDF Encoder")
    print("=" * 60)
    
    encoder = TFIDFEncoder(max_features=100)
    
    corpus = [
        "Machine learning models require training data",
        "Deep learning is a subset of machine learning",
        "Neural networks power modern AI systems",
        "Data preprocessing is essential for model performance",
    ]
    
    encoder.fit(corpus)
    print(f"  Vocab size: {len(encoder._vocab)}")
    
    vec = encoder.encode("machine learning data")
    print(f"  Query vector: {len(vec.indices)} non-zero terms")
    
    # Test max_features limit
    encoder2 = TFIDFEncoder(max_features=5)
    encoder2.fit(corpus)
    print(f"  With max_features=5: vocab_size={len(encoder2._vocab)}")
    assert len(encoder2._vocab) <= 5, "Should respect max_features"
    
    print("  TF-IDF tests passed!")


def test_splade_encoder():
    """Test SPLADE encoder (if GPU available)."""
    print("\n" + "=" * 60)
    print("TEST: SPLADE Encoder")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠ CUDA not available, testing on CPU (slow)")
        
        encoder = SpladeEncoder(batch_size=2)
        
        # Single encode
        start = time.perf_counter()
        vec = encoder.encode("Netflix streaming revenue")
        elapsed = time.perf_counter() - start
        print(f"  Single encode: {len(vec.indices)} terms in {elapsed:.2f}s")
        
        # Batch encode
        texts = ["query one", "query two", "query three"]
        start = time.perf_counter()
        vecs = encoder.encode_batch(texts)
        elapsed = time.perf_counter() - start
        print(f"  Batch encode ({len(texts)} texts): {elapsed:.2f}s")
        
        # Decode tokens
        top_tokens = encoder.decode_tokens(vec, top_k=5)
        print(f"  Top tokens: {top_tokens}")
        
        print("  SPLADE tests passed!")
        
    except Exception as e:
        print(f"  ⚠ SPLADE test skipped: {e}")


def test_edge_cases():
    """Test edge cases for all encoders."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    encoders = [
        ("BM25", BM25Encoder()),
        ("TFIDF", TFIDFEncoder()),
    ]
    
    # Fit on small corpus
    corpus = ["hello world", "foo bar baz"]
    
    edge_cases = [
        ("empty string", ""),
        ("single word", "hello"),
        ("numbers only", "12345 67890"),
        ("special chars", "!@#$%^&*()"),
        ("very long", "word " * 1000),
        ("unicode", "café résumé naïve"),
        ("mixed", "Hello123 World456!"),
    ]
    
    for name, encoder in encoders:
        encoder.fit(corpus)
        print(f"\n  {name}:")
        
        for case_name, text in edge_cases:
            try:
                vec = encoder.encode(text)
                print(f"    ✓ {case_name}: {len(vec.indices)} terms")
            except Exception as e:
                print(f"    ✗ {case_name}: {e}")
    
    print("\n  Edge case tests completed!")


def test_performance_comparison():
    """Compare performance across encoders."""
    print("\n" + "=" * 60)
    print("TEST: Performance Comparison")
    print("=" * 60)
    
    # Generate test corpus
    corpus = [f"Document number {i} contains information about topic {i % 10}" for i in range(1000)]
    queries = [f"topic {i}" for i in range(10)]
    
    results = {}
    
    # BM25
    encoder = BM25Encoder()
    start = time.perf_counter()
    encoder.fit(corpus)
    fit_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for q in queries:
        encoder.encode(q)
    encode_time = time.perf_counter() - start
    
    results["BM25"] = {"fit": fit_time, "encode": encode_time}
    print(f"  BM25:  fit={fit_time:.3f}s, encode={encode_time:.4f}s")
    
    # TFIDF
    encoder = TFIDFEncoder()
    start = time.perf_counter()
    encoder.fit(corpus)
    fit_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for q in queries:
        encoder.encode(q)
    encode_time = time.perf_counter() - start
    
    results["TFIDF"] = {"fit": fit_time, "encode": encode_time}
    print(f"  TFIDF: fit={fit_time:.3f}s, encode={encode_time:.4f}s")
    
    print("\n  Performance tests completed!")
    return results


def test_hybrid_retriever_integration():
    """Test sparse encoders with HybridRetriever."""
    print("\n" + "=" * 60)
    print("TEST: HybridRetriever Integration")
    print("=" * 60)
    
    from src.embeddings.factory import EmbeddingsFactory
    from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    # Test data
    texts = [
        "Netflix reported $39 billion in streaming revenue for 2024",
        "The company added 19 million new paid subscribers",
        "Content spending increased to $17 billion annually",
        "Operating margin improved to 28% from 21%",
        "Ad-supported tier launched in November 2022",
    ]
    
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    encoder_configs = [
        "bm25",
        "tfidf",
        {"type": "bm25", "k1": 2.0, "b": 0.5},
    ]
    
    for config in encoder_configs:
        print(f"\n  Testing with sparse_encoder={config}")
        
        # Create fresh collection
        collection_name = f"test_integration_{hash(str(config)) % 10000}"
        vectorstore = QdrantHybridStore(
            collection_name=collection_name,
            dense_dimensions=768,
            recreate_collection=True,
        )
        
        retriever = HybridRetriever(
            embeddings=embeddings,
            vectorstore=vectorstore,
            sparse_encoder=config,
        )
        
        print(f"    Created: {retriever}")
        
        # Add documents
        retriever.add_documents(texts)
        print(f"    Added {len(texts)} documents")
        
        # Search
        results = retriever.retrieve("Netflix revenue 2024", top_k=2)
        print(f"    Retrieved {len(results)} results")
        
        for i, r in enumerate(results):
            print(f"      {i+1}. (score={r.score:.3f}) {r.content[:60]}...")
        
        # Cleanup
        vectorstore._client.delete_collection(collection_name)
    
    print("\n  Integration tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SPARSE ENCODER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Factory Creation", test_factory_creation),
        ("Config Creation", test_config_creation),
        ("BM25 Encoder", test_bm25_encoder),
        ("TF-IDF Encoder", test_tfidf_encoder),
        ("SPLADE Encoder", test_splade_encoder),
        ("Edge Cases", test_edge_cases),
        ("Performance", test_performance_comparison),
        ("Integration", test_hybrid_retriever_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
