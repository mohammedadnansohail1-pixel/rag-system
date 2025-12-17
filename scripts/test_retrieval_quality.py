#!/usr/bin/env python
"""Test retrieval quality across different query types."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore


def test_retrieval(collection_name: str = "sec_filings_hybrid_test"):
    """Run retrieval quality tests."""
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768,
    )
    
    vectorstore = QdrantHybridStore(
        collection_name=collection_name,
        host="localhost",
        port=6333,
        dense_dimensions=768,
        recreate_collection=False,
    )
    
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
    )
    
    # Test cases: (query, expected_ticker, query_type)
    test_cases = [
        # Entity-specific queries
        ("Netflix subscriber growth", "NFLX", "entity"),
        ("Apple revenue breakdown", "AAPL", "entity"),
        ("Netflix content spending", "NFLX", "entity"),
        ("Apple services business", "AAPL", "entity"),
        
        # Conceptual queries (any company is valid)
        ("risks from global economic conditions", None, "concept"),
        ("supply chain disruption risks", None, "concept"),
        ("cybersecurity threats and data protection", None, "concept"),
        ("competition in technology industry", None, "concept"),
        
        # Mixed queries
        ("how does Netflix make money", "NFLX", "mixed"),
        ("Apple's competitive advantages", "AAPL", "mixed"),
    ]
    
    print("=" * 70)
    print("RETRIEVAL QUALITY TEST")
    print("=" * 70)
    
    results = {"entity": [], "concept": [], "mixed": []}
    
    for query, expected, qtype in test_cases:
        retrieved = retriever.retrieve(query, top_k=3, mode="hybrid")
        
        if not retrieved:
            correct = False
            top_ticker = "NONE"
        else:
            top_ticker = retrieved[0].metadata.get("ticker", "?")
            if expected is None:
                correct = True  # Any result is fine for conceptual
            else:
                correct = top_ticker == expected
        
        results[qtype].append(correct)
        status = "✅" if correct else "❌"
        
        print(f"\n{status} {query}")
        print(f"   Expected: {expected or 'any'} | Got: {top_ticker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for qtype in ["entity", "concept", "mixed"]:
        correct = sum(results[qtype])
        total = len(results[qtype])
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {qtype.capitalize():10} queries: {correct}/{total} ({pct:.0f}%)")
    
    all_correct = sum(sum(v) for v in results.values())
    all_total = sum(len(v) for v in results.values())
    overall = all_correct / all_total * 100 if all_total > 0 else 0
    
    print(f"\n  {'Overall':10}: {all_correct}/{all_total} ({overall:.0f}%)")
    
    return overall >= 80  # Pass threshold


if __name__ == "__main__":
    success = test_retrieval()
    sys.exit(0 if success else 1)
