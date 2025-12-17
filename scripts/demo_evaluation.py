"""
Demo: Retrieval Evaluation Framework

Compares different retrieval strategies on the same dataset:
- Dense only
- Hybrid (BM25 + Dense)
- Hybrid (TFIDF + Dense)

Shows how to measure and compare retrieval quality systematically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvaluationDataset,
    TestCase,
    RetrievalEvaluator,
)
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.base import RetrievalResult


class DenseRetriever:
    """Simple dense-only retriever for comparison."""
    
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
    print("=" * 60)
    print("RETRIEVAL EVALUATION DEMO")
    print("=" * 60)
    
    # Test corpus - Netflix financial data
    corpus = [
        "Netflix reported $39 billion in streaming revenue for 2024, a 16% increase year over year.",
        "The company added 19 million new paid subscribers in 2024, exceeding analyst expectations.",
        "Content spending increased to $17 billion annually, focused on original productions.",
        "Operating margin improved to 28% from 21%, driven by cost optimization.",
        "Ad-supported tier launched in November 2022, now has 40 million monthly active users.",
        "Ted Sarandos and Greg Peters serve as co-CEOs of Netflix since 2023.",
        "Netflix was founded in 1997 by Reed Hastings and Marc Randolph in Scotts Valley, California.",
        "The company operates in over 190 countries with 280 million paid memberships worldwide.",
        "Netflix stock price reached all-time high of $900 in late 2024.",
        "The company's main competitors include Disney+, Amazon Prime Video, and HBO Max.",
        "Netflix invested heavily in gaming, acquiring several game studios in 2023.",
        "International markets now account for over 60% of Netflix's total revenue.",
    ]
    
    print(f"\nCorpus: {len(corpus)} documents")
    
    # Initialize embeddings
    print("\nInitializing components...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    # Create retrievers
    print("Setting up retrievers...")
    
    # 1. Dense-only retriever
    dense_store = QdrantVectorStore(
        collection_name="eval_dense",
        dimensions=768,
    )
    dense_retriever = DenseRetriever(embeddings, dense_store)
    dense_ids = dense_retriever.add_documents(corpus)
    print(f"  Dense retriever: indexed {len(dense_ids)} docs")
    
    # 2. Hybrid with BM25
    hybrid_bm25_store = QdrantHybridStore(
        collection_name="eval_hybrid_bm25",
        dense_dimensions=768,
        recreate_collection=True,
    )
    hybrid_bm25 = HybridRetriever(
        embeddings=embeddings,
        vectorstore=hybrid_bm25_store,
        sparse_encoder="bm25",
    )
    hybrid_bm25.add_documents(corpus)
    print(f"  Hybrid (BM25) retriever: indexed")
    
    # 3. Hybrid with TFIDF
    hybrid_tfidf_store = QdrantHybridStore(
        collection_name="eval_hybrid_tfidf",
        dense_dimensions=768,
        recreate_collection=True,
    )
    hybrid_tfidf = HybridRetriever(
        embeddings=embeddings,
        vectorstore=hybrid_tfidf_store,
        sparse_encoder="tfidf",
    )
    hybrid_tfidf.add_documents(corpus)
    print(f"  Hybrid (TFIDF) retriever: indexed")
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    # Use content hash for consistent IDs across retrievers
    corpus_ids = [str(hash(doc)) for doc in corpus]
    
    dataset = EvaluationDataset(
        name="netflix_financial",
        description="Questions about Netflix financial and company data",
        test_cases=[
            TestCase(
                query="What was Netflix's revenue in 2024?",
                relevant_ids={corpus_ids[0]},
            ),
            TestCase(
                query="How many new subscribers did Netflix add?",
                relevant_ids={corpus_ids[1]},
            ),
            TestCase(
                query="Who is the CEO of Netflix?",
                relevant_ids={corpus_ids[5]},
            ),
            TestCase(
                query="When and where was Netflix founded?",
                relevant_ids={corpus_ids[6]},
            ),
            TestCase(
                query="What is Netflix's content budget?",
                relevant_ids={corpus_ids[2]},
            ),
            TestCase(
                query="How many countries does Netflix operate in?",
                relevant_ids={corpus_ids[7]},
            ),
            TestCase(
                query="What are Netflix's main competitors?",
                relevant_ids={corpus_ids[9]},
            ),
            TestCase(
                query="What is Netflix's operating margin?",
                relevant_ids={corpus_ids[3]},
            ),
            TestCase(
                query="How many ad-tier users does Netflix have?",
                relevant_ids={corpus_ids[4]},
            ),
            TestCase(
                query="What is Netflix's international revenue share?",
                relevant_ids={corpus_ids[11]},
            ),
        ],
    )
    
    print(f"{dataset.summary()}")
    
    # Save dataset for reuse
    dataset.save("data/eval_datasets/netflix_financial.json")
    print("  Saved to data/eval_datasets/netflix_financial.json")
    
    # Evaluate each retriever
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    retrievers = {
        "Dense Only": dense_retriever,
        "Hybrid (BM25)": hybrid_bm25,
        "Hybrid (TFIDF)": hybrid_tfidf,
    }
    
    results = {}
    for name, retriever in retrievers.items():
        print(f"\nEvaluating: {name}")
        evaluator = RetrievalEvaluator(
            retriever,
            id_extractor=lambda r: str(hash(r.content)),
        )
        results[name] = evaluator.evaluate(dataset, k=3, verbose=False)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(RetrievalEvaluator.comparison_table(results))
    
    # Print detailed results for best performer
    best_name = max(results.keys(), key=lambda k: results[k].mean_mrr)
    print(f"\nBest performer: {best_name}")
    print(results[best_name].summary())
    
    # Show example queries
    print("\n" + "=" * 60)
    print("EXAMPLE RETRIEVALS")
    print("=" * 60)
    
    example_queries = [
        "Netflix revenue 2024",
        "Who runs Netflix?",
    ]
    
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        for name, retriever in retrievers.items():
            results_list = retriever.retrieve(query, top_k=1)
            if results_list:
                print(f"  {name}: {results_list[0].content[:60]}...")
    
    # Cleanup
    print("\n\nCleaning up...")
    dense_store._client.delete_collection("eval_dense")
    hybrid_bm25_store._client.delete_collection("eval_hybrid_bm25")
    hybrid_tfidf_store._client.delete_collection("eval_hybrid_tfidf")
    
    print("Done!")


if __name__ == "__main__":
    # Ensure output directory exists
    Path("data/eval_datasets").mkdir(parents=True, exist_ok=True)
    main()
