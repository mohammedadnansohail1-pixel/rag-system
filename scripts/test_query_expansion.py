"""
Test query expansion impact on retrieval.

Compares:
1. No expansion (baseline)
2. Synonym expansion
3. HyDE expansion
"""
import sys
import hashlib
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import SECLoader
from src.chunkers.factory import ChunkerFactory
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.generation.factory import LLMFactory
from src.retrieval.query_expansion import SynonymExpander, HyDEExpander
from src.retrieval.base import RetrievalResult
from src.evaluation import EvaluationDataset, RetrievalEvaluator


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


class ExpandedRetriever:
    """Retriever with query expansion."""
    
    def __init__(self, embeddings, vectorstore, expander=None, use_hyde=False):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.expander = expander
        self.use_hyde = use_hyde
    
    def retrieve(self, query, top_k=5):
        if self.expander:
            expanded = self.expander.expand(query)
            # For HyDE, embed the hypothetical doc
            # For others, embed the expanded query
            search_text = expanded.expanded if self.use_hyde else expanded.expanded
        else:
            search_text = query
        
        query_emb = self.embeddings.embed_text(search_text)
        results = self.vectorstore.search(query_emb, top_k=top_k)
        
        return [
            RetrievalResult(content=r.content, metadata=r.metadata, score=r.score)
            for r in results
        ]
    
    def health_check(self):
        return True


def main():
    print("=" * 70)
    print("QUERY EXPANSION EVALUATION")
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
    llm = LLMFactory.create("ollama", model="llama3.2")
    
    # Create vector store and index
    print("\n[3/4] Indexing...")
    vectorstore = QdrantVectorStore(collection_name="expansion_test", dimensions=768)
    emb_list = embeddings.embed_batch(clean_texts)
    vectorstore.add(texts=clean_texts, embeddings=emb_list)
    print(f"  Indexed {len(clean_texts)} chunks")
    
    # Create expanders
    synonym_expander = SynonymExpander()
    hyde_expander = HyDEExpander(llm=llm, domain="financial")
    
    # Create retrievers
    retrievers = {
        "No Expansion": ExpandedRetriever(embeddings, vectorstore),
        "Synonym Expansion": ExpandedRetriever(embeddings, vectorstore, synonym_expander),
        "HyDE (Financial)": ExpandedRetriever(embeddings, vectorstore, hyde_expander, use_hyde=True),
    }
    
    # Test
    print("\n[4/4] Evaluating...")
    results = {}
    
    for name, retriever in retrievers.items():
        print(f"\n  Testing: {name}")
        
        evaluator = RetrievalEvaluator(
            retriever,
            id_extractor=lambda r: stable_hash(r.content),
        )
        
        start = time.perf_counter()
        result = evaluator.evaluate(dataset, k=10, verbose=False)
        elapsed = time.perf_counter() - start
        
        results[name] = result
        print(f"    MRR: {result.mean_mrr:.3f}, Hit Rate: {result.hit_rate:.1%}, Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(RetrievalEvaluator.comparison_table(results))
    
    # Detailed comparison
    print("\n" + "-" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("-" * 70)
    
    baseline = results["No Expansion"]
    
    for name, result in results.items():
        if name == "No Expansion":
            continue
        
        mrr_delta = result.mean_mrr - baseline.mean_mrr
        hit_delta = result.hit_rate - baseline.hit_rate
        recall_delta = result.mean_recall - baseline.mean_recall
        
        print(f"\n{name}:")
        print(f"  MRR: {baseline.mean_mrr:.3f} → {result.mean_mrr:.3f} ({mrr_delta:+.3f})")
        print(f"  Hit Rate: {baseline.hit_rate:.1%} → {result.hit_rate:.1%} ({hit_delta:+.1%})")
        print(f"  Recall@10: {baseline.mean_recall:.3f} → {result.mean_recall:.3f} ({recall_delta:+.3f})")
    
    # Show example expansions
    print("\n" + "-" * 70)
    print("EXAMPLE EXPANSIONS")
    print("-" * 70)
    
    test_queries = [
        "What was Netflix revenue in 2024?",
        "How many employees does Netflix have?",
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        print(f"  Synonym: {synonym_expander.expand(q).expanded}")
        hyde_result = hyde_expander.expand(q)
        print(f"  HyDE: {hyde_result.expanded[:100]}...")
    
    # Cleanup
    print("\n\nCleaning up...")
    vectorstore._client.delete_collection("expansion_test")
    print("Done!")


if __name__ == "__main__":
    main()
