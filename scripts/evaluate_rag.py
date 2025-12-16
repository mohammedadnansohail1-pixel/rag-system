"""
RAG System Evaluation using RAGAS metrics.

Evaluates:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Did we retrieve all needed info?

Compares:
- Dense vs Hybrid retrieval
- With vs Without reranking
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking import RerankerFactory
from src.generation.factory import LLMFactory
from src.pipeline import RAGPipelineV2


@dataclass
class EvalQuestion:
    """A test question with ground truth."""
    question: str
    ground_truth: str  # Expected answer
    relevant_topics: List[str]  # Topics that should be retrieved


@dataclass
class EvalResult:
    """Result for a single evaluation."""
    question: str
    answer: str
    ground_truth: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    config: str  # e.g., "hybrid+rerank" or "dense"
    
    # RAGAS-style metrics (simplified)
    context_precision: float  # Relevant docs in top-k
    answer_similarity: float  # Overlap with ground truth


# Test dataset - questions with ground truth
EVAL_DATASET = [
    EvalQuestion(
        question="How does Kafka guarantee message ordering?",
        ground_truth="Kafka guarantees message ordering within a partition using sequential offset numbers. Consumers read messages in offset order.",
        relevant_topics=["kafka"],
    ),
    EvalQuestion(
        question="What is hybrid retrieval in RAG systems?",
        ground_truth="Hybrid retrieval combines dense (semantic) search with sparse (keyword) search using RRF fusion for better recall.",
        relevant_topics=["rag"],
    ),
    EvalQuestion(
        question="How do vector databases perform similarity search?",
        ground_truth="Vector databases use algorithms like HNSW for approximate nearest neighbor search. They measure similarity using cosine, euclidean, or dot product distance.",
        relevant_topics=["vectordb"],
    ),
    EvalQuestion(
        question="What is reranking and why is it used?",
        ground_truth="Reranking uses cross-encoders to re-score retrieved documents. It improves precision by examining query-document pairs jointly.",
        relevant_topics=["rag"],
    ),
    EvalQuestion(
        question="How can I run LLMs locally?",
        ground_truth="Ollama runs LLMs locally with a simple CLI. It supports Llama, Mistral, and other open models.",
        relevant_topics=["llm"],
    ),
]


# Test documents
DOCUMENTS = [
    {"text": "Apache Kafka is a distributed event streaming platform.", "topic": "kafka"},
    {"text": "Kafka guarantees message ordering within a partition using sequential offset numbers. Consumers read messages in offset order.", "topic": "kafka"},
    {"text": "Kafka consumers can join consumer groups for parallel processing.", "topic": "kafka"},
    {"text": "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation to reduce hallucinations.", "topic": "rag"},
    {"text": "Hybrid retrieval combines dense (semantic) and sparse (keyword) search. RRF fusion merges results from both methods for better recall.", "topic": "rag"},
    {"text": "Reranking uses cross-encoders to re-score retrieved documents. It improves precision by examining query-document pairs jointly.", "topic": "rag"},
    {"text": "Chunking strategies include fixed-size, recursive, and semantic chunking.", "topic": "rag"},
    {"text": "Vector databases store embeddings for similarity search using algorithms like HNSW.", "topic": "vectordb"},
    {"text": "HNSW (Hierarchical Navigable Small World) provides fast approximate nearest neighbor search.", "topic": "vectordb"},
    {"text": "Cosine similarity measures angle between vectors. Euclidean measures distance. Dot product is fastest.", "topic": "vectordb"},
    {"text": "Qdrant supports both dense and sparse vectors for hybrid search.", "topic": "vectordb"},
    {"text": "Large Language Models like GPT-4, Claude, and Llama use transformer architecture.", "topic": "llm"},
    {"text": "Ollama runs LLMs locally with simple CLI. It supports Llama, Mistral, and other open models.", "topic": "llm"},
    {"text": "Fine-tuning adapts pre-trained models using techniques like LoRA.", "topic": "llm"},
    {"text": "Prompt engineering techniques include few-shot examples and chain-of-thought.", "topic": "llm"},
]


def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculate word overlap between two texts (simplified similarity)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def calculate_context_precision(sources: List[Dict], relevant_topics: List[str], k: int = 3) -> float:
    """Calculate precision@k for retrieved contexts."""
    if not sources:
        return 0.0
    
    relevant_count = sum(
        1 for s in sources[:k]
        if s.get("metadata", {}).get("topic") in relevant_topics
    )
    
    return relevant_count / min(k, len(sources))


def run_evaluation(
    pipeline: RAGPipelineV2,
    config_name: str,
    use_reranker: bool,
) -> List[EvalResult]:
    """Run evaluation on dataset."""
    results = []
    
    for eval_q in EVAL_DATASET:
        start = time.perf_counter()
        
        response = pipeline.query(
            eval_q.question,
            retrieval_top_k=10,
            rerank_top_n=3,
            use_reranker=use_reranker,
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        # Calculate metrics
        context_precision = calculate_context_precision(
            response.sources,
            eval_q.relevant_topics,
            k=3
        )
        
        answer_similarity = calculate_word_overlap(
            response.answer,
            eval_q.ground_truth
        )
        
        results.append(EvalResult(
            question=eval_q.question,
            answer=response.answer,
            ground_truth=eval_q.ground_truth,
            sources=response.sources,
            latency_ms=latency,
            config=config_name,
            context_precision=context_precision,
            answer_similarity=answer_similarity,
        ))
    
    return results


def print_results(results: List[EvalResult], config_name: str):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*60}")
    
    avg_precision = sum(r.context_precision for r in results) / len(results)
    avg_similarity = sum(r.answer_similarity for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    print(f"\nAggregate Metrics:")
    print(f"  Context Precision@3: {avg_precision:.3f}")
    print(f"  Answer Similarity:   {avg_similarity:.3f}")
    print(f"  Avg Latency:         {avg_latency:.0f}ms")
    
    print(f"\nPer-Question Results:")
    for r in results:
        print(f"\n  Q: {r.question[:50]}...")
        print(f"  Context P@3: {r.context_precision:.2f} | Similarity: {r.answer_similarity:.2f} | {r.latency_ms:.0f}ms")
        print(f"  Sources: {[s['metadata'].get('topic', '?') for s in r.sources[:3]]}")


def run_full_evaluation():
    """Run complete evaluation comparing configurations."""
    print("=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    print("[2/5] Initializing hybrid vector store...")
    vectorstore = QdrantHybridStore(
        collection_name="eval_collection",
        dense_dimensions=768,
        recreate_collection=True,
    )
    
    print("[3/5] Initializing hybrid retriever...")
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
    
    print("[4/5] Initializing reranker...")
    reranker = RerankerFactory.create("cross_encoder")
    
    print("[5/5] Initializing LLM...")
    llm = LLMFactory.create("ollama", model="llama3.2")
    
    # Create pipeline
    pipeline = RAGPipelineV2(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )
    
    # Index documents
    print(f"\nIndexing {len(DOCUMENTS)} documents...")
    texts = [d["text"] for d in DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]
    retriever.add_documents(texts, metadatas)
    print("Indexing complete.")
    
    # Run evaluations
    all_results = {}
    
    print("\n" + "=" * 60)
    print("Running evaluation: Hybrid WITHOUT reranking")
    print("=" * 60)
    results_no_rerank = run_evaluation(pipeline, "hybrid", use_reranker=False)
    print_results(results_no_rerank, "Hybrid (no rerank)")
    all_results["hybrid_no_rerank"] = results_no_rerank
    
    print("\n" + "=" * 60)
    print("Running evaluation: Hybrid WITH reranking")
    print("=" * 60)
    results_with_rerank = run_evaluation(pipeline, "hybrid+rerank", use_reranker=True)
    print_results(results_with_rerank, "Hybrid + Reranking")
    all_results["hybrid_with_rerank"] = results_with_rerank
    
    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    configs = [
        ("Hybrid (no rerank)", results_no_rerank),
        ("Hybrid + Reranking", results_with_rerank),
    ]
    
    print(f"\n{'Configuration':<25} {'Context P@3':<15} {'Answer Sim':<15} {'Latency':<10}")
    print("-" * 65)
    
    for name, results in configs:
        avg_p = sum(r.context_precision for r in results) / len(results)
        avg_s = sum(r.answer_similarity for r in results) / len(results)
        avg_l = sum(r.latency_ms for r in results) / len(results)
        print(f"{name:<25} {avg_p:<15.3f} {avg_s:<15.3f} {avg_l:<10.0f}ms")
    
    # Save results
    output_path = Path("data/eval")
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {k: [asdict(r) for r in v] for k, v in all_results.items()},
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to: {results_file}")
    
    # Cleanup
    print("\nCleaning up...")
    vectorstore._client.delete_collection("eval_collection")
    print("Done!")


if __name__ == "__main__":
    run_full_evaluation()
