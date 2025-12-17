"""Retrieval evaluation framework."""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import statistics

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    f1_at_k,
    average_precision,
)
from src.evaluation.dataset import EvaluationDataset, TestCase

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Results for a single query evaluation."""
    query: str
    retrieved_ids: List[str]
    retrieved_contents: List[str]
    scores: List[float]
    relevant_ids: set
    latency_ms: float
    
    # Metrics (computed after creation)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mrr_score: float = 0.0
    ndcg: float = 0.0
    hit: bool = False


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    dataset_name: str
    num_queries: int
    k: int
    
    # Aggregate metrics
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_mrr: float
    mean_ndcg: float
    hit_rate: float
    map_score: float  # Mean Average Precision
    
    # Latency stats
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    
    # Per-query results (optional, can be large)
    query_results: List[QueryResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
╔══════════════════════════════════════════════════════════╗
║  RETRIEVAL EVALUATION RESULTS                            ║
╠══════════════════════════════════════════════════════════╣
║  Dataset: {self.dataset_name:<46} ║
║  Queries: {self.num_queries:<46} ║
║  K: {self.k:<52} ║
╠══════════════════════════════════════════════════════════╣
║  RETRIEVAL QUALITY                                       ║
║  ──────────────────────────────────────────────────────  ║
║  Precision@{self.k}: {self.mean_precision:<43.3f} ║
║  Recall@{self.k}: {self.mean_recall:<46.3f} ║
║  F1@{self.k}: {self.mean_f1:<50.3f} ║
║  MRR: {self.mean_mrr:<51.3f} ║
║  NDCG@{self.k}: {self.mean_ndcg:<48.3f} ║
║  Hit Rate@{self.k}: {self.hit_rate:<44.3f} ║
║  MAP: {self.map_score:<51.3f} ║
╠══════════════════════════════════════════════════════════╣
║  LATENCY                                                 ║
║  ──────────────────────────────────────────────────────  ║
║  P50: {self.latency_p50_ms:<48.1f}ms ║
║  P95: {self.latency_p95_ms:<48.1f}ms ║
║  P99: {self.latency_p99_ms:<48.1f}ms ║
║  Mean: {self.latency_mean_ms:<47.1f}ms ║
╚══════════════════════════════════════════════════════════╝
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "num_queries": self.num_queries,
            "k": self.k,
            "metrics": {
                "precision": self.mean_precision,
                "recall": self.mean_recall,
                "f1": self.mean_f1,
                "mrr": self.mean_mrr,
                "ndcg": self.mean_ndcg,
                "hit_rate": self.hit_rate,
                "map": self.map_score,
            },
            "latency": {
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
                "mean_ms": self.latency_mean_ms,
            },
        }


class RetrievalEvaluator:
    """
    Evaluate retrieval system quality.
    
    Usage:
        evaluator = RetrievalEvaluator(retriever)
        
        # Evaluate on dataset
        results = evaluator.evaluate(dataset, k=5)
        print(results.summary())
        
        # Evaluate with custom ID extractor
        results = evaluator.evaluate(
            dataset,
            id_extractor=lambda r: r.metadata.get('chunk_id'),
        )
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        id_extractor: Optional[Callable[[RetrievalResult], str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            retriever: Retriever to evaluate
            id_extractor: Function to extract doc ID from RetrievalResult
                          Default uses hash of content
        """
        self.retriever = retriever
        self.id_extractor = id_extractor or self._default_id_extractor
    
    def _default_id_extractor(self, result: RetrievalResult) -> str:
        """Default: use metadata id or content hash."""
        if "id" in result.metadata:
            return result.metadata["id"]
        if "chunk_id" in result.metadata:
            return result.metadata["chunk_id"]
        if "source" in result.metadata:
            # Use source + content hash for uniqueness
            return f"{result.metadata['source']}:{hash(result.content)}"
        return str(hash(result.content))
    
    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 5,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate retriever on dataset.
        
        Args:
            dataset: Evaluation dataset with test cases
            k: Number of results to retrieve per query
            verbose: Whether to print progress
            
        Returns:
            EvaluationResult with aggregate metrics
        """
        logger.info(f"Evaluating on dataset '{dataset.name}' ({len(dataset)} queries)")
        
        query_results = []
        latencies = []
        
        for i, test_case in enumerate(dataset):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(dataset)}")
            
            # Retrieve with timing
            start = time.perf_counter()
            results = self.retriever.retrieve(test_case.query, top_k=k)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            
            # Extract IDs
            retrieved_ids = [self.id_extractor(r) for r in results]
            retrieved_contents = [r.content for r in results]
            scores = [r.score for r in results]
            
            # Compute per-query metrics
            qr = QueryResult(
                query=test_case.query,
                retrieved_ids=retrieved_ids,
                retrieved_contents=retrieved_contents,
                scores=scores,
                relevant_ids=test_case.relevant_ids,
                latency_ms=latency_ms,
            )
            
            qr.precision = precision_at_k(retrieved_ids, test_case.relevant_ids, k)
            qr.recall = recall_at_k(retrieved_ids, test_case.relevant_ids, k)
            qr.f1 = f1_at_k(retrieved_ids, test_case.relevant_ids, k)
            qr.mrr_score = mrr(retrieved_ids, test_case.relevant_ids)
            qr.ndcg = ndcg_at_k(retrieved_ids, test_case.relevance_scores, k)
            qr.hit = hit_rate(retrieved_ids, test_case.relevant_ids, k) > 0
            
            query_results.append(qr)
        
        # Aggregate metrics
        n = len(query_results)
        
        result = EvaluationResult(
            dataset_name=dataset.name,
            num_queries=n,
            k=k,
            mean_precision=sum(qr.precision for qr in query_results) / n if n else 0,
            mean_recall=sum(qr.recall for qr in query_results) / n if n else 0,
            mean_f1=sum(qr.f1 for qr in query_results) / n if n else 0,
            mean_mrr=sum(qr.mrr_score for qr in query_results) / n if n else 0,
            mean_ndcg=sum(qr.ndcg for qr in query_results) / n if n else 0,
            hit_rate=sum(1 for qr in query_results if qr.hit) / n if n else 0,
            map_score=sum(
                average_precision(qr.retrieved_ids, qr.relevant_ids)
                for qr in query_results
            ) / n if n else 0,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=self._percentile(latencies, 95) if latencies else 0,
            latency_p99_ms=self._percentile(latencies, 99) if latencies else 0,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            query_results=query_results,
        )
        
        logger.info(f"Evaluation complete: P@{k}={result.mean_precision:.3f}, MRR={result.mean_mrr:.3f}")
        return result
    
    def evaluate_single(
        self,
        query: str,
        relevant_ids: set,
        k: int = 5,
    ) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            query: Search query
            relevant_ids: Set of relevant document IDs
            k: Number of results
            
        Returns:
            QueryResult with metrics
        """
        start = time.perf_counter()
        results = self.retriever.retrieve(query, top_k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        
        retrieved_ids = [self.id_extractor(r) for r in results]
        
        qr = QueryResult(
            query=query,
            retrieved_ids=retrieved_ids,
            retrieved_contents=[r.content for r in results],
            scores=[r.score for r in results],
            relevant_ids=relevant_ids,
            latency_ms=latency_ms,
        )
        
        qr.precision = precision_at_k(retrieved_ids, relevant_ids, k)
        qr.recall = recall_at_k(retrieved_ids, relevant_ids, k)
        qr.f1 = f1_at_k(retrieved_ids, relevant_ids, k)
        qr.mrr_score = mrr(retrieved_ids, relevant_ids)
        qr.hit = hit_rate(retrieved_ids, relevant_ids, k) > 0
        
        return qr
    
    def compare_retrievers(
        self,
        retrievers: Dict[str, BaseRetriever],
        dataset: EvaluationDataset,
        k: int = 5,
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple retrievers on same dataset.
        
        Args:
            retrievers: Dict of name -> retriever
            dataset: Evaluation dataset
            k: Number of results
            
        Returns:
            Dict of name -> EvaluationResult
        """
        results = {}
        
        for name, retriever in retrievers.items():
            logger.info(f"Evaluating retriever: {name}")
            evaluator = RetrievalEvaluator(retriever, self.id_extractor)
            results[name] = evaluator.evaluate(dataset, k=k, verbose=False)
        
        return results
    
    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]
    
    @staticmethod
    def comparison_table(results: Dict[str, EvaluationResult]) -> str:
        """Generate comparison table for multiple retrievers."""
        if not results:
            return "No results to compare"
        
        headers = ["Retriever", "P@K", "R@K", "F1@K", "MRR", "NDCG", "Hit Rate", "Latency"]
        rows = []
        
        for name, r in results.items():
            rows.append([
                name,
                f"{r.mean_precision:.3f}",
                f"{r.mean_recall:.3f}",
                f"{r.mean_f1:.3f}",
                f"{r.mean_mrr:.3f}",
                f"{r.mean_ndcg:.3f}",
                f"{r.hit_rate:.3f}",
                f"{r.latency_p50_ms:.0f}ms",
            ])
        
        # Format table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        
        def format_row(row):
            return "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) + " |"
        
        lines = [separator, format_row(headers), separator]
        for row in rows:
            lines.append(format_row(row))
        lines.append(separator)
        
        return "\n".join(lines)
