"""Evaluation module for measuring retrieval and RAG quality."""
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
    f1_at_k,
)
from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    EvaluationResult,
    QueryResult,
    TestCase,
)
from src.evaluation.dataset import (
    EvaluationDataset,
    create_dataset_from_qa_pairs,
)

__all__ = [
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "f1_at_k",
    # Evaluator
    "RetrievalEvaluator",
    "EvaluationResult",
    "QueryResult",
    "TestCase",
    # Dataset
    "EvaluationDataset",
    "create_dataset_from_qa_pairs",
]
