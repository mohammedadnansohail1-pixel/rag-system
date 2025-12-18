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

# Semantic faithfulness (better than n-gram)
from src.evaluation.metrics import calculate_faithfulness_semantic

# NLI-based faithfulness (production-grade)
from src.evaluation.faithfulness_nli import (
    NLIFaithfulnessEvaluator,
    FaithfulnessResult,
    calculate_faithfulness_nli,
    NLI_AVAILABLE,
)
