"""RAG evaluation metrics."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single RAG response.
    
    Attributes:
        query: Original question
        answer: Generated answer
        context_relevance: How relevant retrieved context is to query (0-1)
        answer_relevance: How relevant answer is to query (0-1)
        faithfulness: How grounded answer is in context (0-1)
        overall_score: Combined score (0-1)
    """
    query: str
    answer: str
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    overall_score: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"overall={self.overall_score:.2f}, "
            f"context_rel={self.context_relevance:.2f}, "
            f"answer_rel={self.answer_relevance:.2f}, "
            f"faithfulness={self.faithfulness:.2f})"
        )


@dataclass 
class BatchEvaluationResult:
    """
    Result of evaluating multiple RAG responses.
    
    Attributes:
        results: Individual evaluation results
        avg_context_relevance: Average context relevance
        avg_answer_relevance: Average answer relevance
        avg_faithfulness: Average faithfulness
        avg_overall: Average overall score
        num_samples: Number of samples evaluated
    """
    results: List[EvaluationResult]
    avg_context_relevance: float
    avg_answer_relevance: float
    avg_faithfulness: float
    avg_overall: float
    num_samples: int
    
    def __repr__(self) -> str:
        return (
            f"BatchEvaluationResult("
            f"n={self.num_samples}, "
            f"overall={self.avg_overall:.2f}, "
            f"context={self.avg_context_relevance:.2f}, "
            f"answer={self.avg_answer_relevance:.2f}, "
            f"faith={self.avg_faithfulness:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_samples": self.num_samples,
            "avg_overall": self.avg_overall,
            "avg_context_relevance": self.avg_context_relevance,
            "avg_answer_relevance": self.avg_answer_relevance,
            "avg_faithfulness": self.avg_faithfulness,
        }


def calculate_context_relevance(
    query: str,
    contexts: List[str],
    scores: List[float],
) -> float:
    """
    Calculate context relevance score.
    
    Uses retrieval scores as proxy for relevance.
    Higher scores = more relevant context retrieved.
    
    Args:
        query: Original query
        contexts: Retrieved context chunks
        scores: Retrieval scores for each chunk
        
    Returns:
        Relevance score (0-1)
    """
    if not scores:
        return 0.0
    
    # Use average of top scores, weighted toward higher scores
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    # Weighted combination: 60% average, 40% max
    relevance = 0.6 * avg_score + 0.4 * max_score
    
    return min(1.0, max(0.0, relevance))


def calculate_answer_relevance_simple(
    query: str,
    answer: str,
) -> float:
    """
    Simple answer relevance using keyword overlap.
    
    For production, use LLM-based evaluation.
    
    Args:
        query: Original query
        answer: Generated answer
        
    Returns:
        Relevance score (0-1)
    """
    if not answer or not query:
        return 0.0
    
    # Simple keyword overlap
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
    query_words -= stop_words
    answer_words -= stop_words
    
    if not query_words:
        return 0.5  # Default if query is all stop words
    
    overlap = len(query_words & answer_words)
    relevance = min(1.0, overlap / len(query_words))
    
    # Penalize very short answers
    if len(answer.split()) < 10:
        relevance *= 0.8
    
    # Bonus for longer, detailed answers
    if len(answer.split()) > 50:
        relevance = min(1.0, relevance * 1.1)
    
    return relevance


def calculate_faithfulness_simple(
    answer: str,
    contexts: List[str],
) -> float:
    """
    Simple faithfulness check using content overlap.
    
    Measures how much of the answer content appears in context.
    For production, use LLM-based evaluation.
    
    Args:
        answer: Generated answer
        contexts: Retrieved context chunks
        
    Returns:
        Faithfulness score (0-1)
    """
    if not answer or not contexts:
        return 0.0
    
    # Combine all contexts
    combined_context = " ".join(contexts).lower()
    
    # Get significant words from answer (longer words more likely to be content words)
    answer_words = [w.lower() for w in answer.split() if len(w) > 4]
    
    if not answer_words:
        return 0.5
    
    # Count how many answer words appear in context
    found = sum(1 for w in answer_words if w in combined_context)
    faithfulness = found / len(answer_words)
    
    # Check for uncertainty indicators (good sign of not hallucinating)
    uncertainty_phrases = [
        "don't have enough information",
        "not mentioned in",
        "cannot find",
        "based on the context",
        "according to the",
    ]
    
    has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
    if has_uncertainty:
        faithfulness = min(1.0, faithfulness + 0.1)
    
    return min(1.0, max(0.0, faithfulness))


def calculate_overall_score(
    context_relevance: float,
    answer_relevance: float,
    faithfulness: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate weighted overall score.
    
    Args:
        context_relevance: Context relevance score
        answer_relevance: Answer relevance score  
        faithfulness: Faithfulness score
        weights: Optional custom weights
        
    Returns:
        Overall score (0-1)
    """
    if weights is None:
        weights = {
            "context_relevance": 0.3,
            "answer_relevance": 0.3,
            "faithfulness": 0.4,  # Faithfulness weighted highest
        }
    
    total_weight = sum(weights.values())
    
    score = (
        weights.get("context_relevance", 0.3) * context_relevance +
        weights.get("answer_relevance", 0.3) * answer_relevance +
        weights.get("faithfulness", 0.4) * faithfulness
    ) / total_weight
    
    return score
