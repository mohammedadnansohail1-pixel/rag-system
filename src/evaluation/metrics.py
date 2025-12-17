"""Retrieval evaluation metrics."""
import math
from typing import List, Set, Union


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Precision@K: What fraction of retrieved docs are relevant?
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision score between 0 and 1
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
    return relevant_retrieved / len(retrieved_at_k)


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Recall@K: What fraction of relevant docs were retrieved?
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_ids:
        return 1.0  # No relevant docs means perfect recall
    
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def f1_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    F1@K: Harmonic mean of precision and recall.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        F1 score between 0 and 1
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant result.
    
    Higher is better (1.0 means first result is relevant).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        
    Returns:
        MRR score between 0 and 1
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Hit Rate@K: Did we find at least one relevant doc in top K?
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        1.0 if hit, 0.0 if miss
    """
    retrieved_at_k = set(retrieved_ids[:k])
    return 1.0 if (retrieved_at_k & relevant_ids) else 0.0


def dcg_at_k(
    relevance_scores: List[float],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at K.
    
    Args:
        relevance_scores: List of relevance scores for each position
        k: Number of positions to consider
        
    Returns:
        DCG score
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        # Using log2(i + 2) for positions starting at 0
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_map: dict,
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    Measures ranking quality considering graded relevance.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevance_map: Dict mapping doc_id -> relevance score (0-3 typical)
        k: Number of top results to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    # Get relevance scores for retrieved docs
    relevance_scores = [
        relevance_map.get(doc_id, 0.0)
        for doc_id in retrieved_ids[:k]
    ]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate ideal DCG (perfect ranking)
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def average_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Average Precision: Mean of precision at each relevant doc position.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        
    Returns:
        AP score between 0 and 1
    """
    if not relevant_ids:
        return 1.0
    
    num_relevant = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(relevant_ids)
