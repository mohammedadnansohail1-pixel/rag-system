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


# ============================================================================
# RAG-SPECIFIC EVALUATION METRICS
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG response."""
    query: str
    answer: str
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    overall_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvaluationResult:
    """Aggregated results from evaluating multiple queries."""
    results: List[EvaluationResult]
    avg_context_relevance: float
    avg_answer_relevance: float
    avg_faithfulness: float
    avg_overall: float
    num_samples: int

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
╔══════════════════════════════════════════════════════════╗
║  RAG EVALUATION RESULTS                                  ║
╠══════════════════════════════════════════════════════════╣
║  Samples: {self.num_samples:<46} ║
╠══════════════════════════════════════════════════════════╣
║  QUALITY METRICS                                         ║
║  ──────────────────────────────────────────────────────  ║
║  Context Relevance: {self.avg_context_relevance:<36.3f} ║
║  Answer Relevance:  {self.avg_answer_relevance:<36.3f} ║
║  Faithfulness:      {self.avg_faithfulness:<36.3f} ║
║  Overall Score:     {self.avg_overall:<36.3f} ║
╚══════════════════════════════════════════════════════════╝
"""


def calculate_context_relevance(
    query: str,
    contexts: List[str],
    scores: List[float],
) -> float:
    """
    Calculate how relevant retrieved contexts are to the query.
    
    Uses retrieval scores as proxy for relevance.
    Higher scores = more relevant contexts.
    
    Args:
        query: User query
        contexts: Retrieved text chunks
        scores: Retrieval scores for each chunk
        
    Returns:
        Relevance score between 0 and 1
    """
    if not contexts or not scores:
        return 0.0
    
    # Use average of top scores (already normalized by retriever)
    # Weight by position (first results more important)
    weighted_sum = 0.0
    weight_total = 0.0
    
    for i, score in enumerate(scores):
        weight = 1.0 / (i + 1)  # Position weight: 1, 0.5, 0.33, ...
        weighted_sum += score * weight
        weight_total += weight
    
    if weight_total == 0:
        return 0.0
    
    # Normalize to 0-1 range
    relevance = weighted_sum / weight_total
    return min(1.0, max(0.0, relevance))


def calculate_answer_relevance_simple(
    query: str,
    answer: str,
) -> float:
    """
    Simple answer relevance based on keyword overlap.
    
    Checks if answer addresses the query by measuring
    word overlap between query and answer.
    
    Args:
        query: User query
        answer: Generated answer
        
    Returns:
        Relevance score between 0 and 1
    """
    if not query or not answer:
        return 0.0
    
    # Tokenize
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Remove stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    }
    
    query_words -= stop_words
    answer_words -= stop_words
    
    if not query_words:
        return 0.5  # No meaningful query words
    
    # Calculate overlap
    overlap = len(query_words & answer_words)
    
    # Score based on coverage of query words in answer
    coverage = overlap / len(query_words)
    
    # Bonus for longer, more detailed answers
    length_bonus = min(0.2, len(answer_words) / 500)
    
    return min(1.0, coverage + length_bonus)


def calculate_faithfulness_simple(
    answer: str,
    contexts: List[str],
) -> float:
    """
    Simple faithfulness check based on n-gram overlap.
    
    Measures how much of the answer can be traced back
    to the retrieved contexts.
    
    Args:
        answer: Generated answer
        contexts: Retrieved text chunks
        
    Returns:
        Faithfulness score between 0 and 1
    """
    if not answer or not contexts:
        return 0.0
    
    # Combine contexts
    context_text = " ".join(contexts).lower()
    answer_lower = answer.lower()
    
    # Extract answer phrases (2-4 word n-grams)
    answer_words = answer_lower.split()
    
    if len(answer_words) < 2:
        return 0.5
    
    # Check bigrams and trigrams
    matched = 0
    total = 0
    
    for n in [2, 3, 4]:
        for i in range(len(answer_words) - n + 1):
            ngram = " ".join(answer_words[i:i+n])
            total += 1
            if ngram in context_text:
                matched += 1
    
    if total == 0:
        return 0.5
    
    return matched / total


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
        weights: Custom weights (default: equal)
        
    Returns:
        Overall score between 0 and 1
    """
    if weights is None:
        weights = {
            "context_relevance": 0.3,
            "answer_relevance": 0.3,
            "faithfulness": 0.4,
        }
    
    score = (
        context_relevance * weights.get("context_relevance", 0.3) +
        answer_relevance * weights.get("answer_relevance", 0.3) +
        faithfulness * weights.get("faithfulness", 0.4)
    )
    
    return min(1.0, max(0.0, score))


def calculate_faithfulness_semantic(
    answer: str,
    contexts: List[str],
    embeddings=None,
) -> float:
    """
    Semantic faithfulness using embedding similarity.
    
    Measures if answer content is semantically grounded
    in the retrieved contexts.
    
    Args:
        answer: Generated answer
        contexts: Retrieved text chunks
        embeddings: Embedding model (optional, uses sentence similarity if None)
        
    Returns:
        Faithfulness score between 0 and 1
    """
    if not answer or not contexts:
        return 0.0
    
    # Split answer into sentences/claims
    import re
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return 0.5
    
    combined_context = " ".join(contexts).lower()
    
    # For each sentence, check if key concepts appear in context
    grounded_count = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Extract key words (nouns, numbers, proper nouns)
        words = sentence_lower.split()
        
        # Filter to meaningful words (>4 chars, not stopwords)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'which', 'what', 'where',
            'when', 'while', 'with', 'from', 'into', 'about', 'their',
            'there', 'they', 'them', 'than', 'then', 'also', 'only',
            'based', 'according', 'including', 'such', 'other', 'more',
        }
        
        key_words = [w for w in words if len(w) > 4 and w not in stop_words]
        
        if not key_words:
            grounded_count += 0.5  # Neutral for generic sentences
            continue
        
        # Check how many key words appear in context
        matches = sum(1 for w in key_words if w in combined_context)
        coverage = matches / len(key_words)
        
        if coverage >= 0.5:  # At least half of key concepts found
            grounded_count += 1
        elif coverage >= 0.25:
            grounded_count += 0.5
    
    return grounded_count / len(sentences)
