"""Rule-based query complexity classification."""
import re
from typing import Tuple, List
from .base import QueryComplexity


class RuleBasedClassifier:
    """
    Classify query complexity using pattern matching and heuristics.
    
    Fast (~1ms) classification based on:
    - Regex patterns for simple/complex queries
    - Word count heuristics
    - Keyword matching
    """
    
    SIMPLE_PATTERNS: List[str] = [
        r"^(what is|what's|define|definition of)\s+[\w\s]{1,30}\??$",
        r"^(who is|who's)\s+[\w\s]{1,20}\??$",
        r"^(when did|when was)\s+.{1,30}\??$",
        r"^\w+\s+(meaning|definition)\??$",
        r"^(show|list|find|get)\s+\w+\s*\??$",
    ]
    
    COMPLEX_PATTERNS: List[str] = [
        r"\b(compare|comparison|versus|vs\.?|differ|difference)\b",
        r"\b(why|explain why|reason)\b.*\b(and|but|however)\b",
        r"\b(if|assuming|given that|suppose)\b.*\b(then|would|should)\b",
        r"\b(pros and cons|advantages and disadvantages|trade-?offs?)\b",
        r"\b(step by step|how to)\b.*\b(and|then|after|next)\b",
        r"\b(relationship|correlation|impact|affect|effect)\b.*\b(between|on)\b",
        r"(\?.*){2,}",
        r"\b(analyze|evaluate|assess|recommend)\b.*\b(and|or|with)\b",
    ]
    
    COMPLEXITY_BOOSTERS: List[str] = [
        "analyze", "evaluate", "assess", "recommend", "suggest",
        "implications", "consequences", "strategy", "optimize",
        "multiple", "several", "various", "all", "each",
    ]
    
    SIMPLICITY_INDICATORS: List[str] = [
        "what", "who", "when", "where", "define", "name",
        "list", "show", "find", "get", "tell", "is",
    ]
    
    def __init__(self):
        """Compile regex patterns for efficiency."""
        self._simple_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS
        ]
        self._complex_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS
        ]
    
    def classify(self, query: str) -> Tuple[QueryComplexity, float]:
        """Classify query complexity."""
        if not query or not query.strip():
            return QueryComplexity.SIMPLE, 1.0
        
        query = query.strip()
        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)
        
        simple_score = self._calculate_simple_score(query, query_lower, words, word_count)
        complex_score = self._calculate_complex_score(query, query_lower, words, word_count)
        
        return self._decide(simple_score, complex_score)
    
    def _calculate_simple_score(
        self, query: str, query_lower: str, words: List[str], word_count: int
    ) -> float:
        score = 0.0
        
        for pattern in self._simple_patterns:
            if pattern.search(query):
                score += 0.4
                break
        
        if word_count <= 4:
            score += 0.3
        elif word_count <= 6:
            score += 0.15
        
        connectors = {"and", "or", "vs", "versus", "compare"}
        if word_count <= 4 and not any(w in connectors for w in words):
            score += 0.2
        
        matches = sum(1 for kw in self.SIMPLICITY_INDICATORS if kw in words)
        score += min(matches * 0.1, 0.25)
        
        return min(score, 1.0)
    
    def _calculate_complex_score(
        self, query: str, query_lower: str, words: List[str], word_count: int
    ) -> float:
        score = 0.0
        
        for pattern in self._complex_patterns:
            if pattern.search(query):
                score += 0.4
                break
        
        if word_count >= 15:
            score += 0.3
        elif word_count >= 10:
            score += 0.15
        
        connector_count = sum(1 for w in words if w in {"and", "or", "vs", "versus"})
        score += min(connector_count * 0.2, 0.4)
        
        matches = sum(1 for kw in self.COMPLEXITY_BOOSTERS if kw in query_lower)
        score += min(matches * 0.15, 0.3)
        
        if query.count("?") > 1:
            score += 0.25
        
        return min(score, 1.0)
    
    def _decide(self, simple_score: float, complex_score: float) -> Tuple[QueryComplexity, float]:
        if simple_score >= 0.5 and complex_score < 0.25:
            confidence = 0.3 + (simple_score - complex_score)
            return QueryComplexity.SIMPLE, min(confidence, 0.95)
        
        if complex_score >= 0.45 and simple_score < 0.35:
            confidence = 0.3 + (complex_score - simple_score)
            return QueryComplexity.COMPLEX, min(confidence, 0.95)
        
        score_diff = abs(simple_score - complex_score)
        confidence = 0.4 + (score_diff * 0.3)
        return QueryComplexity.MEDIUM, min(confidence, 0.75)
