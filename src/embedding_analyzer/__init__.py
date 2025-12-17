"""
Embedding Analyzer - Production-grade text analysis for RAG systems.

Usage:
    from src.embedding_analyzer import EmbeddingAnalyzer

    analyzer = EmbeddingAnalyzer()
    report = analyzer.analyze("Your text here")
    print(report.summary())
"""

from src.embedding_analyzer.analyzer import EmbeddingAnalyzer
from src.embedding_analyzer.base import BaseAnalyzer
from src.embedding_analyzer.models import (
    AnalysisReport,
    AnalyzerResult,
    Issue,
    IssueCategory,
    Severity,
)

__all__ = [
    "EmbeddingAnalyzer",
    "BaseAnalyzer",
    "AnalysisReport",
    "AnalyzerResult",
    "Issue",
    "IssueCategory",
    "Severity",
]
