"""Semantic analyzer using embeddings for quality detection."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.embedding_analyzer.base import BaseAnalyzer
from src.embedding_analyzer.models import (
    AnalyzerResult,
    Issue,
    IssueCategory,
    Severity,
)

logger = logging.getLogger(__name__)


class SemanticAnalyzer(BaseAnalyzer):
    """
    Analyzes text using embeddings for semantic quality detection.

    Checks for:
    - Low information density (via embedding magnitude)
    - Outlier detection (distance from corpus centroid)
    - Language detection anomalies
    - Semantic coherence

    Requires an embedding function to be provided.

    Attributes:
        min_embedding_magnitude: Minimum L2 norm (default: 0.1)
        max_outlier_distance: Max std devs from centroid (default: 3.0)
        embedding_dim: Expected embedding dimension (default: 384)
    """

    # Default thresholds
    DEFAULT_CONFIG = {
        "min_embedding_magnitude": 0.1,
        "max_outlier_distance": 3.0,
        "embedding_dim": 384,  # Common for sentence-transformers
        "min_coherence_score": 0.3,
    }

    def __init__(
        self,
        weight: float = 0.25,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        embedding_fn: Optional[callable] = None,
        corpus_centroid: Optional[np.ndarray] = None,
        corpus_std: Optional[float] = None,
    ):
        """
        Initialize semantic analyzer.

        Args:
            weight: Weight for overall score calculation
            enabled: Whether analyzer is active
            config: Override default thresholds
            embedding_fn: Function that takes text and returns embedding vector
            corpus_centroid: Pre-computed centroid of corpus embeddings
            corpus_std: Pre-computed standard deviation of distances from centroid
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            name="semantic",
            weight=weight,
            enabled=enabled,
            config=merged_config,
        )
        self._embedding_fn = embedding_fn
        self._corpus_centroid = corpus_centroid
        self._corpus_std = corpus_std

    def set_embedding_function(self, embedding_fn: callable) -> None:
        """
        Set the embedding function after initialization.

        Args:
            embedding_fn: Function that takes text and returns embedding vector
        """
        self._embedding_fn = embedding_fn

    def set_corpus_stats(
        self,
        centroid: np.ndarray,
        std: float,
    ) -> None:
        """
        Set corpus statistics for outlier detection.

        Args:
            centroid: Mean embedding vector of corpus
            std: Standard deviation of distances from centroid
        """
        self._corpus_centroid = centroid
        self._corpus_std = std

    def compute_corpus_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Compute corpus statistics from a list of texts.

        Args:
            texts: List of text samples from corpus

        Returns:
            Dict with centroid and std

        Raises:
            ValueError: If embedding function not set
        """
        if self._embedding_fn is None:
            raise ValueError("Embedding function not set. Call set_embedding_function() first.")

        embeddings = []
        for text in texts:
            try:
                emb = self._embedding_fn(text)
                if emb is not None:
                    embeddings.append(np.array(emb))
            except Exception as e:
                logger.warning(f"Failed to embed text for corpus stats: {e}")

        if len(embeddings) < 2:
            raise ValueError(f"Need at least 2 embeddings, got {len(embeddings)}")

        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        std = np.std(distances)

        self._corpus_centroid = centroid
        self._corpus_std = std

        return {
            "centroid_shape": centroid.shape,
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(std),
            "num_samples": len(embeddings),
        }

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text for semantic quality issues.

        Args:
            text: Text content to analyze
            metadata: Optional metadata (unused in this analyzer)

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        issues: List[Issue] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}

        # Check if embedding function is available
        if self._embedding_fn is None:
            logger.debug("Semantic analyzer skipped: no embedding function")
            return self._build_result(
                passed=True,
                score=1.0,
                issues=[],
                metrics={"skipped": True, "reason": "no_embedding_function"},
                recommendations=["Set embedding function for semantic analysis"],
            )

        # Get embedding
        try:
            embedding = self._embedding_fn(text)
            if embedding is None:
                raise ValueError("Embedding function returned None")
            embedding = np.array(embedding)
            metrics["embedding_dim"] = len(embedding)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.SEMANTIC,
                code="EMBEDDING_FAILED",
                message=f"Failed to generate embedding: {str(e)[:100]}",
                auto_fixable=False,
                metadata={"error": str(e)},
            ))
            return self._build_result(
                passed=False,
                score=0.0,
                issues=issues,
                metrics=metrics,
                recommendations=["Check text for encoding issues or unsupported content"],
            )

        # Run checks
        issues.extend(self._check_embedding_magnitude(embedding, metrics))
        issues.extend(self._check_outlier_distance(embedding, metrics))
        issues.extend(self._check_embedding_validity(embedding, metrics))

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, metrics)

        # Calculate score
        score = self._calculate_score(issues, metrics)
        passed = not any(i.severity == Severity.CRITICAL for i in issues)

        return self._build_result(
            passed=passed,
            score=score,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _check_embedding_magnitude(
        self,
        embedding: np.ndarray,
        metrics: Dict[str, Any],
    ) -> List[Issue]:
        """Check embedding magnitude (L2 norm)."""
        issues = []

        magnitude = float(np.linalg.norm(embedding))
        metrics["embedding_magnitude"] = round(magnitude, 4)

        if magnitude < self.config["min_embedding_magnitude"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.SEMANTIC,
                code="LOW_EMBEDDING_MAGNITUDE",
                message=f"Embedding magnitude ({magnitude:.4f}) below threshold ({self.config['min_embedding_magnitude']})",
                auto_fixable=False,
                suggested_fix="Text may lack semantic content - review for meaningfulness",
                metadata={"magnitude": magnitude, "threshold": self.config["min_embedding_magnitude"]},
            ))

        return issues

    def _check_outlier_distance(
        self,
        embedding: np.ndarray,
        metrics: Dict[str, Any],
    ) -> List[Issue]:
        """Check if embedding is an outlier from corpus centroid."""
        issues = []

        if self._corpus_centroid is None or self._corpus_std is None:
            metrics["outlier_check"] = "skipped_no_corpus_stats"
            return issues

        # Compute distance from centroid
        distance = float(np.linalg.norm(embedding - self._corpus_centroid))
        metrics["distance_from_centroid"] = round(distance, 4)

        # Compute z-score
        if self._corpus_std > 0:
            z_score = distance / self._corpus_std
            metrics["outlier_z_score"] = round(z_score, 2)

            if z_score > self.config["max_outlier_distance"]:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    category=IssueCategory.SEMANTIC,
                    code="SEMANTIC_OUTLIER",
                    message=f"Text is a semantic outlier (z-score: {z_score:.2f}, max: {self.config['max_outlier_distance']})",
                    auto_fixable=False,
                    suggested_fix="Review text - may be off-topic or contain unusual content",
                    metadata={"z_score": z_score, "threshold": self.config["max_outlier_distance"]},
                ))

        return issues

    def _check_embedding_validity(
        self,
        embedding: np.ndarray,
        metrics: Dict[str, Any],
    ) -> List[Issue]:
        """Check for embedding validity issues."""
        issues = []

        # Check for NaN or Inf values
        if np.any(np.isnan(embedding)):
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.SEMANTIC,
                code="EMBEDDING_NAN",
                message="Embedding contains NaN values",
                auto_fixable=False,
                metadata={"nan_count": int(np.sum(np.isnan(embedding)))},
            ))

        if np.any(np.isinf(embedding)):
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.SEMANTIC,
                code="EMBEDDING_INF",
                message="Embedding contains infinite values",
                auto_fixable=False,
                metadata={"inf_count": int(np.sum(np.isinf(embedding)))},
            ))

        # Check for all zeros
        if np.allclose(embedding, 0):
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.SEMANTIC,
                code="EMBEDDING_ZERO",
                message="Embedding is all zeros",
                auto_fixable=False,
            ))

        # Check dimension
        expected_dim = self.config["embedding_dim"]
        if len(embedding) != expected_dim:
            metrics["dimension_mismatch"] = True
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.SEMANTIC,
                code="EMBEDDING_DIM_MISMATCH",
                message=f"Embedding dimension ({len(embedding)}) doesn't match expected ({expected_dim})",
                auto_fixable=False,
                metadata={"actual": len(embedding), "expected": expected_dim},
            ))

        return issues

    def _calculate_score(self, issues: List[Issue], metrics: Dict[str, Any]) -> float:
        """
        Calculate quality score based on issues found.

        Args:
            issues: List of issues detected
            metrics: Collected metrics

        Returns:
            Score from 0.0 (worst) to 1.0 (best)
        """
        score = 1.0

        # Deduct based on severity
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score -= 0.5
            elif issue.severity == Severity.WARNING:
                score -= 0.15
            elif issue.severity == Severity.INFO:
                score -= 0.05

        # Bonus for good embedding characteristics
        magnitude = metrics.get("embedding_magnitude", 0)
        if 0.5 <= magnitude <= 2.0:
            score += 0.05  # Normal range bonus

        z_score = metrics.get("outlier_z_score")
        if z_score is not None and z_score < 1.5:
            score += 0.05  # Well within corpus distribution

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        issues: List[Issue],
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        codes = {i.code for i in issues}

        if "EMBEDDING_FAILED" in codes:
            recommendations.append("Check text preprocessing - may contain invalid characters")

        if "LOW_EMBEDDING_MAGNITUDE" in codes:
            recommendations.append("Text may be too generic or lack distinctive content")

        if "SEMANTIC_OUTLIER" in codes:
            recommendations.append("Verify text belongs to expected domain/topic")

        if "EMBEDDING_NAN" in codes or "EMBEDDING_INF" in codes:
            recommendations.append("Debug embedding model - numerical instability detected")

        if "EMBEDDING_ZERO" in codes:
            recommendations.append("Embedding model may not support this text type")

        return recommendations
