"""Abstract base class for text analyzers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.embedding_analyzer.models import AnalyzerResult, Issue

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class that all analyzers must implement.

    Ensures consistent interface across:
    - Text quality analyzer (encoding, whitespace)
    - Structural analyzer (sentences, boundaries)
    - Token analyzer (limits, explosion)
    - Semantic analyzer (embedding-based outliers)
    - Content analyzer (tables, code, boilerplate)

    Attributes:
        name: Unique identifier for this analyzer
        weight: Importance weight for overall score (0.0-1.0)
        enabled: Whether this analyzer is active
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base analyzer.

        Args:
            name: Unique name for this analyzer
            weight: Weight for scoring (0.0-1.0)
            enabled: Whether analyzer is active
            config: Optional configuration dictionary
        """
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration. Override in subclasses.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text and return results.

        Args:
            text: Text content to analyze
            metadata: Optional metadata about the text (source, page, etc.)

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        pass

    def can_fix(self, issue: Issue) -> bool:
        """
        Check if this analyzer can auto-fix an issue.

        Args:
            issue: The issue to check

        Returns:
            True if auto-fixable by this analyzer
        """
        return issue.auto_fixable

    def fix(self, text: str, issue: Issue) -> str:
        """
        Attempt to fix an issue in the text.

        Args:
            text: Original text
            issue: Issue to fix

        Returns:
            Fixed text (or original if unfixable)

        Raises:
            NotImplementedError: If fix not implemented for this issue
        """
        raise NotImplementedError(
            f"Fix not implemented for issue code: {issue.code}"
        )

    def _build_result(
        self,
        passed: bool,
        score: float,
        issues: list,
        metrics: Dict[str, Any],
        recommendations: list,
    ) -> AnalyzerResult:
        """
        Helper to build AnalyzerResult.

        Args:
            passed: Whether analysis passed
            score: Quality score (0.0-1.0)
            issues: List of Issue objects
            metrics: Measurement data
            recommendations: Suggested improvements

        Returns:
            AnalyzerResult object
        """
        return AnalyzerResult(
            analyzer_name=self.name,
            passed=passed,
            score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
        )

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight}, {status})"
