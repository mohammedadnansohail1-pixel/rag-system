"""Data models for embedding analysis results."""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"  # Must fix - will cause embedding failure
    WARNING = "warning"    # Should fix - degrades quality
    INFO = "info"          # Nice to know - minor impact


class IssueCategory(str, Enum):
    """Categories of issues detected."""
    ENCODING = "encoding"
    STRUCTURE = "structure"
    TOKEN = "token"
    SEMANTIC = "semantic"
    CONTENT = "content"


@dataclass
class Issue:
    """
    A single issue found during analysis.

    Attributes:
        severity: How critical the issue is
        category: Type of issue (encoding, structure, etc.)
        code: Machine-readable code (e.g., "INVALID_UTF8")
        message: Human-readable description
        location: Optional (start, end) character positions
        auto_fixable: Whether this can be automatically fixed
        suggested_fix: Description of how to fix
        metadata: Additional issue-specific data
    """
    severity: Severity
    category: IssueCategory
    code: str
    message: str
    location: Optional[Tuple[int, int]] = None
    auto_fixable: bool = False
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Issue({self.severity.value.upper()}: "
            f"[{self.code}] {self.message[:50]})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "auto_fixable": self.auto_fixable,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
        }


@dataclass
class AnalyzerResult:
    """
    Result from a single analyzer.

    Attributes:
        analyzer_name: Name of the analyzer that produced this
        passed: Whether the text passed this analyzer's checks
        score: Quality score from 0.0 (worst) to 1.0 (best)
        issues: List of issues found
        metrics: Analyzer-specific measurements
        recommendations: Suggested improvements
    """
    analyzer_name: str
    passed: bool
    score: float
    issues: List[Issue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"AnalyzerResult({status} {self.analyzer_name}: "
            f"score={self.score:.2f}, issues={len(self.issues)})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analyzer": self.analyzer_name,
            "passed": self.passed,
            "score": round(self.score, 3),
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "recommendations": self.recommendations,
        }


@dataclass
class AnalysisReport:
    """
    Complete analysis report for a text chunk.

    Attributes:
        text_hash: MD5 hash for identification
        text_preview: First 100 characters
        char_count: Total character count
        overall_passed: True if all critical checks passed
        overall_score: Weighted average of analyzer scores (0.0-1.0)
        analyzer_results: Results from each analyzer
        all_issues: Flattened list of all issues found
        critical_count: Number of critical issues
        warning_count: Number of warnings
        info_count: Number of info-level issues
    """
    text_hash: str
    text_preview: str
    char_count: int
    overall_passed: bool
    overall_score: float
    analyzer_results: List[AnalyzerResult] = field(default_factory=list)
    all_issues: List[Issue] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def __repr__(self) -> str:
        status = "PASS" if self.overall_passed else "FAIL"
        preview = self.text_preview[:30] + "..." if len(self.text_preview) > 30 else self.text_preview
        return (
            f"AnalysisReport({status}, score={self.overall_score:.2f}, "
            f"issues={self.critical_count}C/{self.warning_count}W/{self.info_count}I, "
            f"text='{preview}')"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text_hash": self.text_hash,
            "text_preview": self.text_preview,
            "char_count": self.char_count,
            "overall_passed": self.overall_passed,
            "overall_score": round(self.overall_score, 3),
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "analyzer_results": [r.to_dict() for r in self.analyzer_results],
            "all_issues": [i.to_dict() for i in self.all_issues],
        }

    def summary(self) -> str:
        """
        Generate one-line summary of the report.

        Returns:
            Human-readable summary string
        """
        status = "✅ PASS" if self.overall_passed else "❌ FAIL"
        return (
            f"{status} | Score: {self.overall_score:.2f} | "
            f"Issues: {self.critical_count}C/{self.warning_count}W/{self.info_count}I | "
            f"{self.char_count} chars"
        )

    @staticmethod
    def compute_hash(text: str) -> str:
        """
        Compute MD5 hash for text identification.

        Args:
            text: Input text

        Returns:
            MD5 hex digest
        """
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
