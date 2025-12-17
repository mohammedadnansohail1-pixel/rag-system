"""Text quality analyzer for encoding, whitespace, and character issues."""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.embedding_analyzer.base import BaseAnalyzer
from src.embedding_analyzer.models import (
    AnalyzerResult,
    Issue,
    IssueCategory,
    Severity,
)

logger = logging.getLogger(__name__)


class TextQualityAnalyzer(BaseAnalyzer):
    """
    Analyzes text for quality issues that affect embedding.

    Checks for:
    - Invalid UTF-8 / encoding issues
    - Excessive whitespace
    - Special character density
    - Empty or near-empty content
    - Control characters
    - Unicode normalization issues

    Attributes:
        min_chars: Minimum character count (default: 10)
        max_whitespace_ratio: Maximum whitespace ratio (default: 0.5)
        max_special_char_ratio: Maximum special char ratio (default: 0.3)
        max_control_char_count: Maximum control characters allowed (default: 5)
    """

    # Default thresholds
    DEFAULT_CONFIG = {
        "min_chars": 10,
        "max_whitespace_ratio": 0.5,
        "max_special_char_ratio": 0.3,
        "max_control_char_count": 5,
        "max_consecutive_newlines": 5,
        "max_repetition_ratio": 0.5,
    }

    # Patterns
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    WHITESPACE_PATTERN = re.compile(r'\s')
    SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)
    CONSECUTIVE_NEWLINE_PATTERN = re.compile(r'\n{3,}')
    REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{10,}')

    def __init__(
        self,
        weight: float = 0.25,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize text quality analyzer.

        Args:
            weight: Weight for overall score calculation
            enabled: Whether analyzer is active
            config: Override default thresholds
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            name="text_quality",
            weight=weight,
            enabled=enabled,
            config=merged_config,
        )

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.config.get("min_chars", 0) < 0:
            raise ValueError("min_chars must be non-negative")
        if not 0 <= self.config.get("max_whitespace_ratio", 0) <= 1:
            raise ValueError("max_whitespace_ratio must be between 0 and 1")
        if not 0 <= self.config.get("max_special_char_ratio", 0) <= 1:
            raise ValueError("max_special_char_ratio must be between 0 and 1")

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text for quality issues.

        Args:
            text: Text content to analyze
            metadata: Optional metadata (unused in this analyzer)

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        issues: List[Issue] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}

        # Run all checks
        issues.extend(self._check_empty_content(text, metrics))
        issues.extend(self._check_encoding(text, metrics))
        issues.extend(self._check_whitespace(text, metrics))
        issues.extend(self._check_special_chars(text, metrics))
        issues.extend(self._check_control_chars(text, metrics))
        issues.extend(self._check_repetition(text, metrics))

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

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

    def _check_empty_content(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for empty or near-empty content."""
        issues = []
        char_count = len(text)
        stripped_count = len(text.strip())

        metrics["char_count"] = char_count
        metrics["stripped_char_count"] = stripped_count

        if char_count == 0:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.CONTENT,
                code="EMPTY_TEXT",
                message="Text is completely empty",
                auto_fixable=False,
            ))
        elif stripped_count == 0:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.CONTENT,
                code="WHITESPACE_ONLY",
                message="Text contains only whitespace",
                auto_fixable=False,
            ))
        elif stripped_count < self.config["min_chars"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="TOO_SHORT",
                message=f"Text too short ({stripped_count} chars, min: {self.config['min_chars']})",
                auto_fixable=False,
                metadata={"char_count": stripped_count, "min_required": self.config["min_chars"]},
            ))

        return issues

    def _check_encoding(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for encoding issues."""
        issues = []

        # Check for replacement characters (indicates encoding problems)
        replacement_count = text.count('\ufffd')
        metrics["replacement_char_count"] = replacement_count

        if replacement_count > 0:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.ENCODING,
                code="REPLACEMENT_CHARS",
                message=f"Found {replacement_count} replacement characters (encoding issues)",
                auto_fixable=True,
                suggested_fix="Remove or replace invalid characters",
                metadata={"count": replacement_count},
            ))

        # Check for surrogate pairs (broken emoji/unicode)
        surrogate_count = len(re.findall(r'[\ud800-\udfff]', text))
        metrics["surrogate_count"] = surrogate_count

        if surrogate_count > 0:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.ENCODING,
                code="SURROGATE_CHARS",
                message=f"Found {surrogate_count} surrogate characters (broken unicode)",
                auto_fixable=True,
                suggested_fix="Remove surrogate pairs",
                metadata={"count": surrogate_count},
            ))

        # Check for mixed unicode normalization
        nfc_text = unicodedata.normalize('NFC', text)
        if nfc_text != text:
            metrics["needs_normalization"] = True
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.ENCODING,
                code="UNNORMALIZED_UNICODE",
                message="Text contains unnormalized unicode (NFC)",
                auto_fixable=True,
                suggested_fix="Apply NFC normalization",
            ))
        else:
            metrics["needs_normalization"] = False

        return issues

    def _check_whitespace(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for whitespace issues."""
        issues = []

        if len(text) == 0:
            return issues

        # Calculate whitespace ratio
        whitespace_count = len(self.WHITESPACE_PATTERN.findall(text))
        whitespace_ratio = whitespace_count / len(text)
        metrics["whitespace_ratio"] = round(whitespace_ratio, 3)

        if whitespace_ratio > self.config["max_whitespace_ratio"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="EXCESSIVE_WHITESPACE",
                message=f"Whitespace ratio too high ({whitespace_ratio:.1%}, max: {self.config['max_whitespace_ratio']:.0%})",
                auto_fixable=True,
                suggested_fix="Collapse multiple whitespace",
                metadata={"ratio": whitespace_ratio, "max": self.config["max_whitespace_ratio"]},
            ))

        # Check consecutive newlines
        consecutive_matches = self.CONSECUTIVE_NEWLINE_PATTERN.findall(text)
        max_consecutive = max((len(m) for m in consecutive_matches), default=0)
        metrics["max_consecutive_newlines"] = max_consecutive

        if max_consecutive > self.config["max_consecutive_newlines"]:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.STRUCTURE,
                code="EXCESSIVE_NEWLINES",
                message=f"Found {max_consecutive} consecutive newlines (max: {self.config['max_consecutive_newlines']})",
                auto_fixable=True,
                suggested_fix="Collapse multiple newlines",
            ))

        return issues

    def _check_special_chars(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for special character density."""
        issues = []

        if len(text) == 0:
            return issues

        special_chars = self.SPECIAL_CHAR_PATTERN.findall(text)
        special_ratio = len(special_chars) / len(text)
        metrics["special_char_ratio"] = round(special_ratio, 3)

        if special_ratio > self.config["max_special_char_ratio"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="HIGH_SPECIAL_CHAR_RATIO",
                message=f"Special character ratio too high ({special_ratio:.1%}, max: {self.config['max_special_char_ratio']:.0%})",
                auto_fixable=False,
                metadata={"ratio": special_ratio, "max": self.config["max_special_char_ratio"]},
            ))

        return issues

    def _check_control_chars(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for control characters."""
        issues = []

        control_chars = self.CONTROL_CHAR_PATTERN.findall(text)
        control_count = len(control_chars)
        metrics["control_char_count"] = control_count

        if control_count > self.config["max_control_char_count"]:
            # Find positions
            positions = [(m.start(), m.end()) for m in self.CONTROL_CHAR_PATTERN.finditer(text)]
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.ENCODING,
                code="CONTROL_CHARACTERS",
                message=f"Found {control_count} control characters",
                auto_fixable=True,
                suggested_fix="Remove control characters",
                metadata={"count": control_count, "positions": positions[:10]},  # First 10
            ))

        return issues

    def _check_repetition(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for repetitive content."""
        issues = []

        if len(text) == 0:
            return issues

        # Check for repeated characters (like "aaaaaaaaaaa")
        repeated_matches = self.REPEATED_CHAR_PATTERN.findall(text)
        metrics["repeated_char_sequences"] = len(repeated_matches)

        if repeated_matches:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.CONTENT,
                code="REPEATED_CHARACTERS",
                message=f"Found {len(repeated_matches)} repeated character sequences",
                auto_fixable=True,
                suggested_fix="Collapse repeated characters",
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
                score -= 0.4
            elif issue.severity == Severity.WARNING:
                score -= 0.15
            elif issue.severity == Severity.INFO:
                score -= 0.05

        return max(0.0, score)

    def _generate_recommendations(self, issues: List[Issue]) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        codes = {i.code for i in issues}

        if "EMPTY_TEXT" in codes or "WHITESPACE_ONLY" in codes:
            recommendations.append("Skip this chunk - no meaningful content")
        if "TOO_SHORT" in codes:
            recommendations.append("Consider merging with adjacent chunks")
        if "REPLACEMENT_CHARS" in codes or "SURROGATE_CHARS" in codes:
            recommendations.append("Re-extract text with proper encoding handling")
        if "EXCESSIVE_WHITESPACE" in codes:
            recommendations.append("Normalize whitespace before embedding")
        if "HIGH_SPECIAL_CHAR_RATIO" in codes:
            recommendations.append("Review source - may be code or corrupted text")

        return recommendations

    def fix(self, text: str, issue: Issue) -> str:
        """
        Fix an issue in the text.

        Args:
            text: Original text
            issue: Issue to fix

        Returns:
            Fixed text
        """
        if issue.code == "REPLACEMENT_CHARS":
            return text.replace('\ufffd', '')

        elif issue.code == "SURROGATE_CHARS":
            return re.sub(r'[\ud800-\udfff]', '', text)

        elif issue.code == "UNNORMALIZED_UNICODE":
            return unicodedata.normalize('NFC', text)

        elif issue.code == "EXCESSIVE_WHITESPACE":
            # Collapse multiple spaces to single
            text = re.sub(r'[^\S\n]+', ' ', text)
            return text.strip()

        elif issue.code == "EXCESSIVE_NEWLINES":
            return re.sub(r'\n{3,}', '\n\n', text)

        elif issue.code == "CONTROL_CHARACTERS":
            return self.CONTROL_CHAR_PATTERN.sub('', text)

        elif issue.code == "REPEATED_CHARACTERS":
            return re.sub(r'(.)\1{10,}', r'\1\1\1', text)  # Reduce to 3

        else:
            raise NotImplementedError(f"Fix not implemented for: {issue.code}")
