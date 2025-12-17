"""Structural analyzer for sentence boundaries and content organization."""

import logging
import re
from typing import Any, Dict, List, Optional

from src.embedding_analyzer.base import BaseAnalyzer
from src.embedding_analyzer.models import (
    AnalyzerResult,
    Issue,
    IssueCategory,
    Severity,
)

logger = logging.getLogger(__name__)


class StructuralAnalyzer(BaseAnalyzer):
    """
    Analyzes text for structural issues affecting embedding quality.

    Checks for:
    - Incomplete sentences (mid-sentence splits)
    - Orphan headers (headers without content)
    - Table fragments (broken table structure)
    - List fragments (incomplete lists)
    - Sentence count and density

    Attributes:
        min_sentences: Minimum complete sentences (default: 1)
        max_header_distance: Max chars between header and content (default: 50)
        min_words_per_sentence: Minimum words for a valid sentence (default: 3)
    """

    # Default thresholds
    DEFAULT_CONFIG = {
        "min_sentences": 1,
        "max_header_distance": 50,
        "min_words_per_sentence": 3,
        "max_incomplete_sentence_ratio": 0.3,
    }

    # Patterns
    SENTENCE_END_PATTERN = re.compile(r'[.!?](?:\s|$)')
    HEADER_PATTERN = re.compile(
        r'^(?:#{1,6}\s+.+|[A-Z][A-Za-z\s]{2,50}:?\s*$|(?:Section|Chapter|Part)\s+\d+)',
        re.MULTILINE
    )
    TABLE_INDICATOR_PATTERN = re.compile(r'\t{2,}|(?:\s{2,}\|)|(?:\|\s{2,})')
    LIST_ITEM_PATTERN = re.compile(r'^[\s]*[-•*]\s+|^\s*\d+[.)]\s+', re.MULTILINE)
    INCOMPLETE_START_PATTERN = re.compile(r'^[a-z]|^[,;:]')
    INCOMPLETE_END_PATTERN = re.compile(r'[,;:\-]$|[a-z]$(?<![.!?])')

    def __init__(
        self,
        weight: float = 0.25,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize structural analyzer.

        Args:
            weight: Weight for overall score calculation
            enabled: Whether analyzer is active
            config: Override default thresholds
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            name="structural",
            weight=weight,
            enabled=enabled,
            config=merged_config,
        )

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text for structural issues.

        Args:
            text: Text content to analyze
            metadata: Optional metadata (unused in this analyzer)

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        issues: List[Issue] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}

        # Skip if text is too short for structural analysis
        if len(text.strip()) < 10:
            return self._build_result(
                passed=True,
                score=1.0,
                issues=[],
                metrics={"skipped": True, "reason": "text_too_short"},
                recommendations=[],
            )

        # Run all checks
        issues.extend(self._check_sentence_completeness(text, metrics))
        issues.extend(self._check_orphan_headers(text, metrics))
        issues.extend(self._check_table_fragments(text, metrics))
        issues.extend(self._check_list_fragments(text, metrics))
        issues.extend(self._check_sentence_boundaries(text, metrics))

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

    def _check_sentence_completeness(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for complete sentences."""
        issues = []
        text_stripped = text.strip()

        # Count sentence endings
        sentence_ends = len(self.SENTENCE_END_PATTERN.findall(text))
        metrics["sentence_count"] = sentence_ends

        # Check for incomplete start
        starts_incomplete = bool(self.INCOMPLETE_START_PATTERN.match(text_stripped))
        metrics["starts_incomplete"] = starts_incomplete

        # Check for incomplete end
        ends_incomplete = bool(self.INCOMPLETE_END_PATTERN.search(text_stripped))
        metrics["ends_incomplete"] = ends_incomplete

        if starts_incomplete:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.STRUCTURE,
                code="INCOMPLETE_START",
                message="Text appears to start mid-sentence",
                location=(0, min(50, len(text))),
                auto_fixable=False,
                suggested_fix="Include beginning of sentence from previous chunk",
            ))

        if ends_incomplete and sentence_ends == 0:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.STRUCTURE,
                code="INCOMPLETE_END",
                message="Text appears to end mid-sentence",
                location=(max(0, len(text) - 50), len(text)),
                auto_fixable=False,
                suggested_fix="Include end of sentence from next chunk",
            ))

        if sentence_ends < self.config["min_sentences"]:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.STRUCTURE,
                code="FEW_SENTENCES",
                message=f"Only {sentence_ends} complete sentence(s) found (min: {self.config['min_sentences']})",
                auto_fixable=False,
                metadata={"count": sentence_ends, "min": self.config["min_sentences"]},
            ))

        return issues

    def _check_orphan_headers(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for headers without associated content."""
        issues = []

        headers = list(self.HEADER_PATTERN.finditer(text))
        metrics["header_count"] = len(headers)

        if not headers:
            return issues

        for match in headers:
            header_end = match.end()
            remaining_text = text[header_end:].strip()

            # Check if there's content after the header
            if len(remaining_text) < self.config["max_header_distance"]:
                # Header at end of chunk with little content
                issues.append(Issue(
                    severity=Severity.WARNING,
                    category=IssueCategory.STRUCTURE,
                    code="ORPHAN_HEADER",
                    message=f"Header '{match.group()[:30]}...' has little/no following content",
                    location=(match.start(), match.end()),
                    auto_fixable=False,
                    suggested_fix="Include more content with this header or move to next chunk",
                    metadata={"header": match.group()[:50]},
                ))

        return issues

    def _check_table_fragments(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for fragmented table content."""
        issues = []

        # Look for table indicators
        table_indicators = self.TABLE_INDICATOR_PATTERN.findall(text)
        metrics["table_indicators"] = len(table_indicators)

        if len(table_indicators) > 2:
            # Likely a table fragment - check if it looks complete
            lines = text.strip().split('\n')
            tab_lines = [l for l in lines if '\t' in l or '|' in l]

            if len(tab_lines) > 0 and len(tab_lines) < 3:
                issues.append(Issue(
                    severity=Severity.INFO,
                    category=IssueCategory.STRUCTURE,
                    code="TABLE_FRAGMENT",
                    message=f"Possible table fragment detected ({len(tab_lines)} rows)",
                    auto_fixable=False,
                    suggested_fix="Consider keeping table rows together",
                    metadata={"row_count": len(tab_lines)},
                ))

        return issues

    def _check_list_fragments(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for fragmented list content."""
        issues = []

        list_items = self.LIST_ITEM_PATTERN.findall(text)
        metrics["list_items"] = len(list_items)

        if len(list_items) == 1:
            # Single list item might be a fragment
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.STRUCTURE,
                code="LIST_FRAGMENT",
                message="Single list item found - possible list fragment",
                auto_fixable=False,
                suggested_fix="Consider keeping list items together",
            ))

        return issues

    def _check_sentence_boundaries(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for mid-word splits and other boundary issues."""
        issues = []
        text_stripped = text.strip()

        # Check for hyphenated word at end (common in PDF extraction)
        if re.search(r'\w-$', text_stripped):
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.STRUCTURE,
                code="HYPHENATED_WORD_SPLIT",
                message="Text ends with hyphenated word (likely mid-word split)",
                location=(len(text) - 20, len(text)),
                auto_fixable=False,
                suggested_fix="Join with next chunk to complete word",
            ))

        # Check for split words (word starting with lowercase after space at start)
        if re.match(r'^[a-z]{1,3}\s', text_stripped):
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.STRUCTURE,
                code="POSSIBLE_WORD_FRAGMENT",
                message="Text may start with word fragment",
                location=(0, 10),
                auto_fixable=False,
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

        # Bonus for complete sentences
        if metrics.get("sentence_count", 0) >= 2:
            score += 0.1

        # Penalty for both incomplete start and end
        if metrics.get("starts_incomplete") and metrics.get("ends_incomplete"):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        issues: List[Issue],
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        codes = {i.code for i in issues}

        if "INCOMPLETE_START" in codes or "INCOMPLETE_END" in codes:
            recommendations.append("Adjust chunk boundaries to align with sentence boundaries")

        if "ORPHAN_HEADER" in codes:
            recommendations.append("Increase chunk overlap or adjust chunking strategy")

        if "TABLE_FRAGMENT" in codes or "LIST_FRAGMENT" in codes:
            recommendations.append("Consider semantic chunking that respects document structure")

        if "HYPHENATED_WORD_SPLIT" in codes:
            recommendations.append("Check PDF extraction settings for hyphenation handling")

        return recommendations
