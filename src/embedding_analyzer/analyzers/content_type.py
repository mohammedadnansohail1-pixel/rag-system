"""Content type analyzer for detecting HTML, code, and boilerplate."""

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


class ContentTypeAnalyzer(BaseAnalyzer):
    """
    Analyzes text to detect content types that may not embed well.

    Checks for:
    - HTML/XML markup
    - CSS styles
    - Code/scripts
    - SEC EDGAR headers
    - Boilerplate/legal text
    - Base64/binary data

    Attributes:
        max_markup_ratio: Max ratio of markup to text (default: 0.3)
        max_code_ratio: Max ratio of code-like content (default: 0.4)
    """

    DEFAULT_CONFIG = {
        "max_markup_ratio": 0.3,
        "max_code_ratio": 0.4,
        "max_boilerplate_ratio": 0.5,
    }

    # Patterns
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    CSS_PATTERN = re.compile(r'\{[^}]*:[^}]*\}|[\w-]+\s*:\s*[^;]+;')
    XML_HEADER_PATTERN = re.compile(r'<\?xml|<!DOCTYPE|xmlns[=:]')
    SEC_HEADER_PATTERN = re.compile(r'<SEC-|<DOCUMENT>|<TYPE>|ACCESSION NUMBER:|CONFORMED')
    SCRIPT_PATTERN = re.compile(r'<script|function\s*\(|var\s+\w+\s*=|const\s+\w+\s*=')
    BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{100,}={0,2}')
    
    # Boilerplate indicators
    BOILERPLATE_PHRASES = [
        "all rights reserved",
        "terms and conditions",
        "privacy policy",
        "cookie policy",
        "unauthorized",
        "strictly prohibited",
        "subject to prosecution",
    ]

    def __init__(
        self,
        weight: float = 0.15,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize content type analyzer.

        Args:
            weight: Weight for overall score calculation
            enabled: Whether analyzer is active
            config: Override default thresholds
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            name="content_type",
            weight=weight,
            enabled=enabled,
            config=merged_config,
        )

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text for content type issues.

        Args:
            text: Text content to analyze
            metadata: Optional metadata

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        issues: List[Issue] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}

        if len(text.strip()) < 10:
            return self._build_result(
                passed=True,
                score=1.0,
                issues=[],
                metrics={"skipped": True, "reason": "text_too_short"},
                recommendations=[],
            )

        # Run all checks
        issues.extend(self._check_html_xml(text, metrics))
        issues.extend(self._check_css(text, metrics))
        issues.extend(self._check_sec_headers(text, metrics))
        issues.extend(self._check_code(text, metrics))
        issues.extend(self._check_base64(text, metrics))
        issues.extend(self._check_boilerplate(text, metrics))

        # Determine content type
        content_type = self._determine_content_type(metrics)
        metrics["detected_content_type"] = content_type

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, content_type)

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

    def _check_html_xml(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for HTML/XML markup."""
        issues = []

        # Count HTML tags
        tags = self.HTML_TAG_PATTERN.findall(text)
        tag_chars = sum(len(t) for t in tags)
        markup_ratio = tag_chars / len(text) if len(text) > 0 else 0
        metrics["html_tag_count"] = len(tags)
        metrics["markup_ratio"] = round(markup_ratio, 3)

        # Check for XML/HTML headers
        has_xml_header = bool(self.XML_HEADER_PATTERN.search(text))
        metrics["has_xml_header"] = has_xml_header

        if has_xml_header:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="XML_HTML_DOCUMENT",
                message="Text appears to be XML/HTML document markup",
                auto_fixable=False,
                suggested_fix="Strip HTML tags before embedding",
            ))

        if markup_ratio > self.config["max_markup_ratio"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="HIGH_MARKUP_RATIO",
                message=f"Markup ratio ({markup_ratio:.1%}) exceeds threshold ({self.config['max_markup_ratio']:.0%})",
                auto_fixable=True,
                suggested_fix="Strip HTML/XML tags",
                metadata={"ratio": markup_ratio},
            ))

        return issues

    def _check_css(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for CSS content."""
        issues = []

        css_matches = self.CSS_PATTERN.findall(text)
        css_chars = sum(len(m) for m in css_matches)
        css_ratio = css_chars / len(text) if len(text) > 0 else 0
        metrics["css_ratio"] = round(css_ratio, 3)

        if css_ratio > 0.2:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.CONTENT,
                code="CSS_CONTENT",
                message=f"Text contains CSS styling ({css_ratio:.1%})",
                auto_fixable=False,
                suggested_fix="Remove CSS before embedding",
                metadata={"ratio": css_ratio},
            ))

        return issues

    def _check_sec_headers(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for SEC EDGAR headers."""
        issues = []

        has_sec_header = bool(self.SEC_HEADER_PATTERN.search(text))
        metrics["has_sec_header"] = has_sec_header

        if has_sec_header:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.CONTENT,
                code="SEC_EDGAR_HEADER",
                message="Text contains SEC EDGAR filing headers",
                auto_fixable=False,
                suggested_fix="Extract document content, skip headers",
            ))

        return issues

    def _check_code(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for code/script content."""
        issues = []

        has_script = bool(self.SCRIPT_PATTERN.search(text))
        metrics["has_script"] = has_script

        # Check for code-like patterns
        code_indicators = [
            len(re.findall(r'[{}\[\]();]', text)),
            len(re.findall(r'function|var |const |let |=>|return ', text)),
            len(re.findall(r'import |from |class |def ', text)),
        ]
        code_score = sum(code_indicators)
        code_ratio = code_score / len(text) if len(text) > 0 else 0
        metrics["code_indicator_score"] = code_score

        if has_script or code_ratio > self.config["max_code_ratio"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.CONTENT,
                code="CODE_CONTENT",
                message="Text appears to contain code/scripts",
                auto_fixable=False,
                suggested_fix="Separate code from text content",
            ))

        return issues

    def _check_base64(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for base64/binary data."""
        issues = []

        base64_matches = self.BASE64_PATTERN.findall(text)
        metrics["base64_count"] = len(base64_matches)

        if base64_matches:
            total_base64 = sum(len(m) for m in base64_matches)
            ratio = total_base64 / len(text)
            metrics["base64_ratio"] = round(ratio, 3)

            if ratio > 0.3:
                issues.append(Issue(
                    severity=Severity.CRITICAL,
                    category=IssueCategory.CONTENT,
                    code="BASE64_CONTENT",
                    message=f"Text contains base64/binary data ({ratio:.1%})",
                    auto_fixable=False,
                    suggested_fix="Remove binary data before embedding",
                ))

        return issues

    def _check_boilerplate(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for boilerplate/legal text."""
        issues = []

        text_lower = text.lower()
        boilerplate_count = sum(
            1 for phrase in self.BOILERPLATE_PHRASES
            if phrase in text_lower
        )
        metrics["boilerplate_phrases"] = boilerplate_count

        if boilerplate_count >= 3:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.CONTENT,
                code="BOILERPLATE_CONTENT",
                message=f"Text appears to be boilerplate/legal ({boilerplate_count} phrases)",
                auto_fixable=False,
                suggested_fix="Consider filtering boilerplate sections",
            ))

        return issues

    def _determine_content_type(self, metrics: Dict[str, Any]) -> str:
        """Determine the most likely content type."""
        if metrics.get("css_ratio", 0) > 0.1:
            return "css"
        if metrics.get("has_xml_header"):
            return "html_xml"
        if metrics.get("markup_ratio", 0) > 0.2:
            return "markup"
        if metrics.get("has_sec_header"):
            return "sec_filing"
        if metrics.get("has_script") or metrics.get("code_indicator_score", 0) > 50:
            return "code"
        if metrics.get("base64_ratio", 0) > 0.1:
            return "binary"
        if metrics.get("boilerplate_phrases", 0) >= 3:
            return "boilerplate"
        return "text"

    def _calculate_score(self, issues: List[Issue], metrics: Dict[str, Any]) -> float:
        """Calculate quality score."""
        score = 1.0

        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score -= 0.4
            elif issue.severity == Severity.WARNING:
                score -= 0.15
            elif issue.severity == Severity.INFO:
                score -= 0.05

        # Bonus for clean text content
        if metrics.get("detected_content_type") == "text":
            score += 0.1

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        issues: List[Issue],
        content_type: str,
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        codes = {i.code for i in issues}

        if content_type in ("html_xml", "markup"):
            recommendations.append("Use BeautifulSoup or similar to extract text from HTML")

        if "CSS_CONTENT" in codes:
            recommendations.append("This chunk is CSS styling - skip or filter")

        if "SEC_EDGAR_HEADER" in codes:
            recommendations.append("Use SEC EDGAR parser to extract document sections")

        if "BASE64_CONTENT" in codes:
            recommendations.append("Remove embedded images/attachments before chunking")

        return recommendations
