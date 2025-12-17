"""Token analyzer for embedding model limits and token explosion detection."""

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


class TokenAnalyzer(BaseAnalyzer):
    """
    Analyzes text for token-related issues.

    Checks for:
    - Token count vs model limits
    - Token explosion (URLs, code, special patterns)
    - Extremely short content
    - Token-to-character ratio anomalies

    Attributes:
        model_max_tokens: Maximum tokens for embedding model (default: 512)
        min_tokens: Minimum tokens for meaningful embedding (default: 5)
        token_char_ratio_max: Max ratio indicating explosion (default: 0.5)
    """

    # Default thresholds
    DEFAULT_CONFIG = {
        "model_max_tokens": 512,       # Common for sentence-transformers
        "min_tokens": 5,               # Below this, embedding quality suffers
        "token_char_ratio_max": 0.5,   # Typical is ~0.25, >0.5 indicates explosion
        "warn_at_percent": 0.8,        # Warn at 80% of limit
    }

    # Patterns that cause token explosion
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
    CODE_PATTERN = re.compile(r'[{}\[\]();=<>]{3,}')  # Code-like sequences
    HEX_PATTERN = re.compile(r'\b0x[0-9a-fA-F]+\b|#[0-9a-fA-F]{6,8}\b')
    BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{50,}={0,2}')

    def __init__(
        self,
        weight: float = 0.25,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize token analyzer.

        Args:
            weight: Weight for overall score calculation
            enabled: Whether analyzer is active
            config: Override default thresholds
            tokenizer: Optional tokenizer instance (uses estimate if None)
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(
            name="token_analysis",
            weight=weight,
            enabled=enabled,
            config=merged_config,
        )
        self._tokenizer = tokenizer
        self._tiktoken_encoding = None

    def _get_token_count(self, text: str) -> int:
        """
        Get token count for text.

        Uses tiktoken if available, falls back to estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (exact or estimated)
        """
        # Try custom tokenizer first
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Custom tokenizer failed: {e}, using estimation")

        # Try tiktoken
        if self._tiktoken_encoding is None:
            try:
                import tiktoken
                self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._tiktoken_encoding = "unavailable"
                logger.debug("tiktoken not available, using estimation")

        if self._tiktoken_encoding != "unavailable":
            try:
                return len(self._tiktoken_encoding.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken failed: {e}, using estimation")

        # Fallback: estimate ~4 chars per token (rough average)
        return len(text) // 4

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerResult:
        """
        Analyze text for token-related issues.

        Args:
            text: Text content to analyze
            metadata: Optional metadata (unused in this analyzer)

        Returns:
            AnalyzerResult with score, issues, and recommendations
        """
        issues: List[Issue] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}

        # Get token count
        token_count = self._get_token_count(text)
        char_count = len(text)
        metrics["token_count"] = token_count
        metrics["char_count"] = char_count

        # Run all checks
        issues.extend(self._check_token_limits(token_count, metrics))
        issues.extend(self._check_token_explosion(text, token_count, char_count, metrics))
        issues.extend(self._check_explosion_patterns(text, metrics))

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

    def _check_token_limits(self, token_count: int, metrics: Dict[str, Any]) -> List[Issue]:
        """Check token count against model limits."""
        issues = []
        max_tokens = self.config["model_max_tokens"]
        min_tokens = self.config["min_tokens"]
        warn_threshold = int(max_tokens * self.config["warn_at_percent"])

        metrics["max_tokens"] = max_tokens
        metrics["utilization"] = round(token_count / max_tokens, 3) if max_tokens > 0 else 0

        if token_count > max_tokens:
            overflow = token_count - max_tokens
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.TOKEN,
                code="TOKEN_LIMIT_EXCEEDED",
                message=f"Token count ({token_count}) exceeds model limit ({max_tokens}) by {overflow}",
                auto_fixable=False,
                suggested_fix="Split into smaller chunks or truncate",
                metadata={"token_count": token_count, "limit": max_tokens, "overflow": overflow},
            ))
        elif token_count > warn_threshold:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.TOKEN,
                code="TOKEN_LIMIT_NEAR",
                message=f"Token count ({token_count}) near limit ({max_tokens}), {metrics['utilization']:.0%} utilized",
                auto_fixable=False,
                metadata={"token_count": token_count, "limit": max_tokens},
            ))

        if token_count < min_tokens:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.TOKEN,
                code="TOO_FEW_TOKENS",
                message=f"Token count ({token_count}) below minimum ({min_tokens})",
                auto_fixable=False,
                suggested_fix="Merge with adjacent chunks",
                metadata={"token_count": token_count, "min": min_tokens},
            ))

        return issues

    def _check_token_explosion(
        self,
        text: str,
        token_count: int,
        char_count: int,
        metrics: Dict[str, Any],
    ) -> List[Issue]:
        """Check for token explosion (high token-to-char ratio)."""
        issues = []

        if char_count == 0:
            return issues

        ratio = token_count / char_count
        metrics["token_char_ratio"] = round(ratio, 3)
        metrics["expected_ratio"] = 0.25  # Typical English text

        if ratio > self.config["token_char_ratio_max"]:
            issues.append(Issue(
                severity=Severity.WARNING,
                category=IssueCategory.TOKEN,
                code="TOKEN_EXPLOSION",
                message=f"Token-to-char ratio ({ratio:.2f}) indicates token explosion (max: {self.config['token_char_ratio_max']})",
                auto_fixable=False,
                suggested_fix="Check for URLs, code, or special patterns",
                metadata={"ratio": ratio, "max": self.config["token_char_ratio_max"]},
            ))

        return issues

    def _check_explosion_patterns(self, text: str, metrics: Dict[str, Any]) -> List[Issue]:
        """Check for patterns known to cause token explosion."""
        issues = []
        patterns_found = []

        # Check each pattern
        url_count = len(self.URL_PATTERN.findall(text))
        if url_count > 0:
            patterns_found.append(f"{url_count} URLs")

        email_count = len(self.EMAIL_PATTERN.findall(text))
        if email_count > 0:
            patterns_found.append(f"{email_count} emails")

        code_matches = len(self.CODE_PATTERN.findall(text))
        if code_matches > 0:
            patterns_found.append(f"{code_matches} code-like sequences")

        hex_count = len(self.HEX_PATTERN.findall(text))
        if hex_count > 0:
            patterns_found.append(f"{hex_count} hex values")

        base64_count = len(self.BASE64_PATTERN.findall(text))
        if base64_count > 0:
            patterns_found.append(f"{base64_count} base64 strings")

        metrics["explosion_patterns"] = {
            "urls": url_count,
            "emails": email_count,
            "code_sequences": code_matches,
            "hex_values": hex_count,
            "base64_strings": base64_count,
        }

        total_patterns = url_count + email_count + code_matches + hex_count + base64_count

        if total_patterns > 5:
            issues.append(Issue(
                severity=Severity.INFO,
                category=IssueCategory.TOKEN,
                code="EXPLOSION_PATTERNS_FOUND",
                message=f"Found patterns that may cause token explosion: {', '.join(patterns_found)}",
                auto_fixable=False,
                metadata=metrics["explosion_patterns"],
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

        # Bonus for good token utilization (not too short, not too long)
        utilization = metrics.get("utilization", 0)
        if 0.3 <= utilization <= 0.8:
            score += 0.1  # Ideal range bonus

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        issues: List[Issue],
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on issues."""
        recommendations = []
        codes = {i.code for i in issues}

        if "TOKEN_LIMIT_EXCEEDED" in codes:
            overflow = metrics.get("token_count", 0) - metrics.get("max_tokens", 512)
            recommendations.append(f"Split chunk - {overflow} tokens over limit")

        if "TOO_FEW_TOKENS" in codes:
            recommendations.append("Merge with adjacent chunk for better embedding quality")

        if "TOKEN_EXPLOSION" in codes or "EXPLOSION_PATTERNS_FOUND" in codes:
            patterns = metrics.get("explosion_patterns", {})
            if patterns.get("urls", 0) > 0:
                recommendations.append("Consider removing or shortening URLs")
            if patterns.get("base64_strings", 0) > 0:
                recommendations.append("Remove base64-encoded content")
            if patterns.get("hex_values", 0) > 0:
                recommendations.append("Consider removing hex values if not essential")

        return recommendations
