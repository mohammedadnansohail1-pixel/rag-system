"""Main embedding analyzer orchestrator."""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.embedding_analyzer.base import BaseAnalyzer
from src.embedding_analyzer.models import (
    AnalysisReport,
    AnalyzerResult,
    Issue,
    Severity,
)
from src.embedding_analyzer.analyzers.text_quality import TextQualityAnalyzer
from src.embedding_analyzer.analyzers.token_analysis import TokenAnalyzer
from src.embedding_analyzer.analyzers.structural import StructuralAnalyzer
from src.embedding_analyzer.analyzers.semantic import SemanticAnalyzer

logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    """
    Main orchestrator for text embedding analysis.

    Combines multiple analyzers to produce comprehensive quality reports.
    Supports pluggable analyzers, configurable thresholds, and auto-fixing.

    Usage:
        analyzer = EmbeddingAnalyzer()
        report = analyzer.analyze("Some text to check")
        print(report.summary())

        # With custom config
        analyzer = EmbeddingAnalyzer.from_config(config_dict)

        # With embedding function for semantic analysis
        analyzer = EmbeddingAnalyzer(embedding_fn=my_embed_func)

    Attributes:
        analyzers: List of active analyzers
        fail_threshold: Score below which analysis fails (default: 0.5)
        auto_fix_enabled: Whether to attempt auto-fixes (default: False)
    """

    def __init__(
        self,
        analyzers: Optional[List[BaseAnalyzer]] = None,
        fail_threshold: float = 0.5,
        auto_fix_enabled: bool = False,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize embedding analyzer.

        Args:
            analyzers: Custom list of analyzers (uses defaults if None)
            fail_threshold: Score threshold for pass/fail
            auto_fix_enabled: Enable automatic fixing of issues
            embedding_fn: Function for semantic analysis
            config: Configuration dictionary for analyzers
        """
        self.fail_threshold = fail_threshold
        self.auto_fix_enabled = auto_fix_enabled
        self._embedding_fn = embedding_fn
        self._config = config or {}

        if analyzers is not None:
            self.analyzers = analyzers
        else:
            self.analyzers = self._create_default_analyzers()

        # Set embedding function on semantic analyzer if provided
        if embedding_fn is not None:
            self._configure_semantic_analyzer()

    def _create_default_analyzers(self) -> List[BaseAnalyzer]:
        """Create default set of analyzers."""
        analyzer_configs = self._config.get("analyzers", {})

        analyzers = []

        # Text Quality Analyzer
        tq_config = analyzer_configs.get("text_quality", {})
        if tq_config.get("enabled", True):
            analyzers.append(TextQualityAnalyzer(
                weight=tq_config.get("weight", 0.25),
                config=tq_config.get("thresholds", {}),
            ))

        # Token Analyzer
        ta_config = analyzer_configs.get("token_analysis", {})
        if ta_config.get("enabled", True):
            analyzers.append(TokenAnalyzer(
                weight=ta_config.get("weight", 0.25),
                config=ta_config.get("thresholds", {}),
            ))

        # Structural Analyzer
        st_config = analyzer_configs.get("structural", {})
        if st_config.get("enabled", True):
            analyzers.append(StructuralAnalyzer(
                weight=st_config.get("weight", 0.25),
                config=st_config.get("thresholds", {}),
            ))

        # Semantic Analyzer
        se_config = analyzer_configs.get("semantic", {})
        if se_config.get("enabled", True):
            analyzers.append(SemanticAnalyzer(
                weight=se_config.get("weight", 0.25),
                config=se_config.get("thresholds", {}),
                embedding_fn=self._embedding_fn,
            ))

        return analyzers

    def _configure_semantic_analyzer(self) -> None:
        """Configure semantic analyzer with embedding function."""
        for analyzer in self.analyzers:
            if isinstance(analyzer, SemanticAnalyzer):
                analyzer.set_embedding_function(self._embedding_fn)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EmbeddingAnalyzer":
        """
        Create analyzer from configuration dictionary.

        Args:
            config: Configuration with analyzer settings

        Returns:
            Configured EmbeddingAnalyzer instance
        """
        return cls(
            fail_threshold=config.get("fail_threshold", 0.5),
            auto_fix_enabled=config.get("auto_fix", {}).get("enabled", False),
            config=config,
        )

    def set_embedding_function(self, embedding_fn: Callable[[str], List[float]]) -> None:
        """
        Set embedding function for semantic analysis.

        Args:
            embedding_fn: Function that takes text and returns embedding vector
        """
        self._embedding_fn = embedding_fn
        self._configure_semantic_analyzer()

    def add_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """
        Add a custom analyzer.

        Args:
            analyzer: Analyzer instance to add
        """
        self.analyzers.append(analyzer)
        logger.info(f"Added analyzer: {analyzer.name}")

    def remove_analyzer(self, name: str) -> bool:
        """
        Remove an analyzer by name.

        Args:
            name: Name of analyzer to remove

        Returns:
            True if removed, False if not found
        """
        for i, analyzer in enumerate(self.analyzers):
            if analyzer.name == name:
                self.analyzers.pop(i)
                logger.info(f"Removed analyzer: {name}")
                return True
        return False

    def analyze(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalysisReport:
        """
        Analyze text with all enabled analyzers.

        Args:
            text: Text to analyze
            metadata: Optional metadata about the text

        Returns:
            Complete AnalysisReport
        """
        analyzer_results: List[AnalyzerResult] = []
        all_issues: List[Issue] = []

        # Run each analyzer
        for analyzer in self.analyzers:
            if not analyzer.enabled:
                continue

            try:
                result = analyzer.analyze(text, metadata)
                analyzer_results.append(result)
                all_issues.extend(result.issues)
            except Exception as e:
                logger.error(f"Analyzer {analyzer.name} failed: {e}")
                # Create error result
                analyzer_results.append(AnalyzerResult(
                    analyzer_name=analyzer.name,
                    passed=False,
                    score=0.0,
                    issues=[Issue(
                        severity=Severity.CRITICAL,
                        category=result.issues[0].category if result.issues else "content",
                        code="ANALYZER_ERROR",
                        message=f"Analyzer failed: {str(e)[:100]}",
                    )],
                ))

        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(analyzer_results)

        # Count issues by severity
        critical_count = sum(1 for i in all_issues if i.severity == Severity.CRITICAL)
        warning_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
        info_count = sum(1 for i in all_issues if i.severity == Severity.INFO)

        # Determine pass/fail
        overall_passed = (
            overall_score >= self.fail_threshold
            and critical_count == 0
        )

        # Build report
        report = AnalysisReport(
            text_hash=AnalysisReport.compute_hash(text),
            text_preview=text[:100] if text else "",
            char_count=len(text),
            overall_passed=overall_passed,
            overall_score=overall_score,
            analyzer_results=analyzer_results,
            all_issues=all_issues,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
        )

        return report

    def analyze_many(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AnalysisReport]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze
            metadatas: Optional list of metadata dicts

        Returns:
            List of AnalysisReports
        """
        if metadatas is None:
            metadatas = [None] * len(texts)

        reports = []
        for text, metadata in zip(texts, metadatas):
            reports.append(self.analyze(text, metadata))

        return reports

    def auto_fix(self, text: str, report: AnalysisReport) -> str:
        """
        Attempt to auto-fix issues in text.

        Args:
            text: Original text
            report: Analysis report with issues

        Returns:
            Fixed text (or original if no fixes applied)
        """
        if not self.auto_fix_enabled:
            logger.warning("Auto-fix not enabled")
            return text

        fixed_text = text
        fixes_applied = 0

        for issue in report.all_issues:
            if not issue.auto_fixable:
                continue

            # Find analyzer that can fix this issue
            for analyzer in self.analyzers:
                if analyzer.can_fix(issue):
                    try:
                        fixed_text = analyzer.fix(fixed_text, issue)
                        fixes_applied += 1
                        logger.debug(f"Applied fix for {issue.code}")
                    except NotImplementedError:
                        pass
                    except Exception as e:
                        logger.warning(f"Fix failed for {issue.code}: {e}")
                    break

        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} auto-fixes")

        return fixed_text

    def _calculate_overall_score(self, results: List[AnalyzerResult]) -> float:
        """
        Calculate weighted average score from analyzer results.

        Args:
            results: List of AnalyzerResults

        Returns:
            Overall score (0.0-1.0)
        """
        if not results:
            return 1.0

        total_weight = 0.0
        weighted_score = 0.0

        for result in results:
            # Find analyzer to get weight
            weight = 1.0
            for analyzer in self.analyzers:
                if analyzer.name == result.analyzer_name:
                    weight = analyzer.weight
                    break

            weighted_score += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return 1.0

        return weighted_score / total_weight

    def get_summary_stats(self, reports: List[AnalysisReport]) -> Dict[str, Any]:
        """
        Get summary statistics for multiple reports.

        Args:
            reports: List of analysis reports

        Returns:
            Summary statistics dictionary
        """
        if not reports:
            return {}

        passed_count = sum(1 for r in reports if r.overall_passed)
        scores = [r.overall_score for r in reports]
        total_critical = sum(r.critical_count for r in reports)
        total_warning = sum(r.warning_count for r in reports)
        total_info = sum(r.info_count for r in reports)

        # Count issues by code
        issue_codes: Dict[str, int] = {}
        for report in reports:
            for issue in report.all_issues:
                issue_codes[issue.code] = issue_codes.get(issue.code, 0) + 1

        return {
            "total_analyzed": len(reports),
            "passed": passed_count,
            "failed": len(reports) - passed_count,
            "pass_rate": round(passed_count / len(reports), 3),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "total_critical": total_critical,
            "total_warning": total_warning,
            "total_info": total_info,
            "top_issues": dict(sorted(issue_codes.items(), key=lambda x: -x[1])[:10]),
        }

    def __repr__(self) -> str:
        analyzer_names = [a.name for a in self.analyzers if a.enabled]
        return (
            f"EmbeddingAnalyzer(analyzers={analyzer_names}, "
            f"fail_threshold={self.fail_threshold})"
        )
