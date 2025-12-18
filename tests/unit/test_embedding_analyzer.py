"""Unit tests for embedding analyzer."""

import pytest

from src.embedding_analyzer import (
    EmbeddingAnalyzer,
    AnalysisReport,
    Issue,
    Severity,
    IssueCategory,
)
from src.embedding_analyzer.analyzers import (
    TextQualityAnalyzer,
    TokenAnalyzer,
    StructuralAnalyzer,
)
from src.embedding_analyzer.config_loader import load_config, list_available_configs


class TestTextQualityAnalyzer:
    """Tests for TextQualityAnalyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TextQualityAnalyzer()

    def test_good_text_passes(self):
        """Well-formed text should pass."""
        text = "This is a well-formed sentence. It has proper structure."
        result = self.analyzer.analyze(text)
        assert result.passed is True
        assert result.score >= 0.8

    def test_empty_text_fails(self):
        """Empty text should fail with critical issue."""
        result = self.analyzer.analyze("")
        assert result.passed is False
        assert any(i.code == "EMPTY_TEXT" for i in result.issues)

    def test_whitespace_only_fails(self):
        """Whitespace-only text should fail."""
        result = self.analyzer.analyze("   \n\t  ")
        assert result.passed is False
        assert any(i.code == "WHITESPACE_ONLY" for i in result.issues)

    def test_surrogate_chars_detected(self):
        """Surrogate characters should be detected."""
        text = "Hello \ud800 world"
        result = self.analyzer.analyze(text)
        assert any(i.code == "SURROGATE_CHARS" for i in result.issues)

    def test_replacement_chars_detected(self):
        """Replacement characters should be detected."""
        text = "Hello \ufffd world"
        result = self.analyzer.analyze(text)
        assert any(i.code == "REPLACEMENT_CHARS" for i in result.issues)

    def test_excessive_whitespace_detected(self):
        """Excessive whitespace should be detected."""
        text = "a    b    c    d    e"
        result = self.analyzer.analyze(text)
        assert result.metrics["whitespace_ratio"] > 0.5

    def test_fix_surrogate_chars(self):
        """Should fix surrogate characters."""
        text = "Hello \ud800 world"
        issue = Issue(
            severity=Severity.CRITICAL,
            category=IssueCategory.ENCODING,
            code="SURROGATE_CHARS",
            message="test",
            auto_fixable=True,
        )
        fixed = self.analyzer.fix(text, issue)
        assert "\ud800" not in fixed


class TestTokenAnalyzer:
    """Tests for TokenAnalyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TokenAnalyzer()

    def test_normal_text_passes(self):
        """Normal text should pass."""
        text = "This is a normal sentence with enough tokens to pass."
        result = self.analyzer.analyze(text)
        assert result.passed is True

    def test_too_few_tokens_detected(self):
        """Very short text should trigger warning."""
        result = self.analyzer.analyze("Hi")
        assert any(i.code == "TOO_FEW_TOKENS" for i in result.issues)

    def test_token_limit_exceeded(self):
        """Text exceeding token limit should fail."""
        # Create analyzer with low limit
        analyzer = TokenAnalyzer(config={"model_max_tokens": 10})
        text = "This is a sentence with many words that will exceed the very low token limit we set."
        result = analyzer.analyze(text)
        assert any(i.code == "TOKEN_LIMIT_EXCEEDED" for i in result.issues)

    def test_url_patterns_detected(self):
        """URLs should be detected as explosion patterns."""
        text = "Check out https://example.com/path/to/resource?query=value for more info."
        result = self.analyzer.analyze(text)
        assert result.metrics["explosion_patterns"]["urls"] > 0


class TestStructuralAnalyzer:
    """Tests for StructuralAnalyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = StructuralAnalyzer()

    def test_complete_sentences_pass(self):
        """Complete sentences should pass."""
        text = "This is a complete sentence. Here is another one."
        result = self.analyzer.analyze(text)
        assert result.passed is True
        assert result.metrics["sentence_count"] >= 2

    def test_incomplete_start_detected(self):
        """Text starting mid-sentence should be detected."""
        text = "and then the results were calculated."
        result = self.analyzer.analyze(text)
        assert any(i.code == "INCOMPLETE_START" for i in result.issues)

    def test_hyphenated_word_split_detected(self):
        """Hyphenated word at end should be detected."""
        text = "This is a sentence with a hyphen-"
        result = self.analyzer.analyze(text)
        assert any(i.code == "HYPHENATED_WORD_SPLIT" for i in result.issues)

    def test_short_text_skipped(self):
        """Very short text should skip structural analysis."""
        result = self.analyzer.analyze("Hi")
        assert result.metrics.get("skipped") is True


class TestEmbeddingAnalyzer:
    """Tests for main EmbeddingAnalyzer orchestrator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = EmbeddingAnalyzer()

    def test_analyzer_creation(self):
        """Analyzer should create with default analyzers."""
        assert len(self.analyzer.analyzers) == 5  # 5 default analyzers

    def test_good_text_passes(self):
        """Well-formed text should pass overall."""
        text = "This is a well-formed paragraph. It contains multiple sentences. The content is meaningful."
        report = self.analyzer.analyze(text)
        assert report.overall_passed is True
        assert report.overall_score >= 0.8

    def test_empty_text_fails(self):
        """Empty text should fail overall."""
        report = self.analyzer.analyze("")
        assert report.overall_passed is False
        assert report.critical_count > 0

    def test_report_has_all_fields(self):
        """Report should have all required fields."""
        report = self.analyzer.analyze("Test text.")
        assert report.text_hash is not None
        assert report.char_count > 0
        assert len(report.analyzer_results) > 0

    def test_summary_generation(self):
        """Summary should be generated correctly."""
        report = self.analyzer.analyze("Test text with content.")
        summary = report.summary()
        assert "Score:" in summary
        assert "Issues:" in summary

    def test_analyze_many(self):
        """Should analyze multiple texts."""
        texts = ["First text.", "Second text.", "Third text."]
        reports = self.analyzer.analyze_many(texts)
        assert len(reports) == 3

    def test_summary_stats(self):
        """Should generate summary statistics."""
        texts = ["Good text here.", "", "Another good one."]
        reports = self.analyzer.analyze_many(texts)
        stats = self.analyzer.get_summary_stats(reports)
        assert stats["total_analyzed"] == 3
        assert "pass_rate" in stats

    def test_auto_fix(self):
        """Auto-fix should work when enabled."""
        analyzer = EmbeddingAnalyzer(auto_fix_enabled=True)
        text = "Hello \ufffd world"
        report = analyzer.analyze(text)
        fixed = analyzer.auto_fix(text, report)
        assert "\ufffd" not in fixed

    def test_custom_fail_threshold(self):
        """Custom fail threshold should be respected."""
        analyzer = EmbeddingAnalyzer(fail_threshold=0.9)
        text = "Short"  # Will have some warnings
        report = analyzer.analyze(text)
        # With high threshold, even minor issues can cause failure
        assert analyzer.fail_threshold == 0.9


class TestConfigLoader:
    """Tests for configuration loader."""

    def test_load_default_config(self):
        """Should load default config."""
        config = load_config("default")
        assert "analyzers" in config
        assert "fail_threshold" in config

    def test_load_financial_config(self):
        """Should load financial config."""
        config = load_config("financial")
        assert config["fail_threshold"] == 0.4  # Financial is more lenient

    def test_list_configs(self):
        """Should list available configs."""
        configs = list_available_configs()
        assert "default" in configs
        assert "financial" in configs

    def test_missing_config_raises(self):
        """Should raise error for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent")

    def test_from_config(self):
        """Should create analyzer from config."""
        config = load_config("default")
        analyzer = EmbeddingAnalyzer.from_config(config)
        assert analyzer.fail_threshold == config["fail_threshold"]
