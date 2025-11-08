"""Test suite for search result formatting and validation.

Tests cover:
- SearchResult dataclass creation and validation
- Score normalization and range validation
- Result formatting (JSON, dict, text)
- Metadata filtering
- Ranking validation
- Score normalization for vector and BM25
- Hybrid score combination
- Result deduplication
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from src.search.results import (
    SearchResult,
    SearchResultFormatter,
    RankingValidator,
)


# Fixtures for test data
@pytest.fixture
def valid_search_result() -> SearchResult:
    """Create a valid SearchResult for testing."""
    return SearchResult(
        chunk_id=1,
        chunk_text="This is a test chunk with important information.",
        similarity_score=0.85,
        bm25_score=0.72,
        hybrid_score=0.81,
        rank=1,
        score_type="hybrid",
        source_file="docs/guide.md",
        source_category="guide",
        document_date=None,
        context_header="guide.md > Section > Subsection",
        chunk_index=0,
        total_chunks=10,
        chunk_token_count=512,
        metadata={"tags": ["important"], "author": "test"},
    )


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create multiple SearchResult objects for testing."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="Machine learning fundamentals and theory.",
            similarity_score=0.95,
            bm25_score=0.88,
            hybrid_score=0.92,
            rank=1,
            score_type="hybrid",
            source_file="docs/ml.md",
            source_category="ml",
            document_date=None,
            context_header="ml.md > Intro",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"tags": ["ml", "ai"]},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Deep learning neural networks explained.",
            similarity_score=0.88,
            bm25_score=0.80,
            hybrid_score=0.85,
            rank=2,
            score_type="hybrid",
            source_file="docs/ml.md",
            source_category="ml",
            document_date=None,
            context_header="ml.md > Advanced",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"tags": ["ml", "deep"]},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Computer vision and image processing.",
            similarity_score=0.75,
            bm25_score=0.65,
            hybrid_score=0.71,
            rank=3,
            score_type="hybrid",
            source_file="docs/cv.md",
            source_category="cv",
            document_date=None,
            context_header="cv.md > Overview",
            chunk_index=0,
            total_chunks=3,
            chunk_token_count=512,
            metadata={"tags": ["cv", "vision"]},
        ),
    ]


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass functionality."""

    def test_result_creation_valid(self, valid_search_result: SearchResult) -> None:
        """Test creating a valid SearchResult."""
        assert valid_search_result.chunk_id == 1
        assert valid_search_result.rank == 1
        assert valid_search_result.score_type == "hybrid"
        assert valid_search_result.chunk_text == "This is a test chunk with important information."

    def test_result_score_validation_similarity(self) -> None:
        """Test that similarity_score must be 0-1."""
        with pytest.raises(ValueError, match="similarity_score must be 0-1"):
            SearchResult(
                chunk_id=1,
                chunk_text="test",
                similarity_score=1.5,  # Invalid: > 1
                bm25_score=0.5,
                hybrid_score=0.5,
                rank=1,
                score_type="vector",
                source_file="test.md",
                source_category="test",
                document_date=None,
                context_header="test",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=10,
            )

    def test_result_score_validation_bm25(self) -> None:
        """Test that bm25_score must be 0-1."""
        with pytest.raises(ValueError, match="bm25_score must be 0-1"):
            SearchResult(
                chunk_id=1,
                chunk_text="test",
                similarity_score=0.5,
                bm25_score=-0.1,  # Invalid: < 0
                hybrid_score=0.5,
                rank=1,
                score_type="bm25",
                source_file="test.md",
                source_category="test",
                document_date=None,
                context_header="test",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=10,
            )

    def test_result_rank_validation(self) -> None:
        """Test that rank must be >= 1."""
        with pytest.raises(ValueError, match="rank must be >= 1"):
            SearchResult(
                chunk_id=1,
                chunk_text="test",
                similarity_score=0.5,
                bm25_score=0.5,
                hybrid_score=0.5,
                rank=0,  # Invalid: < 1
                score_type="vector",
                source_file="test.md",
                source_category="test",
                document_date=None,
                context_header="test",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=10,
            )

    def test_result_empty_text_validation(self) -> None:
        """Test that chunk_text cannot be empty."""
        with pytest.raises(ValueError, match="chunk_text cannot be empty"):
            SearchResult(
                chunk_id=1,
                chunk_text="",  # Invalid: empty
                similarity_score=0.5,
                bm25_score=0.5,
                hybrid_score=0.5,
                rank=1,
                score_type="vector",
                source_file="test.md",
                source_category="test",
                document_date=None,
                context_header="test",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=10,
            )

    def test_result_chunk_index_validation(self) -> None:
        """Test that chunk_index must be valid."""
        with pytest.raises(ValueError, match="chunk_index must be"):
            SearchResult(
                chunk_id=1,
                chunk_text="test",
                similarity_score=0.5,
                bm25_score=0.5,
                hybrid_score=0.5,
                rank=1,
                score_type="vector",
                source_file="test.md",
                source_category="test",
                document_date=None,
                context_header="test",
                chunk_index=5,  # Invalid: >= total_chunks
                total_chunks=5,
                chunk_token_count=10,
            )


class TestSearchResultFormatting:
    """Tests for SearchResult formatting methods."""

    def test_to_dict(self, valid_search_result: SearchResult) -> None:
        """Test converting SearchResult to dictionary."""
        result_dict = valid_search_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["chunk_id"] == 1
        assert result_dict["chunk_text"] == "This is a test chunk with important information."
        assert result_dict["similarity_score"] == 0.85
        assert result_dict["rank"] == 1
        assert result_dict["metadata"] == {"tags": ["important"], "author": "test"}

    def test_to_json(self, valid_search_result: SearchResult) -> None:
        """Test converting SearchResult to JSON string."""
        json_str = valid_search_result.to_json()

        assert isinstance(json_str, str)
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["chunk_id"] == 1
        assert parsed["rank"] == 1

    def test_to_text_with_metadata(self, valid_search_result: SearchResult) -> None:
        """Test converting SearchResult to text format with metadata."""
        text = valid_search_result.to_text(include_metadata=True)

        assert isinstance(text, str)
        assert "Rank: 1" in text
        assert "Source: docs/guide.md" in text
        assert "Category: guide" in text
        assert "Metadata:" in text

    def test_to_text_without_metadata(self, valid_search_result: SearchResult) -> None:
        """Test converting SearchResult to text format without metadata."""
        text = valid_search_result.to_text(include_metadata=False)

        assert isinstance(text, str)
        assert "Rank: 1" in text
        assert "Metadata:" not in text

    def test_to_text_long_chunk(self) -> None:
        """Test text formatting truncates long chunks."""
        long_text = "word " * 100
        result = SearchResult(
            chunk_id=1,
            chunk_text=long_text,
            similarity_score=0.5,
            bm25_score=0.5,
            hybrid_score=0.5,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=100,
        )

        text = result.to_text()
        assert "..." in text  # Should be truncated


class TestMetadataFiltering:
    """Tests for metadata filtering functionality."""

    def test_matches_filters_category_match(self, valid_search_result: SearchResult) -> None:
        """Test category filter matching."""
        filters = {"category": "guide"}
        assert valid_search_result.matches_filters(filters) is True

    def test_matches_filters_category_no_match(self, valid_search_result: SearchResult) -> None:
        """Test category filter not matching."""
        filters = {"category": "other"}
        assert valid_search_result.matches_filters(filters) is False

    def test_matches_filters_tags(self) -> None:
        """Test tag filtering."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="test",
            similarity_score=0.5,
            bm25_score=0.5,
            hybrid_score=0.5,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
            metadata={"tags": ["important", "urgent"]},
        )

        filters = {"tags": ["important"]}
        assert result.matches_filters(filters) is True

        filters = {"tags": ["other"]}
        assert result.matches_filters(filters) is False

    def test_matches_filters_source_file(self, valid_search_result: SearchResult) -> None:
        """Test source file filter (partial match)."""
        filters = {"source_file": "guide"}
        assert valid_search_result.matches_filters(filters) is True

        filters = {"source_file": "other"}
        assert valid_search_result.matches_filters(filters) is False

    def test_matches_filters_multiple(self, valid_search_result: SearchResult) -> None:
        """Test multiple filters combined (AND logic)."""
        filters = {"category": "guide", "source_file": "guide.md"}
        assert valid_search_result.matches_filters(filters) is True

        filters = {"category": "guide", "source_file": "other.md"}
        assert valid_search_result.matches_filters(filters) is False


class TestSearchResultFormatter:
    """Tests for SearchResultFormatter functionality."""

    def test_formatter_initialization(self) -> None:
        """Test SearchResultFormatter initialization."""
        formatter = SearchResultFormatter(
            deduplication_enabled=True,
            min_score_threshold=0.5,
            max_results=50,
        )

        assert formatter.deduplication_enabled is True
        assert formatter.min_score_threshold == 0.5
        assert formatter.max_results == 50

    def test_formatter_invalid_threshold(self) -> None:
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError, match="min_score_threshold must be 0-1"):
            SearchResultFormatter(min_score_threshold=1.5)

    def test_formatter_invalid_max_results(self) -> None:
        """Test that invalid max_results raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be >= 1"):
            SearchResultFormatter(max_results=0)

    def test_format_results_dict(self, sample_results: list[SearchResult]) -> None:
        """Test formatting results as dictionaries."""
        formatter = SearchResultFormatter()
        formatted = formatter.format_results(sample_results, format_type="dict")

        assert len(formatted) == 3
        assert all(isinstance(r, dict) for r in formatted)
        assert formatted[0]["chunk_id"] == 1

    def test_format_results_json(self, sample_results: list[SearchResult]) -> None:
        """Test formatting results as JSON strings."""
        formatter = SearchResultFormatter()
        formatted = formatter.format_results(sample_results, format_type="json")

        assert len(formatted) == 3
        assert all(isinstance(r, str) for r in formatted)
        parsed = json.loads(formatted[0])
        assert parsed["chunk_id"] == 1

    def test_format_results_text(self, sample_results: list[SearchResult]) -> None:
        """Test formatting results as text."""
        formatter = SearchResultFormatter()
        formatted = formatter.format_results(sample_results, format_type="text")

        assert len(formatted) == 3
        assert all(isinstance(r, str) for r in formatted)
        assert "Rank: 1" in formatted[0]

    def test_format_results_deduplication(self) -> None:
        """Test result deduplication."""
        result1 = SearchResult(
            chunk_id=1,
            chunk_text="test",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        result2 = SearchResult(
            chunk_id=1,  # Duplicate ID
            chunk_text="test",
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=2,
            score_type="hybrid",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        results = [result1, result2]
        formatter = SearchResultFormatter(deduplication_enabled=True)
        formatted = formatter.format_results(results, apply_deduplication=True)

        assert len(formatted) == 1  # Duplicate removed
        assert formatted[0]["chunk_id"] == 1

    def test_format_results_threshold_filtering(self, sample_results: list[SearchResult]) -> None:
        """Test threshold-based filtering."""
        formatter = SearchResultFormatter(min_score_threshold=0.80)
        formatted = formatter.format_results(sample_results, apply_threshold=True)

        # Only results with score >= 0.80 should be included
        assert len(formatted) <= 3
        for result in formatted:
            score = result["hybrid_score"] if result["score_type"] == "hybrid" else result["similarity_score"]
            assert score >= 0.80

    def test_format_results_max_results(self, sample_results: list[SearchResult]) -> None:
        """Test max results limiting."""
        formatter = SearchResultFormatter(max_results=2)
        formatted = formatter.format_results(sample_results, apply_threshold=False)

        assert len(formatted) == 2


class TestScoreNormalization:
    """Tests for score normalization functions."""

    def test_normalize_vector_scores_standard(self) -> None:
        """Test normalizing vector scores from -1 to 1 range."""
        scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
        normalized = SearchResultFormatter.normalize_vector_scores(scores)

        assert len(normalized) == 5
        assert normalized[0] == 0.0  # -1 -> 0
        assert normalized[2] == 0.5  # 0 -> 0.5
        assert normalized[4] == 1.0  # 1 -> 1

    def test_normalize_vector_scores_custom_range(self) -> None:
        """Test normalizing vector scores with custom range."""
        scores = [0.0, 0.5, 1.0]
        normalized = SearchResultFormatter.normalize_vector_scores(scores, score_range=(0.0, 1.0))

        assert normalized[0] == 0.0
        assert normalized[1] == 0.5
        assert normalized[2] == 1.0

    def test_normalize_bm25_scores(self) -> None:
        """Test normalizing BM25 scores."""
        scores = [0.0, 5.0, 10.0, 15.0, 20.0]
        normalized = SearchResultFormatter.normalize_bm25_scores(scores, percentile_99=10.0)

        assert normalized[0] == 0.0
        assert normalized[2] == 1.0  # 10 / 10 = 1.0
        assert normalized[4] == 1.0  # 20 / 10 = 1.0, clamped to 1.0

    def test_combine_hybrid_scores(self) -> None:
        """Test combining vector and BM25 scores."""
        vector_scores = [0.9, 0.8, 0.7]
        bm25_scores = [0.7, 0.8, 0.9]
        weights = (0.6, 0.4)

        hybrid = SearchResultFormatter.combine_hybrid_scores(vector_scores, bm25_scores, weights)

        assert len(hybrid) == 3
        expected = [
            0.9 * 0.6 + 0.7 * 0.4,  # 0.82
            0.8 * 0.6 + 0.8 * 0.4,  # 0.8
            0.7 * 0.6 + 0.9 * 0.4,  # 0.78
        ]
        assert hybrid == expected

    def test_combine_hybrid_scores_weight_validation(self) -> None:
        """Test that weights must sum to 1.0."""
        vector_scores = [0.9, 0.8]
        bm25_scores = [0.7, 0.8]
        weights = (0.5, 0.3)  # Sum to 0.8, not 1.0

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            SearchResultFormatter.combine_hybrid_scores(vector_scores, bm25_scores, weights)

    def test_combine_hybrid_scores_length_mismatch(self) -> None:
        """Test that score lists must have same length."""
        vector_scores = [0.9, 0.8]
        bm25_scores = [0.7]  # Different length

        with pytest.raises(ValueError, match="Score lists must have same length"):
            SearchResultFormatter.combine_hybrid_scores(vector_scores, bm25_scores)


class TestRankingValidator:
    """Tests for RankingValidator functionality."""

    def test_validate_ranking_sorted(self, sample_results: list[SearchResult]) -> None:
        """Test validating properly sorted results."""
        validation = RankingValidator.validate_ranking(sample_results)

        assert validation["is_sorted"] is True
        assert validation["rank_correctness"] is True
        assert validation["has_duplicates"] is False

    def test_validate_ranking_unsorted(self) -> None:
        """Test validating unsorted results."""
        result1 = SearchResult(
            chunk_id=1,
            chunk_text="test",
            similarity_score=0.7,  # Lower score
            bm25_score=0.7,
            hybrid_score=0.7,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        result2 = SearchResult(
            chunk_id=2,
            chunk_text="test",
            similarity_score=0.9,  # Higher score but rank 2
            bm25_score=0.9,
            hybrid_score=0.9,
            rank=2,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        results = [result1, result2]
        validation = RankingValidator.validate_ranking(results)

        assert validation["is_sorted"] is False

    def test_validate_ranking_with_duplicates(self) -> None:
        """Test validating results with duplicate chunk IDs."""
        result1 = SearchResult(
            chunk_id=1,
            chunk_text="test",
            similarity_score=0.9,
            bm25_score=0.9,
            hybrid_score=0.9,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        result2 = SearchResult(
            chunk_id=1,  # Duplicate
            chunk_text="test",
            similarity_score=0.8,
            bm25_score=0.8,
            hybrid_score=0.8,
            rank=2,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        results = [result1, result2]
        validation = RankingValidator.validate_ranking(results)

        assert validation["has_duplicates"] is True

    def test_rank_correlation_calculation(self) -> None:
        """Test Spearman rank correlation calculation."""
        actual_order = [1, 2, 3, 4, 5]
        expected_order = [1, 2, 3, 4, 5]

        correlation = RankingValidator._calculate_rank_correlation(actual_order, expected_order)
        assert correlation == 1.0  # Perfect correlation

    def test_rank_correlation_reverse_order(self) -> None:
        """Test rank correlation with reverse order."""
        actual_order = [5, 4, 3, 2, 1]
        expected_order = [1, 2, 3, 4, 5]

        correlation = RankingValidator._calculate_rank_correlation(actual_order, expected_order)
        assert correlation == -1.0  # Perfect negative correlation
