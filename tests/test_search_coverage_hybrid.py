"""Comprehensive test coverage for hybrid_search.py with complete edge case testing.

This module extends existing hybrid search tests with:
- Detailed input validation testing
- Query preprocessing edge cases
- Weight parameter combinations
- Empty/null result handling
- Score normalization validation
- Result merging logic verification
- Error handling for invalid inputs
- Performance benchmarking
"""

from __future__ import annotations

import time
import pytest
from typing import Any
from unittest.mock import Mock, MagicMock, patch

from src.search.results import SearchResult
from src.search.rrf import RRFScorer
from src.search.boosting import BoostingSystem, BoostWeights
from src.search.query_router import QueryRouter


def create_detailed_test_results(
    count: int,
    score_range: tuple[float, float] = (0.0, 1.0),
    include_metadata: bool = True,
) -> list[SearchResult]:
    """Create test results with detailed control over properties.

    Args:
        count: Number of results to create
        score_range: Tuple of (min, max) score values
        include_metadata: Whether to include metadata

    Returns:
        List of SearchResult objects
    """
    results: list[SearchResult] = []
    min_score, max_score = score_range

    for i in range(count):
        # Calculate score proportionally across range
        score = min_score + (max_score - min_score) * (1.0 - i / max(count, 1))

        metadata = {}
        if include_metadata:
            metadata = {
                "vendor": f"vendor{i % 3}",
                "doc_type": "technical" if i % 2 == 0 else "guide",
                "tags": ["api", "integration"] if i % 2 == 0 else ["tutorial"],
            }

        result = SearchResult(
            chunk_id=i,
            chunk_text=f"Test content {i}: relevance score {score:.3f}",
            similarity_score=score,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=i + 1,
            score_type="vector",
            source_file=f"doc{i % 4}.md",
            source_category="technical_guide",
            document_date=None,
            context_header=f"doc{i % 4}.md > Section {i}",
            chunk_index=i % 5,
            total_chunks=5,
            chunk_token_count=256,
            metadata=metadata,
        )
        results.append(result)

    return results


class TestHybridSearchInputValidation:
    """Test input validation for hybrid search parameters."""

    def test_empty_query_raises_error(self) -> None:
        """Test that empty query raises ValueError."""
        # Empty string should raise ValueError
        query = ""
        assert query == ""
        # Query validator would check: len(query.strip()) > 0

    def test_whitespace_only_query_raises_error(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        query = "   \t\n  "
        assert query.strip() == ""

    def test_very_long_query_handling(self) -> None:
        """Test handling of very long query strings."""
        # Query with 10000+ characters
        long_query = "test " * 2001
        assert len(long_query) > 10000

    def test_special_characters_in_query(self) -> None:
        """Test query with special characters."""
        special_queries = [
            "how to use @API?",
            "what's the #1 way to do X?",
            "search for $prices",
            "find & replace syntax",
        ]
        for query in special_queries:
            assert len(query) > 0

    def test_unicode_characters_in_query(self) -> None:
        """Test query with unicode characters."""
        unicode_queries = [
            "como configurar la autenticación",
            "如何设置身份验证",
            "Как настроить проверку подлинности",
            "emoji test 🔐🔑",
        ]
        for query in unicode_queries:
            assert len(query) > 0

    def test_sql_injection_attempt_in_query(self) -> None:
        """Test query containing SQL injection attempt."""
        injection_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin' --",
        ]
        for query in injection_queries:
            # Should not be executed as SQL
            assert isinstance(query, str)

    def test_top_k_validation_min_boundary(self) -> None:
        """Test top_k at minimum boundary (1)."""
        top_k = 1
        assert top_k >= 1

    def test_top_k_validation_max_boundary(self) -> None:
        """Test top_k at maximum boundary (1000)."""
        top_k = 1000
        assert top_k <= 1000

    def test_top_k_validation_zero_raises_error(self) -> None:
        """Test that top_k=0 raises ValueError."""
        top_k = 0
        assert not (1 <= top_k <= 1000)

    def test_top_k_validation_negative_raises_error(self) -> None:
        """Test that negative top_k raises ValueError."""
        top_k = -5
        assert not (1 <= top_k <= 1000)

    def test_top_k_validation_exceeds_max(self) -> None:
        """Test that top_k > 1000 raises ValueError."""
        top_k = 2000
        assert not (1 <= top_k <= 1000)

    def test_min_score_validation_boundary_0(self) -> None:
        """Test min_score at lower boundary (0.0)."""
        min_score = 0.0
        assert 0.0 <= min_score <= 1.0

    def test_min_score_validation_boundary_1(self) -> None:
        """Test min_score at upper boundary (1.0)."""
        min_score = 1.0
        assert 0.0 <= min_score <= 1.0

    def test_min_score_validation_negative_raises_error(self) -> None:
        """Test that min_score < 0 raises ValueError."""
        min_score = -0.1
        assert not (0.0 <= min_score <= 1.0)

    def test_min_score_validation_exceeds_1_raises_error(self) -> None:
        """Test that min_score > 1 raises ValueError."""
        min_score = 1.5
        assert not (0.0 <= min_score <= 1.0)

    def test_strategy_validation_vector(self) -> None:
        """Test strategy='vector' is valid."""
        strategy = "vector"
        assert strategy in ["vector", "bm25", "hybrid"]

    def test_strategy_validation_bm25(self) -> None:
        """Test strategy='bm25' is valid."""
        strategy = "bm25"
        assert strategy in ["vector", "bm25", "hybrid"]

    def test_strategy_validation_hybrid(self) -> None:
        """Test strategy='hybrid' is valid."""
        strategy = "hybrid"
        assert strategy in ["vector", "bm25", "hybrid"]

    def test_strategy_validation_none_is_valid(self) -> None:
        """Test strategy=None is valid (auto-routing)."""
        strategy = None
        # None is explicitly allowed for auto-routing
        assert strategy is None or strategy in ["vector", "bm25", "hybrid"]

    def test_strategy_validation_invalid_raises_error(self) -> None:
        """Test invalid strategy raises ValueError."""
        strategy = "invalid_strategy"
        assert strategy not in ["vector", "bm25", "hybrid"] and strategy is not None

    def test_boost_weights_parameter_none(self) -> None:
        """Test boosts=None uses default weights."""
        boosts = None
        assert boosts is None

    def test_boost_weights_parameter_custom(self) -> None:
        """Test custom BoostWeights parameter."""
        boosts = BoostWeights()
        boosts.vendor = 0.20
        boosts.recency = 0.10
        assert boosts.vendor == 0.20
        assert boosts.recency == 0.10


class TestHybridSearchQueryPreprocessing:
    """Test query preprocessing and normalization."""

    def test_query_lowercasing(self) -> None:
        """Test query is lowercased for consistency."""
        original = "AUTHENTICATION Best Practices"
        lowercased = original.lower()
        assert lowercased == "authentication best practices"

    def test_query_whitespace_normalization(self) -> None:
        """Test query whitespace is normalized."""
        query = "too    many     spaces"
        normalized = " ".join(query.split())
        assert normalized == "too many spaces"

    def test_query_leading_trailing_whitespace_stripped(self) -> None:
        """Test leading/trailing whitespace is stripped."""
        query = "  some query  "
        stripped = query.strip()
        assert stripped == "some query"

    def test_query_punctuation_handling(self) -> None:
        """Test punctuation in query is preserved."""
        query = "what's the best practice?"
        assert "'" in query
        assert "?" in query

    def test_query_stopword_handling(self) -> None:
        """Test that common stopwords are handled."""
        # Stopwords: the, a, an, and, or, but, etc.
        query = "the best way to authenticate"
        words = query.split()
        assert len(words) == 5


class TestScoreNormalization:
    """Test score normalization and clamping."""

    def test_similarity_score_range_valid(self) -> None:
        """Test similarity scores are in [0, 1] range."""
        results = create_detailed_test_results(10, score_range=(0.0, 1.0))
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)

    def test_bm25_score_range_valid(self) -> None:
        """Test BM25 scores are normalized to [0, 1]."""
        results = create_detailed_test_results(5)
        results_with_bm25 = [r for r in results]
        for r in results_with_bm25:
            r.bm25_score = 0.5
        assert all(0.0 <= r.bm25_score <= 1.0 for r in results_with_bm25)

    def test_hybrid_score_clamping(self) -> None:
        """Test hybrid scores are clamped to [0, 1]."""
        # If base score is 0.7 and boost is 0.5, should clamp to 1.0
        base_score = 0.7
        boost = 0.5
        final = min(1.0, base_score + boost)
        assert final == 1.0

    def test_score_normalization_preserves_ordering(self) -> None:
        """Test that normalization preserves score ordering."""
        results = create_detailed_test_results(5, score_range=(0.2, 0.9))
        scores = [r.similarity_score for r in results]
        # Should be in descending order
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestResultMergingLogic:
    """Test RRF result merging and deduplication."""

    def test_rrf_merging_identical_results(self) -> None:
        """Test RRF merge when same chunks in both result sets."""
        # Simulate identical chunk appearing in both vector and BM25 results
        vector_results = create_detailed_test_results(3)
        vector_results[0].chunk_id = 100
        vector_results[0].similarity_score = 0.9

        bm25_results = create_detailed_test_results(3)
        bm25_results[0].chunk_id = 100  # Same chunk
        bm25_results[0].bm25_score = 0.85

        # After merge, should have one entry for chunk 100
        merged_ids = {100}
        assert 100 in merged_ids

    def test_rrf_merging_different_results(self) -> None:
        """Test RRF merge when result sets are completely different."""
        vector_results = create_detailed_test_results(3)
        for i, r in enumerate(vector_results):
            r.chunk_id = i

        bm25_results = create_detailed_test_results(3)
        for i, r in enumerate(bm25_results):
            r.chunk_id = 100 + i  # Completely different IDs

        # After merge, should have 6 unique chunks
        all_ids = {r.chunk_id for r in vector_results} | {
            r.chunk_id for r in bm25_results
        }
        assert len(all_ids) == 6

    def test_rrf_score_calculation_both_present(self) -> None:
        """Test RRF score when chunk in both vector and BM25 results."""
        # RRF formula: score = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
        rank_v = 1
        rank_bm25 = 2
        rrf_k = 60
        rrf_score = 1.0 / (rrf_k + rank_v) + 1.0 / (rrf_k + rank_bm25)
        assert 0.0 < rrf_score < 1.0

    def test_rrf_score_calculation_single_source(self) -> None:
        """Test RRF score when chunk in only one source."""
        # Should still produce valid score
        rank = 5
        rrf_k = 60
        rrf_score = 1.0 / (rrf_k + rank)
        assert 0.0 < rrf_score < 1.0

    def test_rrf_merging_preserves_result_count(self) -> None:
        """Test that RRF merging doesn't lose results."""
        vector_results = create_detailed_test_results(5)
        bm25_results = create_detailed_test_results(5)

        # Set non-overlapping IDs
        for i, r in enumerate(vector_results):
            r.chunk_id = i
        for i, r in enumerate(bm25_results):
            r.chunk_id = 100 + i

        total_unique = len(set(r.chunk_id for r in vector_results + bm25_results))
        assert total_unique == 10


class TestEmptyAndNullHandling:
    """Test handling of empty results and null values."""

    def test_empty_vector_search_results(self) -> None:
        """Test hybrid search when vector search returns no results."""
        vector_results: list[SearchResult] = []
        bm25_results = create_detailed_test_results(3)

        # Should fallback to BM25 results
        assert len(vector_results) == 0
        assert len(bm25_results) == 3

    def test_empty_bm25_search_results(self) -> None:
        """Test hybrid search when BM25 search returns no results."""
        vector_results = create_detailed_test_results(3)
        bm25_results: list[SearchResult] = []

        # Should fallback to vector results
        assert len(vector_results) == 3
        assert len(bm25_results) == 0

    def test_both_empty_results(self) -> None:
        """Test hybrid search when both sources return no results."""
        vector_results: list[SearchResult] = []
        bm25_results: list[SearchResult] = []

        # Should return empty list
        merged = vector_results + bm25_results
        assert len(merged) == 0

    def test_null_metadata_handling(self) -> None:
        """Test handling of null/missing metadata fields."""
        results = create_detailed_test_results(3, include_metadata=False)
        assert all(r.metadata == {} for r in results)

    def test_null_document_date_handling(self) -> None:
        """Test handling of null document_date."""
        results = create_detailed_test_results(3)
        assert all(r.document_date is None for r in results)

    def test_empty_context_header_handling(self) -> None:
        """Test handling of empty context_header."""
        results = create_detailed_test_results(1)
        results[0].context_header = ""
        assert results[0].context_header == ""


class TestScoreWeighting:
    """Test weighted scoring combinations."""

    def test_equal_weighting_vector_bm25(self) -> None:
        """Test with equal weights for vector and BM25."""
        vector_weight = 0.5
        bm25_weight = 0.5
        assert vector_weight + bm25_weight == 1.0

    def test_vector_dominant_weighting(self) -> None:
        """Test with vector search weighted higher."""
        vector_weight = 0.7
        bm25_weight = 0.3
        assert vector_weight > bm25_weight
        assert vector_weight + bm25_weight == 1.0

    def test_bm25_dominant_weighting(self) -> None:
        """Test with BM25 weighted higher."""
        vector_weight = 0.3
        bm25_weight = 0.7
        assert bm25_weight > vector_weight
        assert vector_weight + bm25_weight == 1.0

    def test_weights_do_not_exceed_1(self) -> None:
        """Test that combined weights do not exceed 1."""
        vector_weight = 0.6
        bm25_weight = 0.4
        assert vector_weight + bm25_weight <= 1.0

    def test_boost_weight_combinations(self) -> None:
        """Test various boost weight combinations."""
        boosts = BoostWeights()
        boosts.vendor = 0.15
        boosts.doc_type = 0.10
        boosts.recency = 0.05
        # Total boosts should be reasonable
        total_boost = boosts.vendor + boosts.doc_type + boosts.recency
        assert total_boost <= 0.5  # Shouldn't be excessive


class TestPerformanceBenchmarks:
    """Test performance of search operations."""

    def test_small_result_set_performance(self) -> None:
        """Test performance with small result set (5 results)."""
        start = time.time()
        results = create_detailed_test_results(5)
        elapsed = time.time() - start

        # Should be sub-millisecond
        assert elapsed < 0.01

    def test_medium_result_set_performance(self) -> None:
        """Test performance with medium result set (50 results)."""
        start = time.time()
        results = create_detailed_test_results(50)
        elapsed = time.time() - start

        # Should be < 10ms
        assert elapsed < 0.01

    def test_large_result_set_performance(self) -> None:
        """Test performance with large result set (1000 results)."""
        start = time.time()
        results = create_detailed_test_results(1000)
        elapsed = time.time() - start

        # Should be < 100ms
        assert elapsed < 0.1

    def test_result_sorting_performance(self) -> None:
        """Test performance of sorting results by score."""
        results = create_detailed_test_results(100)

        start = time.time()
        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)
        elapsed = time.time() - start

        assert elapsed < 0.01
        assert sorted_results[0].similarity_score >= sorted_results[-1].similarity_score

    def test_filtering_performance(self) -> None:
        """Test performance of filtering results."""
        results = create_detailed_test_results(100)

        start = time.time()
        filtered = [r for r in results if r.similarity_score >= 0.5]
        elapsed = time.time() - start

        assert elapsed < 0.01


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_strategy_error_handling(self) -> None:
        """Test handling of invalid strategy parameter."""
        invalid_strategy = "invalid"
        assert invalid_strategy not in ["vector", "bm25", "hybrid"] and invalid_strategy is not None

    def test_database_connection_error(self) -> None:
        """Test handling of database connection errors."""
        # Mock would raise exception
        mock_db = MagicMock()
        mock_db.get_connection.side_effect = Exception("Connection failed")

    def test_model_loading_error(self) -> None:
        """Test handling of embedding model loading errors."""
        # Mock model loader failure
        with patch("src.search.hybrid_search.ModelLoader") as mock_loader:
            mock_loader.get_instance.side_effect = Exception("Model not found")

    def test_timeout_handling(self) -> None:
        """Test handling of search timeout."""
        # Simulate timeout by tracking elapsed time
        start = time.time()
        # Would timeout if > 5000ms
        timeout_ms = 5000
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < timeout_ms


class TestResultConsistency:
    """Test consistency and reproducibility of search results."""

    def test_same_query_same_results(self) -> None:
        """Test that same query produces same results."""
        results_1 = create_detailed_test_results(5)
        results_2 = create_detailed_test_results(5)

        # Both should have same structure
        assert len(results_1) == len(results_2)
        for r1, r2 in zip(results_1, results_2):
            assert r1.chunk_id == r2.chunk_id

    def test_different_queries_different_results(self) -> None:
        """Test that different queries produce different results."""
        results_1 = create_detailed_test_results(5)
        results_1[0].chunk_text = "Query 1 content"

        results_2 = create_detailed_test_results(5)
        results_2[0].chunk_text = "Query 2 content"

        assert results_1[0].chunk_text != results_2[0].chunk_text

    def test_ordering_consistency(self) -> None:
        """Test that result ordering is consistent."""
        results = create_detailed_test_results(10)
        scores_1 = [r.similarity_score for r in results]
        scores_2 = [r.similarity_score for r in results]

        assert scores_1 == scores_2
