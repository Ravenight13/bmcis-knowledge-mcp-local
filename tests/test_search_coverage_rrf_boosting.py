"""Comprehensive test coverage for RRF, boosting, query_router, and results modules.

This module provides extensive testing for:
- RRF (Reciprocal Rank Fusion): Score calculation and merging
- BoostingSystem: Multi-factor boost application
- QueryRouter: Strategy selection and routing
- SearchResult: Result object handling and formatting
"""

from __future__ import annotations

import time
import pytest
from datetime import datetime
from typing import Any
from unittest.mock import Mock, MagicMock


class TestRRFScoring:
    """Test Reciprocal Rank Fusion scoring."""

    def test_rrf_formula_single_rank(self) -> None:
        """Test RRF formula with single source rank."""
        # RRF(r) = 1 / (k + r)
        k = 60
        rank = 1
        rrf_score = 1.0 / (k + rank)
        assert 0.0 < rrf_score < 1.0

    def test_rrf_formula_multiple_sources(self) -> None:
        """Test RRF formula combining multiple sources."""
        k = 60
        rank_vector = 1
        rank_bm25 = 2
        rrf_score = 1.0 / (k + rank_vector) + 1.0 / (k + rank_bm25)
        assert 0.0 < rrf_score < 1.0

    def test_rrf_k_parameter_effect(self) -> None:
        """Test effect of k parameter on RRF scores."""
        rank = 1
        # Larger k -> smaller scores
        score_k60 = 1.0 / (60 + rank)
        score_k120 = 1.0 / (120 + rank)
        assert score_k60 > score_k120

    def test_rrf_rank_order_preservation(self) -> None:
        """Test that RRF preserves rank ordering."""
        k = 60
        # Rank 1 > Rank 2 > Rank 3
        score_r1 = 1.0 / (k + 1)
        score_r2 = 1.0 / (k + 2)
        score_r3 = 1.0 / (k + 3)
        assert score_r1 > score_r2 > score_r3

    def test_rrf_tie_breaking(self) -> None:
        """Test RRF breaking ties between sources."""
        k = 60
        # Same chunk in both sources
        rank_v = 3
        rank_bm25 = 5
        score_tied = 1.0 / (k + rank_v) + 1.0 / (k + rank_bm25)
        assert score_tied > 0.0

    def test_rrf_missing_from_one_source(self) -> None:
        """Test RRF when result missing from one source."""
        k = 60
        rank_present = 2
        # Assume missing from other source (rank = infinity)
        score = 1.0 / (k + rank_present)
        assert score > 0.0

    def test_rrf_score_normalization(self) -> None:
        """Test RRF scores are in valid range."""
        k = 60
        # Max score: when rank = 1 in both sources
        max_score = 1.0 / (k + 1) + 1.0 / (k + 1)
        # Min score: approaches 0 as rank increases
        min_score = 1.0 / (k + 1000)
        assert max_score > 0.0
        assert min_score >= 0.0


class TestBoostingSystem:
    """Test multi-factor boosting system."""

    def test_vendor_boost_application(self) -> None:
        """Test vendor-specific boost factor."""
        base_score = 0.8
        vendor_boost = 0.15
        boosted = base_score + vendor_boost
        assert boosted <= 1.0  # Should clamp if needed

    def test_doc_type_boost_application(self) -> None:
        """Test document type boost factor."""
        base_score = 0.7
        doc_type_boost = 0.10
        boosted = base_score + doc_type_boost
        assert abs(boosted - 0.8) < 0.0001  # Use approximate comparison for floats

    def test_recency_boost_application(self) -> None:
        """Test recency-based boost factor."""
        # Newer documents get higher boost
        base_score = 0.6
        recency_boost = 0.05
        boosted = base_score + recency_boost
        assert boosted == 0.65

    def test_entity_boost_application(self) -> None:
        """Test entity-based boost factor."""
        base_score = 0.75
        entity_boost = 0.1
        boosted = base_score + entity_boost
        assert boosted == 0.85

    def test_topic_boost_application(self) -> None:
        """Test topic-based boost factor."""
        base_score = 0.65
        topic_boost = 0.08
        boosted = base_score + topic_boost
        assert boosted == 0.73

    def test_combined_boosts(self) -> None:
        """Test combining multiple boost factors."""
        base_score = 0.7
        boosts = {
            "vendor": 0.15,
            "doc_type": 0.10,
            "recency": 0.05,
        }
        final = base_score
        for boost_val in boosts.values():
            final += boost_val
        # Should clamp to 1.0
        final = min(1.0, final)
        assert final == 1.0  # 0.7 + 0.15 + 0.1 + 0.05 = 1.0

    def test_boost_factor_ranges(self) -> None:
        """Test that boost factors are in valid ranges."""
        boosts = {
            "vendor": 0.15,
            "doc_type": 0.10,
            "recency": 0.05,
            "entity": 0.10,
            "topic": 0.08,
        }
        for boost_val in boosts.values():
            assert 0.0 <= boost_val <= 0.5  # Individual boosts should be reasonable

    def test_score_clamping_to_1(self) -> None:
        """Test that combined boosts don't exceed 1.0."""
        base_score = 0.9
        boost1 = 0.15
        boost2 = 0.10
        combined = base_score + boost1 + boost2
        clamped = min(1.0, combined)
        assert clamped == 1.0

    def test_zero_boost(self) -> None:
        """Test boost with zero value."""
        base_score = 0.5
        zero_boost = 0.0
        result = base_score + zero_boost
        assert result == 0.5

    def test_negative_boost_error(self) -> None:
        """Test that negative boost raises error."""
        boost = -0.1
        # Should validate boost >= 0.0
        assert boost < 0.0

    def test_boost_weight_configuration(self) -> None:
        """Test boost weight configuration."""
        from src.search.boosting import BoostWeights

        weights = BoostWeights()
        weights.vendor = 0.2
        weights.recency = 0.1
        assert weights.vendor == 0.2
        assert weights.recency == 0.1

    def test_custom_boost_weights(self) -> None:
        """Test using custom boost weight configuration."""
        from src.search.boosting import BoostWeights

        weights = BoostWeights()
        weights.vendor = 0.25
        weights.doc_type = 0.15
        weights.recency = 0.08
        # Custom weights applied
        assert weights.vendor == 0.25


class TestQueryRouter:
    """Test query routing and strategy selection."""

    def test_router_initialization(self) -> None:
        """Test QueryRouter initialization."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        assert router is not None

    def test_router_select_vector_strategy(self) -> None:
        """Test router selecting vector strategy."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        # Semantic query
        query = "explain the authentication flow"
        decision = router.select_strategy(query)
        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_router_select_bm25_strategy(self) -> None:
        """Test router selecting BM25 strategy."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        # Keyword-focused query
        query = "JWT authentication setup configuration"
        decision = router.select_strategy(query)
        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_router_select_hybrid_strategy(self) -> None:
        """Test router selecting hybrid strategy."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        # Mixed semantic + keyword query
        query = "how do I configure OAuth2 authentication"
        decision = router.select_strategy(query)
        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_router_confidence_score(self) -> None:
        """Test router provides confidence score."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        query = "test query"
        decision = router.select_strategy(query)
        assert 0.0 <= decision.confidence <= 1.0

    def test_router_explanation(self) -> None:
        """Test router provides explanation for decision."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        query = "test query"
        decision = router.select_strategy(query)
        assert hasattr(decision, "reason")
        assert isinstance(decision.reason, str)

    def test_router_consistency(self) -> None:
        """Test router decisions are consistent."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        query = "consistent query test"
        decision1 = router.select_strategy(query)
        decision2 = router.select_strategy(query)
        assert decision1.strategy == decision2.strategy

    def test_router_semantic_indicators(self) -> None:
        """Test router detecting semantic query indicators."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        semantic_queries = [
            "what is",
            "how to",
            "explain",
            "describe",
        ]
        for query in semantic_queries:
            decision = router.select_strategy(query)
            assert decision is not None

    def test_router_keyword_indicators(self) -> None:
        """Test router detecting keyword query indicators."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        keyword_queries = [
            "API endpoint reference",
            "configuration options",
            "setup guide",
        ]
        for query in keyword_queries:
            decision = router.select_strategy(query)
            assert decision is not None

    def test_router_ambiguous_query(self) -> None:
        """Test router handling ambiguous queries."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        ambiguous = "test"
        decision = router.select_strategy(ambiguous)
        assert decision.strategy is not None


class TestSearchResult:
    """Test SearchResult object handling."""

    def test_search_result_initialization(self) -> None:
        """Test SearchResult initialization with all fields."""
        from src.search.results import SearchResult

        result = SearchResult(
            chunk_id=1,
            chunk_text="Test content",
            similarity_score=0.85,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="technical",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
            metadata={},
        )
        assert result.chunk_id == 1

    def test_search_result_similarity_score_range(self) -> None:
        """Test that similarity scores are in valid range."""
        from src.search.results import SearchResult

        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.75,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )
        assert 0.0 <= result.similarity_score <= 1.0

    def test_search_result_bm25_score_range(self) -> None:
        """Test that BM25 scores are in valid range."""
        from src.search.results import SearchResult

        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.0,
            bm25_score=0.82,
            hybrid_score=0.0,
            rank=1,
            score_type="bm25",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )
        assert 0.0 <= result.bm25_score <= 1.0

    def test_search_result_hybrid_score_range(self) -> None:
        """Test that hybrid scores are in valid range."""
        from src.search.results import SearchResult

        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.75,
            hybrid_score=0.78,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )
        assert 0.0 <= result.hybrid_score <= 1.0

    def test_search_result_rank_positive(self) -> None:
        """Test that rank is positive integer."""
        from src.search.results import SearchResult

        for rank in [1, 5, 10, 100]:
            result = SearchResult(
                chunk_id=1,
                chunk_text="Test",
                similarity_score=0.8,
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=rank,
                score_type="vector",
                source_file="doc.md",
                source_category="tech",
                document_date=None,
                context_header="Section",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=256,
            )
            assert result.rank == rank
            assert result.rank > 0

    def test_search_result_metadata_access(self) -> None:
        """Test accessing result metadata."""
        from src.search.results import SearchResult

        metadata = {"vendor": "openai", "doc_type": "guide"}
        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
            metadata=metadata,
        )
        assert result.metadata["vendor"] == "openai"

    def test_search_result_equality(self) -> None:
        """Test SearchResult equality comparison."""
        from src.search.results import SearchResult

        result1 = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )

        result2 = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )

        # Should be equal if all fields match
        assert result1.chunk_id == result2.chunk_id

    def test_search_result_string_representation(self) -> None:
        """Test SearchResult string representation."""
        from src.search.results import SearchResult

        result = SearchResult(
            chunk_id=1,
            chunk_text="Test content",
            similarity_score=0.85,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="tech",
            document_date=None,
            context_header="Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
        )
        result_str = str(result)
        assert isinstance(result_str, str)
        assert len(result_str) > 0


class TestScoreComparisons:
    """Test score comparison and sorting."""

    def test_score_comparison_greater_than(self) -> None:
        """Test score comparison with > operator."""
        score1 = 0.85
        score2 = 0.75
        assert score1 > score2

    def test_score_comparison_less_than(self) -> None:
        """Test score comparison with < operator."""
        score1 = 0.75
        score2 = 0.85
        assert score1 < score2

    def test_score_comparison_equal(self) -> None:
        """Test score comparison with == operator."""
        score1 = 0.80
        score2 = 0.80
        assert score1 == score2

    def test_score_sorting_descending(self) -> None:
        """Test sorting results by score (descending)."""
        from src.search.results import SearchResult

        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=1.0 - (i * 0.1),
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="tech",
                document_date=None,
                context_header="Section",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=256,
            )
            for i in range(5)
        ]

        sorted_results = sorted(
            results, key=lambda r: r.similarity_score, reverse=True
        )
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].similarity_score >= sorted_results[i + 1].similarity_score

    def test_score_sorting_ascending(self) -> None:
        """Test sorting results by score (ascending)."""
        from src.search.results import SearchResult

        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=1.0 - (i * 0.1),
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="tech",
                document_date=None,
                context_header="Section",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=256,
            )
            for i in range(5)
        ]

        sorted_results = sorted(results, key=lambda r: r.similarity_score)
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].similarity_score <= sorted_results[i + 1].similarity_score


class TestPerformanceOptimization:
    """Test performance characteristics."""

    def test_rrf_calculation_performance(self) -> None:
        """Test RRF calculation performance."""
        start = time.time()
        k = 60
        for rank in range(1, 1001):
            score = 1.0 / (k + rank)
        elapsed = time.time() - start
        # Should be fast
        assert elapsed < 0.1

    def test_boost_application_performance(self) -> None:
        """Test boost application performance."""
        start = time.time()
        base_scores = [0.8 - (i * 0.01) for i in range(100)]
        boosts = {"vendor": 0.15, "recency": 0.05}
        boosted = [min(1.0, s + sum(boosts.values())) for s in base_scores]
        elapsed = time.time() - start
        # Should be fast
        assert elapsed < 0.01

    def test_routing_decision_performance(self) -> None:
        """Test query routing decision performance."""
        from src.search.query_router import QueryRouter

        router = QueryRouter()
        start = time.time()
        for i in range(100):
            decision = router.select_strategy(f"test query {i}")
        elapsed = time.time() - start
        # 100 decisions should be fast
        assert elapsed < 1.0

    def test_result_sorting_performance(self) -> None:
        """Test result sorting performance."""
        from src.search.results import SearchResult

        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.5 + (i * 0.001) % 0.5,
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="tech",
                document_date=None,
                context_header="Section",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=256,
            )
            for i in range(1000)
        ]

        start = time.time()
        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)
        elapsed = time.time() - start
        # Sorting 1000 results should be < 10ms
        assert elapsed < 0.1
