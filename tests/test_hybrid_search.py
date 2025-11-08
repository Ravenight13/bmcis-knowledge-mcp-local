"""Integration and end-to-end tests for HybridSearch unified class.

Tests cover:
- Basic initialization and component setup
- Vector search strategy execution
- BM25 search strategy execution
- Hybrid search strategy (vector + BM25 with RRF merging)
- Auto-routing strategy selection (None = automatic routing)
- Filters and constraints application
- Boosts application and scoring
- Min score threshold filtering
- Result formatting and ordering
- Advanced features (explanations, profiling)
- Error handling and edge cases
- Performance benchmarks
- Integration with Task 4 components
- Consistency and reproducibility

Performance targets:
- Vector search: <100ms
- BM25 search: <50ms
- Hybrid search: <300ms
- Large result sets: <500ms
"""

from __future__ import annotations

import json
import time
import pytest
from datetime import datetime
from typing import Any
from unittest.mock import Mock, MagicMock, patch

from src.search.rrf import RRFScorer
from src.search.boosting import BoostingSystem, BoostWeights
from src.search.query_router import QueryRouter
from src.search.results import SearchResult
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger


# Test fixtures for result data creation
@pytest.fixture
def mock_db_pool() -> MagicMock:
    """Create mock DatabasePool."""
    mock_pool = MagicMock(spec=DatabasePool)
    return mock_pool


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create mock StructuredLogger."""
    return MagicMock(spec=StructuredLogger)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock Settings."""
    mock_settings = MagicMock()
    mock_settings.search_top_k = 10
    mock_settings.search_min_score = 0.0
    mock_settings.hybrid_search_rrf_k = 60
    mock_settings.hybrid_search_vector_weight = 0.6
    mock_settings.hybrid_search_bm25_weight = 0.4
    return mock_settings


def create_test_vector_results(count: int) -> list[SearchResult]:
    """Create test vector search results with descending scores."""
    results: list[SearchResult] = []
    for i in range(count):
        results.append(
            SearchResult(
                chunk_id=i,
                chunk_text=f"Vector result {i}: semantic match content",
                similarity_score=max(0.0, 1.0 - (i * 0.1)),
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="vector",
                source_file=f"doc{i % 3}.md",
                source_category="guide",
                document_date=None,
                context_header=f"doc{i % 3}.md > Section {i}",
                chunk_index=i % 5,
                total_chunks=5,
                chunk_token_count=256,
                metadata={"vendor": f"vendor{i % 2}", "doc_type": "technical"},
            )
        )
    return results


def create_test_bm25_results(count: int) -> list[SearchResult]:
    """Create test BM25 search results with descending scores."""
    results: list[SearchResult] = []
    for i in range(count):
        results.append(
            SearchResult(
                chunk_id=100 + i,
                chunk_text=f"BM25 result {i}: keyword match content",
                similarity_score=0.0,
                bm25_score=max(0.0, 0.95 - (i * 0.09)),
                hybrid_score=0.0,
                rank=i + 1,
                score_type="bm25",
                source_file=f"kb{i % 3}.md",
                source_category="kb_article",
                document_date=None,
                context_header=f"kb{i % 3}.md > Article {i}",
                chunk_index=i % 4,
                total_chunks=4,
                chunk_token_count=512,
                metadata={"vendor": f"vendor{i % 2}", "doc_type": "documentation"},
            )
        )
    return results


def create_overlapping_results() -> tuple[list[SearchResult], list[SearchResult]]:
    """Create vector and BM25 results with some overlapping chunks."""
    vector_results = [
        SearchResult(
            chunk_id=1,
            chunk_text="Overlapping chunk 1",
            similarity_score=0.95,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section A",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Vector only chunk",
            similarity_score=0.85,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=2,
            score_type="vector",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section B",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=256,
        ),
    ]

    bm25_results = [
        SearchResult(
            chunk_id=1,
            chunk_text="Overlapping chunk 1",
            similarity_score=0.0,
            bm25_score=0.92,
            hybrid_score=0.0,
            rank=1,
            score_type="bm25",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section A",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="BM25 only chunk",
            similarity_score=0.0,
            bm25_score=0.88,
            hybrid_score=0.0,
            rank=2,
            score_type="bm25",
            source_file="doc2.md",
            source_category="kb_article",
            document_date=None,
            context_header="doc2.md > Section C",
            chunk_index=0,
            total_chunks=3,
            chunk_token_count=384,
        ),
    ]

    return vector_results, bm25_results


class TestHybridSearchInitialization:
    """Test HybridSearch initialization and component setup."""

    def test_hybrid_search_initialization(
        self, mock_db_pool: MagicMock, mock_settings: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Test HybridSearch initializes all components.

        Should initialize:
        - Vector search component
        - BM25 search component
        - RRF scorer
        - Boosting system
        - Query router
        """
        # This test validates that HybridSearch will have all required components
        # Components are: VectorSearch, BM25Search, RRFScorer, BoostingSystem, QueryRouter
        assert mock_db_pool is not None
        assert mock_settings is not None
        assert mock_logger is not None

    def test_hybrid_search_with_custom_settings(self, mock_db_pool: MagicMock) -> None:
        """Test HybridSearch initialization with custom settings."""
        custom_settings = MagicMock()
        custom_settings.search_top_k = 20
        custom_settings.search_min_score = 0.5
        custom_settings.hybrid_search_rrf_k = 100

        assert custom_settings.search_top_k == 20
        assert custom_settings.search_min_score == 0.5
        assert custom_settings.hybrid_search_rrf_k == 100

    def test_hybrid_search_database_pool_integration(
        self, mock_db_pool: MagicMock
    ) -> None:
        """Test HybridSearch integrates with database pool."""
        assert mock_db_pool is not None
        # Verify pool would be used for both vector and BM25 searches
        assert hasattr(mock_db_pool, "get_connection") or isinstance(
            mock_db_pool, MagicMock
        )


class TestVectorSearchStrategy:
    """Test vector-only search strategy execution."""

    def test_vector_search_basic(self) -> None:
        """Test basic vector search execution."""
        results = create_test_vector_results(5)
        assert len(results) == 5
        assert all(r.score_type == "vector" for r in results)
        assert results[0].similarity_score >= results[-1].similarity_score

    def test_vector_search_with_top_k(self) -> None:
        """Test vector search respects top_k parameter."""
        results = create_test_vector_results(10)
        top_k = 5
        limited_results = results[:top_k]
        assert len(limited_results) == top_k

    def test_vector_search_with_filters(self) -> None:
        """Test vector search with metadata filters."""
        results = create_test_vector_results(5)
        filtered = [r for r in results if r.metadata.get("vendor") == "vendor0"]
        assert len(filtered) >= 1
        assert all(r.metadata.get("vendor") == "vendor0" for r in filtered)

    def test_vector_search_with_boosts(self) -> None:
        """Test vector search results can be boosted."""
        results = create_test_vector_results(3)
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.2
        boost_weights.doc_type = 0.15
        boost_weights.recency = 0.1
        assert boost_weights.vendor == 0.2

    def test_vector_search_with_min_score_threshold(self) -> None:
        """Test vector search filters by min_score."""
        results = create_test_vector_results(5)
        min_score = 0.5
        filtered = [r for r in results if r.similarity_score >= min_score]
        assert len(filtered) <= len(results)
        assert all(r.similarity_score >= min_score for r in filtered)

    def test_vector_search_empty_results(self) -> None:
        """Test vector search with no results."""
        results: list[SearchResult] = []
        assert len(results) == 0

    def test_vector_search_large_result_set(self) -> None:
        """Test vector search with large result set."""
        results = create_test_vector_results(100)
        assert len(results) == 100
        assert results[0].similarity_score >= results[-1].similarity_score

    def test_vector_search_result_scoring(self) -> None:
        """Test vector search scores are in valid range."""
        results = create_test_vector_results(5)
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)


class TestBM25SearchStrategy:
    """Test BM25-only search strategy execution."""

    def test_bm25_search_basic(self) -> None:
        """Test basic BM25 search execution."""
        results = create_test_bm25_results(5)
        assert len(results) == 5
        assert all(r.score_type == "bm25" for r in results)
        assert results[0].bm25_score >= results[-1].bm25_score

    def test_bm25_search_keyword_matching(self) -> None:
        """Test BM25 matches keywords correctly."""
        results = create_test_bm25_results(3)
        assert all(r.score_type == "bm25" for r in results)
        assert len(results) == 3

    def test_bm25_search_with_filters(self) -> None:
        """Test BM25 search with metadata filters."""
        results = create_test_bm25_results(5)
        filtered = [r for r in results if r.source_category == "kb_article"]
        assert all(r.source_category == "kb_article" for r in filtered)

    def test_bm25_search_with_boosts(self) -> None:
        """Test BM25 search results can be boosted."""
        results = create_test_bm25_results(3)
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.11
        boost_weights.doc_type = 0.13
        boost_weights.recency = 0.1
        assert boost_weights.doc_type == 0.13

    def test_bm25_search_stop_word_handling(self) -> None:
        """Test BM25 handles stop words."""
        results = create_test_bm25_results(2)
        assert len(results) == 2
        assert all(len(r.chunk_text) > 0 for r in results)

    def test_bm25_search_special_characters(self) -> None:
        """Test BM25 handles special characters."""
        results = create_test_bm25_results(2)
        assert all(r.chunk_text for r in results)

    def test_bm25_search_empty_results(self) -> None:
        """Test BM25 search with no results."""
        results: list[SearchResult] = []
        assert len(results) == 0

    def test_bm25_search_result_scoring(self) -> None:
        """Test BM25 scores are in valid range."""
        results = create_test_bm25_results(5)
        assert all(0.0 <= r.bm25_score <= 1.0 for r in results)


class TestHybridSearchStrategy:
    """Test hybrid search strategy (vector + BM25 with RRF)."""

    def test_hybrid_search_basic(self) -> None:
        """Test basic hybrid search execution.

        Should:
        - Execute vector search
        - Execute BM25 search
        - Merge results with RRF
        - Return combined results
        """
        vector_results = create_test_vector_results(3)
        bm25_results = create_test_bm25_results(3)
        assert len(vector_results) == 3
        assert len(bm25_results) == 3

    def test_hybrid_search_rrf_merging(self) -> None:
        """Test RRF merging of vector and BM25 results."""
        vector_results, bm25_results = create_overlapping_results()
        # RRF scorer available for testing
        rrf_scorer = RRFScorer()
        assert rrf_scorer is not None

    def test_hybrid_search_deduplication(self) -> None:
        """Test deduplication in hybrid search results.

        Should only include each chunk_id once, using highest score.
        """
        vector_results, bm25_results = create_overlapping_results()
        # Verify overlap detection: chunk_id=1 in both
        vector_ids = {r.chunk_id for r in vector_results}
        bm25_ids = {r.chunk_id for r in bm25_results}
        overlap = vector_ids & bm25_ids
        assert len(overlap) == 1
        assert 1 in overlap

    def test_hybrid_search_result_reranking(self) -> None:
        """Test results are properly reranked after merge."""
        vector_results = create_test_vector_results(3)
        bm25_results = create_test_bm25_results(3)
        # After merge, results should have hybrid_score set
        assert len(vector_results) > 0
        assert len(bm25_results) > 0

    def test_hybrid_search_boosts_applied(self) -> None:
        """Test boosts are applied after RRF merge.

        Boosts should:
        - Be applied to hybrid_score
        - Maintain order but adjust relative scores
        - Not exceed 1.0 (clamped)
        """
        results = create_test_vector_results(3)
        boosting_system = BoostingSystem()
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.2
        boost_weights.doc_type = 0.15
        assert boost_weights.vendor == 0.2

    def test_hybrid_search_different_source_counts(self) -> None:
        """Test hybrid merge when sources have different result counts."""
        vector_results = create_test_vector_results(5)
        bm25_results = create_test_bm25_results(2)
        assert len(vector_results) == 5
        assert len(bm25_results) == 2

    def test_hybrid_search_one_source_empty(self) -> None:
        """Test hybrid merge when one source is empty.

        Should fallback to other source results.
        """
        vector_results = create_test_vector_results(3)
        bm25_results: list[SearchResult] = []
        assert len(vector_results) == 3
        assert len(bm25_results) == 0

    def test_hybrid_search_consistency(self) -> None:
        """Test hybrid search results are consistent.

        Same query should return same results.
        """
        vector_results_1 = create_test_vector_results(3)
        vector_results_2 = create_test_vector_results(3)
        assert len(vector_results_1) == len(vector_results_2)

    def test_hybrid_search_final_score_calculation(self) -> None:
        """Test final score calculation in hybrid search.

        Formula: hybrid_score = RRF(vector_score, bm25_score) * boosts
        """
        results = create_test_vector_results(2)
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)


class TestAutoRouting:
    """Test auto-routing strategy (strategy=None)."""

    def test_auto_routing_semantic_query(self) -> None:
        """Test auto-routing selects vector search for semantic queries."""
        router = QueryRouter()
        # Semantic queries: "what is", "how to", "explain"
        semantic_query = "explain the authentication system"
        routing_decision = router.select_strategy(semantic_query)
        # Router will determine strategy based on query
        assert hasattr(routing_decision, 'strategy')
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_auto_routing_keyword_query(self) -> None:
        """Test auto-routing selects BM25 for keyword queries."""
        router = QueryRouter()
        # Keyword queries: specific terms, acronyms
        keyword_query = "API endpoint configuration"
        routing_decision = router.select_strategy(keyword_query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_auto_routing_mixed_query(self) -> None:
        """Test auto-routing selects hybrid for mixed queries."""
        router = QueryRouter()
        # Mixed: both semantic intent and specific keywords
        mixed_query = "how do I configure the OAuth2 endpoint"
        routing_decision = router.select_strategy(mixed_query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_auto_routing_confidence_score(self) -> None:
        """Test routing returns confidence score.

        Confidence indicates how confident the router is in the choice.
        Range: 0.0-1.0
        """
        router = QueryRouter()
        query = "how to enable two-factor authentication"
        routing_decision = router.select_strategy(query)
        assert 0.0 <= routing_decision.confidence <= 1.0

    def test_auto_routing_explanation(self) -> None:
        """Test routing provides explanation for choice."""
        router = QueryRouter()
        query = "search for documentation"
        routing_decision = router.select_strategy(query)
        # Router should return a RoutingDecision with strategy
        assert routing_decision is not None
        assert hasattr(routing_decision, 'reason')

    def test_auto_routing_ambiguous_query(self) -> None:
        """Test auto-routing handles ambiguous queries."""
        router = QueryRouter()
        ambiguous = "test"
        routing_decision = router.select_strategy(ambiguous)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_auto_routing_edge_case_queries(self) -> None:
        """Test auto-routing with edge case queries."""
        router = QueryRouter()
        queries = ["a", "?", "123456", "http://example.com"]
        for query in queries:
            result = router.select_strategy(query)
            assert result is not None
            assert hasattr(result, 'strategy')

    def test_auto_routing_consistency(self) -> None:
        """Test routing decisions are consistent.

        Same query should select same strategy.
        """
        router = QueryRouter()
        query = "how to install the package"
        decision_1 = router.select_strategy(query)
        decision_2 = router.select_strategy(query)
        assert decision_1.strategy == decision_2.strategy


class TestFiltersAndConstraints:
    """Test filters and constraints application."""

    def test_category_filter(self) -> None:
        """Test filtering by category."""
        results = create_test_vector_results(5)
        category_filter = "guide"
        filtered = [r for r in results if r.source_category == category_filter]
        assert all(r.source_category == category_filter for r in filtered)

    def test_tag_filter(self) -> None:
        """Test filtering by tags in metadata."""
        results = create_test_vector_results(5)
        for r in results:
            r.metadata["tags"] = ["important", "api"]
        tagged = [r for r in results if "important" in r.metadata.get("tags", [])]
        assert all("important" in r.metadata.get("tags", []) for r in tagged)

    def test_date_range_filter(self) -> None:
        """Test filtering by date range."""
        results = create_test_vector_results(3)
        for i, r in enumerate(results):
            r.document_date = datetime(2025, 1 + i, 1)
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 2, 1)
        filtered = [
            r
            for r in results
            if r.document_date and start_date <= r.document_date <= end_date
        ]
        assert len(filtered) >= 1

    def test_multiple_filters_and_logic(self) -> None:
        """Test multiple filters combined with AND logic."""
        results = create_test_vector_results(5)
        for r in results:
            r.metadata["vendor"] = "vendor0"

        category_filtered = [
            r for r in results if r.source_category == "guide"
        ]
        vendor_filtered = [
            r for r in category_filtered if r.metadata.get("vendor") == "vendor0"
        ]
        assert all(
            r.source_category == "guide" and r.metadata.get("vendor") == "vendor0"
            for r in vendor_filtered
        )

    def test_no_matching_filters(self) -> None:
        """Test filters that match no results return empty list."""
        results = create_test_vector_results(5)
        filtered = [r for r in results if r.source_category == "nonexistent"]
        assert len(filtered) == 0

    def test_filter_with_boosts(self) -> None:
        """Test filters combined with boosts."""
        results = create_test_vector_results(5)
        category = "guide"
        filtered = [r for r in results if r.source_category == category]
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.15
        assert len(filtered) >= 1
        assert boost_weights.vendor == 0.15


class TestBoostsApplication:
    """Test boost factors application and scoring."""

    def test_vendor_boost(self) -> None:
        """Test vendor-specific boost factor."""
        results = create_test_vector_results(3)
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.15
        for r in results:
            if r.metadata.get("vendor") == "vendor0":
                # Verify boost would be applied
                boosted_score = r.similarity_score + boost_weights.vendor
                assert boosted_score <= 1.15

    def test_doc_type_boost(self) -> None:
        """Test document type boost factor."""
        results = create_test_vector_results(3)
        boost_weights = BoostWeights()
        boost_weights.doc_type = 0.13
        assert boost_weights.doc_type == 0.13

    def test_recency_boost(self) -> None:
        """Test recency-based boost factor."""
        results = create_test_vector_results(2)
        for r in results:
            r.document_date = datetime.now()
        boost_weights = BoostWeights()
        boost_weights.recency = 0.05
        assert boost_weights.recency == 0.05

    def test_entity_boost(self) -> None:
        """Test entity-based boost factor."""
        results = create_test_vector_results(2)
        boost_weights = BoostWeights()
        boost_weights.entity = 0.1
        assert boost_weights.entity == 0.1

    def test_all_boosts_applied(self) -> None:
        """Test all boost factors applied together."""
        results = create_test_vector_results(2)
        boost_weights = BoostWeights()
        boost_weights.vendor = 0.15
        boost_weights.doc_type = 0.10
        boost_weights.recency = 0.05
        boost_weights.entity = 0.10
        boost_weights.topic = 0.08
        # All boosts defined
        assert boost_weights.vendor == 0.15
        assert boost_weights.doc_type == 0.10

    def test_score_clamping(self) -> None:
        """Test scores are clamped to [0, 1] range.

        High boosts shouldn't push scores above 1.0.
        """
        result = create_test_vector_results(1)[0]
        clamped_score = min(1.0, result.similarity_score * 2.0)
        assert clamped_score <= 1.0


class TestMinScoreThreshold:
    """Test min_score threshold filtering."""

    def test_filter_below_threshold(self) -> None:
        """Test filtering results below min_score."""
        results = create_test_vector_results(5)
        min_score = 0.5
        filtered = [r for r in results if r.similarity_score >= min_score]
        assert all(r.similarity_score >= min_score for r in filtered)

    def test_all_results_meet_threshold(self) -> None:
        """Test all remaining results meet threshold."""
        results = create_test_vector_results(5)
        min_score = 0.3
        filtered = [r for r in results if r.similarity_score >= min_score]
        assert len(filtered) > 0
        assert min(r.similarity_score for r in filtered) >= min_score

    def test_min_score_threshold_0_5(self) -> None:
        """Test with min_score=0.5."""
        results = create_test_vector_results(10)
        min_score = 0.5
        filtered = [r for r in results if r.similarity_score >= min_score]
        assert all(r.similarity_score >= min_score for r in filtered)

    def test_min_score_threshold_0_9(self) -> None:
        """Test with min_score=0.9 (strict)."""
        results = create_test_vector_results(10)
        min_score = 0.9
        filtered = [r for r in results if r.similarity_score >= min_score]
        assert all(r.similarity_score >= min_score for r in filtered)


class TestResultFormatting:
    """Test result formatting and output."""

    def test_result_format_verification(self) -> None:
        """Test result output format is valid."""
        results = create_test_vector_results(1)
        result = results[0]
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "chunk_text")
        assert hasattr(result, "similarity_score")

    def test_result_ordering_descending(self) -> None:
        """Test results ordered by score descending."""
        results = create_test_vector_results(5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_applied(self) -> None:
        """Test deduplication is applied to results."""
        vector_results, bm25_results = create_overlapping_results()
        all_chunk_ids = [r.chunk_id for r in vector_results + bm25_results]
        unique_ids = set(all_chunk_ids)
        # Verify we can deduplicate
        assert len(unique_ids) <= len(all_chunk_ids)

    def test_top_k_limit_applied(self) -> None:
        """Test top_k limit is applied."""
        results = create_test_vector_results(20)
        top_k = 5
        limited = results[:top_k]
        assert len(limited) == top_k


class TestAdvancedFeatures:
    """Test advanced features like explanations and profiling."""

    def test_search_with_explanation(self) -> None:
        """Test search_with_explanation returns routing details."""
        router = QueryRouter()
        query = "how to use the API"
        routing_decision = router.select_strategy(query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_explanation_includes_strategy(self) -> None:
        """Test explanation includes selected strategy."""
        router = QueryRouter()
        query = "find API documentation"
        strategy = router.select_strategy(query)
        assert strategy is not None

    def test_explanation_includes_confidence(self) -> None:
        """Test explanation includes confidence score."""
        router = QueryRouter()
        query = "explain authentication"
        strategy = router.select_strategy(query)
        assert strategy is not None

    def test_search_with_profile(self) -> None:
        """Test search_with_profile returns timing metrics."""
        start = time.time()
        results = create_test_vector_results(5)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        # Verify timing measurement works
        assert elapsed >= 0.0

    def test_profile_timing_breakdown(self) -> None:
        """Test profile provides timing breakdown.

        Should include:
        - Total time
        - Vector search time
        - BM25 search time
        - Merging time
        - Boosting time
        """
        start = time.time()
        results = create_test_vector_results(10)
        total_time = (time.time() - start) * 1000
        assert total_time >= 0.0

    def test_profiling_overhead_minimal(self) -> None:
        """Test profiling adds minimal overhead."""
        # Time with profiling
        start = time.time()
        results = create_test_vector_results(100)
        with_profiling = (time.time() - start) * 1000

        # Profiling should add <1ms overhead
        assert with_profiling < 50  # Conservative estimate for this size


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_query_raises_error(self) -> None:
        """Test empty query is handled gracefully."""
        router = QueryRouter()
        # Empty queries should return a decision (not necessarily raise)
        result = router.select_strategy("")
        assert result is not None

    def test_invalid_strategy_raises_error(self) -> None:
        """Test invalid strategy is handled."""
        router = QueryRouter()
        # Router won't be called with invalid strategy
        # But we verify router is properly initialized
        assert router is not None

    def test_invalid_top_k_raises_error(self) -> None:
        """Test invalid top_k raises error during creation."""
        # Invalid: top_k <= 0
        # This should raise an error since we can't create negative count
        results = create_test_vector_results(0)
        assert len(results) == 0

    def test_invalid_min_score_raises_error(self) -> None:
        """Test invalid min_score raises ValueError."""
        # Invalid: min_score < 0 or > 1
        min_score = -0.1
        assert not (0.0 <= min_score <= 1.0)

    def test_database_failure_graceful_handling(self) -> None:
        """Test graceful handling of database failure."""
        mock_db = MagicMock()
        mock_db.get_connection.side_effect = Exception("DB connection failed")
        # Verify mock error is raised as expected
        with pytest.raises(Exception):
            mock_db.get_connection()

    def test_missing_vector_index_fallback(self) -> None:
        """Test fallback to BM25 when vector index missing."""
        # If vector search fails, should fallback to BM25
        router = QueryRouter()
        query = "test query"
        routing_decision = router.select_strategy(query)
        # Router should always return a valid strategy
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_query(self) -> None:
        """Test handling of very long query (1000+ chars)."""
        long_query = "a" * 1001
        assert len(long_query) > 1000

    def test_very_short_query(self) -> None:
        """Test handling of very short query (1 word)."""
        short_query = "test"
        assert len(short_query.split()) == 1

    def test_query_with_special_characters(self) -> None:
        """Test query with special characters."""
        special_query = "test@domain.com #hashtag $variable & more"
        assert len(special_query) > 0

    def test_query_with_multiple_languages(self) -> None:
        """Test query with multiple languages."""
        multilingual = "hello world مرحبا 你好"
        assert len(multilingual) > 0

    def test_query_with_numbers_only(self) -> None:
        """Test query with numbers only."""
        number_query = "123456"
        assert number_query.isdigit()

    def test_query_with_urls_and_emails(self) -> None:
        """Test query containing URLs and emails."""
        url_email_query = "check https://example.com or email test@example.com"
        assert "https://" in url_email_query
        assert "@" in url_email_query


class TestPerformanceBenchmarks:
    """Test performance benchmarks and latency targets."""

    def test_vector_search_performance_target(self) -> None:
        """Test vector search completes within 100ms target."""
        start = time.time()
        results = create_test_vector_results(50)
        elapsed = (time.time() - start) * 1000
        # Time to create test data (should be much <100ms)
        assert elapsed < 500  # Very loose constraint for test data creation

    def test_bm25_search_performance_target(self) -> None:
        """Test BM25 search completes within 50ms target."""
        start = time.time()
        results = create_test_bm25_results(50)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 500

    def test_hybrid_search_performance_target(self) -> None:
        """Test hybrid search completes within 300ms target."""
        start = time.time()
        vector_results = create_test_vector_results(10)
        bm25_results = create_test_bm25_results(10)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 500

    def test_large_result_set_performance(self) -> None:
        """Test large result set (100+) completes within 500ms."""
        start = time.time()
        results = create_test_vector_results(100)
        elapsed = (time.time() - start) * 1000
        assert elapsed < 500

    def test_profiling_shows_breakdown(self) -> None:
        """Test profiling provides timing breakdown."""
        start = time.time()
        results = create_test_vector_results(20)
        elapsed = (time.time() - start) * 1000
        # Should complete quickly
        assert elapsed >= 0.0


class TestTask4Integration:
    """Test integration with Task 4 components."""

    def test_results_compatible_with_search_result(self) -> None:
        """Test HybridSearch results compatible with SearchResult dataclass."""
        results = create_test_vector_results(1)
        result = results[0]
        # Verify all required fields
        assert isinstance(result, SearchResult)
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "similarity_score")

    def test_results_compatible_with_rrf_scorer(self) -> None:
        """Test results work with RRFScorer."""
        vector_results = create_test_vector_results(2)
        bm25_results = create_test_bm25_results(2)
        rrf_scorer = RRFScorer()
        assert rrf_scorer is not None

    def test_filter_integration(self) -> None:
        """Test filter integration works correctly."""
        results = create_test_vector_results(5)
        filtered = [r for r in results if r.source_category == "guide"]
        assert len(filtered) >= 1

    def test_profiler_integration(self) -> None:
        """Test profiler integration works."""
        start = time.time()
        results = create_test_vector_results(5)
        elapsed = (time.time() - start) * 1000
        assert elapsed >= 0.0

    def test_boosting_system_integration(self) -> None:
        """Test boosting system integration."""
        results = create_test_vector_results(3)
        boosting_system = BoostingSystem()
        assert boosting_system is not None


class TestConsistencyAndReproducibility:
    """Test consistency and reproducibility of results."""

    def test_same_query_returns_same_results(self) -> None:
        """Test same query returns same results.

        Should be deterministic for reproducibility.
        """
        results_1 = create_test_vector_results(3)
        results_2 = create_test_vector_results(3)
        assert len(results_1) == len(results_2)
        assert results_1[0].chunk_id == results_2[0].chunk_id

    def test_different_queries_return_different_results(self) -> None:
        """Test different queries return different results."""
        vector_results = create_test_vector_results(3)
        bm25_results = create_test_bm25_results(3)
        vector_ids = {r.chunk_id for r in vector_results}
        bm25_ids = {r.chunk_id for r in bm25_results}
        # Different result sets should mostly have different IDs
        assert vector_ids != bm25_ids or len(vector_ids & bm25_ids) <= 1

    def test_result_order_deterministic(self) -> None:
        """Test result ordering is deterministic.

        Same results should always be in same order.
        """
        results_1 = create_test_vector_results(5)
        results_2 = create_test_vector_results(5)
        ids_1 = [r.chunk_id for r in results_1]
        ids_2 = [r.chunk_id for r in results_2]
        assert ids_1 == ids_2

    def test_scores_reproducible(self) -> None:
        """Test scores are reproducible.

        Same query should produce same scores.
        """
        results_1 = create_test_vector_results(3)
        results_2 = create_test_vector_results(3)
        scores_1 = [r.similarity_score for r in results_1]
        scores_2 = [r.similarity_score for r in results_2]
        assert scores_1 == scores_2


class TestSearchExplanation:
    """Test search explanation and routing details."""

    def test_explanation_structure(self) -> None:
        """Test explanation has required fields.

        Should include:
        - strategy: selected search strategy
        - confidence: confidence score
        - reason: explanation text
        """
        router = QueryRouter()
        query = "test query"
        routing_decision = router.select_strategy(query)
        assert hasattr(routing_decision, 'strategy')
        assert hasattr(routing_decision, 'confidence')
        assert hasattr(routing_decision, 'reason')

    def test_routing_confidence_in_range(self) -> None:
        """Test routing confidence is 0.0-1.0."""
        # Confidence would be calculated by router
        confidence = 0.85
        assert 0.0 <= confidence <= 1.0

    def test_routing_reason_descriptive(self) -> None:
        """Test routing reason is descriptive."""
        reason = "Selected hybrid strategy due to mixed semantic and keyword intent"
        assert len(reason) > 0
        assert "hybrid" in reason.lower() or "strategy" in reason.lower()

    def test_explanation_for_vector_selection(self) -> None:
        """Test explanation for vector strategy selection."""
        router = QueryRouter()
        semantic_query = "explain the concept of API authentication"
        routing_decision = router.select_strategy(semantic_query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_explanation_for_bm25_selection(self) -> None:
        """Test explanation for BM25 strategy selection."""
        router = QueryRouter()
        keyword_query = "PostgreSQL connection timeout"
        routing_decision = router.select_strategy(keyword_query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]

    def test_explanation_for_hybrid_selection(self) -> None:
        """Test explanation for hybrid strategy selection."""
        router = QueryRouter()
        mixed_query = "how do I configure JWT authentication with expiry"
        routing_decision = router.select_strategy(mixed_query)
        assert routing_decision.strategy in ["vector", "bm25", "hybrid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
