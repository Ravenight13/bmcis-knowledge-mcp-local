"""Test response_formatter helper functions.

Tests token estimation, confidence scoring, ranking context generation,
envelope wrapping, and warning generation.

Test Coverage:
- Token estimation across all response modes
- Confidence score calculation from score distributions
- Ranking context with percentile calculations
- Deduplication detection
- Response warning generation
- Envelope wrapping for semantic_search and find_vendor_info
"""

from datetime import datetime

import pytest

from src.mcp.models import (
    ExecutionContext,
    MCPResponseEnvelope,
    PaginationMetadata,
    ResponseMetadata,
    SearchResultMetadata,
    VendorInfoMetadata,
    VendorStatistics,
)
from src.mcp.response_formatter import (
    RESPONSE_SIZE_WARNING_THRESHOLD,
    VENDOR_ENTITY_WARNING_THRESHOLD,
    apply_compression_to_envelope,
    calculate_confidence_scores,
    detect_duplicates,
    estimate_response_tokens,
    generate_ranking_context,
    generate_response_warnings,
    wrap_semantic_search_response,
    wrap_vendor_info_response,
)
from src.search.results import SearchResult


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample SearchResult list for testing."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="High relevance content",
            similarity_score=0.95,
            bm25_score=0.85,
            hybrid_score=0.90,
            rank=1,
            score_type="hybrid",
            source_file="docs/auth.md",
            source_category="security",
            document_date=datetime(2024, 1, 1),
            context_header="auth.md > Security",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
            metadata={},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Medium relevance content",
            similarity_score=0.75,
            bm25_score=0.65,
            hybrid_score=0.70,
            rank=2,
            score_type="hybrid",
            source_file="docs/api.md",
            source_category="api",
            document_date=datetime(2024, 1, 1),
            context_header="api.md > Reference",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
            metadata={},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Low relevance content",
            similarity_score=0.55,
            bm25_score=0.45,
            hybrid_score=0.50,
            rank=3,
            score_type="hybrid",
            source_file="docs/guide.md",
            source_category="docs",
            document_date=datetime(2024, 1, 1),
            context_header="guide.md > Tutorial",
            chunk_index=0,
            total_chunks=3,
            chunk_token_count=128,
            metadata={},
        ),
    ]


class TestTokenEstimation:
    """Test token estimation functions."""

    def test_estimate_ids_only(self) -> None:
        """Test token estimation for ids_only mode."""
        tokens = estimate_response_tokens(10, "ids_only", include_metadata=False)
        assert tokens == 100  # 10 * 10

    def test_estimate_metadata(self) -> None:
        """Test token estimation for metadata mode."""
        tokens = estimate_response_tokens(10, "metadata", include_metadata=False)
        assert tokens == 2000  # 10 * 200

    def test_estimate_preview(self) -> None:
        """Test token estimation for preview mode."""
        tokens = estimate_response_tokens(10, "preview", include_metadata=False)
        assert tokens == 8000  # 10 * 800

    def test_estimate_full(self) -> None:
        """Test token estimation for full mode."""
        tokens = estimate_response_tokens(10, "full", include_metadata=False)
        assert tokens == 15000  # 10 * 1500

    def test_estimate_with_metadata_overhead(self) -> None:
        """Test token estimation includes envelope metadata overhead."""
        tokens = estimate_response_tokens(10, "metadata", include_metadata=True)
        assert tokens == 2300  # 10 * 200 + 300

    def test_estimate_unknown_mode_defaults_to_metadata(self) -> None:
        """Test unknown response_mode defaults to metadata."""
        tokens = estimate_response_tokens(10, "unknown", include_metadata=False)
        assert tokens == 2000  # Same as metadata


class TestConfidenceScores:
    """Test confidence score calculation."""

    def test_calculate_confidence_scores(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test confidence scores calculated from score distribution."""
        confidence_map = calculate_confidence_scores(sample_search_results)

        # Check all results have confidence scores
        assert len(confidence_map) == 3
        assert 1 in confidence_map
        assert 2 in confidence_map
        assert 3 in confidence_map

        # Top result should have highest score_reliability
        assert confidence_map[1].score_reliability > confidence_map[2].score_reliability
        assert confidence_map[2].score_reliability > confidence_map[3].score_reliability

        # All should have fixed source_quality and recency
        for conf in confidence_map.values():
            assert conf.source_quality == 0.9
            assert conf.recency == 0.85

    def test_calculate_confidence_empty_results(self) -> None:
        """Test confidence calculation with empty results."""
        confidence_map = calculate_confidence_scores([])
        assert confidence_map == {}

    def test_calculate_confidence_single_result(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test confidence calculation with single result."""
        confidence_map = calculate_confidence_scores([sample_search_results[0]])
        assert len(confidence_map) == 1
        # Single result gets 100% percentile (1.0 score_reliability)
        assert confidence_map[1].score_reliability == 1.0


class TestRankingContext:
    """Test ranking context generation."""

    def test_generate_ranking_context(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test ranking context with percentiles and explanations."""
        ranking_map = generate_ranking_context(sample_search_results)

        # Check all results have ranking context
        assert len(ranking_map) == 3

        # Top result should have 100 percentile
        assert ranking_map[1].percentile == 100
        assert ranking_map[1].explanation == "Highest combined semantic + keyword match"

        # Second result should have lower percentile
        assert ranking_map[2].percentile < 100
        assert "high relevance" in ranking_map[2].explanation.lower()

        # Third result should have lowest percentile
        assert ranking_map[3].percentile < ranking_map[2].percentile

        # All should have hybrid score_method
        for ranking in ranking_map.values():
            assert ranking.score_method == "hybrid"

    def test_generate_ranking_empty_results(self) -> None:
        """Test ranking generation with empty results."""
        ranking_map = generate_ranking_context([])
        assert ranking_map == {}


class TestDeduplication:
    """Test duplicate detection."""

    def test_detect_duplicates_no_duplicates(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test deduplication with well-separated scores."""
        dedup_map = detect_duplicates(sample_search_results, similarity_threshold=0.95)

        # No results should be marked as duplicates
        for dedup_info in dedup_map.values():
            assert not dedup_info.is_duplicate
            assert dedup_info.confidence == 0.85

    def test_detect_duplicates_similar_scores(self) -> None:
        """Test deduplication detects similar scores."""
        # Create results with very similar scores (within 1% difference)
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Content 1",
                similarity_score=0.90,
                bm25_score=0.90,
                hybrid_score=0.90,
                rank=1,
                score_type="hybrid",
                source_file="doc1.md",
                source_category="docs",
                document_date=datetime(2024, 1, 1),
                context_header="doc1",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=100,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Content 2",
                similarity_score=0.895,
                bm25_score=0.895,
                hybrid_score=0.895,
                rank=2,
                score_type="hybrid",
                source_file="doc2.md",
                source_category="docs",
                document_date=datetime(2024, 1, 1),
                context_header="doc2",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=100,
                metadata={},
            ),
        ]

        # Use 0.99 threshold (1% difference allowed)
        dedup_map = detect_duplicates(results, similarity_threshold=0.99)

        # Second result should be marked as duplicate (score diff < 0.01, lower rank)
        assert dedup_map[2].is_duplicate
        assert 1 in dedup_map[2].similar_chunk_ids

    def test_detect_duplicates_empty_results(self) -> None:
        """Test deduplication with empty results."""
        dedup_map = detect_duplicates([])
        assert dedup_map == {}


class TestWarningGeneration:
    """Test response warning generation."""

    def test_generate_warnings_no_issues(self) -> None:
        """Test warning generation with normal response."""
        warnings = generate_response_warnings(
            response_size_bytes=10_000,
            result_count=10,
            response_mode="metadata",
            entity_count=None,
        )
        assert len(warnings) == 0

    def test_generate_warnings_large_response(self) -> None:
        """Test warning for large response size."""
        warnings = generate_response_warnings(
            response_size_bytes=RESPONSE_SIZE_WARNING_THRESHOLD + 1000,
            result_count=10,
            response_mode="metadata",
        )
        assert len(warnings) == 1
        assert warnings[0].code == "RESPONSE_SIZE_LARGE"
        assert warnings[0].level == "warning"

    def test_generate_warnings_excessive_response(self) -> None:
        """Test error for excessive response size."""
        warnings = generate_response_warnings(
            response_size_bytes=150_000,
            result_count=10,
            response_mode="full",
        )
        # Should have excessive size error
        assert any(w.code == "RESPONSE_SIZE_EXCESSIVE" for w in warnings)
        excessive_warning = next(w for w in warnings if w.code == "RESPONSE_SIZE_EXCESSIVE")
        assert excessive_warning.level == "error"

    def test_generate_warnings_large_entity_count(self) -> None:
        """Test warning for large vendor entity count."""
        warnings = generate_response_warnings(
            response_size_bytes=10_000,
            result_count=1,
            response_mode="metadata",
            entity_count=VENDOR_ENTITY_WARNING_THRESHOLD + 10,
        )
        assert len(warnings) == 1
        assert warnings[0].code == "ENTITY_GRAPH_LARGE"
        assert warnings[0].level == "warning"

    def test_generate_warnings_high_result_count_full_mode(self) -> None:
        """Test warning for high result count with full mode."""
        warnings = generate_response_warnings(
            response_size_bytes=10_000,
            result_count=25,
            response_mode="full",
        )
        assert len(warnings) == 1
        assert warnings[0].code == "RESULT_COUNT_HIGH"
        assert warnings[0].level == "warning"


class TestSemanticSearchEnvelope:
    """Test semantic search envelope wrapping."""

    def test_wrap_semantic_search_response(self) -> None:
        """Test wrapping semantic search results in envelope."""
        results = [
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/auth.md",
                source_category="security",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=10,
            )
        ]

        envelope = wrap_semantic_search_response(
            results=results,
            total_found=42,
            execution_time_ms=245.3,
            cache_hit=True,
            response_mode="metadata",
        )

        # Check envelope structure
        assert isinstance(envelope, MCPResponseEnvelope)
        assert isinstance(envelope.metadata, ResponseMetadata)
        assert envelope.metadata.operation == "semantic_search"
        assert envelope.metadata.version == "1.0"
        assert envelope.metadata.status == "success"

        # Check execution context
        assert isinstance(envelope.execution_context, ExecutionContext)
        assert envelope.execution_context.cache_hit is True
        assert envelope.execution_context.execution_time_ms == 245.3
        assert envelope.execution_context.tokens_estimated > 0

        # Check results data
        assert isinstance(envelope.results, dict)
        assert envelope.results["total_found"] == 42
        assert envelope.results["strategy_used"] == "hybrid"
        assert len(envelope.results["results"]) == 1

    def test_wrap_semantic_search_with_pagination(self) -> None:
        """Test envelope with pagination metadata."""
        import base64
        import json

        results = [
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/auth.md",
                source_category="security",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=10,
            )
        ]

        # Create valid cursor (base64-encoded JSON)
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata"
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        envelope = wrap_semantic_search_response(
            results=results,
            total_found=42,
            execution_time_ms=245.3,
            cache_hit=False,
            pagination=pagination,
            response_mode="metadata",
        )

        # Check pagination is included
        assert envelope.pagination is not None
        assert envelope.pagination.cursor == cursor
        assert envelope.pagination.has_more is True


class TestVendorInfoEnvelope:
    """Test vendor info envelope wrapping."""

    def test_wrap_vendor_info_response(self) -> None:
        """Test wrapping vendor info in envelope."""
        vendor_stats = VendorStatistics(
            entity_count=25,
            relationship_count=15,
            entity_type_distribution={"ORG": 10, "PRODUCT": 15},
            relationship_type_distribution={"PARTNER": 10, "SUPPLIER": 5},
        )

        vendor_info = VendorInfoMetadata(
            vendor_name="Acme Corp",
            statistics=vendor_stats,
            last_updated="2025-11-09T12:00:00Z",
        )

        envelope = wrap_vendor_info_response(
            vendor_name="Acme Corp",
            results=vendor_info,
            execution_time_ms=320.5,
            cache_hit=False,
            entity_count=25,
        )

        # Check envelope structure
        assert isinstance(envelope, MCPResponseEnvelope)
        assert envelope.metadata.operation == "find_vendor_info"
        assert envelope.metadata.version == "1.0"

        # Check execution context
        assert envelope.execution_context.cache_hit is False
        assert envelope.execution_context.execution_time_ms == 320.5

        # Check results data
        assert isinstance(envelope.results, dict)
        assert envelope.results["vendor_name"] == "Acme Corp"

    def test_wrap_vendor_info_with_warnings(self) -> None:
        """Test vendor info envelope generates warnings for large graphs."""
        vendor_stats = VendorStatistics(
            entity_count=75,
            relationship_count=50,
        )

        vendor_info = VendorInfoMetadata(
            vendor_name="Large Corp",
            statistics=vendor_stats,
        )

        envelope = wrap_vendor_info_response(
            vendor_name="Large Corp",
            results=vendor_info,
            execution_time_ms=500.0,
            cache_hit=False,
            entity_count=75,
        )

        # Should have warning about large entity count
        assert len(envelope.warnings) > 0
        assert any(w.code == "ENTITY_GRAPH_LARGE" for w in envelope.warnings)


class TestCompressionConfig:
    """Test envelope compression configuration."""

    def test_apply_compression_no_op(self) -> None:
        """Test compression is currently a no-op placeholder."""
        results = [
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/auth.md",
                source_category="security",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=10,
            )
        ]

        envelope = wrap_semantic_search_response(
            results=results,
            total_found=10,
            execution_time_ms=100.0,
            cache_hit=False,
            response_mode="metadata",
        )

        # Apply compression (should be no-op)
        compressed = apply_compression_to_envelope(envelope, config={"max_tokens": 5000})

        # Should be unchanged
        assert compressed == envelope
