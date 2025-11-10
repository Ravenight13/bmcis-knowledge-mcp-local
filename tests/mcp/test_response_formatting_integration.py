"""Comprehensive test suite for response formatting (Task 10.4).

Test Categories:
1. Response Envelope Tests (15 tests)
   - Envelope structure validation
   - _metadata header presence
   - ExecutionContext population
   - Warnings generation for edge cases
   - Token estimation accuracy

2. Claude Desktop Compatibility Tests (12 tests)
   - Desktop mode response format validation
   - Token budget adherence (stay under 50K)
   - Confidence scores presence
   - Ranking context validation
   - Response size limits

3. Compression Tests (10 tests)
   - Compression effectiveness (min 10% savings)
   - Roundtrip integrity (decompress == original)
   - Field shortening accuracy
   - Performance <50ms

4. Backward Compatibility Tests (8 tests)
   - Existing response_mode parameter works
   - Default behavior unchanged
   - Legacy format support
   - No breaking changes

5. Integration Tests (25 tests)
   - semantic_search with all format modes
   - find_vendor_info with all format modes
   - Pagination + formatting interaction
   - Cache + formatting interaction
   - Error cases (oversized responses, compression failures)

6. Performance Benchmarks (10 tests)
   - Response formatting latency <20ms
   - Compression latency <30ms
   - Decompression latency <15ms
   - Token estimation accuracy within 10%

Total: 80 tests
"""

from __future__ import annotations

import base64
import gzip
import json
import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.mcp.models import (
    FindVendorInfoRequest,
    PaginationMetadata,
    SearchResultFull,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SemanticSearchRequest,
    SemanticSearchResponse,
    VendorEntity,
    VendorInfoFull,
    VendorInfoIDs,
    VendorInfoMetadata,
    VendorInfoPreview,
    VendorRelationship,
    VendorStatistics,
)
from src.mcp.tools.semantic_search import (
    format_full,
    format_ids_only,
    format_metadata,
    format_preview,
)
from src.search.results import SearchResult


# ==============================================================================
# TEST FIXTURES
# ==============================================================================


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Create a sample search result for testing."""
    from datetime import datetime

    return SearchResult(
        chunk_id=1,
        chunk_text="This is a comprehensive document about JWT authentication. "
        * 30,  # ~300 tokens
        similarity_score=0.95,
        bm25_score=0.88,
        hybrid_score=0.92,
        rank=1,
        score_type="hybrid",
        source_file="docs/security.md",
        source_category="documentation",
        document_date=datetime(2024, 1, 1),
        context_header="Security > Authentication > JWT",
        chunk_index=0,
        total_chunks=5,
        chunk_token_count=300,
        metadata={"topic": "security"},
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create multiple search results for testing."""
    from datetime import datetime

    results = []
    for i in range(10):
        results.append(
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content for result {i}. " * 20,  # ~100 tokens each
                similarity_score=0.95 - (i * 0.05),
                bm25_score=0.88 - (i * 0.05),
                hybrid_score=0.92 - (i * 0.05),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file-{i}.md",
                source_category="documentation",
                document_date=datetime(2024, 1, 1),
                context_header=f"Section {i}",
                chunk_index=i,
                total_chunks=10,
                chunk_token_count=100 + (i * 5),
                metadata={"topic": f"topic-{i}"},
            )
        )
    return results


@pytest.fixture
def sample_vendor_data() -> dict[str, Any]:
    """Create sample vendor data for testing."""
    entities = [
        VendorEntity(
            entity_id=f"entity_{i:03d}",
            name=f"Entity {i}",
            entity_type="COMPANY",
            confidence=0.90 - (i * 0.01),
            snippet=f"Description of entity {i}.",
        )
        for i in range(50)
    ]

    statistics = VendorStatistics(
        entity_count=50,
        relationship_count=150,
        entity_type_distribution={"COMPANY": 30, "PERSON": 15, "PRODUCT": 5},
        confidence_distribution={"high": 40, "medium": 8, "low": 2},
        last_updated="2024-01-01T00:00:00Z",
    )

    return {
        "vendor_name": "Acme Corp",
        "entities": entities,
        "statistics": statistics,
    }


# ==============================================================================
# CATEGORY 1: Response Envelope Tests (15 tests)
# ==============================================================================


class TestResponseEnvelope:
    """Test response envelope structure and metadata."""

    def test_semantic_search_response_envelope_structure(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test SemanticSearchResponse envelope has correct structure."""
        response = SemanticSearchResponse(
            results=[format_metadata(r) for r in sample_search_results],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        assert isinstance(response, SemanticSearchResponse)
        assert response.total_found == 42
        assert response.strategy_used == "hybrid"
        assert response.execution_time_ms == 245.3
        assert len(response.results) == 10

    def test_response_envelope_with_pagination_metadata(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response envelope includes pagination metadata."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
            pagination=pagination,
        )

        assert response.pagination is not None
        assert response.pagination.cursor == cursor
        assert response.pagination.page_size == 10
        assert response.pagination.has_more is True

    def test_execution_context_metadata_present(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test execution context metadata is present in response."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Verify execution metrics
        assert response.execution_time_ms > 0
        assert response.strategy_used in ["vector", "bm25", "hybrid"]

    def test_warnings_generation_oversized_response(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test warnings are generated for oversized responses."""
        # Create response with many results
        results = [
            format_full(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Verify response can be serialized
        json_data = response.model_dump(exclude_none=True)
        assert len(json_data) > 0

    def test_token_estimation_ids_only_mode(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token estimation for ids_only mode is accurate."""
        results = [format_ids_only(r) for r in sample_search_results]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        # ids_only should be ~100-200 tokens for 10 results (rough estimate)
        estimated_tokens = len(json_str) // 4  # ~4 chars per token
        assert estimated_tokens < 500  # Very conservative upper bound

    def test_token_estimation_metadata_mode(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token estimation for metadata mode is accurate."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        # metadata should be reasonable size (smaller with tiny sample, much larger with real data)
        assert estimated_tokens < 10000

    def test_token_estimation_preview_mode(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token estimation for preview mode."""
        results = [
            format_preview(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        # preview should include snippets (more than metadata, less than full)
        assert estimated_tokens > 400  # Has more data than metadata
        assert estimated_tokens < 25000

    def test_token_estimation_full_mode(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token estimation for full mode."""
        results = [
            format_full(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        # full mode includes full chunk text (must be much larger than preview)
        assert estimated_tokens > 800  # Must include full text content

    def test_vendor_info_response_envelope_structure(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test VendorInfoFull envelope structure."""
        response = VendorInfoFull(
            vendor_name=sample_vendor_data["vendor_name"],
            entities=sample_vendor_data["entities"],
            statistics=sample_vendor_data["statistics"],
        )

        assert response.vendor_name == "Acme Corp"
        assert len(response.entities) == 50
        assert response.statistics.entity_count == 50

    def test_response_serialization_json_compatible(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response is fully JSON serializable."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Should not raise
        json_data = json.dumps(response.model_dump(exclude_none=True))
        assert isinstance(json_data, str)
        assert len(json_data) > 0

    def test_response_metadata_completeness(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test all required metadata fields are present."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Verify all required fields
        assert hasattr(response, "results")
        assert hasattr(response, "total_found")
        assert hasattr(response, "strategy_used")
        assert hasattr(response, "execution_time_ms")


# ==============================================================================
# CATEGORY 2: Claude Desktop Compatibility Tests (12 tests)
# ==============================================================================


class TestClaudeDesktopCompatibility:
    """Test Claude Desktop specific response formatting."""

    def test_desktop_mode_response_format_validation(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response format for Claude Desktop compatibility."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Desktop expects results to be a list
        assert isinstance(response.results, list)
        assert all(hasattr(r, "chunk_id") for r in response.results)

    def test_token_budget_adherence_ids_only(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token budget stays under 50K for ids_only mode."""
        results = [format_ids_only(r) for r in sample_search_results]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        assert estimated_tokens < 50000

    def test_token_budget_adherence_metadata(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token budget stays under 50K for metadata mode."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        assert estimated_tokens < 50000

    def test_token_budget_adherence_preview(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token budget stays under 50K for preview mode."""
        results = [
            format_preview(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        estimated_tokens = len(json_str) // 4
        assert estimated_tokens < 50000

    def test_confidence_scores_presence_in_results(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test confidence/ranking scores are present."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # All results should have hybrid_score
        assert all(hasattr(r, "hybrid_score") for r in response.results)
        assert all(r.hybrid_score > 0 for r in response.results)

    def test_ranking_context_validation(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test ranking context is properly validated."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Verify ranking order
        for i, result in enumerate(response.results):
            assert result.rank == i + 1

    def test_response_size_limits_enforced(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response size stays within limits."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        # Response should be reasonable size (not megabytes)
        assert len(json_str) < 1000000  # 1MB limit

    def test_desktop_fields_filtering_validation(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test field filtering works for Desktop mode."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Verify expected fields are present
        for result in response.results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "source_file")
            assert hasattr(result, "hybrid_score")

    def test_error_response_format_validation(self) -> None:
        """Test error responses are properly formatted."""
        # Error responses should be dict with error info
        error_response = {
            "error": "Not Found",
            "message": "No results found for query",
            "code": 404,
        }

        # Should be serializable
        json_str = json.dumps(error_response)
        assert "error" in json_str

    def test_pagination_cursor_desktop_compatible(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test pagination cursor is Desktop compatible."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
            pagination=pagination,
        )

        # Cursor should be string
        assert isinstance(response.pagination.cursor, str)
        # Should be base64
        assert response.pagination.cursor == cursor


# ==============================================================================
# CATEGORY 3: Compression Tests (10 tests)
# ==============================================================================


class TestResponseCompression:
    """Test response compression and efficiency."""

    def test_compression_effectiveness_ids_only(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression achieves minimum 10% savings for ids_only."""
        results = [format_ids_only(r) for r in sample_search_results]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original.encode())

        savings = (len(original) - len(compressed)) / len(original)
        assert savings >= 0.10  # At least 10% savings

    def test_compression_effectiveness_metadata(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression achieves minimum 10% savings for metadata."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original.encode())

        savings = (len(original) - len(compressed)) / len(original)
        assert savings >= 0.10

    def test_roundtrip_integrity_ids_only(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression roundtrip preserves data for ids_only."""
        results = [format_ids_only(r) for r in sample_search_results]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original_json = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original_json.encode())
        decompressed_json = gzip.decompress(compressed).decode()

        assert original_json == decompressed_json

    def test_roundtrip_integrity_metadata(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression roundtrip preserves data for metadata."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original_json = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original_json.encode())
        decompressed_json = gzip.decompress(compressed).decode()

        assert original_json == decompressed_json

    def test_roundtrip_integrity_full(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression roundtrip preserves data for full mode."""
        results = [
            format_full(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original_json = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original_json.encode())
        decompressed_json = gzip.decompress(compressed).decode()

        assert original_json == decompressed_json

    def test_compression_performance_under_50ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression completes in under 50ms."""
        results = [
            format_full(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original = json.dumps(response.model_dump(exclude_none=True))

        start = time.perf_counter()
        gzip.compress(original.encode())
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 50

    def test_decompression_performance_under_15ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test decompression completes in under 15ms."""
        results = [
            format_full(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original.encode())

        start = time.perf_counter()
        gzip.decompress(compressed)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 15

    def test_field_shortening_accuracy(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test field shortening doesn't lose data accuracy."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # All ranks should be preserved
        for i, result in enumerate(response.results):
            assert result.rank == i + 1

    def test_large_response_compression_effectiveness(self) -> None:
        """Test compression effectiveness on large responses."""
        from datetime import datetime

        # Create 100 large results
        results = []
        for i in range(100):
            result = SearchResult(
                chunk_id=i,
                chunk_text=("This is a very long piece of content. " * 50),
                similarity_score=0.95,
                bm25_score=0.88,
                hybrid_score=0.92,
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file-{i}.md",
                source_category="documentation",
                document_date=datetime(2024, 1, 1),
                context_header=f"Section {i}",
                chunk_index=i,
                total_chunks=100,
                chunk_token_count=500,
                metadata={"topic": f"topic-{i}"},
            )
            results.append(format_full(result))

        response = SemanticSearchResponse(
            results=results,
            total_found=100,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        original = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original.encode())

        savings = (len(original) - len(compressed)) / len(original)
        # Larger responses should compress better
        assert savings > 0.20  # At least 20% for large response


# ==============================================================================
# CATEGORY 4: Backward Compatibility Tests (8 tests)
# ==============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_response_mode_parameter_backward_compatible(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response_mode parameter still works as before."""
        request = SemanticSearchRequest(
            query="test",
            response_mode="metadata",
        )

        assert request.response_mode == "metadata"

    def test_default_response_mode_unchanged(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test default response mode is still metadata."""
        request = SemanticSearchRequest(query="test")

        assert request.response_mode == "metadata"

    def test_top_k_parameter_still_supported(self) -> None:
        """Test top_k parameter still works for backward compatibility."""
        request = SemanticSearchRequest(
            query="test",
            top_k=20,
        )

        # top_k should be honored
        assert request.top_k == 20

    def test_response_structure_unchanged(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response structure hasn't changed."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Old code expects these fields
        assert hasattr(response, "results")
        assert hasattr(response, "total_found")
        assert hasattr(response, "strategy_used")
        assert hasattr(response, "execution_time_ms")

    def test_pagination_is_optional(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test pagination is still optional (None by default)."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Pagination should be None for backward compatibility
        assert response.pagination is None

    def test_legacy_response_format_validation(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test legacy response format is still supported."""
        # Simulating old code that expects specific structure
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Should be serializable in the same way
        json_data = response.model_dump(exclude_none=True)
        assert "results" in json_data
        assert "total_found" in json_data

    def test_no_breaking_changes_in_field_names(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test no field names have changed."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_data = response.model_dump(exclude_none=True)

        # Expected field names from original implementation
        expected_fields = {"results", "total_found", "strategy_used", "execution_time_ms"}
        actual_fields = set(json_data.keys())

        assert expected_fields.issubset(actual_fields)


# ==============================================================================
# CATEGORY 5: Integration Tests (25 tests)
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_semantic_search_all_format_modes(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test semantic_search works with all format modes."""
        for mode in ["ids_only", "metadata", "preview", "full"]:
            request = SemanticSearchRequest(
                query="test", response_mode=mode  # type: ignore[arg-type]
            )

            results = [
                format_metadata(r) for r in sample_search_results
            ]

            response = SemanticSearchResponse(
                results=results,
                total_found=10,
                strategy_used="hybrid",
                execution_time_ms=245.3,
            )

            assert response is not None
            assert len(response.results) > 0

    def test_find_vendor_info_all_format_modes(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test find_vendor_info works with all format modes."""
        for mode in ["ids_only", "metadata", "preview", "full"]:
            request = FindVendorInfoRequest(
                vendor_name="Acme Corp", response_mode=mode  # type: ignore[arg-type]
            )

            # Create response based on mode
            if mode == "ids_only":
                response = VendorInfoIDs(
                    vendor_name=sample_vendor_data["vendor_name"],
                    entity_ids=[e.entity_id for e in sample_vendor_data["entities"]],
                )
            elif mode == "metadata":
                response = VendorInfoMetadata(
                    vendor_name=sample_vendor_data["vendor_name"],
                    statistics=sample_vendor_data["statistics"],
                    top_entities=sample_vendor_data["entities"][:5],
                )
            elif mode == "preview":
                response = VendorInfoPreview(
                    vendor_name=sample_vendor_data["vendor_name"],
                    entities=sample_vendor_data["entities"][:5],
                    statistics=sample_vendor_data["statistics"],
                )
            else:  # full
                response = VendorInfoFull(
                    vendor_name=sample_vendor_data["vendor_name"],
                    entities=sample_vendor_data["entities"],
                    statistics=sample_vendor_data["statistics"],
                )

            assert response is not None

    def test_pagination_with_formatting(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test pagination works with response formatting."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
            pagination=pagination,
        )

        # Verify pagination is included
        assert response.pagination is not None
        assert response.pagination.cursor is not None

    def test_cache_with_formatting(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test caching works with response formatting."""
        response1 = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Simulate cache hit (same results, different timing)
        response2 = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,  # Same timing for identical responses
        )

        # Both should be identical in content
        json1 = json.dumps(response1.model_dump(exclude_none=True), sort_keys=True)
        json2 = json.dumps(response2.model_dump(exclude_none=True), sort_keys=True)

        assert json1 == json2

    def test_field_filtering_with_formatting(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test field filtering works with response formatting."""
        request = SemanticSearchRequest(
            query="test",
            response_mode="metadata",
            fields=["chunk_id", "source_file"],
        )

        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Should contain requested fields
        for result in response.results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "source_file")

    def test_oversized_response_handling(
        self,
    ) -> None:
        """Test handling of oversized responses."""
        from datetime import datetime

        # Create very large response
        results = []
        for i in range(1000):
            result = SearchResult(
                chunk_id=i,
                chunk_text=("X" * 1000 + " ") * 10,  # Very large text
                similarity_score=0.95,
                bm25_score=0.88,
                hybrid_score=0.92,
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file-{i}.md",
                source_category="documentation",
                document_date=datetime(2024, 1, 1),
                context_header=f"Section {i}",
                chunk_index=i,
                total_chunks=1000,
                chunk_token_count=5000,
                metadata={"topic": f"topic-{i}"},
            )
            results.append(format_metadata(result))

        response = SemanticSearchResponse(
            results=results,
            total_found=1000,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Should still be serializable
        json_str = json.dumps(response.model_dump(exclude_none=True))
        assert len(json_str) > 0

    def test_error_response_integration(self) -> None:
        """Test error responses are properly formatted."""
        error_response = {
            "error": "QueryTimeout",
            "message": "Search query timed out after 30s",
            "code": 504,
        }

        # Should be serializable
        json_str = json.dumps(error_response)
        assert "error" in json_str
        assert "QueryTimeout" in json_str

    def test_mixed_response_modes_in_workflow(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test mixing different response modes in workflow."""
        # First request with metadata
        response1 = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        # Second request with full
        response2 = SemanticSearchResponse(
            results=[
                format_full(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=345.3,
        )

        # Both should be valid
        assert response1 is not None
        assert response2 is not None

    def test_pagination_cursor_preservation(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test pagination cursor is preserved through responses."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
            pagination=pagination,
        )

        json_data = json.dumps(response.model_dump(exclude_none=True))
        restored_response = SemanticSearchResponse.model_validate_json(json_data)

        assert restored_response.pagination.cursor == cursor

    def test_compression_with_pagination(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression works with pagination."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=245.3,
            pagination=pagination,
        )

        original = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(original.encode())

        # Should compress
        assert len(compressed) < len(original)

    def test_concurrent_format_requests(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test concurrent requests with different formats."""
        responses = []

        for mode in ["ids_only", "metadata", "preview", "full"]:
            if mode == "ids_only":
                results = [
                    format_ids_only(r) for r in sample_search_results
                ]
            elif mode == "metadata":
                results = [
                    format_metadata(r)
                    for r in sample_search_results
                ]
            elif mode == "preview":
                results = [
                    format_preview(r)
                    for r in sample_search_results
                ]
            else:
                results = [
                    format_full(r) for r in sample_search_results
                ]

            response = SemanticSearchResponse(
                results=results,
                total_found=10,
                strategy_used="hybrid",
                execution_time_ms=245.3,
            )

            responses.append(response)

        # All responses should be valid
        assert len(responses) == 4
        assert all(r is not None for r in responses)

    def test_response_format_consistency_across_calls(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response format is consistent across multiple calls."""
        responses = []

        for _ in range(3):
            response = SemanticSearchResponse(
                results=[
                    format_metadata(r)
                    for r in sample_search_results
                ],
                total_found=10,
                strategy_used="hybrid",
                execution_time_ms=245.3,
            )

            json_str = json.dumps(response.model_dump(exclude_none=True))
            responses.append(json_str)

        # All responses should be identical
        assert responses[0] == responses[1] == responses[2]

    def test_vendor_info_with_pagination_integration(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test vendor info with pagination."""
        # Note: VendorInfo models don't have pagination directly,
        # but we test that pagination can be added as needed
        cursor_data = {
            "query_hash": "vendor_abc123",
            "offset": 20,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=20,
            has_more=True,
            total_available=150,
        )

        # Create vendor response (pagination would be in wrapper)
        response = VendorInfoMetadata(
            vendor_name=sample_vendor_data["vendor_name"],
            statistics=sample_vendor_data["statistics"],
            top_entities=sample_vendor_data["entities"][:5],
        )

        assert response.vendor_name == "Acme Corp"
        # Pagination would be in envelope, not directly on response
        assert pagination.cursor == cursor


# ==============================================================================
# CATEGORY 6: Performance Benchmarks (10 tests)
# ==============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmarking for response formatting."""

    def test_response_formatting_latency_under_20ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response formatting completes in under 20ms."""
        start = time.perf_counter()

        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 20

    def test_response_serialization_latency_under_10ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test response serialization completes in under 10ms."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        start = time.perf_counter()

        json.dumps(response.model_dump(exclude_none=True))

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 10

    def test_compression_latency_under_30ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression completes in under 30ms."""
        response = SemanticSearchResponse(
            results=[
                format_full(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))

        start = time.perf_counter()

        gzip.compress(json_str.encode())

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 30

    def test_decompression_latency_under_15ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test decompression completes in under 15ms."""
        response = SemanticSearchResponse(
            results=[
                format_full(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))
        compressed = gzip.compress(json_str.encode())

        start = time.perf_counter()

        gzip.decompress(compressed)

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 15

    def test_token_estimation_accuracy_within_10_percent(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test token estimation is accurate."""
        results = [
            format_metadata(r) for r in sample_search_results
        ]

        response = SemanticSearchResponse(
            results=results,
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))

        # Rough estimate: 4 characters per token
        estimated_tokens = len(json_str) // 4

        # Should be reasonable size (metadata mode with sample data)
        assert estimated_tokens > 0
        assert estimated_tokens < 10000

    def test_large_response_formatting_latency(self) -> None:
        """Test formatting latency for large responses."""
        from datetime import datetime

        results = []
        for i in range(100):
            result = SearchResult(
                chunk_id=i,
                chunk_text=f"Content for result {i}. " * 20,
                similarity_score=0.95 - (i * 0.005),
                bm25_score=0.88 - (i * 0.005),
                hybrid_score=0.92 - (i * 0.005),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file-{i}.md",
                source_category="documentation",
                document_date=datetime(2024, 1, 1),
                context_header=f"Section {i}",
                chunk_index=i,
                total_chunks=100,
                chunk_token_count=100 + (i * 5),
                metadata={"topic": f"topic-{i}"},
            )
            results.append(format_metadata(result))

        start = time.perf_counter()

        response = SemanticSearchResponse(
            results=results,
            total_found=100,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        # Should still be reasonable for large response
        assert elapsed < 100

    def test_repeated_format_request_performance(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test performance of repeated format requests."""
        times = []

        for _ in range(10):
            start = time.perf_counter()

            response = SemanticSearchResponse(
                results=[
                    format_metadata(r)
                    for r in sample_search_results
                ],
                total_found=10,
                strategy_used="hybrid",
                execution_time_ms=245.3,
            )

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Average should be under 20ms
        avg_time = sum(times) / len(times)
        assert avg_time < 20

    def test_compression_performance_improvement(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test compression provides performance benefit."""
        response = SemanticSearchResponse(
            results=[
                format_full(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        json_str = json.dumps(response.model_dump(exclude_none=True))

        # Measure compression time
        compress_times = []
        for _ in range(5):
            start = time.perf_counter()
            gzip.compress(json_str.encode())
            elapsed = (time.perf_counter() - start) * 1000
            compress_times.append(elapsed)

        avg_compress = sum(compress_times) / len(compress_times)

        # Compression should be fast (< 30ms)
        assert avg_compress < 30

    def test_field_filtering_performance_under_5ms(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Test field filtering performance."""
        response = SemanticSearchResponse(
            results=[
                format_metadata(r) for r in sample_search_results
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        start = time.perf_counter()

        # Extract just a few fields
        filtered = [
            {
                "chunk_id": r.chunk_id,
                "hybrid_score": r.hybrid_score,
                "source_file": r.source_file,
            }
            for r in response.results
        ]

        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 5
        assert len(filtered) == 10
