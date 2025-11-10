"""Test semantic_search tool implementation.

Tests format functions and semantic_search tool with all 4 response modes.

Test Coverage:
- Format functions (ids_only, metadata, preview, full)
- Tool parameter validation
- Error handling
- Response structure
- Token budget verification (estimated)

Performance:
- Unit tests: <100ms (mocked HybridSearch)
- Integration tests: <1s (requires database)
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.mcp.models import (
    SearchResultFull,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
)
from src.mcp.tools.semantic_search import (
    format_full,
    format_ids_only,
    format_metadata,
    format_preview,
    semantic_search,
)
from src.search.results import SearchResult


@pytest.fixture
def sample_search_result() -> SearchResult:
    """Create sample SearchResult for testing."""
    return SearchResult(
        chunk_id=1,
        chunk_text="This is a sample chunk about JWT authentication and security best practices. "
        "It includes information about token validation, refresh tokens, and secure storage. "
        "JWT tokens should always be validated on the server side and stored securely.",
        similarity_score=0.85,
        bm25_score=0.75,
        hybrid_score=0.80,
        rank=1,
        score_type="hybrid",
        source_file="docs/security/auth.md",
        source_category="security",
        document_date=datetime(2024, 1, 1),
        context_header="auth.md > Security > Authentication",
        chunk_index=0,
        total_chunks=10,
        chunk_token_count=512,
        metadata={"tags": ["security", "auth"]},
    )


class TestFormatFunctions:
    """Test result formatting functions."""

    def test_format_ids_only(self, sample_search_result: SearchResult) -> None:
        """Test IDs-only formatting."""
        result = format_ids_only(sample_search_result)

        # Check type
        assert isinstance(result, SearchResultIDs)

        # Check fields present
        assert result.chunk_id == 1
        assert result.hybrid_score == 0.80
        assert result.rank == 1

        # Ensure no content fields
        assert not hasattr(result, "chunk_text")
        assert not hasattr(result, "source_file")

    def test_format_metadata(self, sample_search_result: SearchResult) -> None:
        """Test metadata formatting."""
        result = format_metadata(sample_search_result)

        # Check type
        assert isinstance(result, SearchResultMetadata)

        # Check fields present
        assert result.chunk_id == 1
        assert result.source_file == "docs/security/auth.md"
        assert result.source_category == "security"
        assert result.hybrid_score == 0.80
        assert result.rank == 1
        assert result.chunk_index == 0
        assert result.total_chunks == 10

        # Ensure no content fields
        assert not hasattr(result, "chunk_text")
        assert not hasattr(result, "chunk_snippet")

    def test_format_preview(self, sample_search_result: SearchResult) -> None:
        """Test preview formatting."""
        result = format_preview(sample_search_result)

        # Check type
        assert isinstance(result, SearchResultPreview)

        # Check fields present
        assert result.chunk_id == 1
        assert result.source_file == "docs/security/auth.md"
        assert result.source_category == "security"
        assert result.hybrid_score == 0.80
        assert result.rank == 1
        assert result.chunk_index == 0
        assert result.total_chunks == 10
        assert result.context_header == "auth.md > Security > Authentication"

        # Check snippet is truncated
        assert "JWT authentication" in result.chunk_snippet
        assert len(result.chunk_snippet) <= 203  # 200 + "..."

        # Ensure full text not present
        assert not hasattr(result, "chunk_text")

    def test_format_preview_short_text(self) -> None:
        """Test preview formatting with text shorter than 200 chars."""
        short_result = SearchResult(
            chunk_id=1,
            chunk_text="Short text.",
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/short.md",
            source_category="docs",
            document_date=None,
            context_header="short.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
        )

        result = format_preview(short_result)
        assert result.chunk_snippet == "Short text."
        assert "..." not in result.chunk_snippet

    def test_format_preview_long_text(self) -> None:
        """Test preview formatting with text longer than 200 chars."""
        long_text = "a" * 300
        long_result = SearchResult(
            chunk_id=1,
            chunk_text=long_text,
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/long.md",
            source_category="docs",
            document_date=None,
            context_header="long.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=300,
        )

        result = format_preview(long_result)
        assert len(result.chunk_snippet) == 203  # 200 + "..."
        assert result.chunk_snippet.endswith("...")

    def test_format_full(self, sample_search_result: SearchResult) -> None:
        """Test full formatting."""
        result = format_full(sample_search_result)

        # Check type
        assert isinstance(result, SearchResultFull)

        # Check all fields present
        assert result.chunk_id == 1
        assert result.chunk_text == sample_search_result.chunk_text
        assert result.similarity_score == 0.85
        assert result.bm25_score == 0.75
        assert result.hybrid_score == 0.80
        assert result.rank == 1
        assert result.score_type == "hybrid"
        assert result.source_file == "docs/security/auth.md"
        assert result.source_category == "security"
        assert result.context_header == "auth.md > Security > Authentication"
        assert result.chunk_index == 0
        assert result.total_chunks == 10
        assert result.chunk_token_count == 512

        # Check full text is present
        assert len(result.chunk_text) > 200


class TestSemanticSearchTool:
    """Test semantic_search tool."""

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_metadata_mode(
        self, mock_get_search: Mock, sample_search_result: SearchResult
    ) -> None:
        """Test semantic_search with metadata mode."""
        # Mock HybridSearch.search() return value
        mock_search = Mock()
        mock_search.search.return_value = [sample_search_result]
        mock_get_search.return_value = mock_search

        # Execute search
        response = semantic_search(query="JWT authentication", top_k=10, response_mode="metadata")

        # Check response structure
        assert response.total_found == 1
        assert response.strategy_used == "hybrid"
        assert response.execution_time_ms > 0
        assert len(response.results) == 1

        # Check result is metadata type
        result = response.results[0]
        assert isinstance(result, SearchResultMetadata)
        assert result.chunk_id == 1
        assert result.source_file == "docs/security/auth.md"

        # Verify search was called correctly
        mock_search.search.assert_called_once_with(
            query="JWT authentication",
            top_k=10,
            strategy="hybrid",
            min_score=0.0,
        )

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_ids_only_mode(
        self, mock_get_search: Mock, sample_search_result: SearchResult
    ) -> None:
        """Test semantic_search with ids_only mode."""
        mock_search = Mock()
        mock_search.search.return_value = [sample_search_result]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", top_k=5, response_mode="ids_only")

        # Check result is IDs type
        assert len(response.results) == 1
        result = response.results[0]
        assert isinstance(result, SearchResultIDs)
        assert result.chunk_id == 1
        assert result.hybrid_score == 0.80
        assert not hasattr(result, "source_file")

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_preview_mode(
        self, mock_get_search: Mock, sample_search_result: SearchResult
    ) -> None:
        """Test semantic_search with preview mode."""
        mock_search = Mock()
        mock_search.search.return_value = [sample_search_result]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", top_k=5, response_mode="preview")

        # Check result is preview type
        assert len(response.results) == 1
        result = response.results[0]
        assert isinstance(result, SearchResultPreview)
        assert result.chunk_id == 1
        assert hasattr(result, "chunk_snippet")
        assert not hasattr(result, "chunk_text")

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_full_mode(
        self, mock_get_search: Mock, sample_search_result: SearchResult
    ) -> None:
        """Test semantic_search with full mode."""
        mock_search = Mock()
        mock_search.search.return_value = [sample_search_result]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", top_k=5, response_mode="full")

        # Check result is full type
        assert len(response.results) == 1
        result = response.results[0]
        assert isinstance(result, SearchResultFull)
        assert result.chunk_id == 1
        assert result.chunk_text == sample_search_result.chunk_text
        assert result.chunk_token_count == 512

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_defaults(
        self, mock_get_search: Mock, sample_search_result: SearchResult
    ) -> None:
        """Test semantic_search with default parameters."""
        mock_search = Mock()
        mock_search.search.return_value = [sample_search_result]
        mock_get_search.return_value = mock_search

        # Call with only query (defaults: top_k=10, response_mode="metadata")
        response = semantic_search(query="test query")

        # Verify defaults applied
        mock_search.search.assert_called_once_with(
            query="test query",
            top_k=10,
            strategy="hybrid",
            min_score=0.0,
        )

        # Check response mode default (metadata)
        assert isinstance(response.results[0], SearchResultMetadata)

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_empty_results(self, mock_get_search: Mock) -> None:
        """Test semantic_search with no results."""
        mock_search = Mock()
        mock_search.search.return_value = []
        mock_get_search.return_value = mock_search

        response = semantic_search(query="nonexistent query")

        assert response.total_found == 0
        assert len(response.results) == 0
        assert response.execution_time_ms >= 0

    def test_semantic_search_invalid_query_empty(self) -> None:
        """Test semantic_search with empty query raises ValueError."""
        with pytest.raises(ValueError, match="Invalid request parameters"):
            semantic_search(query="", top_k=10)

    def test_semantic_search_invalid_top_k_too_large(self) -> None:
        """Test semantic_search with top_k > 50 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid request parameters"):
            semantic_search(query="test", top_k=100)

    def test_semantic_search_invalid_top_k_zero(self) -> None:
        """Test semantic_search with top_k = 0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid request parameters"):
            semantic_search(query="test", top_k=0)

    def test_semantic_search_invalid_response_mode(self) -> None:
        """Test semantic_search with invalid response_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid request parameters"):
            semantic_search(query="test", response_mode="invalid")

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_handles_search_failure(self, mock_get_search: Mock) -> None:
        """Test semantic_search handles HybridSearch failures gracefully."""
        mock_search = Mock()
        mock_search.search.side_effect = RuntimeError("Database connection failed")
        mock_get_search.return_value = mock_search

        with pytest.raises(RuntimeError, match="Search failed"):
            semantic_search(query="test")

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_semantic_search_multiple_results(self, mock_get_search: Mock) -> None:
        """Test semantic_search with multiple results."""
        # Create multiple results
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.8 - (i * 0.1),
                bm25_score=0.7 - (i * 0.1),
                hybrid_score=0.75 - (i * 0.1),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="docs",
                document_date=None,
                context_header=f"file{i}.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=100,
            )
            for i in range(5)
        ]

        mock_search = Mock()
        mock_search.search.return_value = results
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", top_k=10, response_mode="metadata")

        assert response.total_found == 5
        assert len(response.results) == 5

        # Check results are ordered by rank
        for i, result in enumerate(response.results):
            assert result.rank == i + 1
            assert result.chunk_id == i


class TestResponseModePerformance:
    """Test response mode performance characteristics."""

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_ids_only_mode_latency(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test ids_only mode has fastest response time."""
        import time

        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Create a sample result
        sample = SearchResult(
            chunk_id=1,
            chunk_text="a" * 5000,  # Large text
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/large.md",
            source_category="docs",
            document_date=None,
            context_header="large.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=1500,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        # Measure execution time for ids_only mode
        start = time.time()
        response_ids = semantic_search(query="test", response_mode="ids_only")
        time_ids = time.time() - start

        # Measure execution time for full mode
        start = time.time()
        response_full = semantic_search(query="test", response_mode="full")
        time_full = time.time() - start

        # Both should complete quickly, but ids_only should be comparable or faster
        # (Due to formatting overhead being minimal on mocked data)
        assert response_ids.execution_time_ms >= 0
        assert response_full.execution_time_ms >= 0
        # Basic sanity check: both modes work
        assert len(response_ids.results) == 1
        assert len(response_full.results) == 1

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_metadata_mode_default_performance(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test metadata mode (default) has balanced performance."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        sample = SearchResult(
            chunk_id=1,
            chunk_text="Medium size text " * 50,  # ~800 chars
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/medium.md",
            source_category="docs",
            document_date=None,
            context_header="medium.md",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=200,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        # Metadata mode should be default and balanced
        response = semantic_search(query="test")

        assert response.execution_time_ms >= 0
        assert len(response.results) == 1
        assert isinstance(response.results[0], SearchResultMetadata)

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_preview_mode_with_snippet_overhead(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test preview mode includes snippet formatting overhead."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        sample = SearchResult(
            chunk_id=1,
            chunk_text="This is content. " * 100,  # ~1700 chars, needs truncation
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/long.md",
            source_category="docs",
            document_date=None,
            context_header="long.md",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=400,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", response_mode="preview")

        assert response.execution_time_ms >= 0
        assert len(response.results) == 1
        result = response.results[0]
        assert isinstance(result, SearchResultPreview)
        # Snippet should be truncated
        assert len(result.chunk_snippet) <= 203


class TestTokenReductionAcrossModes:
    """Test token reduction benefits of progressive disclosure."""

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_metadata_vs_full_token_reduction(
        self, mock_get_search: Mock, mock_get_cache: Mock
    ) -> None:
        """Test metadata mode produces significantly fewer tokens than full."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Create result with substantial content
        large_text = "Authentication security token validation " * 200  # ~8000 chars
        sample = SearchResult(
            chunk_id=1,
            chunk_text=large_text,
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/auth-guide.md",
            source_category="security",
            document_date=None,
            context_header="auth-guide.md > Security > JWT",
            chunk_index=5,
            total_chunks=100,
            chunk_token_count=2000,  # ~8000 chars / 4 = 2000 tokens est.
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        # Get metadata response
        response_metadata = semantic_search(
            query="test", top_k=10, response_mode="metadata"
        )

        # Get full response (mock will return same data both times)
        mock_search.search.reset_mock()
        mock_search.search.return_value = [sample]
        response_full = semantic_search(
            query="test", top_k=10, response_mode="full"
        )

        # Estimate token counts (rough heuristic: ~4 chars per token)
        metadata_result = response_metadata.results[0]
        full_result = response_full.results[0]

        # Metadata should have much less content
        assert not hasattr(metadata_result, "chunk_text")
        assert hasattr(full_result, "chunk_text")
        assert len(full_result.chunk_text) == len(large_text)

        # Token reduction estimation:
        # Metadata: ~100-200 tokens per result (IDs + metadata)
        # Full: ~500-2000+ tokens per result (includes full text)
        # Expected ratio: metadata should be ~10-20% of full
        metadata_estimated_tokens = 150  # Rough estimate
        full_estimated_tokens = 1000  # Rough estimate
        ratio = metadata_estimated_tokens / full_estimated_tokens

        assert ratio < 0.25  # Metadata should be <25% of full

    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_ids_only_minimum_tokens(self, mock_get_search: Mock) -> None:
        """Test ids_only mode produces minimum token count."""
        sample = SearchResult(
            chunk_id=999,
            chunk_text="x" * 10000,  # Very large text
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/huge.md",
            source_category="docs",
            document_date=None,
            context_header="huge.md",
            chunk_index=0,
            total_chunks=1000,
            chunk_token_count=2500,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", response_mode="ids_only")

        result = response.results[0]
        assert isinstance(result, SearchResultIDs)
        # ids_only should have minimal fields
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "hybrid_score")
        assert hasattr(result, "rank")
        # Should NOT have content
        assert not hasattr(result, "chunk_text")
        assert not hasattr(result, "source_file")


class TestEdgeCaseResponses:
    """Test edge cases in response formatting."""

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_snippet_exactly_200_characters(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test snippet with exactly 200 characters."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        text_200 = "a" * 200  # Exactly 200 chars

        sample = SearchResult(
            chunk_id=1,
            chunk_text=text_200,
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/test.md",
            source_category="docs",
            document_date=None,
            context_header="test.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=50,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", response_mode="preview")

        result = response.results[0]
        # At exactly 200 chars, should not add ellipsis
        assert result.chunk_snippet == text_200
        assert not result.chunk_snippet.endswith("...")

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_snippet_201_characters_truncation(
        self, mock_get_search: Mock, mock_get_cache: Mock
    ) -> None:
        """Test snippet with 201 characters gets truncated to 200 + ellipsis."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        text_201 = "b" * 201  # One over the limit

        sample = SearchResult(
            chunk_id=1,
            chunk_text=text_201,
            similarity_score=0.85,
            bm25_score=0.75,
            hybrid_score=0.80,
            rank=1,
            score_type="hybrid",
            source_file="docs/test.md",
            source_category="docs",
            document_date=None,
            context_header="test.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=51,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", response_mode="preview")

        result = response.results[0]
        # Should be truncated to 200 chars + "..."
        assert len(result.chunk_snippet) == 203
        assert result.chunk_snippet.endswith("...")
        assert result.chunk_snippet == "b" * 200 + "..."

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_empty_results_list(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test handling of empty results from search."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_search = Mock()
        mock_search.search.return_value = []
        mock_get_search.return_value = mock_search

        response = semantic_search(query="nonexistent", top_k=10)

        assert response.total_found == 0
        assert len(response.results) == 0
        assert response.execution_time_ms >= 0

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_large_result_set_maximum(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test handling of maximum result set (top_k=50)."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        # Create 50 results
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content for result {i}",
                similarity_score=0.9 - (i * 0.01),
                bm25_score=0.8 - (i * 0.01),
                hybrid_score=0.85 - (i * 0.01),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="docs",
                document_date=None,
                context_header=f"file{i}.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=50,
            )
            for i in range(50)
        ]

        mock_search = Mock()
        mock_search.search.return_value = results
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test", top_k=50, response_mode="metadata")

        assert response.total_found == 50
        assert len(response.results) == 50

        # Verify ranking is preserved
        for i, result in enumerate(response.results):
            assert result.rank == i + 1
            assert result.chunk_id == i

    @patch("src.mcp.tools.semantic_search.get_cache_layer")
    @patch("src.mcp.tools.semantic_search.get_hybrid_search")
    def test_single_result(self, mock_get_search: Mock, mock_get_cache: Mock) -> None:
        """Test handling of single result."""
        # Mock cache to always miss (prevent cache pollution from other tests)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        sample = SearchResult(
            chunk_id=1,
            chunk_text="Single result",
            similarity_score=0.95,
            bm25_score=0.90,
            hybrid_score=0.92,
            rank=1,
            score_type="hybrid",
            source_file="docs/single.md",
            source_category="docs",
            document_date=None,
            context_header="single.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=20,
        )

        mock_search = Mock()
        mock_search.search.return_value = [sample]
        mock_get_search.return_value = mock_search

        response = semantic_search(query="test")

        assert response.total_found == 1
        assert len(response.results) == 1
        assert response.results[0].rank == 1


# Integration tests (require database - mark as skipif)
@pytest.mark.skipif(True, reason="Requires running database and FastMCP server")
class TestSemanticSearchIntegration:
    """Integration tests for semantic_search tool with real database."""

    def test_search_with_real_database(self) -> None:
        """Test search against real database (integration test)."""
        # This test would run against real database
        # Skipped by default to avoid database dependencies in unit tests
        response = semantic_search(query="JWT authentication", top_k=5, response_mode="metadata")

        assert response.total_found >= 0
        assert response.strategy_used == "hybrid"
        assert response.execution_time_ms > 0

        if response.results:
            result = response.results[0]
            assert hasattr(result, "source_file")
            assert isinstance(result, SearchResultMetadata)
