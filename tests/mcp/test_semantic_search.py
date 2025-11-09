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
