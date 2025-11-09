"""Unit tests for BM25 full-text search functionality.

Tests cover:
- Basic search functionality
- Phrase search
- Category filtering
- Score thresholds
- Error handling
- Result ordering and scoring
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from src.search.bm25_search import BM25Search, SearchResult


class TestSearchResult:
    """Test cases for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating SearchResult with all fields."""
        result = SearchResult(
            id=1,
            chunk_text="Test chunk text content",
            context_header="file.md > Section",
            source_file="docs/test.md",
            source_category="product_docs",
            document_date=date(2025, 1, 1),
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
            metadata={"author": "Test Author"},
            similarity=0.85,
        )

        assert result.id == 1
        assert result.chunk_text == "Test chunk text content"
        assert result.similarity == 0.85
        assert result.chunk_token_count == 256

    def test_search_result_repr(self) -> None:
        """Test SearchResult string representation."""
        result = SearchResult(
            id=1,
            chunk_text="Test",
            context_header="file.md",
            source_file="test.md",
            source_category=None,
            document_date=None,
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=10,
            metadata={},
            similarity=0.75,
        )

        repr_str = repr(result)
        assert "id=1" in repr_str
        assert "source_file='test.md'" in repr_str
        assert "similarity=0.7500" in repr_str


class TestBM25Search:
    """Test cases for BM25Search class."""

    @pytest.fixture
    def mock_db_pool(self) -> tuple[Mock, Mock, Mock]:
        """Create mock database pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup context manager for connection
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.__exit__ = MagicMock(return_value=None)
        mock_pool.get_connection = MagicMock(return_value=mock_conn_ctx)

        # Setup context manager for cursor
        mock_cursor_ctx = MagicMock()
        mock_cursor_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor_ctx.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)

        return mock_pool, mock_conn, mock_cursor

    @pytest.fixture
    def bm25_search(self, mock_db_pool: tuple[Mock, Mock, Mock]) -> BM25Search:
        """Create BM25Search instance with mocked database."""
        mock_pool, _, _ = mock_db_pool
        with patch("src.search.bm25_search.DatabasePool", mock_pool):
            search = BM25Search()
            search._db_pool = mock_pool
        return search

    def test_init(self) -> None:
        """Test BM25Search initialization."""
        search = BM25Search()
        assert search._db_pool is not None

    def test_search_basic(self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]) -> None:
        """Test basic search functionality."""
        _, _, mock_cursor = mock_db_pool

        # Mock database response
        mock_cursor.fetchall.return_value = [
            (
                1,
                "Authentication using JWT tokens",
                "auth.md > JWT",
                "docs/auth.md",
                "product_docs",
                date(2025, 1, 1),
                0,
                3,
                128,
                {"author": "Test"},
                0.85,
            ),
            (
                2,
                "Token validation process",
                "auth.md > Validation",
                "docs/auth.md",
                "product_docs",
                date(2025, 1, 1),
                1,
                3,
                96,
                {},
                0.72,
            ),
        ]

        results = bm25_search.search("authentication jwt", top_k=10)

        assert len(results) == 2
        assert results[0].id == 1
        assert results[0].similarity == 0.85
        assert results[1].id == 2
        assert results[1].similarity == 0.72
        assert "JWT" in results[0].chunk_text

    def test_search_with_category_filter(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test search with category filtering."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = [
            (
                1,
                "Product documentation chunk",
                "product.md > Section",
                "docs/product.md",
                "product_docs",
                None,
                0,
                1,
                100,
                {},
                0.80,
            ),
        ]

        results = bm25_search.search(
            "documentation",
            top_k=10,
            category_filter="product_docs",
        )

        assert len(results) == 1
        assert results[0].source_category == "product_docs"

        # Verify SQL call included category filter
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == ("documentation", 10, "product_docs", 0.0)

    def test_search_with_min_score(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test search with minimum score threshold."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = [
            (1, "High relevance", "h.md", "h.md", None, None, 0, 1, 50, {}, 0.90),
        ]

        results = bm25_search.search("test", top_k=10, min_score=0.8)

        assert len(results) == 1
        assert results[0].similarity >= 0.8

        # Verify min_score parameter passed
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][3] == 0.8

    def test_search_empty_query(self, bm25_search: BM25Search) -> None:
        """Test search with empty query raises ValueError."""
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            bm25_search.search("")

        with pytest.raises(ValueError, match="query_text cannot be empty"):
            bm25_search.search("   ")

    def test_search_invalid_top_k(self, bm25_search: BM25Search) -> None:
        """Test search with invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            bm25_search.search("test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            bm25_search.search("test", top_k=-1)

    def test_search_phrase_basic(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test phrase search functionality."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = [
            (
                1,
                "JWT authentication token validation",
                "auth.md > JWT",
                "docs/auth.md",
                "product_docs",
                None,
                0,
                1,
                128,
                {},
                0.92,
            ),
        ]

        results = bm25_search.search_phrase("JWT authentication")

        assert len(results) == 1
        assert results[0].similarity == 0.92
        assert "JWT authentication" in results[0].chunk_text

    def test_search_phrase_with_category(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test phrase search with category filter."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = []

        results = bm25_search.search_phrase(
            "exact phrase",
            top_k=5,
            category_filter="kb_article",
        )

        assert len(results) == 0

        # Verify SQL call
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == ("exact phrase", 5, "kb_article")

    def test_search_phrase_empty(self, bm25_search: BM25Search) -> None:
        """Test phrase search with empty phrase raises ValueError."""
        with pytest.raises(ValueError, match="phrase cannot be empty"):
            bm25_search.search_phrase("")

        with pytest.raises(ValueError, match="phrase cannot be empty"):
            bm25_search.search_phrase("   ")

    def test_search_phrase_invalid_top_k(self, bm25_search: BM25Search) -> None:
        """Test phrase search with invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            bm25_search.search_phrase("test phrase", top_k=0)

    def test_search_no_results(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test search with no matching results."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = []

        results = bm25_search.search("nonexistent query")

        assert len(results) == 0
        assert isinstance(results, list)

    def test_search_result_ordering(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test that search results are ordered by similarity descending."""
        _, _, mock_cursor = mock_db_pool

        # Return results in descending similarity order
        mock_cursor.fetchall.return_value = [
            (1, "High", "h.md", "h.md", None, None, 0, 1, 10, {}, 0.95),
            (2, "Medium", "m.md", "m.md", None, None, 0, 1, 10, {}, 0.75),
            (3, "Low", "l.md", "l.md", None, None, 0, 1, 10, {}, 0.55),
        ]

        results = bm25_search.search("test")

        assert len(results) == 3
        assert results[0].similarity > results[1].similarity
        assert results[1].similarity > results[2].similarity

    def test_row_to_result_with_nulls(self, bm25_search: BM25Search) -> None:
        """Test converting database row with NULL values."""
        row = (
            1,
            "Test chunk",
            "context",
            "file.md",
            None,  # source_category
            None,  # document_date
            0,
            1,
            None,  # chunk_token_count
            None,  # metadata
            0.80,
        )

        result = bm25_search._row_to_result(row)

        assert result.id == 1
        assert result.source_category is None
        assert result.document_date is None
        assert result.chunk_token_count is None
        assert result.metadata == {}
        assert result.similarity == 0.80

    def test_search_with_special_characters(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test search with special characters in query."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = [
            (1, "C++ programming", "cpp.md", "cpp.md", None, None, 0, 1, 50, {}, 0.88),
        ]

        results = bm25_search.search("C++ programming")

        assert len(results) == 1
        assert "C++" in results[0].chunk_text

    def test_search_stop_words(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test search with stop words (handled by PostgreSQL)."""
        _, _, mock_cursor = mock_db_pool

        mock_cursor.fetchall.return_value = [
            (1, "Authentication guide", "auth.md", "auth.md", None, None, 0, 1, 50, {}, 0.82),
        ]

        # Query with stop words (the, is, and)
        results = bm25_search.search("what is the authentication process")

        assert len(results) == 1

    def test_search_metadata_preservation(
        self, bm25_search: BM25Search, mock_db_pool: tuple[Mock, Mock, Mock]
    ) -> None:
        """Test that metadata is preserved in search results."""
        _, _, mock_cursor = mock_db_pool

        metadata = {"author": "John Doe", "tags": ["auth", "security"], "version": 2}

        mock_cursor.fetchall.return_value = [
            (1, "Test", "ctx", "f.md", "kb", None, 0, 1, 50, metadata, 0.90),
        ]

        results = bm25_search.search("test")

        assert len(results) == 1
        assert results[0].metadata == metadata
        assert results[0].metadata["author"] == "John Doe"
        assert "auth" in results[0].metadata["tags"]


class TestBM25SearchIntegration:
    """Integration tests requiring database connection."""

    @pytest.mark.integration
    def test_search_real_database(self) -> None:
        """Test search against real database with sample data.

        This test requires a running PostgreSQL database with knowledge_base
        table and sample data inserted. Mark as integration test to skip
        in unit test runs.
        """
        search = BM25Search()

        # This would fail if database not available
        # results = search.search("authentication", top_k=5)
        # assert isinstance(results, list)

        # Placeholder for integration testing
        pytest.skip("Requires database setup with sample data")

    @pytest.mark.integration
    def test_phrase_search_real_database(self) -> None:
        """Test phrase search against real database."""
        search = BM25Search()

        pytest.skip("Requires database setup with sample data")
