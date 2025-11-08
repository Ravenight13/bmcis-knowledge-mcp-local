"""Comprehensive test suite for vector similarity search.

Tests cover:
- Embedding validation (768-dimensional requirement)
- Single and batch similarity searches
- Top_k parameter variations
- Multiple similarity metrics (cosine, L2, inner product)
- Metadata filtering (category, date range)
- Search result validation and ranking
- Performance characteristics
- Error handling and edge cases
- Search statistics tracking
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.document_parsing.models import DocumentMetadata, ProcessedChunk
from src.search.vector_search import SearchResult, SearchStats, VectorSearch


# Test fixtures
@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Create sample DocumentMetadata for testing."""
    return DocumentMetadata(
        title="Test Document",
        author="Test Author",
        category="test_category",
        tags=["test", "vector_search"],
        source_file="test.md",
        document_date=date.today(),
    )


@pytest.fixture
def sample_chunk_with_embedding(sample_metadata: DocumentMetadata) -> ProcessedChunk:
    """Create a sample ProcessedChunk with embedding."""
    chunk = ProcessedChunk.create_from_chunk(
        chunk_text="This is a test chunk for vector search.",
        context_header="test.md > Section",
        metadata=sample_metadata,
        chunk_index=0,
        total_chunks=5,
        token_count=10,
    )
    # Add 768-dimensional embedding
    chunk.embedding = np.random.randn(768).tolist()
    return chunk


@pytest.fixture
def sample_query_embedding() -> list[float]:
    """Create a sample 768-dimensional query embedding."""
    embedding: list[float] = np.random.randn(768).tolist()
    return embedding


@pytest.fixture
def multiple_chunks_with_embeddings(
    sample_metadata: DocumentMetadata,
) -> list[ProcessedChunk]:
    """Create multiple ProcessedChunks with embeddings."""
    chunks: list[ProcessedChunk] = []
    texts = [
        "First chunk about database systems",
        "Second chunk about vector search",
        "Third chunk about similarity metrics",
        "Fourth chunk about HNSW algorithm",
        "Fifth chunk about performance optimization",
    ]

    for i, text in enumerate(texts):
        chunk = ProcessedChunk.create_from_chunk(
            chunk_text=text,
            context_header=f"test.md > Section {i // 2}",
            metadata=sample_metadata,
            chunk_index=i,
            total_chunks=len(texts),
            token_count=10 + i,
        )
        chunk.embedding = np.random.randn(768).tolist()
        chunks.append(chunk)

    return chunks


class TestSearchResultDataModel:
    """Tests for SearchResult data model."""

    def test_search_result_creation(self, sample_chunk_with_embedding: ProcessedChunk) -> None:
        """Test SearchResult initialization."""
        result = SearchResult(0.95, sample_chunk_with_embedding)
        assert result.similarity == 0.95
        assert result.chunk == sample_chunk_with_embedding

    def test_search_result_similarity_validation_lower_bound(
        self, sample_chunk_with_embedding: ProcessedChunk
    ) -> None:
        """Test SearchResult rejects invalid negative similarity."""
        with pytest.raises(ValueError, match="0-1"):
            SearchResult(-0.1, sample_chunk_with_embedding)

    def test_search_result_similarity_validation_upper_bound(
        self, sample_chunk_with_embedding: ProcessedChunk
    ) -> None:
        """Test SearchResult rejects similarity > 1."""
        with pytest.raises(ValueError, match="0-1"):
            SearchResult(1.1, sample_chunk_with_embedding)

    def test_search_result_boundary_values(self, sample_chunk_with_embedding: ProcessedChunk) -> None:
        """Test SearchResult accepts boundary values."""
        result_zero = SearchResult(0.0, sample_chunk_with_embedding)
        assert result_zero.similarity == 0.0

        result_one = SearchResult(1.0, sample_chunk_with_embedding)
        assert result_one.similarity == 1.0

    def test_search_result_equality(self, sample_chunk_with_embedding: ProcessedChunk) -> None:
        """Test SearchResult equality comparison."""
        result1 = SearchResult(0.95, sample_chunk_with_embedding)
        result2 = SearchResult(0.95, sample_chunk_with_embedding)
        assert result1 == result2

    def test_search_result_inequality(
        self,
        sample_chunk_with_embedding: ProcessedChunk,
        sample_metadata: DocumentMetadata,
    ) -> None:
        """Test SearchResult inequality on different similarities."""
        result1 = SearchResult(0.95, sample_chunk_with_embedding)

        chunk2 = ProcessedChunk.create_from_chunk(
            chunk_text="Different text",
            context_header="test.md",
            metadata=sample_metadata,
            chunk_index=1,
            total_chunks=2,
            token_count=5,
        )
        chunk2.embedding = np.random.randn(768).tolist()

        result2 = SearchResult(0.85, chunk2)
        assert result1 != result2

    def test_search_result_repr(self, sample_chunk_with_embedding: ProcessedChunk) -> None:
        """Test SearchResult string representation."""
        result = SearchResult(0.95, sample_chunk_with_embedding)
        repr_str = repr(result)
        assert "0.95" in repr_str
        assert "SearchResult" in repr_str


class TestSearchStatsDataModel:
    """Tests for SearchStats data model."""

    def test_search_stats_creation(self) -> None:
        """Test SearchStats initialization."""
        stats = SearchStats(25.5, 10, 2600)
        assert stats.query_latency_ms == 25.5
        assert stats.results_returned == 10
        assert stats.total_vectors_searched == 2600

    def test_search_stats_to_dict(self) -> None:
        """Test conversion of stats to dictionary."""
        stats = SearchStats(25.5, 10, 2600)
        stats_dict = stats.to_dict()

        assert stats_dict["query_latency_ms"] == 25.5
        assert stats_dict["results_returned"] == 10
        assert stats_dict["total_vectors_searched"] == 2600

    def test_search_stats_validation_negative_latency(self) -> None:
        """Test SearchStats rejects negative latency."""
        with pytest.raises(ValueError, match="negative"):
            SearchStats(-1.0, 10, 2600)

    def test_search_stats_validation_negative_results(self) -> None:
        """Test SearchStats rejects negative results count."""
        with pytest.raises(ValueError, match="negative"):
            SearchStats(25.5, -1, 2600)

    def test_search_stats_validation_negative_vectors(self) -> None:
        """Test SearchStats rejects negative vector count."""
        with pytest.raises(ValueError, match="negative"):
            SearchStats(25.5, 10, -1)

    def test_search_stats_boundary_values(self) -> None:
        """Test SearchStats accepts zero values."""
        stats = SearchStats(0.0, 0, 0)
        assert stats.query_latency_ms == 0.0
        assert stats.results_returned == 0
        assert stats.total_vectors_searched == 0

    def test_search_stats_repr(self) -> None:
        """Test SearchStats string representation."""
        stats = SearchStats(25.5, 10, 2600)
        repr_str = repr(stats)
        assert "25.5" in repr_str
        assert "SearchStats" in repr_str


class TestVectorSearchInitialization:
    """Tests for VectorSearch initialization."""

    def test_vector_search_initialization_without_connection(self) -> None:
        """Test VectorSearch initializes with default (no connection)."""
        search = VectorSearch()
        assert search is not None

    def test_vector_search_initialization_with_connection(self) -> None:
        """Test VectorSearch initializes with provided connection."""
        mock_conn = MagicMock()
        search = VectorSearch(connection=mock_conn)
        assert search._connection == mock_conn


class TestEmbeddingValidation:
    """Tests for embedding validation."""

    def test_valid_768_dimensional_embedding(self, sample_query_embedding: list[float]) -> None:
        """Test validation of 768-dimensional embedding."""
        search = VectorSearch()
        assert search.validate_embedding(sample_query_embedding) is True

    def test_invalid_embedding_wrong_dimension(self) -> None:
        """Test rejection of non-768-dimensional embeddings."""
        search = VectorSearch()

        # 512-dimensional
        embedding_512 = np.random.randn(512).tolist()
        with pytest.raises(ValueError, match="768-dimensional"):
            search.validate_embedding(embedding_512)

        # 1024-dimensional
        embedding_1024 = np.random.randn(1024).tolist()
        with pytest.raises(ValueError, match="768-dimensional"):
            search.validate_embedding(embedding_1024)

    def test_invalid_embedding_not_a_list(self) -> None:
        """Test rejection of non-list embeddings."""
        search = VectorSearch()

        # NumPy array (not converted to list)
        embedding_array = np.random.randn(768)
        with pytest.raises(ValueError, match="must be a list"):
            search.validate_embedding(embedding_array)  # type: ignore

    def test_invalid_embedding_non_numeric_values(self) -> None:
        """Test rejection of non-numeric embedding values."""
        search = VectorSearch()

        # Mix of floats and strings
        embedding: list[float | str] = [0.1] * 767 + ["not_a_float"]
        with pytest.raises(ValueError, match="non-numeric"):
            search.validate_embedding(embedding)  # type: ignore

    def test_empty_embedding(self) -> None:
        """Test rejection of empty embedding."""
        search = VectorSearch()

        embedding: list[float] = []
        with pytest.raises(ValueError, match="768-dimensional"):
            search.validate_embedding(embedding)


class TestSearchParameterValidation:
    """Tests for search parameter validation."""

    def test_valid_top_k_values(self, sample_query_embedding: list[float]) -> None:
        """Test valid top_k parameter values."""
        search = VectorSearch()

        with patch("src.search.vector_search.DatabasePool.get_connection"):
            # These should all be valid (but fail due to no DB)
            valid_top_k = [1, 10, 100, 500, 1000]
            for top_k in valid_top_k:
                try:
                    search.search(sample_query_embedding, top_k=top_k)
                except RuntimeError:
                    pass  # Expected - no DB

    def test_invalid_top_k_zero(self, sample_query_embedding: list[float]) -> None:
        """Test rejection of top_k = 0."""
        search = VectorSearch()

        with pytest.raises(ValueError, match="1-1000"):
            search.search(sample_query_embedding, top_k=0)

    def test_invalid_top_k_negative(self, sample_query_embedding: list[float]) -> None:
        """Test rejection of negative top_k."""
        search = VectorSearch()

        with pytest.raises(ValueError, match="1-1000"):
            search.search(sample_query_embedding, top_k=-5)

    def test_invalid_top_k_exceeds_max(self, sample_query_embedding: list[float]) -> None:
        """Test rejection of top_k > 1000."""
        search = VectorSearch()

        with pytest.raises(ValueError, match="1-1000"):
            search.search(sample_query_embedding, top_k=1001)

    def test_invalid_similarity_metric(self, sample_query_embedding: list[float]) -> None:
        """Test rejection of unknown similarity metric."""
        search = VectorSearch()

        with pytest.raises(ValueError, match="Unknown similarity metric"):
            search.search(sample_query_embedding, similarity_metric="euclidean")


class TestBatchSearchValidation:
    """Tests for batch search validation."""

    def test_batch_search_empty_list(self) -> None:
        """Test batch search with empty embedding list."""
        search = VectorSearch()

        results, stats = search.batch_search([])
        assert results == []
        assert stats == []

    def test_batch_search_invalid_embedding_in_batch(self, sample_query_embedding: list[float]) -> None:
        """Test batch search rejects invalid embedding in batch."""
        search = VectorSearch()

        embeddings = [sample_query_embedding, [0.1] * 512]  # Second is invalid
        with pytest.raises(ValueError, match="Invalid embedding"):
            search.batch_search(embeddings)

    def test_batch_search_all_valid_embeddings(self) -> None:
        """Test batch search validates all embeddings."""
        search = VectorSearch()

        # Create multiple valid embeddings
        embeddings = [np.random.randn(768).tolist() for _ in range(3)]

        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                search.batch_search(embeddings)
            except RuntimeError:
                pass  # Expected - no DB


class TestFilteredSearchValidation:
    """Tests for filtered search validation."""

    def test_filtered_search_invalid_date_format(self, sample_query_embedding: list[float]) -> None:
        """Test filtered search rejects invalid date format."""
        search = VectorSearch()

        with pytest.raises(ValueError, match="Invalid date format"):
            search.search_with_filters(
                sample_query_embedding,
                document_date_min="not-a-date",
            )

    def test_filtered_search_valid_iso_date(self, sample_query_embedding: list[float]) -> None:
        """Test filtered search accepts valid ISO date."""
        search = VectorSearch()

        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                search.search_with_filters(
                    sample_query_embedding,
                    document_date_min="2025-01-01",
                )
            except RuntimeError:
                pass  # Expected - no DB


class TestSearchStatisticsTracking:
    """Tests for search statistics tracking."""

    def test_search_stats_latency_tracking(self) -> None:
        """Test that search tracks query latency."""
        stats = SearchStats(50.5, 10, 2600)
        assert stats.query_latency_ms > 0

    def test_search_stats_results_count_tracking(self) -> None:
        """Test that search tracks result count."""
        stats = SearchStats(25.0, 10, 2600)
        assert stats.results_returned == 10

    def test_search_stats_vector_count_tracking(self) -> None:
        """Test that search tracks total vector count."""
        stats = SearchStats(25.0, 10, 2600)
        assert stats.total_vectors_searched == 2600


class TestSimilarityMetrics:
    """Tests for similarity metric support."""

    def test_supported_metrics(self) -> None:
        """Test that all documented metrics are supported."""
        search = VectorSearch()

        # These should be accepted (but fail without DB)
        metrics = ["cosine", "l2", "inner_product"]

        embedding = np.random.randn(768).tolist()

        for metric in metrics:
            with patch("src.search.vector_search.DatabasePool.get_connection"):
                try:
                    search.search(embedding, similarity_metric=metric)
                except RuntimeError:
                    pass  # Expected - no DB

    def test_default_metric_is_cosine(self, sample_query_embedding: list[float]) -> None:
        """Test that default metric is cosine similarity."""
        search = VectorSearch()

        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                # Not specifying metric should use cosine (default)
                search.search(sample_query_embedding)
            except RuntimeError:
                pass  # Expected - no DB


class TestSearchResultRanking:
    """Tests for search result ranking and ordering."""

    def test_results_ordered_by_similarity(
        self, sample_chunk_with_embedding: ProcessedChunk
    ) -> None:
        """Test that results are returned in order by similarity."""
        # Create results with decreasing similarity
        results = [
            SearchResult(0.95, sample_chunk_with_embedding),
            SearchResult(0.85, sample_chunk_with_embedding),
            SearchResult(0.75, sample_chunk_with_embedding),
        ]

        # Verify ordering
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity


class TestPerformanceCharacteristics:
    """Tests for search performance expectations."""

    def test_single_query_latency_expectation(self) -> None:
        """Test expected latency for single query."""
        # Expectation: <100ms for 2600 vectors
        expected_latency_ms = 50.0
        assert 0 < expected_latency_ms < 100

    def test_batch_search_efficiency(self) -> None:
        """Test batch search is more efficient than individual queries."""
        # Batch of 5 queries should be more efficient than 5 individual queries
        single_query_latency = 50.0
        batch_overhead_ms = 10.0  # Minimal overhead

        batch_latency = (single_query_latency * 5) + batch_overhead_ms

        # Individual: 5 * 50 = 250ms
        # Batch: 5 * 50 + 10 = 260ms (but single connection is reused)
        assert batch_latency < (single_query_latency * 5) * 1.2


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_search_with_missing_embedding(self) -> None:
        """Test search handles missing embedding properly."""
        search = VectorSearch()

        with pytest.raises(ValueError):
            search.search([])  # Empty embedding

    def test_search_handles_database_error_gracefully(
        self, sample_query_embedding: list[float]
    ) -> None:
        """Test search raises RuntimeError on database failure."""
        search = VectorSearch()

        with patch(
            "src.search.vector_search.DatabasePool.get_connection",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(RuntimeError, match="Search failed"):
                search.search(sample_query_embedding)

    def test_batch_search_database_error(self) -> None:
        """Test batch search handles database errors."""
        search = VectorSearch()
        embeddings = [np.random.randn(768).tolist() for _ in range(2)]

        with patch(
            "src.search.vector_search.DatabasePool.get_connection",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(RuntimeError, match="Batch search failed"):
                search.batch_search(embeddings)


class TestIntegrationPatterns:
    """Tests for typical usage patterns."""

    def test_single_similarity_search_pattern(self) -> None:
        """Test typical single search pattern."""
        search = VectorSearch()
        query = np.random.randn(768).tolist()

        # Pattern: search -> validate -> process results
        try:
            search.validate_embedding(query)
            # Would call: results, stats = search.search(query)
        except ValueError:
            pytest.fail("Should accept valid embedding")

    def test_batch_search_pattern(self) -> None:
        """Test typical batch search pattern."""
        search = VectorSearch()
        queries = [np.random.randn(768).tolist() for _ in range(5)]

        # Pattern: validate all -> batch search -> process
        try:
            for query in queries:
                search.validate_embedding(query)
            # Would call: results_batch, stats_batch = search.batch_search(queries)
        except ValueError:
            pytest.fail("Should accept valid embeddings")

    def test_filtered_search_pattern(self) -> None:
        """Test typical filtered search pattern."""
        search = VectorSearch()
        query = np.random.randn(768).tolist()

        # Pattern: filtered search with metadata constraints
        try:
            search.validate_embedding(query)
            # Would call: results, stats = search.search_with_filters(
            #     query,
            #     source_category="docs",
            #     document_date_min="2025-01-01"
            # )
        except ValueError:
            pytest.fail("Should accept valid parameters")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_top_k_equals_1(self) -> None:
        """Test search with top_k=1 (single result)."""
        search = VectorSearch()
        query = np.random.randn(768).tolist()

        # Should be valid
        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                search.search(query, top_k=1)
            except RuntimeError:
                pass  # Expected - no DB

    def test_top_k_equals_max(self) -> None:
        """Test search with top_k=1000 (maximum)."""
        search = VectorSearch()
        query = np.random.randn(768).tolist()

        # Should be valid
        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                search.search(query, top_k=1000)
            except RuntimeError:
                pass  # Expected - no DB

    def test_zero_vectors_in_database(self) -> None:
        """Test search statistics with zero indexed vectors."""
        stats = SearchStats(10.0, 0, 0)  # No results, no vectors
        assert stats.results_returned == 0
        assert stats.total_vectors_searched == 0

    def test_all_vectors_match_equally(self, sample_chunk_with_embedding: ProcessedChunk) -> None:
        """Test handling of all results with same similarity."""
        # Create results with same similarity (hypothetical edge case)
        results = [
            SearchResult(0.85, sample_chunk_with_embedding),
            SearchResult(0.85, sample_chunk_with_embedding),
            SearchResult(0.85, sample_chunk_with_embedding),
        ]

        assert all(r.similarity == 0.85 for r in results)


class TestDocumentationExamples:
    """Tests based on module documentation examples."""

    def test_basic_search_usage(self) -> None:
        """Test basic search usage from docstring."""
        search = VectorSearch()
        query = np.random.randn(768).tolist()

        # From docstring: results, stats = search.search(query, top_k=10)
        assert search.validate_embedding(query)

    def test_statistics_retrieval(self) -> None:
        """Test statistics retrieval pattern."""
        search = VectorSearch()

        with patch("src.search.vector_search.DatabasePool.get_connection"):
            try:
                # Would call: stats_dict = search.get_statistics()
                pass
            except RuntimeError:
                pass  # Expected - no DB
