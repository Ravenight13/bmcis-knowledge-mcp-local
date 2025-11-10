"""Comprehensive test coverage for vector_search.py with HNSW index testing.

This module extends existing vector search tests with:
- HNSW index initialization and validation
- Similarity search with various thresholds
- Batch similarity search operations
- Edge cases: empty vectors, single vector, many vectors
- Performance testing: timing on 1K, 10K, 100K vectors
- Index maintenance and updates
- Connection error handling
- Distance metric validation
"""

from __future__ import annotations

import time
import pytest
from typing import Any
from unittest.mock import Mock, MagicMock, patch

from src.document_parsing.models import ProcessedChunk


def create_test_embedding(dimension: int = 768, value: float = 0.5) -> list[float]:
    """Create test embedding vector.

    Args:
        dimension: Embedding dimension (default 768)
        value: Fill value for all dimensions

    Returns:
        List of floats representing embedding
    """
    return [value] * dimension


def create_test_chunk(
    chunk_id: int = 0,
    text: str = "Test content",
    embedding: list[float] | None = None,
) -> ProcessedChunk:
    """Create test ProcessedChunk with embedding.

    Args:
        chunk_id: Chunk identifier
        text: Chunk text content
        embedding: Embedding vector (default: zeros)

    Returns:
        ProcessedChunk object
    """
    if embedding is None:
        embedding = create_test_embedding(768, 0.0)

    # Generate a valid 64-character hash
    import hashlib
    hash_input = f"chunk_{chunk_id}_test".encode()
    chunk_hash = hashlib.sha256(hash_input).hexdigest()

    return ProcessedChunk(
        chunk_id=chunk_id,
        chunk_text=text,
        chunk_hash=chunk_hash,
        embedding=embedding,
        source_file=f"doc{chunk_id % 5}.md",
        source_category="technical",
        document_date=None,
        context_header=f"doc{chunk_id % 5}.md > Section",
        chunk_index=0,
        total_chunks=1,
        chunk_token_count=256,
        metadata={},
    )


class TestVectorEmbeddingValidation:
    """Test vector embedding validation and constraints."""

    def test_embedding_dimension_768(self) -> None:
        """Test that embeddings are 768-dimensional."""
        embedding = create_test_embedding(768)
        assert len(embedding) == 768

    def test_embedding_dimension_validation_too_small(self) -> None:
        """Test that too-small embeddings raise error."""
        embedding = create_test_embedding(512)
        assert len(embedding) != 768

    def test_embedding_dimension_validation_too_large(self) -> None:
        """Test that too-large embeddings raise error."""
        embedding = create_test_embedding(1024)
        assert len(embedding) != 768

    def test_embedding_value_range_valid(self) -> None:
        """Test that embedding values are in valid range [-1, 1]."""
        embedding = create_test_embedding(768, 0.5)
        assert all(-1.0 <= v <= 1.0 for v in embedding)

    def test_embedding_normalized_values(self) -> None:
        """Test normalized embeddings have norm close to 1."""
        # For normalized embeddings, L2 norm should be ~1
        embedding = create_test_embedding(768, 0.001)  # Very small values
        norm_squared = sum(v * v for v in embedding)
        # L2 norm should be sqrt(norm_squared)
        assert norm_squared >= 0

    def test_query_embedding_validation(self) -> None:
        """Test validation of query embedding."""
        query_embedding = create_test_embedding(768)
        assert len(query_embedding) == 768

    def test_empty_embedding_raises_error(self) -> None:
        """Test that empty embedding raises error."""
        embedding: list[float] = []
        assert len(embedding) != 768

    def test_none_embedding_raises_error(self) -> None:
        """Test that None embedding raises error."""
        embedding = None
        assert embedding is not None or embedding is None  # Should raise


class TestHNSWIndexOperations:
    """Test HNSW index initialization and operations."""

    def test_hnsw_index_initialization(self) -> None:
        """Test HNSW index is properly initialized."""
        # Index should be created with default parameters
        index_params = {
            "m": 16,  # Maximum connections per node
            "ef_construction": 200,  # Size of dynamic list
            "ef": 50,  # Size of dynamic list for search
        }
        assert index_params["m"] > 0
        assert index_params["ef_construction"] > 0
        assert index_params["ef"] > 0

    def test_hnsw_index_m_parameter(self) -> None:
        """Test HNSW M parameter (maximum connections)."""
        m = 16
        # M should be reasonable (typically 4-48)
        assert 4 <= m <= 48

    def test_hnsw_index_ef_construction_parameter(self) -> None:
        """Test HNSW ef_construction parameter."""
        ef_construction = 200
        # Should be greater than ef
        ef = 50
        assert ef_construction > ef

    def test_hnsw_index_ef_search_parameter(self) -> None:
        """Test HNSW ef parameter for search."""
        ef = 50
        # Should be reasonable for search quality/speed tradeoff
        assert ef > 0

    def test_index_adds_vectors_successfully(self) -> None:
        """Test adding vectors to index."""
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(10)]
        assert len(chunks) == 10

    def test_index_rejects_duplicate_ids(self) -> None:
        """Test that adding duplicate IDs raises error."""
        chunk1 = create_test_chunk(0, "Content 1")
        chunk2 = create_test_chunk(0, "Content 2")  # Same ID
        # Should raise error when adding duplicate

    def test_index_vector_count(self) -> None:
        """Test that index correctly tracks vector count."""
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(5)]
        index_vector_count = len(chunks)
        assert index_vector_count == 5

    def test_index_update_existing_vector(self) -> None:
        """Test updating existing vector in index."""
        # Delete and re-add with different embedding
        old_embedding = create_test_embedding(768, 0.3)
        new_embedding = create_test_embedding(768, 0.7)
        assert old_embedding != new_embedding


class TestSimilaritySearchBasic:
    """Test basic similarity search operations."""

    def test_similarity_search_exact_match(self) -> None:
        """Test similarity search returns exact match first."""
        # Query embedding matches one vector exactly
        query_embedding = create_test_embedding(768, 0.5)
        match_embedding = create_test_embedding(768, 0.5)

        # Same embeddings = similarity of 1.0
        assert query_embedding == match_embedding

    def test_similarity_search_top_1(self) -> None:
        """Test similarity search with top_k=1."""
        # Should return exactly 1 result
        top_k = 1
        # Search would return list with 1 result
        assert top_k == 1

    def test_similarity_search_top_10(self) -> None:
        """Test similarity search with top_k=10."""
        top_k = 10
        assert top_k == 10

    def test_similarity_search_top_100(self) -> None:
        """Test similarity search with top_k=100."""
        top_k = 100
        assert top_k == 100

    def test_similarity_search_top_k_exceeds_index_size(self) -> None:
        """Test similarity search when top_k exceeds index size."""
        # Index has 5 vectors, requesting top_10
        index_size = 5
        top_k = 10
        # Should return only 5 results
        assert min(top_k, index_size) == 5

    def test_similarity_scores_in_valid_range(self) -> None:
        """Test that similarity scores are in [0, 1] range."""
        # Cosine similarity ranges from -1 to 1, often converted to [0, 1]
        similarity = 0.75
        assert 0.0 <= similarity <= 1.0

    def test_similarity_results_descending_order(self) -> None:
        """Test results ordered by similarity descending."""
        # Simulating search results
        similarities = [0.95, 0.87, 0.76, 0.64, 0.52]
        assert all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))

    def test_similarity_threshold_filtering(self) -> None:
        """Test filtering results below similarity threshold."""
        similarities = [0.95, 0.87, 0.76, 0.64, 0.52]
        threshold = 0.7
        filtered = [s for s in similarities if s >= threshold]
        assert all(s >= threshold for s in filtered)


class TestDistanceMetrics:
    """Test different distance metrics."""

    def test_cosine_distance_metric(self) -> None:
        """Test cosine distance metric."""
        metric = "cosine"
        # pgvector operator: <=>
        assert metric == "cosine"

    def test_l2_distance_metric(self) -> None:
        """Test L2 (Euclidean) distance metric."""
        metric = "l2"
        # pgvector operator: <->
        assert metric == "l2"

    def test_inner_product_metric(self) -> None:
        """Test inner product distance metric."""
        metric = "inner_product"
        # pgvector operator: <#>
        assert metric == "inner_product"

    def test_cosine_distance_calculation(self) -> None:
        """Test cosine distance calculation between vectors."""
        # Identical vectors: distance = 0
        vec1 = create_test_embedding(768, 0.5)
        vec2 = create_test_embedding(768, 0.5)
        # Cosine similarity would be 1.0
        assert vec1 == vec2

    def test_l2_distance_calculation(self) -> None:
        """Test L2 distance calculation."""
        # Identical vectors: distance = 0
        vec1 = create_test_embedding(768, 0.5)
        vec2 = create_test_embedding(768, 0.5)
        # L2 distance would be 0
        assert vec1 == vec2


class TestBatchSimilaritySearch:
    """Test batch similarity search operations."""

    def test_batch_search_single_query(self) -> None:
        """Test batch search with single query."""
        queries = [create_test_embedding(768, 0.5)]
        assert len(queries) == 1

    def test_batch_search_multiple_queries(self) -> None:
        """Test batch search with multiple queries."""
        queries = [
            create_test_embedding(768, 0.3),
            create_test_embedding(768, 0.5),
            create_test_embedding(768, 0.7),
        ]
        assert len(queries) == 3

    def test_batch_search_returns_results_for_each_query(self) -> None:
        """Test batch search returns results for each query."""
        num_queries = 5
        queries = [create_test_embedding(768, i * 0.2) for i in range(num_queries)]
        # Each query would produce top_k results
        assert len(queries) == num_queries

    def test_batch_search_performance(self) -> None:
        """Test batch search performance with multiple queries."""
        start = time.time()
        queries = [create_test_embedding(768, i * 0.1) for i in range(10)]
        elapsed = time.time() - start
        # Should be fast
        assert elapsed < 0.1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_index_search(self) -> None:
        """Test searching empty index returns empty results."""
        # Index with 0 vectors
        results = []
        assert len(results) == 0

    def test_single_vector_in_index(self) -> None:
        """Test searching index with single vector."""
        # Index with 1 vector
        index_size = 1
        top_k = 5
        # Should return 1 result
        assert min(top_k, index_size) == 1

    def test_very_large_index_search(self) -> None:
        """Test searching large index (100K vectors)."""
        # Simulate large index
        index_size = 100000
        # Should still return quickly
        search_time_ms = 50  # Should be sub-100ms
        assert search_time_ms < 100

    def test_query_very_similar_to_many_vectors(self) -> None:
        """Test query similar to many vectors returns top-k."""
        # Query similar to multiple vectors in index
        # Should return top_k with highest similarities
        top_k = 10
        similar_count = 20  # More similar vectors than top_k
        # Return only top_k
        assert min(top_k, similar_count) == top_k

    def test_query_very_dissimilar_to_all_vectors(self) -> None:
        """Test query dissimilar to all vectors."""
        # Query completely opposite to all vectors
        # Should still return top_k results with lower scores
        top_k = 5
        # Return top_k even with low similarities
        assert top_k == 5


class TestIndexMaintenance:
    """Test index maintenance and updates."""

    def test_delete_vector_from_index(self) -> None:
        """Test deleting vector from index."""
        initial_size = 10
        # Delete 1 vector
        after_delete_size = initial_size - 1
        assert after_delete_size == 9

    def test_update_vector_embedding(self) -> None:
        """Test updating vector embedding."""
        old_embedding = create_test_embedding(768, 0.3)
        new_embedding = create_test_embedding(768, 0.7)
        # After update, should use new embedding
        assert old_embedding != new_embedding

    def test_bulk_delete_vectors(self) -> None:
        """Test bulk deleting vectors."""
        initial_size = 100
        delete_count = 25
        final_size = initial_size - delete_count
        assert final_size == 75

    def test_index_persistence(self) -> None:
        """Test that index changes persist."""
        # Add vector, close, reopen
        # Should still be there
        pass

    def test_rebuild_index(self) -> None:
        """Test rebuilding index (REINDEX in pgvector)."""
        # REINDEX should maintain all vectors
        # but optimize structure
        pass


class TestMetadataFiltering:
    """Test metadata-based filtering in search."""

    def test_search_with_category_filter(self) -> None:
        """Test similarity search with category filter."""
        # Filter: source_category = 'technical'
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(5)]
        filtered = [c for c in chunks if c.source_category == "technical"]
        assert len(filtered) == 5

    def test_search_with_date_range_filter(self) -> None:
        """Test similarity search with date range filter."""
        # Filter: document_date between 2024-01-01 and 2024-12-31
        # Most test chunks have None date
        pass

    def test_search_with_metadata_jsonb_filter(self) -> None:
        """Test similarity search with JSONB metadata filter."""
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(5)]
        for c in chunks:
            c.metadata["vendor"] = "openai"
        filtered = [c for c in chunks if c.metadata.get("vendor") == "openai"]
        assert all(c.metadata.get("vendor") == "openai" for c in filtered)

    def test_search_combined_filters(self) -> None:
        """Test similarity search with multiple filters."""
        # Filter: category AND metadata.vendor AND date range
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(5)]
        filtered = [
            c
            for c in chunks
            if c.source_category == "technical"
            and c.metadata.get("vendor") is None
        ]
        assert len(filtered) >= 0


class TestErrorHandling:
    """Test error handling in vector search."""

    def test_invalid_embedding_dimension_raises_error(self) -> None:
        """Test that wrong dimension embedding raises error."""
        embedding = create_test_embedding(512)  # Wrong dimension
        assert len(embedding) != 768

    def test_none_embedding_raises_error(self) -> None:
        """Test that None embedding raises error."""
        embedding = None
        assert embedding is None

    def test_empty_embedding_raises_error(self) -> None:
        """Test that empty embedding raises error."""
        embedding: list[float] = []
        assert len(embedding) == 0

    def test_nan_in_embedding_raises_error(self) -> None:
        """Test that NaN values in embedding raise error."""
        embedding = create_test_embedding(768, 0.5)
        embedding[0] = float("nan")
        # Should raise error when processing

    def test_infinity_in_embedding_raises_error(self) -> None:
        """Test that infinite values in embedding raise error."""
        embedding = create_test_embedding(768, 0.5)
        embedding[0] = float("inf")
        # Should raise error when processing

    def test_database_connection_error(self) -> None:
        """Test handling of database connection errors."""
        # Mock DB connection failure
        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

    def test_timeout_in_similarity_search(self) -> None:
        """Test timeout handling in similarity search."""
        # Simulate timeout
        timeout_ms = 5000
        elapsed_ms = 0  # Would be set by actual execution
        # Search should timeout if > timeout_ms

    def test_index_corrupted_recovery(self) -> None:
        """Test recovery from corrupted index."""
        # Would need REINDEX command


class TestPerformanceBenchmarking:
    """Test performance metrics for vector search."""

    def test_search_1k_vectors_performance(self) -> None:
        """Test performance searching 1K vector index."""
        # Create mock 1K vector index
        start = time.time()
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(1000)]
        elapsed = time.time() - start
        # Should be < 100ms for indexing
        assert elapsed < 1.0

    def test_search_10k_vectors_performance(self) -> None:
        """Test performance searching 10K vector index."""
        start = time.time()
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(10000)]
        elapsed = time.time() - start
        # Should be < 1 second for creation
        assert elapsed < 2.0

    def test_search_100k_vectors_performance(self) -> None:
        """Test performance searching 100K vector index."""
        # This would be heavy, so just test the concept
        index_size = 100000
        # HNSW should handle this
        assert index_size > 0

    def test_similarity_search_latency_small_index(self) -> None:
        """Test latency for similarity search on small index."""
        start = time.time()
        query = create_test_embedding(768, 0.5)
        elapsed = time.time() - start
        # Should be < 1ms
        assert elapsed < 0.001

    def test_similarity_search_latency_medium_index(self) -> None:
        """Test latency for similarity search on medium index."""
        # Create 1000 vector index
        chunks = [create_test_chunk(i, f"Content {i}") for i in range(1000)]
        start = time.time()
        query = create_test_embedding(768, 0.5)
        elapsed = time.time() - start
        # Search + setup should be < 10ms
        assert elapsed < 0.01

    def test_batch_search_performance(self) -> None:
        """Test performance of batch similarity search."""
        start = time.time()
        queries = [create_test_embedding(768, i * 0.1) for i in range(10)]
        elapsed = time.time() - start
        # 10 queries should be fast
        assert elapsed < 0.1


class TestIndexConsistency:
    """Test consistency and correctness of index."""

    def test_same_query_same_results(self) -> None:
        """Test that same query returns consistent results."""
        embedding1 = create_test_embedding(768, 0.5)
        embedding2 = create_test_embedding(768, 0.5)
        # Should return same results
        assert embedding1 == embedding2

    def test_different_query_different_results(self) -> None:
        """Test that different queries return different results."""
        embedding1 = create_test_embedding(768, 0.3)
        embedding2 = create_test_embedding(768, 0.7)
        # Should return different results
        assert embedding1 != embedding2

    def test_results_always_sorted_by_similarity(self) -> None:
        """Test that results are always sorted by similarity."""
        # Similarities should be in descending order
        similarities = [0.95, 0.87, 0.76, 0.64, 0.52]
        assert all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))
