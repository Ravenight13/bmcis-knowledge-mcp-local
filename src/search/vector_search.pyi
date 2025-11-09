"""Type stubs for vector similarity search using HNSW index.

Defines complete type signatures for pgvector cosine similarity search
with HNSW index on 768-dimensional embeddings.
"""

from typing import NamedTuple

from psycopg2.extensions import connection as Connection

from src.document_parsing.models import ProcessedChunk


class SearchResult(NamedTuple):
    """Result of vector similarity search.

    Attributes:
        similarity: Cosine similarity score (0-1, higher is more similar).
        chunk: The matched ProcessedChunk object with metadata.
    """

    similarity: float
    chunk: ProcessedChunk


class SearchStats(NamedTuple):
    """Statistics from vector search operation.

    Attributes:
        query_latency_ms: Query execution time in milliseconds.
        results_returned: Number of results returned.
        total_vectors_searched: Total vectors in knowledge_base.
    """

    query_latency_ms: float
    results_returned: int
    total_vectors_searched: int


class VectorSearch:
    """Vector similarity search using pgvector HNSW index.

    Provides efficient similarity search on 768-dimensional embeddings
    using PostgreSQL pgvector with HNSW index for fast nearest neighbor queries.
    Supports cosine similarity (default) and other distance metrics.
    """

    def __init__(self, connection: Connection | None = None) -> None:
        """Initialize VectorSearch with optional connection.

        Args:
            connection: Optional psycopg2 connection. If not provided,
                       queries will acquire connections from DatabasePool.

        Raises:
            ValueError: If provided connection is invalid.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_metric: str = "cosine",
    ) -> tuple[list[SearchResult], SearchStats]:
        """Search for similar chunks using vector similarity.

        Executes HNSW similarity search for top_k most similar chunks
        to the query embedding. Returns results sorted by similarity
        (highest first).

        Args:
            query_embedding: Query vector (must be 768 dimensions).
            top_k: Number of top results to return (default: 10, max: 1000).
            similarity_metric: Similarity metric to use (default: "cosine").
                             Options: "cosine", "l2", "inner_product".

        Returns:
            Tuple of:
            - list[SearchResult]: Top K results with similarity scores.
            - SearchStats: Query performance statistics.

        Raises:
            ValueError: If query_embedding not 768-dimensional or top_k invalid.
            RuntimeError: If database connection/query fails.

        Example:
            >>> search = VectorSearch()
            >>> query = [0.1, 0.2, ..., 0.5]  # 768 dims
            >>> results, stats = search.search(query, top_k=10)
            >>> for result in results:
            ...     print(f"Similarity: {result.similarity:.4f}")
            ...     print(result.chunk.chunk_text)
        """
        ...

    def search_with_filters(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_category: str | None = None,
        document_date_min: str | None = None,
        similarity_metric: str = "cosine",
    ) -> tuple[list[SearchResult], SearchStats]:
        """Search with optional metadata filters.

        Extends search() with WHERE clause filters on chunk metadata
        before similarity ranking.

        Args:
            query_embedding: Query vector (768 dimensions).
            top_k: Number of top results to return (default: 10).
            source_category: Filter by document category (optional).
            document_date_min: Filter by document date >= this date (ISO format).
            similarity_metric: Similarity metric ("cosine", "l2", "inner_product").

        Returns:
            Tuple of (results, stats) same as search().

        Raises:
            ValueError: If embedding invalid or filters invalid.
            RuntimeError: If database query fails.
        """
        ...

    def batch_search(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 10,
        similarity_metric: str = "cosine",
    ) -> tuple[list[list[SearchResult]], list[SearchStats]]:
        """Batch search for multiple query embeddings.

        Efficient batch processing of multiple similarity searches.
        More efficient than individual search() calls due to connection reuse.

        Args:
            query_embeddings: List of query vectors (each 768 dimensions).
            top_k: Number of top results per query (default: 10).
            similarity_metric: Similarity metric to use.

        Returns:
            Tuple of:
            - list[list[SearchResult]]: Results for each query.
            - list[SearchStats]: Stats for each query.

        Raises:
            ValueError: If any embedding invalid.
            RuntimeError: If database query fails.
        """
        ...

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding has correct dimension (768).

        Args:
            embedding: Vector to validate.

        Returns:
            True if valid 768-dimensional float list.

        Raises:
            ValueError: If embedding invalid.
        """
        ...

    def get_statistics(self) -> dict[str, int | float]:
        """Get statistics about indexed vectors.

        Returns:
            Dictionary with:
            - total_vectors: Total indexed vectors.
            - index_exists: Whether HNSW index exists.
            - index_size_mb: Approximate index size in MB (if known).
        """
        ...
