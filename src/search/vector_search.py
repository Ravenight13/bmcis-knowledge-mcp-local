"""Vector similarity search using HNSW index on pgvector.

Provides efficient similarity search on 768-dimensional embeddings using
PostgreSQL pgvector with HNSW (Hierarchical Navigable Small World) index
for fast nearest-neighbor queries with cosine similarity metric.

The module handles:
- Validation of 768-dimensional query embeddings
- HNSW similarity search with configurable top_k
- Support for multiple distance metrics (cosine, L2, inner product)
- Metadata filtering (category, date range)
- Batch search operations for efficiency
- Comprehensive error handling and logging
- Query performance statistics
"""

import logging
import time
from typing import Any

from psycopg2.extensions import connection as Connection

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.document_parsing.models import ProcessedChunk


# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Constants
EMBEDDING_DIMENSION: int = 768
MAX_TOP_K: int = 1000
DEFAULT_TOP_K: int = 10

# Supported similarity metrics and their pgvector operators
SIMILARITY_METRICS: dict[str, str] = {
    "cosine": "<=>",      # Cosine distance operator
    "l2": "<->",          # L2 distance operator
    "inner_product": "<#>",  # Inner product distance operator
}


class SearchResult:
    """Result of vector similarity search.

    Attributes:
        similarity: Cosine similarity score (0-1, higher is more similar).
        chunk: The matched ProcessedChunk object with metadata.
    """

    def __init__(self, similarity: float, chunk: ProcessedChunk) -> None:
        """Initialize search result.

        Args:
            similarity: Similarity score (0-1 for cosine).
            chunk: The matched ProcessedChunk.

        Raises:
            ValueError: If similarity not in valid range.
        """
        if not 0 <= similarity <= 1:
            raise ValueError(f"Similarity score must be 0-1, got {similarity}")

        self.similarity: float = similarity
        self.chunk: ProcessedChunk = chunk

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"SearchResult(similarity={self.similarity:.4f}, "
            f"chunk_hash={self.chunk.chunk_hash[:16]}...)"
        )

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (
            abs(self.similarity - other.similarity) < 1e-6
            and self.chunk.chunk_hash == other.chunk.chunk_hash
        )


class SearchStats:
    """Statistics from vector search operation.

    Attributes:
        query_latency_ms: Query execution time in milliseconds.
        results_returned: Number of results returned.
        total_vectors_searched: Total vectors in knowledge_base.
    """

    def __init__(
        self,
        query_latency_ms: float,
        results_returned: int,
        total_vectors_searched: int,
    ) -> None:
        """Initialize search statistics.

        Args:
            query_latency_ms: Query execution time in ms.
            results_returned: Number of results.
            total_vectors_searched: Total vectors in index.

        Raises:
            ValueError: If values invalid.
        """
        if query_latency_ms < 0:
            raise ValueError(f"Latency cannot be negative, got {query_latency_ms}")
        if results_returned < 0:
            raise ValueError(f"Results cannot be negative, got {results_returned}")
        if total_vectors_searched < 0:
            raise ValueError(f"Total vectors cannot be negative, got {total_vectors_searched}")

        self.query_latency_ms: float = query_latency_ms
        self.results_returned: int = results_returned
        self.total_vectors_searched: int = total_vectors_searched

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SearchStats(latency={self.query_latency_ms:.2f}ms, "
            f"results={self.results_returned}, "
            f"total_vectors={self.total_vectors_searched})"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Convert stats to dictionary for logging.

        Returns:
            Dictionary with all statistics fields.
        """
        return {
            "query_latency_ms": self.query_latency_ms,
            "results_returned": self.results_returned,
            "total_vectors_searched": self.total_vectors_searched,
        }


class VectorSearch:
    """Vector similarity search using pgvector HNSW index.

    Provides efficient similarity search on 768-dimensional embeddings
    using PostgreSQL pgvector with HNSW index for fast nearest neighbor queries.
    Supports cosine similarity (default) and other distance metrics.

    The class uses DatabasePool for connection management and implements
    comprehensive error handling with detailed logging for debugging.

    Example:
        >>> search = VectorSearch()
        >>> query = [0.1, 0.2, ..., 0.5]  # 768 dims
        >>> results, stats = search.search(query, top_k=10)
        >>> for result in results:
        ...     print(f"Match: {result.chunk.chunk_text[:50]}")
        ...     print(f"Similarity: {result.similarity:.4f}")
    """

    def __init__(self, connection: Connection | None = None) -> None:
        """Initialize VectorSearch with optional connection.

        Args:
            connection: Optional psycopg2 connection. If not provided,
                       queries will acquire connections from DatabasePool.

        Raises:
            ValueError: If provided connection is invalid.
        """
        self._connection: Connection | None = connection
        logger.info("VectorSearch initialized")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_metric: str = "cosine",
    ) -> tuple[list[SearchResult], SearchStats]:
        """Search for similar chunks using vector similarity.

        Executes HNSW similarity search for top_k most similar chunks
        to the query embedding. Returns results sorted by similarity
        (highest first). Uses pgvector distance operators.

        The SQL query uses the <=> operator for cosine distance:
        SELECT * FROM knowledge_base ORDER BY embedding <=> %s LIMIT %s

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
        # Validate inputs
        self.validate_embedding(query_embedding)

        if not isinstance(top_k, int) or top_k < 1 or top_k > MAX_TOP_K:
            msg = f"top_k must be 1-{MAX_TOP_K}, got {top_k}"
            logger.error(msg)
            raise ValueError(msg)

        if similarity_metric not in SIMILARITY_METRICS:
            msg = f"Unknown similarity metric: {similarity_metric}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Starting similarity search: top_k={top_k}, metric={similarity_metric}"
        )

        start_time: float = time.time()
        results: list[SearchResult] = []

        try:
            # Acquire connection (use provided or get from pool)
            if self._connection is not None:
                conn: Connection = self._connection
            else:
                with DatabasePool.get_connection() as conn:
                    results, stats = self._execute_search(
                        conn, query_embedding, top_k, similarity_metric
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    logger.info(f"Search completed in {latency_ms:.2f}ms, {len(results)} results")
                    return results, stats

            # If using provided connection
            results, stats = self._execute_search(
                conn, query_embedding, top_k, similarity_metric
            )
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Search completed in {latency_ms:.2f}ms, {len(results)} results")
            return results, stats

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}") from e

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
        before similarity ranking. Filters reduce the search space.

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
        # Validate inputs
        self.validate_embedding(query_embedding)

        if not isinstance(top_k, int) or top_k < 1 or top_k > MAX_TOP_K:
            msg = f"top_k must be 1-{MAX_TOP_K}, got {top_k}"
            logger.error(msg)
            raise ValueError(msg)

        if similarity_metric not in SIMILARITY_METRICS:
            msg = f"Unknown similarity metric: {similarity_metric}"
            logger.error(msg)
            raise ValueError(msg)

        # Validate filters if provided
        if document_date_min is not None:
            try:
                # Validate ISO format date
                from datetime import datetime
                datetime.fromisoformat(document_date_min)
            except ValueError:
                msg = f"Invalid date format: {document_date_min} (use ISO format)"
                logger.error(msg)
                raise ValueError(msg) from None

        logger.info(
            f"Starting filtered search: top_k={top_k}, "
            f"category={source_category}, date_min={document_date_min}"
        )

        start_time: float = time.time()

        try:
            # Build WHERE clause for filters
            where_clauses: list[str] = ["embedding IS NOT NULL"]
            params: list[Any] = [query_embedding]

            if source_category is not None:
                where_clauses.append("source_category = %s")
                params.append(source_category)

            if document_date_min is not None:
                where_clauses.append("document_date >= %s")
                params.append(document_date_min)

            where_sql = " AND ".join(where_clauses)

            # Build and execute SQL
            if self._connection is not None:
                conn = self._connection
            else:
                with DatabasePool.get_connection() as conn:
                    results, stats = self._execute_filtered_search(
                        conn, params, where_sql, top_k, similarity_metric
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"Filtered search completed in {latency_ms:.2f}ms, "
                        f"{len(results)} results"
                    )
                    return results, stats

            # If using provided connection
            results, stats = self._execute_filtered_search(
                conn, params, where_sql, top_k, similarity_metric
            )
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Filtered search completed in {latency_ms:.2f}ms, "
                f"{len(results)} results"
            )
            return results, stats

        except Exception as e:
            logger.error(f"Filtered search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}") from e

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
        if not query_embeddings:
            logger.warning("Empty query embeddings list provided")
            return [], []

        # Validate all embeddings
        for i, embedding in enumerate(query_embeddings):
            try:
                self.validate_embedding(embedding)
            except ValueError as e:
                msg = f"Invalid embedding at index {i}: {e}"
                logger.error(msg)
                raise ValueError(msg) from e

        logger.info(f"Starting batch search for {len(query_embeddings)} queries")

        all_results: list[list[SearchResult]] = []
        all_stats: list[SearchStats] = []

        try:
            # Reuse single connection for efficiency
            if self._connection is not None:
                conn = self._connection
                for embedding in query_embeddings:
                    results, stats = self._execute_search(
                        conn, embedding, top_k, similarity_metric
                    )
                    all_results.append(results)
                    all_stats.append(stats)
            else:
                with DatabasePool.get_connection() as conn:
                    for embedding in query_embeddings:
                        results, stats = self._execute_search(
                            conn, embedding, top_k, similarity_metric
                        )
                        all_results.append(results)
                        all_stats.append(stats)

            logger.info(
                f"Batch search completed: {len(all_results)} queries, "
                f"avg {sum(s.results_returned for s in all_stats) / len(all_stats):.1f} "
                f"results/query"
            )

            return all_results, all_stats

        except Exception as e:
            logger.error(f"Batch search failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch search failed: {e}") from e

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding has correct dimension (768).

        Args:
            embedding: Vector to validate.

        Returns:
            True if valid 768-dimensional float list.

        Raises:
            ValueError: If embedding invalid.
        """
        if not isinstance(embedding, list):
            msg = f"Embedding must be a list, got {type(embedding).__name__}"
            logger.error(msg)
            raise ValueError(msg)

        if len(embedding) != EMBEDDING_DIMENSION:
            msg = (
                f"Embedding must be {EMBEDDING_DIMENSION}-dimensional, "
                f"got {len(embedding)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Validate all elements are numeric
        invalid_elements = [
            (i, x) for i, x in enumerate(embedding)
            if not isinstance(x, (int, float))
        ]

        if invalid_elements:
            invalid_str = ", ".join(
                f"[{i}]={type(x).__name__}" for i, x in invalid_elements[:5]
            )
            msg = f"Embedding contains non-numeric values: {invalid_str}"
            logger.error(msg)
            raise ValueError(msg)

        return True

    def get_statistics(self) -> dict[str, int | float]:
        """Get statistics about indexed vectors.

        Returns:
            Dictionary with:
            - total_vectors: Total indexed vectors.
            - index_exists: Whether HNSW index exists.

        Raises:
            RuntimeError: If database query fails.
        """
        try:
            if self._connection is not None:
                conn = self._connection
            else:
                with DatabasePool.get_connection() as conn:
                    return self._fetch_statistics(conn)

            return self._fetch_statistics(conn)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            raise RuntimeError(f"Statistics query failed: {e}") from e

    def _execute_search(
        self,
        conn: Connection,
        query_embedding: list[float],
        top_k: int,
        similarity_metric: str,
    ) -> tuple[list[SearchResult], SearchStats]:
        """Execute similarity search query.

        Internal method to run the actual SQL search and convert results.

        Args:
            conn: Database connection.
            query_embedding: Query vector.
            top_k: Number of results.
            similarity_metric: Distance metric.

        Returns:
            Tuple of (results, stats).

        Raises:
            Exception: If query execution fails.
        """
        start_time = time.time()

        # Get the distance operator for the metric
        distance_op = SIMILARITY_METRICS[similarity_metric]

        # Build pgvector format string
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Execute similarity search
        # For cosine: <=> returns distance (0-2), we convert to similarity (0-1)
        sql = f"""
            SELECT
                id,
                chunk_text,
                chunk_hash,
                embedding,
                source_file,
                source_category,
                document_date,
                chunk_index,
                total_chunks,
                context_header,
                chunk_token_count,
                metadata,
                1 - (embedding {distance_op} %s) as similarity
            FROM knowledge_base
            WHERE embedding IS NOT NULL
            ORDER BY embedding {distance_op} %s
            LIMIT %s
        """

        results: list[SearchResult] = []

        try:
            with conn.cursor() as cur:
                # Execute query with parameters
                cur.execute(sql, (embedding_str, embedding_str, top_k))

                rows = cur.fetchall()

                # Convert database rows to SearchResult objects
                for row in rows:
                    # Extract data from row
                    chunk_text = row[1]
                    chunk_hash = row[2]
                    source_file = row[4]
                    source_category = row[5]
                    document_date = row[6]
                    chunk_index = row[7]
                    total_chunks = row[8]
                    context_header = row[9]
                    chunk_token_count = row[10]
                    metadata = row[11]
                    similarity = row[12]

                    # Reconstruct ProcessedChunk from database row
                    chunk = ProcessedChunk(
                        chunk_text=chunk_text,
                        chunk_hash=chunk_hash,
                        context_header=context_header,
                        source_file=source_file,
                        source_category=source_category,
                        document_date=document_date,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        chunk_token_count=chunk_token_count,
                        metadata=metadata if metadata else {},
                        embedding=None,  # Don't return embedding to save memory
                    )

                    # Ensure similarity is in [0, 1] range
                    similarity_score = max(0.0, min(1.0, float(similarity)))

                    result = SearchResult(similarity_score, chunk)
                    results.append(result)

        except Exception as e:
            logger.error(f"Search query execution failed: {e}", exc_info=True)
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Get total vector count for stats
        total_vectors = self._get_total_vectors(conn)

        stats = SearchStats(
            query_latency_ms=latency_ms,
            results_returned=len(results),
            total_vectors_searched=total_vectors,
        )

        return results, stats

    def _execute_filtered_search(
        self,
        conn: Connection,
        params: list[Any],
        where_sql: str,
        top_k: int,
        similarity_metric: str,
    ) -> tuple[list[SearchResult], SearchStats]:
        """Execute filtered similarity search query.

        Args:
            conn: Database connection.
            params: Query parameters (first is embedding).
            where_sql: WHERE clause conditions.
            top_k: Number of results.
            similarity_metric: Distance metric.

        Returns:
            Tuple of (results, stats).
        """
        start_time = time.time()

        distance_op = SIMILARITY_METRICS[similarity_metric]

        # Build SQL with WHERE clause
        sql = f"""
            SELECT
                id,
                chunk_text,
                chunk_hash,
                embedding,
                source_file,
                source_category,
                document_date,
                chunk_index,
                total_chunks,
                context_header,
                chunk_token_count,
                metadata,
                1 - (embedding {distance_op} %s) as similarity
            FROM knowledge_base
            WHERE {where_sql}
            ORDER BY embedding {distance_op} %s
            LIMIT %s
        """

        results: list[SearchResult] = []

        try:
            with conn.cursor() as cur:
                # Add distance operator and embedding to params
                query_embedding = params[0]
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

                execute_params = [embedding_str] + params[1:] + [embedding_str, top_k]

                # Execute query
                cur.execute(sql, execute_params)

                rows = cur.fetchall()

                # Convert rows to SearchResult objects
                for row in rows:
                    chunk_text = row[1]
                    chunk_hash = row[2]
                    source_file = row[4]
                    source_category = row[5]
                    document_date = row[6]
                    chunk_index = row[7]
                    total_chunks = row[8]
                    context_header = row[9]
                    chunk_token_count = row[10]
                    metadata = row[11]
                    similarity = row[12]

                    chunk = ProcessedChunk(
                        chunk_text=chunk_text,
                        chunk_hash=chunk_hash,
                        context_header=context_header,
                        source_file=source_file,
                        source_category=source_category,
                        document_date=document_date,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        chunk_token_count=chunk_token_count,
                        metadata=metadata if metadata else {},
                        embedding=None,
                    )

                    similarity_score = max(0.0, min(1.0, float(similarity)))
                    result = SearchResult(similarity_score, chunk)
                    results.append(result)

        except Exception as e:
            logger.error(f"Filtered search query failed: {e}", exc_info=True)
            raise

        latency_ms = (time.time() - start_time) * 1000
        total_vectors = self._get_total_vectors(conn)

        stats = SearchStats(
            query_latency_ms=latency_ms,
            results_returned=len(results),
            total_vectors_searched=total_vectors,
        )

        return results, stats

    def _get_total_vectors(self, conn: Connection) -> int:
        """Get count of vectors in knowledge_base.

        Args:
            conn: Database connection.

        Returns:
            Count of non-NULL embeddings.
        """
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL")
                result = cur.fetchone()
                if result is None:
                    return 0
                return int(result[0])
        except Exception as e:
            logger.warning(f"Failed to get vector count: {e}")
            return 0

    def _fetch_statistics(self, conn: Connection) -> dict[str, int | float]:
        """Fetch statistics from database.

        Args:
            conn: Database connection.

        Returns:
            Dictionary with statistics.
        """
        try:
            with conn.cursor() as cur:
                # Get vector count
                cur.execute(
                    "SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL"
                )
                result = cur.fetchone()
                total_vectors = int(result[0]) if result else 0

                # Check if index exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE tablename = 'knowledge_base'
                        AND indexname = 'idx_knowledge_embedding'
                    )
                """)
                result = cur.fetchone()
                index_exists = bool(result[0]) if result else False

                return {
                    "total_vectors": total_vectors,
                    "index_exists": 1.0 if index_exists else 0.0,
                }

        except Exception as e:
            logger.warning(f"Failed to fetch statistics: {e}")
            return {"total_vectors": 0, "index_exists": 0.0}
