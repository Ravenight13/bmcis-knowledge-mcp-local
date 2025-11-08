"""BM25 full-text search using PostgreSQL ts_vector with GIN indexing.

Provides keyword-based search with BM25-like relevance ranking using
PostgreSQL's native full-text search capabilities. Uses ts_rank_cd for
sophisticated cover density ranking with document length normalization.

The search leverages the ts_vector column with GIN index for fast lookups
(<50ms for 2,600 chunks) and returns results ranked by relevance score
normalized to 0-1 range.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from psycopg2.extensions import connection as Connection

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger

logger: logging.Logger = StructuredLogger.get_logger(__name__)


@dataclass
class SearchResult:
    """Result from BM25 full-text search.

    Attributes:
        id: Database record ID.
        chunk_text: Full text content of the chunk.
        context_header: Hierarchical context path.
        source_file: Original markdown file path.
        source_category: Document category for filtering.
        document_date: Document publication/update date.
        chunk_index: Position in source document (0-indexed).
        total_chunks: Total chunks in source document.
        chunk_token_count: Number of tokens in chunk.
        metadata: Additional JSONB metadata.
        similarity: BM25 relevance score (0-1 range, higher is better).
    """

    id: int
    chunk_text: str
    context_header: str
    source_file: str
    source_category: str | None
    document_date: date | None
    chunk_index: int
    total_chunks: int
    chunk_token_count: int | None
    metadata: dict[str, Any]
    similarity: float

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"SearchResult(id={self.id}, "
            f"source_file='{self.source_file}', "
            f"chunk_index={self.chunk_index}, "
            f"similarity={self.similarity:.4f})"
        )


class BM25Search:
    """BM25 full-text search using PostgreSQL ts_vector.

    Provides keyword-based search with relevance ranking using PostgreSQL's
    native full-text search with GIN indexing. Uses ts_rank_cd for cover
    density ranking with normalization.

    The search uses the following PostgreSQL features:
    - to_tsvector('english', text): Tokenization with stemming and stop words
    - plainto_tsquery('english', query): Query parsing
    - ts_rank_cd(tsvector, query, normalization): BM25-like ranking
    - GIN index on ts_vector: Fast keyword lookup

    Example:
        >>> search = BM25Search()
        >>> results = search.search("authentication jwt tokens", top_k=10)
        >>> for result in results:
        ...     print(f"{result.source_file}: {result.similarity:.4f}")
    """

    def __init__(self) -> None:
        """Initialize BM25Search with database pool."""
        self._db_pool = DatabasePool

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        category_filter: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform BM25 full-text search.

        Uses PostgreSQL's search_bm25 function with ts_rank_cd for relevance
        ranking. Results are normalized to 0-1 range with higher scores
        indicating better matches.

        Args:
            query_text: Search query (plain text, automatically parsed).
            top_k: Maximum number of results (default: 10).
            category_filter: Optional category filter (e.g., "product_docs").
            min_score: Minimum similarity threshold (default: 0.0).

        Returns:
            List of SearchResult ordered by similarity (descending).

        Raises:
            ValueError: If query_text is empty or top_k < 1.
            DatabaseError: If database connection fails.

        Example:
            >>> search = BM25Search()
            >>> results = search.search(
            ...     "user authentication",
            ...     top_k=5,
            ...     category_filter="kb_article"
            ... )
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        logger.info(
            "BM25 search: query='%s', top_k=%d, category=%s, min_score=%.4f",
            query_text,
            top_k,
            category_filter,
            min_score,
        )

        with self._db_pool.get_connection() as conn:
            results = self._execute_search(
                conn,
                query_text,
                top_k,
                category_filter,
                min_score,
            )

        logger.info("BM25 search returned %d results", len(results))
        return results

    def search_phrase(
        self,
        phrase: str,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[SearchResult]:
        """Perform phrase search for exact matches.

        Uses phraseto_tsquery for exact phrase matching with word order
        preserved. More restrictive than standard search.

        Args:
            phrase: Search phrase (e.g., "user authentication token").
            top_k: Maximum number of results (default: 10).
            category_filter: Optional category filter.

        Returns:
            List of SearchResult ordered by similarity (descending).

        Raises:
            ValueError: If phrase is empty or top_k < 1.
            DatabaseError: If database connection fails.

        Example:
            >>> search = BM25Search()
            >>> results = search.search_phrase("JWT authentication")
        """
        if not phrase or not phrase.strip():
            raise ValueError("phrase cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        logger.info(
            "BM25 phrase search: phrase='%s', top_k=%d, category=%s",
            phrase,
            top_k,
            category_filter,
        )

        with self._db_pool.get_connection() as conn:
            results = self._execute_phrase_search(
                conn,
                phrase,
                top_k,
                category_filter,
            )

        logger.info("BM25 phrase search returned %d results", len(results))
        return results

    def _execute_search(
        self,
        conn: Connection,
        query_text: str,
        top_k: int,
        category_filter: str | None,
        min_score: float,
    ) -> list[SearchResult]:
        """Execute BM25 search query using database function.

        Args:
            conn: Database connection.
            query_text: Search query text.
            top_k: Maximum results.
            category_filter: Optional category filter.
            min_score: Minimum score threshold.

        Returns:
            List of SearchResult objects.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, chunk_text, context_header, source_file,
                    source_category, document_date, chunk_index,
                    total_chunks, chunk_token_count, metadata, similarity
                FROM search_bm25(%s, %s, %s, %s)
                """,
                (query_text, top_k, category_filter, min_score),
            )

            rows = cur.fetchall()

        return [self._row_to_result(row) for row in rows]

    def _execute_phrase_search(
        self,
        conn: Connection,
        phrase: str,
        top_k: int,
        category_filter: str | None,
    ) -> list[SearchResult]:
        """Execute phrase search query using database function.

        Args:
            conn: Database connection.
            phrase: Search phrase.
            top_k: Maximum results.
            category_filter: Optional category filter.

        Returns:
            List of SearchResult objects.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, chunk_text, context_header, source_file,
                    source_category, document_date, chunk_index,
                    total_chunks, chunk_token_count, metadata, similarity
                FROM search_bm25_phrase(%s, %s, %s)
                """,
                (phrase, top_k, category_filter),
            )

            rows = cur.fetchall()

        return [self._row_to_result(row) for row in rows]

    def _row_to_result(self, row: tuple[Any, ...]) -> SearchResult:
        """Convert database row to SearchResult.

        Args:
            row: Database query result row.

        Returns:
            SearchResult object.
        """
        return SearchResult(
            id=row[0],
            chunk_text=row[1],
            context_header=row[2],
            source_file=row[3],
            source_category=row[4],
            document_date=row[5],
            chunk_index=row[6],
            total_chunks=row[7],
            chunk_token_count=row[8],
            metadata=row[9] or {},
            similarity=float(row[10]),
        )
