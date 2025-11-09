"""Type stubs for BM25 full-text search module."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from psycopg2.extensions import connection as Connection

from src.core.database import DatabasePool

logger: logging.Logger

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

    def __repr__(self) -> str: ...

class BM25Search:
    """BM25 full-text search using PostgreSQL ts_vector.

    Provides keyword-based search with relevance ranking using PostgreSQL's
    native full-text search with GIN indexing. Uses ts_rank_cd for cover
    density ranking with normalization.
    """

    _db_pool: type[DatabasePool]

    def __init__(self) -> None:
        """Initialize BM25Search with database pool."""
        ...

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        category_filter: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform BM25 full-text search.

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
        """
        ...

    def search_phrase(
        self,
        phrase: str,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[SearchResult]:
        """Perform phrase search for exact matches.

        Args:
            phrase: Search phrase (e.g., "user authentication token").
            top_k: Maximum number of results (default: 10).
            category_filter: Optional category filter.

        Returns:
            List of SearchResult ordered by similarity (descending).

        Raises:
            ValueError: If phrase is empty or top_k < 1.
            DatabaseError: If database connection fails.
        """
        ...

    def _execute_search(
        self,
        conn: Connection,
        query_text: str,
        top_k: int,
        category_filter: str | None,
        min_score: float,
    ) -> list[SearchResult]:
        """Execute BM25 search query using database function."""
        ...

    def _execute_phrase_search(
        self,
        conn: Connection,
        phrase: str,
        top_k: int,
        category_filter: str | None,
    ) -> list[SearchResult]:
        """Execute phrase search query using database function."""
        ...

    def _row_to_result(self, row: tuple[Any, ...]) -> SearchResult:
        """Convert database row to SearchResult."""
        ...
