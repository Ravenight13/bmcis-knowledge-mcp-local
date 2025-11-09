"""Type stubs for reranker protocol interface.

Defines the Reranker protocol that all reranker implementations must follow,
enabling alternative implementations (LLM-based, ensemble, etc.) to be
composable with HybridSearch.
"""

from __future__ import annotations

from typing import Protocol

from src.search.results import SearchResult


class Reranker(Protocol):
    """Protocol for reranking implementations.

    Any reranker (cross-encoder, LLM-based, ensemble, etc.) should implement
    this interface to be composable with HybridSearch and other search systems.

    Example:
        >>> class MyCustomReranker:
        ...     def rerank(self, query: str, results: list[SearchResult],
        ...                top_k: int = 5) -> list[SearchResult]:
        ...         # Custom reranking logic
        ...         return reranked_results
        >>> reranker = MyCustomReranker()
        >>> # Can now be used with any system expecting a Reranker
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results for a query, returning top-k.

        Args:
            query: User search query string.
            results: List of search results to rerank.
            top_k: Number of top results to return (default: 5).

        Returns:
            Reranked results, sorted by relevance, with confidence scores.
            Length <= top_k (may be less if fewer results available).

        Raises:
            ValueError: If query empty, results empty, or invalid parameters.
            RuntimeError: If reranking computation fails.
        """
        ...
