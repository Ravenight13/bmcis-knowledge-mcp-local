"""Reranker protocol interface for pluggable reranking implementations.

Defines the Reranker protocol that all reranker implementations must follow,
enabling alternative implementations (LLM-based, ensemble, etc.) to be
composable with HybridSearch and other search systems.

This module provides:
- Reranker protocol: Abstract interface for reranking
- Type-safe contract for any reranking implementation
- Documentation of expected behavior and error handling

Example:
    Create a custom reranker implementing the Reranker protocol:

    >>> from src.search.reranker_protocol import Reranker
    >>> from src.search.results import SearchResult
    >>>
    >>> class SimpleReranker:
    ...     '''Simple reranker that reverses result order.'''
    ...     def rerank(
    ...         self,
    ...         query: str,
    ...         results: list[SearchResult],
    ...         top_k: int = 5,
    ...     ) -> list[SearchResult]:
    ...         # Simple example: reverse and take top-k
    ...         return list(reversed(results))[:top_k]
    >>>
    >>> reranker = SimpleReranker()
    >>> # Type checker knows reranker satisfies Reranker protocol
    >>> # Can be used with any system expecting: reranker: Reranker
"""

from __future__ import annotations

from typing import Protocol

from src.search.results import SearchResult


class Reranker(Protocol):
    """Protocol for reranking implementations.

    Any reranker (cross-encoder, LLM-based, ensemble, etc.) should implement
    this interface to be composable with HybridSearch and other search systems.

    The protocol defines a single method: rerank() which takes a query and
    results, and returns a reranked list sorted by relevance with confidence
    scores.

    **Thread Safety**: Implementations should document thread safety guarantees.
    The protocol does not enforce any specific thread safety model.

    **Performance Characteristics**:
    Implementations should document their performance characteristics:
    - Time complexity for typical query/result sizes
    - Memory requirements
    - Batch processing support (if available)

    **Error Handling**: Implementations must raise:
    - ValueError: For invalid inputs (empty query, empty results, invalid top_k)
    - RuntimeError: For computation failures

    Example:
        >>> class CrossEncoderReranker:
        ...     '''Cross-encoder reranker using transformers.'''
        ...
        ...     def rerank(
        ...         self,
        ...         query: str,
        ...         results: list[SearchResult],
        ...         top_k: int = 5,
        ...     ) -> list[SearchResult]:
        ...         # Cross-encoder reranking logic
        ...         return reranked_results
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results for a query, returning top-k.

        Implementations should:
        1. Validate inputs (query not empty, results not empty, top_k >= 1)
        2. Score/rank all input results according to query
        3. Select top-k by relevance score
        4. Update result metadata (rank, confidence, score_type)
        5. Return reranked results sorted by relevance DESC

        Args:
            query: User search query string. Should not be empty or None.
            results: List of search results to rerank. Should not be empty.
            top_k: Number of top results to return (default: 5).
                Must be >= 1. Results returned may be < top_k if fewer inputs.

        Returns:
            Reranked results, sorted by relevance (highest first), with:
            - rank: Updated with new 1-based rank
            - confidence: Confidence score (0-1) if available
            - score_type: Identifier of reranking method used
            Length <= top_k (may be less if fewer results available or
            confidence filtering applied).

        Raises:
            ValueError: If query is empty/None, results is empty/None,
                or top_k < 1.
            RuntimeError: If reranking computation fails (e.g., model error,
                OOM, network error for remote reranker).

        Example:
            >>> reranker: Reranker = ...
            >>> query = "authentication best practices"
            >>> results = hybrid_search(query, top_k=50)
            >>> reranked = reranker.rerank(query, results, top_k=5)
            >>> assert len(reranked) <= 5
            >>> assert all(r1.confidence >= r2.confidence
            ...            for r1, r2 in zip(reranked[:-1], reranked[1:]))
        """
        ...
