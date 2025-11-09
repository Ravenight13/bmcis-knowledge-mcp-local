"""Type stubs for RRF (Reciprocal Rank Fusion) algorithm.

Provides complete type definitions for merging vector and BM25 search results
using the RRF algorithm.
"""

from typing import override

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.results import SearchResult

class RRFScorer:
    """Reciprocal Rank Fusion algorithm for combining search results.

    Merges results from multiple search sources (vector, BM25) using the
    RRF formula: score = 1 / (k + rank), where rank is 1-indexed position.
    """

    k: int

    def __init__(
        self,
        k: int = 60,
        db_pool: DatabasePool | None = None,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize RRF scorer with k parameter.

        Args:
            k: Constant for RRF formula (default 60). Higher k reduces
               the impact of ranking position differences.
            db_pool: Optional database pool for caching results.
            settings: Optional settings for configuration.
            logger: Optional logger for performance metrics.
        """
        ...

    def merge_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[SearchResult]:
        """Merge results from two sources using RRF formula.

        Combines vector search and BM25 results by calculating RRF scores
        for each result in both lists, deduplicating by chunk_id, and
        reranking by combined weighted score.

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 full-text search.
            weights: Tuple of (vector_weight, bm25_weight) normalized to sum
                    to 1.0. Default (0.6, 0.4) gives 60% to vector, 40% to BM25.

        Returns:
            Merged and reranked results sorted by combined RRF score (descending).
            Each result's hybrid_score field contains the merged RRF score.

        Raises:
            ValueError: If weights don't sum to approximately 1.0.
            ValueError: If either input list contains invalid SearchResult objects.
        """
        ...

    def fuse_multiple(
        self,
        results_by_source: dict[str, list[SearchResult]],
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Fuse results from 3+ sources with configurable weights.

        Extends RRF to handle multiple search sources (vector, BM25,
        potentially cross-encoder). Applies RRF scoring to each source
        separately, then combines with provided weights.

        Args:
            results_by_source: Dictionary mapping source names to result lists.
                              Expected keys: "vector", "bm25", optionally others.
            weights: Optional dictionary of source -> weight mappings.
                    If not provided, uses equal weights for all sources.

        Returns:
            Fused and reranked results sorted by combined score (descending).

        Raises:
            ValueError: If weights don't sum to approximately 1.0.
            ValueError: If no results provided.
        """
        ...

    def _calculate_rrf_score(self, rank: int) -> float:
        """Calculate RRF score for a given rank position.

        Formula: score = 1 / (k + rank)

        Args:
            rank: 1-indexed position in result list (1 = first, 2 = second, etc.).

        Returns:
            RRF score as float in range (0, 1).

        Raises:
            ValueError: If rank < 1.
        """
        ...

    def _deduplicate_results(
        self,
        vector_map: dict[int, tuple[SearchResult, float]],
        bm25_map: dict[int, tuple[SearchResult, float]],
        v_weight: float,
        b_weight: float,
    ) -> dict[int, tuple[SearchResult, float]]:
        """Deduplicate and combine scores for results appearing in both sources.

        For each chunk_id appearing in both sources, calculates weighted
        combined score and selects the result with richer metadata.

        Args:
            vector_map: Dict of chunk_id -> (SearchResult, rrf_score).
            bm25_map: Dict of chunk_id -> (SearchResult, rrf_score).
            v_weight: Weight for vector scores.
            b_weight: Weight for BM25 scores.

        Returns:
            Dict of chunk_id -> (merged_SearchResult, combined_score).
        """
        ...

    def _normalize_weights(self, weights: tuple[float, float]) -> tuple[float, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Raw weight tuple.

        Returns:
            Normalized weights that sum to 1.0.

        Raises:
            ValueError: If either weight is negative or both are zero.
        """
        ...
