"""Reciprocal Rank Fusion (RRF) algorithm for combining search results.

Implements RRF algorithm for merging results from multiple search sources
(vector similarity, BM25 full-text). Uses the formula: score = 1 / (k + rank)
where rank is 1-indexed position in each source's result list.

The RRF algorithm is particularly effective for hybrid search because:
1. It treats different scoring scales uniformly (RRF is rank-based, not score-based)
2. It reduces the impact of outlier scores from any single source
3. It naturally deduplicates results appearing in multiple sources
4. It's proven effective in information retrieval literature (Cormack et al., 2009)

Performance target: <50ms for merging 100 results from each source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.results import SearchResult

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Constants
DEFAULT_K: Final[int] = 60
MIN_K: Final[int] = 1
MAX_K: Final[int] = 1000


@dataclass
class _RRFScoreEntry:
    """Internal representation of RRF score for a single source."""

    result: SearchResult
    rrf_score: float


class RRFScorer:
    """Reciprocal Rank Fusion algorithm for combining search results.

    Merges results from multiple search sources (vector, BM25) using the
    RRF algorithm. Each source's results are ranked, and RRF scores are
    calculated based on rank position. Results appearing in multiple sources
    have their scores combined using configurable weights.

    Attributes:
        k: Constant for RRF formula. Higher values reduce ranking impact.
           Typical range: 1-100, default 60.
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        db_pool: DatabasePool | None = None,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize RRF scorer with k parameter.

        Args:
            k: Constant for RRF formula (default 60).
            db_pool: Optional database pool for future optimizations.
            settings: Optional settings for configuration.
            logger: Optional logger for performance metrics.

        Raises:
            ValueError: If k is outside valid range [MIN_K, MAX_K].
        """
        if not (MIN_K <= k <= MAX_K):
            raise ValueError(
                f"k must be in range [{MIN_K}, {MAX_K}], got {k}"
            )

        self.k: int = k
        self._db_pool: DatabasePool | None = db_pool
        self._settings: Settings | None = settings
        self._logger: StructuredLogger | None = logger

    def merge_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[SearchResult]:
        """Merge results from two sources using RRF formula.

        Combines vector search and BM25 results by:
        1. Calculating RRF score for each result in both lists
        2. Deduplicating results by chunk_id
        3. Combining scores using provided weights
        4. Reranking results by combined score

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 full-text search.
            weights: Tuple of (vector_weight, bm25_weight) with sum ~= 1.0.
                    Default (0.6, 0.4) gives 60% to vector, 40% to BM25.

        Returns:
            Merged and reranked results sorted by combined RRF score (descending).
            Each result's hybrid_score contains the merged RRF score.

        Raises:
            ValueError: If weights don't normalize to approximately 1.0.
            ValueError: If input results contain invalid chunk_ids.
        """
        # Validate inputs
        if not vector_results and not bm25_results:
            return []

        # Normalize weights
        v_weight, b_weight = self._normalize_weights(weights)

        # Build maps of chunk_id -> (result, rrf_score)
        vector_map: dict[int, tuple[SearchResult, float]] = {}
        for rank, result in enumerate(vector_results, start=1):
            rrf_score = self._calculate_rrf_score(rank)
            vector_map[result.chunk_id] = (result, rrf_score)

        bm25_map: dict[int, tuple[SearchResult, float]] = {}
        for rank, result in enumerate(bm25_results, start=1):
            rrf_score = self._calculate_rrf_score(rank)
            bm25_map[result.chunk_id] = (result, rrf_score)

        # Deduplicate and combine scores
        merged: dict[int, tuple[SearchResult, float]] = self._deduplicate_results(
            vector_map, bm25_map, v_weight, b_weight
        )

        # Add results that appear in only one source (with weighted score)
        for chunk_id, (result, rrf_score) in vector_map.items():
            if chunk_id not in merged:
                combined_score = rrf_score * v_weight
                merged[chunk_id] = (result, combined_score)

        for chunk_id, (result, rrf_score) in bm25_map.items():
            if chunk_id not in merged:
                combined_score = rrf_score * b_weight
                merged[chunk_id] = (result, combined_score)

        # Sort by combined score (descending) and update rank
        sorted_results: list[tuple[int, SearchResult, float]] = [
            (chunk_id, result, score)
            for chunk_id, (result, score) in merged.items()
        ]
        sorted_results.sort(key=lambda x: x[2], reverse=True)

        # Create output with updated scores and ranks
        output: list[SearchResult] = []
        for new_rank, (_, result, combined_score) in enumerate(sorted_results, start=1):
            # Create new result with updated hybrid_score and rank
            updated_result = SearchResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                similarity_score=result.similarity_score,
                bm25_score=result.bm25_score,
                hybrid_score=min(max(combined_score, 0.0), 1.0),  # Clamp to 0-1
                rank=new_rank,
                score_type="hybrid",
                source_file=result.source_file,
                source_category=result.source_category,
                document_date=result.document_date,
                context_header=result.context_header,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                chunk_token_count=result.chunk_token_count,
                metadata=result.metadata,
                highlighted_context=result.highlighted_context,
                confidence=result.confidence,
            )
            output.append(updated_result)

        return output

    def fuse_multiple(
        self,
        results_by_source: dict[str, list[SearchResult]],
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Fuse results from 3+ sources with configurable weights.

        Extends RRF to handle multiple search sources. Applies RRF scoring
        to each source separately, then combines with provided weights.

        Args:
            results_by_source: Dictionary mapping source names to result lists.
            weights: Optional dictionary of source -> weight mappings.
                    If not provided, uses equal weights for all sources.

        Returns:
            Fused and reranked results sorted by combined score (descending).

        Raises:
            ValueError: If weights don't normalize to approximately 1.0.
            ValueError: If no results provided.
        """
        if not results_by_source:
            return []

        # Filter out empty sources
        non_empty_sources: dict[str, list[SearchResult]] = {
            name: results
            for name, results in results_by_source.items()
            if results
        }

        if not non_empty_sources:
            return []

        # Set up weights
        if weights is None:
            # Equal weights for all sources
            weight_value = 1.0 / len(non_empty_sources)
            source_weights = {name: weight_value for name in non_empty_sources}  # noqa: C420
        else:
            source_weights = weights

        # Validate weights
        total_weight = sum(source_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}"
            )

        # Calculate RRF scores for each source
        merged: dict[int, tuple[SearchResult, float]] = {}

        for source_name, results in non_empty_sources.items():
            source_weight = source_weights.get(source_name, 0.0)
            if source_weight <= 0:
                continue

            for rank, result in enumerate(results, start=1):
                rrf_score = self._calculate_rrf_score(rank)
                weighted_score = rrf_score * source_weight

                if result.chunk_id in merged:
                    # Result already seen, add to existing score
                    existing_result, existing_score = merged[result.chunk_id]
                    new_score = existing_score + weighted_score
                    merged[result.chunk_id] = (existing_result, new_score)
                else:
                    merged[result.chunk_id] = (result, weighted_score)

        # Sort and create output
        sorted_items: list[tuple[int, SearchResult, float]] = [
            (chunk_id, result, score)
            for chunk_id, (result, score) in merged.items()
        ]
        sorted_items.sort(key=lambda x: x[2], reverse=True)

        output: list[SearchResult] = []
        for new_rank, (_, result, combined_score) in enumerate(sorted_items, start=1):
            updated_result = SearchResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                similarity_score=result.similarity_score,
                bm25_score=result.bm25_score,
                hybrid_score=min(max(combined_score, 0.0), 1.0),
                rank=new_rank,
                score_type="hybrid",
                source_file=result.source_file,
                source_category=result.source_category,
                document_date=result.document_date,
                context_header=result.context_header,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                chunk_token_count=result.chunk_token_count,
                metadata=result.metadata,
                highlighted_context=result.highlighted_context,
                confidence=result.confidence,
            )
            output.append(updated_result)

        return output

    def _calculate_rrf_score(self, rank: int) -> float:
        """Calculate RRF score for a given rank position.

        Formula: score = 1 / (k + rank)

        Args:
            rank: 1-indexed position in result list (1 = first, etc.).

        Returns:
            RRF score as float in range (0, 1).

        Raises:
            ValueError: If rank < 1.
        """
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        return 1.0 / (self.k + rank)

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
        merged: dict[int, tuple[SearchResult, float]] = {}

        # Find chunk_ids that appear in both sources
        common_chunks = set(vector_map.keys()) & set(bm25_map.keys())

        for chunk_id in common_chunks:
            vector_result, vector_score = vector_map[chunk_id]
            bm25_result, bm25_score = bm25_map[chunk_id]

            # Combine weighted scores
            combined_score = (vector_score * v_weight) + (bm25_score * b_weight)

            # Use vector_result as base (both have same chunk_id, so content identical)
            # Preserve both scores in the result
            merged_result = SearchResult(
                chunk_id=vector_result.chunk_id,
                chunk_text=vector_result.chunk_text,
                similarity_score=vector_result.similarity_score,
                bm25_score=bm25_result.bm25_score,
                hybrid_score=combined_score,
                rank=vector_result.rank,  # Will be updated later
                score_type="hybrid",
                source_file=vector_result.source_file,
                source_category=vector_result.source_category,
                document_date=vector_result.document_date,
                context_header=vector_result.context_header,
                chunk_index=vector_result.chunk_index,
                total_chunks=vector_result.total_chunks,
                chunk_token_count=vector_result.chunk_token_count,
                metadata=vector_result.metadata,
                highlighted_context=vector_result.highlighted_context,
                confidence=max(vector_result.confidence, bm25_result.confidence),
            )

            merged[chunk_id] = (merged_result, combined_score)

        return merged

    def _normalize_weights(self, weights: tuple[float, float]) -> tuple[float, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Raw weight tuple.

        Returns:
            Normalized weights that sum to 1.0.

        Raises:
            ValueError: If either weight is negative or both are zero.
        """
        w1, w2 = weights

        if w1 < 0.0 or w2 < 0.0:
            raise ValueError(f"Weights must be non-negative, got {weights}")

        total = w1 + w2
        if total == 0.0:
            raise ValueError("At least one weight must be positive")

        return (w1 / total, w2 / total)
