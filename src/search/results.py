"""Search result formatting, ranking, and validation.

Provides standardized result format for vector, BM25, and hybrid search results
with score normalization, deduplication, ranking validation, and formatting
options (JSON, dict, plain text).

The module implements:
- SearchResult dataclass with type-safe fields
- Score normalization to 0-1 scale
- Ranking system for single and hybrid searches
- Deduplication of duplicate chunks
- Threshold filtering based on minimum scores
- Multiple formatting options (JSON, dict, plain text)
- Metadata filtering in results
- Ranking validation against test queries

Performance targets:
- Vector search: <100ms per query
- BM25 search: <50ms per query
- Hybrid search: <150ms per query
- Result formatting: <10ms overhead
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Literal

import psycopg2
from psycopg2.extensions import connection as Connection

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Type aliases
ScoreType = Literal["vector", "bm25", "hybrid", "cross_encoder"]
FormatType = Literal["json", "dict", "text"]
RankingMode = Literal["vector_only", "bm25_only", "hybrid"]


@dataclass
class SearchResult:
    """Standardized search result with formatting and ranking support.

    Represents a single search result from vector, BM25, or hybrid search
    with unified formatting, score normalization, and ranking metadata.

    Attributes:
        chunk_id: Unique database identifier for the chunk.
        chunk_text: The actual text content of the result.
        similarity_score: Similarity score (vector search), normalized to 0-1.
        bm25_score: BM25 relevance score (full-text search), normalized to 0-1.
        hybrid_score: Combined score from hybrid search, normalized to 0-1.
        rank: Position in results (1-indexed).
        score_type: Type of scoring used (vector, bm25, hybrid, cross_encoder).
        source_file: Original document file path.
        source_category: Document category for filtering.
        document_date: Publication/update date of source document.
        context_header: Hierarchical context string.
        chunk_index: Position in source document.
        total_chunks: Total chunks in source document.
        chunk_token_count: Number of tokens in chunk.
        metadata: Additional JSONB metadata from database.
        highlighted_context: Optional matching terms with surrounding context.
        confidence: Confidence score for cross-encoder results (0-1).

    Example:
        >>> result = SearchResult(
        ...     chunk_id=1,
        ...     chunk_text="Important information about X",
        ...     similarity_score=0.85,
        ...     bm25_score=0.72,
        ...     rank=1,
        ...     score_type="hybrid",
        ...     source_file="docs/guide.md",
        ...     source_category="guide",
        ...     document_date=None,
        ...     context_header="guide.md > Section",
        ...     chunk_index=0,
        ...     total_chunks=10,
        ...     chunk_token_count=512,
        ...     metadata={"tags": ["important"]},
        ... )
        >>> print(result.to_dict())
        >>> print(result.to_json())
    """

    chunk_id: int
    chunk_text: str
    similarity_score: float
    bm25_score: float
    hybrid_score: float
    rank: int
    score_type: ScoreType
    source_file: str
    source_category: str | None
    document_date: datetime | None
    context_header: str
    chunk_index: int
    total_chunks: int
    chunk_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    highlighted_context: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate score ranges and rank position after initialization."""
        # Validate score ranges (0-1 after normalization)
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score must be 0-1, got {self.similarity_score}"
            )
        if not (0.0 <= self.bm25_score <= 1.0):
            raise ValueError(f"bm25_score must be 0-1, got {self.bm25_score}")
        if not (0.0 <= self.hybrid_score <= 1.0):
            raise ValueError(f"hybrid_score must be 0-1, got {self.hybrid_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")

        # Validate rank is positive
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")

        # Validate text is not empty
        if not self.chunk_text or not self.chunk_text.strip():
            raise ValueError("chunk_text cannot be empty")

        # Validate chunk index
        if self.chunk_index < 0 or self.chunk_index >= self.total_chunks:
            raise ValueError(
                f"chunk_index must be 0 <= {self.chunk_index} < {self.total_chunks}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format.

        Returns:
            Dictionary representation with all fields, suitable for JSON serialization.
        """
        result_dict = asdict(self)
        # Convert datetime to ISO format string for JSON serialization
        if isinstance(result_dict["document_date"], datetime):
            result_dict["document_date"] = result_dict["document_date"].isoformat()
        return result_dict

    def to_json(self) -> str:
        """Convert result to JSON string format.

        Returns:
            JSON string representation of the result.

        Raises:
            TypeError: If any field contains non-serializable types.
        """
        try:
            return json.dumps(
                self.to_dict(),
                indent=2,
                default=str,  # Fallback for non-serializable objects
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize result to JSON: {e}")
            raise

    def to_text(self, include_metadata: bool = True) -> str:
        """Convert result to human-readable text format.

        Args:
            include_metadata: Whether to include metadata fields (default: True).

        Returns:
            Formatted text representation of the result.
        """
        text_parts: list[str] = [
            f"Rank: {self.rank}",
            f"Score: {self.hybrid_score:.3f}" if self.score_type == "hybrid" else f"Score: {self.similarity_score:.3f}",
            f"Source: {self.source_file}",
            f"Category: {self.source_category}",
            f"Context: {self.context_header}",
            f"Chunk: {self.chunk_index + 1}/{self.total_chunks}",
            "",
            f"Text: {self.chunk_text[:200]}..." if len(self.chunk_text) > 200 else f"Text: {self.chunk_text}",
        ]

        if include_metadata and self.metadata:
            text_parts.append(f"Metadata: {json.dumps(self.metadata, default=str)}")

        return "\n".join(text_parts)

    def matches_filters(self, filters: dict[str, Any]) -> bool:
        """Check if result matches metadata filters.

        Supports filtering by:
        - category: Exact match on source_category
        - min_date: Document date >= this date
        - max_date: Document date <= this date
        - tags: Any tag from metadata['tags'] matches
        - source_file: Partial match on source_file

        Args:
            filters: Dictionary of filter conditions.

        Returns:
            True if result matches all filters, False otherwise.
        """
        # Category filter (exact match)
        if "category" in filters:
            if self.source_category != filters["category"]:
                return False

        # Date range filters
        if "min_date" in filters and self.document_date:
            if self.document_date < filters["min_date"]:
                return False

        if "max_date" in filters and self.document_date:
            if self.document_date > filters["max_date"]:
                return False

        # Tags filter (any tag match)
        if "tags" in filters:
            result_tags = self.metadata.get("tags", [])
            filter_tags = filters["tags"]
            if not any(tag in result_tags for tag in filter_tags):
                return False

        # Source file filter (partial match)
        if "source_file" in filters:
            if filters["source_file"] not in self.source_file:
                return False

        return True


class SearchResultFormatter:
    """Format and deduplicate search results with ranking validation.

    Handles result deduplication, score normalization, ranking validation,
    threshold filtering, and result formatting for all search types.

    Attributes:
        deduplication_enabled: Whether to deduplicate identical chunks.
        min_score_threshold: Minimum score for results to be included (0-1).
        max_results: Maximum number of results to return.
        normalization_mode: How to normalize scores.
    """

    def __init__(
        self,
        deduplication_enabled: bool = True,
        min_score_threshold: float = 0.0,
        max_results: int = 100,
    ) -> None:
        """Initialize result formatter.

        Args:
            deduplication_enabled: Whether to remove duplicate chunks (default: True).
            min_score_threshold: Minimum score for inclusion (default: 0.0).
            max_results: Maximum results to return (default: 100).

        Raises:
            ValueError: If thresholds are out of range.
        """
        if not (0.0 <= min_score_threshold <= 1.0):
            raise ValueError(f"min_score_threshold must be 0-1, got {min_score_threshold}")
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}")

        self.deduplication_enabled = deduplication_enabled
        self.min_score_threshold = min_score_threshold
        self.max_results = max_results
        logger.info(
            f"SearchResultFormatter initialized: "
            f"dedup={deduplication_enabled}, "
            f"min_score={min_score_threshold}, "
            f"max_results={max_results}"
        )

    def format_results(
        self,
        results: list[SearchResult],
        format_type: FormatType = "dict",
        apply_deduplication: bool | None = None,
        apply_threshold: bool = True,
    ) -> list[dict[str, Any]] | list[str]:
        """Format search results for output.

        Args:
            results: List of SearchResult objects.
            format_type: Output format (json, dict, text).
            apply_deduplication: Whether to deduplicate (default: self setting).
            apply_threshold: Whether to apply min_score_threshold (default: True).

        Returns:
            List of formatted results (dicts or strings depending on format_type).

        Example:
            >>> results = [SearchResult(...), SearchResult(...)]
            >>> formatted = formatter.format_results(results, format_type="dict")
            >>> print(formatted)
        """
        # Apply deduplication if enabled
        if apply_deduplication is None:
            apply_deduplication = self.deduplication_enabled

        if apply_deduplication:
            results = self._deduplicate(results)

        # Apply threshold filtering
        if apply_threshold:
            results = [
                r for r in results
                if (r.hybrid_score if r.score_type == "hybrid" else r.similarity_score)
                >= self.min_score_threshold
            ]

        # Apply max results limit
        results = results[: self.max_results]

        # Format results
        if format_type == "json":
            return [json.dumps(r.to_dict(), default=str) for r in results]
        elif format_type == "text":
            return [r.to_text() for r in results]
        else:  # dict
            return [r.to_dict() for r in results]

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate chunks from results.

        Keeps the first occurrence of each chunk_id and removes later occurrences.

        Args:
            results: List of SearchResult objects.

        Returns:
            Deduplicated list of SearchResult objects.
        """
        seen_ids: set[int] = set()
        deduplicated: list[SearchResult] = []

        for result in results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                deduplicated.append(result)
            else:
                logger.debug(
                    f"Deduplicating chunk_id={result.chunk_id} at rank {result.rank}"
                )

        if len(deduplicated) < len(results):
            logger.info(
                f"Deduplication removed {len(results) - len(deduplicated)} duplicates"
            )

        return deduplicated

    @staticmethod
    def normalize_vector_scores(
        raw_scores: list[float], score_range: tuple[float, float] = (-1.0, 1.0)
    ) -> list[float]:
        """Normalize vector similarity scores to 0-1 range.

        Converts cosine similarity scores (typically -1 to 1) to 0-1 scale:
        normalized = (raw - min) / (max - min)

        Args:
            raw_scores: List of raw similarity scores.
            score_range: Tuple of (min, max) in original range (default: -1 to 1).

        Returns:
            List of normalized scores in 0-1 range.

        Example:
            >>> scores = [-0.5, 0.0, 0.5, 1.0]
            >>> normalized = SearchResultFormatter.normalize_vector_scores(scores)
            >>> print(normalized)  # [0.25, 0.5, 0.75, 1.0]
        """
        min_val, max_val = score_range
        if min_val == max_val:
            return [0.5] * len(raw_scores)

        normalized = [
            (score - min_val) / (max_val - min_val) for score in raw_scores
        ]
        return normalized

    @staticmethod
    def normalize_bm25_scores(
        raw_scores: list[float],
        percentile_99: float = 10.0,  # Typical 99th percentile
    ) -> list[float]:
        """Normalize BM25 scores to 0-1 range.

        Uses percentile-based normalization for BM25 scores which are
        theoretically unbounded. Assumes 99th percentile represents scores
        for 'excellent' matches.

        Args:
            raw_scores: List of raw BM25 scores.
            percentile_99: Score value at 99th percentile (default: 10.0).

        Returns:
            List of normalized scores in 0-1 range.
        """
        if percentile_99 <= 0:
            percentile_99 = 1.0

        normalized = [min(1.0, score / percentile_99) for score in raw_scores]
        return normalized

    @staticmethod
    def combine_hybrid_scores(
        vector_scores: list[float],
        bm25_scores: list[float],
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[float]:
        """Combine vector and BM25 scores using weighted average.

        Implements weighted combination: hybrid = vector * w1 + bm25 * w2
        Both input scores should be normalized to 0-1.

        Args:
            vector_scores: List of normalized vector similarity scores (0-1).
            bm25_scores: List of normalized BM25 scores (0-1).
            weights: Tuple of (vector_weight, bm25_weight) (default: 0.6, 0.4).

        Returns:
            List of hybrid scores combining both ranking signals.

        Raises:
            ValueError: If score lists have different lengths or weights don't sum to 1.

        Example:
            >>> vector = [0.9, 0.8, 0.7]
            >>> bm25 = [0.7, 0.8, 0.9]
            >>> hybrid = SearchResultFormatter.combine_hybrid_scores(vector, bm25)
            >>> print(hybrid)  # [0.82, 0.8, 0.78]
        """
        if len(vector_scores) != len(bm25_scores):
            raise ValueError(
                f"Score lists must have same length: {len(vector_scores)} != {len(bm25_scores)}"
            )

        w_vector, w_bm25 = weights
        if abs((w_vector + w_bm25) - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {w_vector} + {w_bm25}")

        hybrid = [v * w_vector + b * w_bm25 for v, b in zip(vector_scores, bm25_scores)]
        return hybrid


def apply_confidence_filtering(
    results: list[SearchResult],
) -> list[SearchResult]:
    """Apply adaptive result limiting based on average confidence score.

    Implements confidence-based result limiting that returns fewer low-confidence
    results to users, improving relevance by filtering out weak matches.

    Logic:
    - If avg_score >= 0.7: returns all results (high confidence)
    - If avg_score >= 0.5: returns top 5 (medium confidence)
    - If avg_score < 0.5: returns top 3 (low confidence)

    Handles edge cases gracefully:
    - Empty results: returns empty list
    - Missing/invalid scores: uses 0.0 as default
    - String scores: attempts conversion, falls back to 0.0

    Args:
        results: List of SearchResult objects to filter.

    Returns:
        Filtered list of SearchResult objects with adaptive limiting applied.

    Example:
        >>> results = [
        ...     SearchResult(..., hybrid_score=0.95),
        ...     SearchResult(..., hybrid_score=0.88),
        ...     SearchResult(..., hybrid_score=0.72),
        ... ]
        >>> filtered = apply_confidence_filtering(results)
        >>> len(filtered)  # Returns all 3 (avg=0.85 >= 0.7)
        3
    """
    # Handle empty results
    if not results:
        return []

    # Calculate average score across all results
    scores: list[float] = []
    for result in results:
        # Use hybrid_score if available, otherwise use similarity_score
        score = result.hybrid_score if result.score_type == "hybrid" else result.similarity_score

        # Handle edge cases: missing/invalid scores
        if isinstance(score, str):
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
        elif score is None:
            score = 0.0

        # Ensure score is in valid range
        if not isinstance(score, (int, float)) or score < 0.0:
            score = 0.0

        scores.append(min(1.0, max(0.0, score)))

    if not scores:
        return []

    avg_score = sum(scores) / len(scores)
    logger.debug(
        f"Confidence filtering: avg_score={avg_score:.3f}, total_results={len(results)}"
    )

    # Apply adaptive limiting based on confidence level
    if avg_score >= 0.7:
        # High confidence: return all results
        logger.debug("High confidence (>=0.7): returning all results")
        return results
    elif avg_score >= 0.5:
        # Medium confidence: return top 5
        logger.debug("Medium confidence (>=0.5): limiting to top 5 results")
        return results[:5]
    else:
        # Low confidence: return top 3
        logger.debug("Low confidence (<0.5): limiting to top 3 results")
        return results[:3]


class RankingValidator:
    """Validate search result ranking quality.

    Implements ranking validation using multiple metrics:
    - Position correlation with expected relevance
    - Score distribution analysis
    - Ranking consistency tests
    - Result deduplication validation
    """

    @staticmethod
    def validate_ranking(
        results: list[SearchResult],
        expected_order: list[int] | None = None,
    ) -> dict[str, Any]:
        """Validate ranking quality and consistency.

        Args:
            results: List of SearchResult objects to validate.
            expected_order: Optional list of chunk_ids in expected order.

        Returns:
            Dictionary with validation metrics including:
            - is_sorted: Whether results are properly sorted by score
            - has_duplicates: Whether duplicate chunk_ids exist
            - score_monotonicity: Whether scores decrease monotonically
            - rank_correctness: Whether ranks match positions (1-indexed)
            - rank_correlation: Spearman correlation with expected order
        """
        validation: dict[str, Any] = {
            "is_sorted": True,
            "has_duplicates": False,
            "score_monotonicity": True,
            "rank_correctness": True,
            "rank_correlation": 1.0,
        }

        if not results:
            return validation

        # Check sorting by score
        scores = [r.hybrid_score if r.score_type == "hybrid" else r.similarity_score for r in results]
        if scores != sorted(scores, reverse=True):
            validation["is_sorted"] = False
            logger.warning("Results not sorted by score in descending order")

        # Check for duplicates
        chunk_ids = [r.chunk_id for r in results]
        if len(chunk_ids) != len(set(chunk_ids)):
            validation["has_duplicates"] = True
            logger.warning(f"Found duplicate chunks in results")

        # Check score monotonicity
        for i in range(1, len(scores)):
            if scores[i] > scores[i - 1]:
                validation["score_monotonicity"] = False
                logger.warning(f"Score increased from position {i-1} to {i}")
                break

        # Check rank correctness (ranks should be 1-indexed position)
        for i, result in enumerate(results):
            if result.rank != i + 1:
                validation["rank_correctness"] = False
                logger.warning(f"Rank mismatch at position {i}: expected {i+1}, got {result.rank}")
                break

        # Calculate rank correlation with expected order if provided
        if expected_order:
            validation["rank_correlation"] = RankingValidator._calculate_rank_correlation(
                chunk_ids, expected_order
            )

        return validation

    @staticmethod
    def _calculate_rank_correlation(
        actual_order: list[int], expected_order: list[int]
    ) -> float:
        """Calculate Spearman rank correlation between actual and expected order.

        Args:
            actual_order: Actual order of chunk_ids from search results.
            expected_order: Expected order of chunk_ids (ground truth).

        Returns:
            Correlation coefficient (-1.0 to 1.0), where 1.0 is perfect correlation.
        """
        # Create position mappings
        expected_positions = {chunk_id: i for i, chunk_id in enumerate(expected_order)}

        # Calculate rank differences
        d_squared_sum = 0
        n = 0

        for actual_pos, chunk_id in enumerate(actual_order):
            if chunk_id in expected_positions:
                expected_pos = expected_positions[chunk_id]
                d = actual_pos - expected_pos
                d_squared_sum += d * d
                n += 1

        if n < 2:
            return 1.0 if n == 1 else 0.0

        # Spearman correlation: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        correlation = 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1))
        return max(-1.0, min(1.0, correlation))  # Clamp to -1 to 1
