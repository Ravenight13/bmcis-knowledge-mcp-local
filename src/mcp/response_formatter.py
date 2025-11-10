"""Response formatting helpers for MCP tools.

Provides helper functions for:
- Token estimation based on response mode
- Confidence score calculation from result distributions
- Ranking context generation with percentiles
- Response envelope wrapping with metadata
- Compression and warning generation

All functions are type-safe and mypy-strict compatible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, TypeVar

from src.mcp.models import (
    ConfidenceScore,
    DeduplicationInfo,
    ExecutionContext,
    MCPResponseEnvelope,
    PaginationMetadata,
    RankingContext,
    ResponseMetadata,
    ResponseWarning,
    SearchResultMetadata,
    VendorInfoMetadata,
)
from src.search.results import SearchResult

T = TypeVar("T")

# Token estimation constants (average tokens per result)
TOKEN_ESTIMATE_IDS_ONLY = 10
TOKEN_ESTIMATE_METADATA = 200
TOKEN_ESTIMATE_PREVIEW = 800
TOKEN_ESTIMATE_FULL = 1500

# Response size thresholds (bytes)
RESPONSE_SIZE_WARNING_THRESHOLD = 20_000  # 20KB
RESPONSE_SIZE_ERROR_THRESHOLD = 100_000  # 100KB

# Vendor info entity thresholds
VENDOR_ENTITY_WARNING_THRESHOLD = 50
VENDOR_ENTITY_ERROR_THRESHOLD = 100


def estimate_response_tokens(
    result_count: int,
    response_mode: str,
    include_metadata: bool = True,
) -> int:
    """Estimate token count for response based on mode and result count.

    Args:
        result_count: Number of results in response
        response_mode: Response detail level (ids_only/metadata/preview/full)
        include_metadata: Include envelope metadata overhead

    Returns:
        Estimated token count

    Example:
        >>> estimate_response_tokens(10, "metadata")
        2300  # 10 * 200 + 300 overhead
    """
    # Base token estimates per mode
    per_result_tokens = {
        "ids_only": TOKEN_ESTIMATE_IDS_ONLY,
        "metadata": TOKEN_ESTIMATE_METADATA,
        "preview": TOKEN_ESTIMATE_PREVIEW,
        "full": TOKEN_ESTIMATE_FULL,
    }

    base_tokens = result_count * per_result_tokens.get(response_mode, TOKEN_ESTIMATE_METADATA)

    # Add envelope metadata overhead (~300 tokens)
    if include_metadata:
        base_tokens += 300

    return base_tokens


def calculate_confidence_scores(
    results: list[SearchResult],
) -> dict[int, ConfidenceScore]:
    """Calculate confidence scores for each result based on score distribution.

    Confidence is calculated from:
    - score_reliability: Position in score distribution (percentile)
    - source_quality: Always 0.9 (assuming high-quality curated knowledge base)
    - recency: Always 0.85 (assuming relatively recent documentation)

    Args:
        results: List of SearchResult objects with hybrid_score

    Returns:
        Dictionary mapping chunk_id to ConfidenceScore

    Example:
        >>> results = [SearchResult(chunk_id=1, hybrid_score=0.95, ...), ...]
        >>> scores = calculate_confidence_scores(results)
        >>> scores[1].score_reliability  # High for top result
        0.98
    """
    if not results:
        return {}

    # Extract scores
    scores = [r.hybrid_score for r in results]
    scores_sorted = sorted(scores, reverse=True)

    confidence_map: dict[int, ConfidenceScore] = {}

    for result in results:
        # Calculate percentile (higher is better)
        rank = scores_sorted.index(result.hybrid_score)
        percentile = 100.0 * (1.0 - rank / len(scores))

        # Convert percentile to score_reliability (0.0-1.0)
        score_reliability = percentile / 100.0

        confidence_map[result.chunk_id] = ConfidenceScore(
            score_reliability=score_reliability,
            source_quality=0.9,  # High quality curated knowledge base
            recency=0.85,  # Relatively recent documentation
        )

    return confidence_map


def generate_ranking_context(
    results: list[SearchResult],
) -> dict[int, RankingContext]:
    """Generate ranking context for each result with percentile and explanation.

    Args:
        results: List of SearchResult objects with hybrid_score and score_type

    Returns:
        Dictionary mapping chunk_id to RankingContext

    Example:
        >>> results = [SearchResult(chunk_id=1, hybrid_score=0.95, ...), ...]
        >>> contexts = generate_ranking_context(results)
        >>> contexts[1].percentile
        100
    """
    if not results:
        return {}

    # Extract scores for percentile calculation
    scores = [r.hybrid_score for r in results]
    scores_sorted = sorted(scores, reverse=True)

    ranking_map: dict[int, RankingContext] = {}

    for result in results:
        # Calculate percentile (1-100, higher is better)
        rank = scores_sorted.index(result.hybrid_score)
        percentile = int(100.0 * (1.0 - rank / len(scores)))

        # Generate explanation based on rank
        if rank == 0:
            explanation = "Highest combined semantic + keyword match"
        elif rank < 3:
            explanation = "Very high relevance score"
        elif rank < 10:
            explanation = "Strong relevance match"
        else:
            explanation = "Moderate relevance match"

        ranking_map[result.chunk_id] = RankingContext(
            percentile=percentile,
            explanation=explanation,
            score_method=result.score_type,
        )

    return ranking_map


def detect_duplicates(
    results: list[SearchResult],
    similarity_threshold: float = 0.95,
) -> dict[int, DeduplicationInfo]:
    """Detect duplicate or highly similar results based on score proximity.

    Args:
        results: List of SearchResult objects
        similarity_threshold: Score difference threshold for similarity (default: 0.95)

    Returns:
        Dictionary mapping chunk_id to DeduplicationInfo

    Example:
        >>> results = [SearchResult(chunk_id=1, hybrid_score=0.90, ...), ...]
        >>> dedup = detect_duplicates(results)
        >>> dedup[1].is_duplicate
        False
    """
    if not results:
        return {}

    dedup_map: dict[int, DeduplicationInfo] = {}

    for i, result in enumerate(results):
        # Find similar results (within threshold)
        similar_ids = []
        for j, other in enumerate(results):
            if i != j and abs(result.hybrid_score - other.hybrid_score) < (1.0 - similarity_threshold):
                similar_ids.append(other.chunk_id)

        # Mark as duplicate if there's a higher-ranked similar result
        is_duplicate = any(
            results[j].rank < result.rank
            for j, other in enumerate(results)
            if other.chunk_id in similar_ids
        )

        dedup_map[result.chunk_id] = DeduplicationInfo(
            is_duplicate=is_duplicate,
            similar_chunk_ids=similar_ids[:5],  # Limit to top 5
            confidence=0.85,  # Fixed confidence for score-based similarity
        )

    return dedup_map


def generate_response_warnings(
    response_size_bytes: int,
    result_count: int,
    response_mode: str,
    entity_count: int | None = None,
) -> list[ResponseWarning]:
    """Generate warnings for oversized responses or configuration issues.

    Args:
        response_size_bytes: Estimated response size in bytes
        result_count: Number of results in response
        response_mode: Response detail level
        entity_count: Number of entities (for vendor info responses)

    Returns:
        List of ResponseWarning objects

    Example:
        >>> warnings = generate_response_warnings(25000, 10, "full")
        >>> warnings[0].code
        'RESPONSE_SIZE_LARGE'
    """
    warnings: list[ResponseWarning] = []

    # Check response size
    if response_size_bytes > RESPONSE_SIZE_ERROR_THRESHOLD:
        warnings.append(
            ResponseWarning(
                level="error",
                code="RESPONSE_SIZE_EXCESSIVE",
                message=f"Response size ({response_size_bytes // 1024}KB) exceeds 100KB limit",
                suggestion="Use pagination with smaller page_size or switch to ids_only mode",
            )
        )
    elif response_size_bytes > RESPONSE_SIZE_WARNING_THRESHOLD:
        warnings.append(
            ResponseWarning(
                level="warning",
                code="RESPONSE_SIZE_LARGE",
                message=f"Response size ({response_size_bytes // 1024}KB) exceeds 20KB",
                suggestion="Consider using pagination or switching to metadata mode",
            )
        )

    # Check vendor entity count
    if entity_count is not None:
        if entity_count > VENDOR_ENTITY_ERROR_THRESHOLD:
            warnings.append(
                ResponseWarning(
                    level="error",
                    code="ENTITY_GRAPH_TOO_LARGE",
                    message=f"Entity graph ({entity_count} entities) exceeds 100 entity limit",
                    suggestion="Result truncated to 100 entities. Use preview mode for smaller response.",
                )
            )
        elif entity_count > VENDOR_ENTITY_WARNING_THRESHOLD:
            warnings.append(
                ResponseWarning(
                    level="warning",
                    code="ENTITY_GRAPH_LARGE",
                    message=f"Entity graph ({entity_count} entities) is large",
                    suggestion="Consider using preview mode (5 entities) for faster response",
                )
            )

    # Check result count vs response mode
    if result_count > 20 and response_mode == "full":
        warnings.append(
            ResponseWarning(
                level="warning",
                code="RESULT_COUNT_HIGH",
                message=f"High result count ({result_count}) with full response mode",
                suggestion="Consider using metadata or preview mode to reduce token usage",
            )
        )

    return warnings


def wrap_semantic_search_response(
    results: list[SearchResultMetadata],
    total_found: int,
    execution_time_ms: float,
    cache_hit: bool,
    pagination: PaginationMetadata | None = None,
    response_mode: str = "metadata",
    enhanced: bool = False,
) -> MCPResponseEnvelope[dict[str, Any]]:
    """Wrap semantic search response in MCPResponseEnvelope with metadata.

    Args:
        results: List of search results (SearchResultMetadata or Enhanced)
        total_found: Total matching results
        execution_time_ms: Execution time in milliseconds
        cache_hit: Whether response was cached
        pagination: Pagination metadata (optional)
        response_mode: Response detail level
        enhanced: Whether to add confidence/ranking metadata

    Returns:
        MCPResponseEnvelope with semantic search response

    Example:
        >>> results = [SearchResultMetadata(...), ...]
        >>> envelope = wrap_semantic_search_response(
        ...     results=results,
        ...     total_found=42,
        ...     execution_time_ms=245.3,
        ...     cache_hit=True
        ... )
    """
    # Generate unique request ID
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # Estimate tokens
    estimated_tokens = estimate_response_tokens(
        result_count=len(results),
        response_mode=response_mode,
        include_metadata=True,
    )

    # Estimate response size (rough approximation: 4 bytes per token)
    response_size_bytes = estimated_tokens * 4

    # Generate warnings
    warnings = generate_response_warnings(
        response_size_bytes=response_size_bytes,
        result_count=len(results),
        response_mode=response_mode,
    )

    # Create response metadata
    metadata = ResponseMetadata(
        operation="semantic_search",
        version="1.0",
        timestamp=datetime.now(UTC).isoformat(),
        request_id=request_id,
        status="success",
        message=None,
    )

    # Create execution context
    execution_context = ExecutionContext(
        tokens_estimated=estimated_tokens,
        tokens_used=None,  # Actual count not available
        cache_hit=cache_hit,
        execution_time_ms=execution_time_ms,
        request_id=request_id,
    )

    # Build response data
    response_data: dict[str, Any] = {
        "results": [r.model_dump() for r in results],
        "total_found": total_found,
        "strategy_used": "hybrid",
        "execution_time_ms": execution_time_ms,
    }

    if pagination:
        response_data["pagination"] = pagination.model_dump()

    return MCPResponseEnvelope(
        metadata=metadata,
        results=response_data,
        pagination=pagination,
        execution_context=execution_context,
        warnings=warnings,
    )


def wrap_vendor_info_response(
    vendor_name: str,
    results: VendorInfoMetadata | dict[str, Any],
    execution_time_ms: float,
    cache_hit: bool,
    pagination: PaginationMetadata | None = None,
    entity_count: int | None = None,
) -> MCPResponseEnvelope[dict[str, Any]]:
    """Wrap vendor info response in MCPResponseEnvelope with metadata.

    Args:
        vendor_name: Vendor name
        results: Vendor info results (VendorInfoMetadata or dict)
        execution_time_ms: Execution time in milliseconds
        cache_hit: Whether response was cached
        pagination: Pagination metadata (optional)
        entity_count: Number of entities in vendor graph

    Returns:
        MCPResponseEnvelope with vendor info response

    Example:
        >>> results = VendorInfoMetadata(...)
        >>> envelope = wrap_vendor_info_response(
        ...     vendor_name="Acme Corp",
        ...     results=results,
        ...     execution_time_ms=320.5,
        ...     cache_hit=False
        ... )
    """
    # Generate unique request ID
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # Estimate tokens (vendor info is typically metadata mode)
    estimated_tokens = 3000  # Base estimate for vendor metadata

    # Estimate response size
    response_size_bytes = estimated_tokens * 4

    # Generate warnings
    warnings = generate_response_warnings(
        response_size_bytes=response_size_bytes,
        result_count=1,
        response_mode="metadata",
        entity_count=entity_count,
    )

    # Create response metadata
    metadata = ResponseMetadata(
        operation="find_vendor_info",
        version="1.0",
        timestamp=datetime.now(UTC).isoformat(),
        request_id=request_id,
        status="success",
        message=None,
    )

    # Create execution context
    execution_context = ExecutionContext(
        tokens_estimated=estimated_tokens,
        tokens_used=None,
        cache_hit=cache_hit,
        execution_time_ms=execution_time_ms,
        request_id=request_id,
    )

    # Convert results to dict if Pydantic model
    results_dict = results.model_dump() if hasattr(results, "model_dump") else results

    # Build response data
    response_data: dict[str, Any] = {
        "vendor_name": vendor_name,
        "results": results_dict,
        "execution_time_ms": execution_time_ms,
    }

    if pagination:
        response_data["pagination"] = pagination.model_dump()

    return MCPResponseEnvelope(
        metadata=metadata,
        results=response_data,
        pagination=pagination,
        execution_context=execution_context,
        warnings=warnings,
    )


def apply_compression_to_envelope(
    envelope: MCPResponseEnvelope[T],
    config: dict[str, Any] | None = None,
) -> MCPResponseEnvelope[T]:
    """Apply compression configuration to response envelope.

    Currently a no-op placeholder for future compression features
    (e.g., field filtering, content truncation, gzip compression).

    Args:
        envelope: Response envelope to compress
        config: Optional compression configuration

    Returns:
        Compressed envelope (currently unchanged)

    Example:
        >>> envelope = wrap_semantic_search_response(...)
        >>> compressed = apply_compression_to_envelope(
        ...     envelope,
        ...     config={"max_tokens": 5000}
        ... )
    """
    # Placeholder for future compression features
    # Could implement:
    # - Field filtering based on config
    # - Content truncation to meet token budgets
    # - Gzip compression for large responses
    # - Adaptive response mode selection

    return envelope
