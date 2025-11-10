"""semantic_search tool implementation for FastMCP.

Hybrid semantic search combining vector similarity and BM25 keyword matching.
Supports 4 progressive disclosure modes (ids_only, metadata, preview, full).

Progressive Disclosure Pattern:
- ids_only: Chunk IDs + scores (~100 tokens for 10 results)
- metadata: IDs + file info (~2-4K tokens for 10 results) - DEFAULT
- preview: metadata + 200-char snippet (~5-10K tokens for 10 results)
- full: Complete chunk content (~15K+ tokens for 10 results)

Token Reduction:
- Traditional (full, 10 results): ~15,000 tokens
- Progressive (metadata, 10 results): ~2,500 tokens (83% reduction)
- Selective (metadata 10 + full 3): ~6,500 tokens (57% reduction)

Example:
    # Metadata-only (default - fast, token-efficient)
    >>> response = semantic_search("JWT authentication", top_k=10)

    # Full content (slower, more tokens)
    >>> response = semantic_search("JWT authentication", top_k=5, response_mode="full")

    # Preview with snippet
    >>> response = semantic_search("authentication patterns", response_mode="preview")
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from src.core.logging import StructuredLogger
from src.mcp.cache import hash_query
from src.mcp.models import (
    PaginationMetadata,
    SearchResultFull,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from src.mcp.server import get_cache_layer, get_hybrid_search, mcp
from src.search.results import SearchResult

logger: logging.Logger = StructuredLogger.get_logger(__name__)


def compute_search_cache_key(query: str, top_k: int, response_mode: str) -> str:
    """Compute cache key for semantic search query.

    Args:
        query: Search query string
        top_k: Number of results requested
        response_mode: Response detail level

    Returns:
        Cache key string (format: "search:hash")

    Example:
        >>> key = compute_search_cache_key("JWT auth", 10, "metadata")
        >>> assert key.startswith("search:")
    """
    params = {"query": query, "top_k": top_k, "response_mode": response_mode}
    return f"search:{hash_query(params)}"


def generate_next_cursor(
    query: str, top_k: int, response_mode: str, offset: int
) -> str:
    """Generate cursor for next page of results.

    Args:
        query: Search query (for hashing)
        top_k: Total results available
        response_mode: Response detail level
        offset: Offset for next page

    Returns:
        Base64-encoded cursor string

    Example:
        >>> cursor = generate_next_cursor("JWT", 10, "metadata", 10)
        >>> data = json.loads(base64.b64decode(cursor))
        >>> assert data["offset"] == 10
    """
    # Create cursor data
    params = {"query": query, "top_k": top_k, "response_mode": response_mode}
    query_hash = hash_query(params)

    cursor_data = {
        "query_hash": query_hash,
        "offset": offset,
        "response_mode": response_mode,
    }

    # Encode as base64 JSON
    cursor_json = json.dumps(cursor_data)
    cursor_bytes = base64.b64encode(cursor_json.encode("utf-8"))
    return cursor_bytes.decode("utf-8")


def parse_cursor(cursor: str) -> dict[str, Any]:
    """Parse cursor to extract pagination state.

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Dictionary with query_hash, offset, response_mode

    Raises:
        ValueError: If cursor is invalid

    Example:
        >>> cursor = generate_next_cursor("JWT", 10, "metadata", 10)
        >>> data = parse_cursor(cursor)
        >>> assert data["offset"] == 10
    """
    try:
        cursor_bytes = base64.b64decode(cursor.encode("utf-8"))
        cursor_json = cursor_bytes.decode("utf-8")
        return json.loads(cursor_json)  # type: ignore[no-any-return]
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid cursor format: {e}") from e


def filter_response_fields(
    result: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    """Filter result dictionary to only include specified fields.

    Args:
        result: Result dictionary (from Pydantic .model_dump())
        fields: List of field names to include

    Returns:
        Filtered dictionary with only specified fields

    Example:
        >>> result = {"chunk_id": 1, "score": 0.9, "text": "..."}
        >>> filtered = filter_response_fields(result, ["chunk_id", "score"])
        >>> assert set(filtered.keys()) == {"chunk_id", "score"}
    """
    return {k: v for k, v in result.items() if k in fields}


def format_ids_only(result: SearchResult) -> SearchResultIDs:
    """Convert SearchResult to IDs-only format (Level 0).

    Token Budget: ~10 tokens per result

    Args:
        result: SearchResult from HybridSearch

    Returns:
        SearchResultIDs with minimal fields

    Example:
        >>> result = SearchResult(chunk_id=1, hybrid_score=0.85, rank=1, ...)
        >>> ids_result = format_ids_only(result)
        >>> assert ids_result.chunk_id == 1
    """
    return SearchResultIDs(
        chunk_id=result.chunk_id,
        hybrid_score=result.hybrid_score,
        rank=result.rank,
    )


def format_metadata(result: SearchResult) -> SearchResultMetadata:
    """Convert SearchResult to metadata-only format (Level 1).

    Token Budget: ~100-200 tokens per result

    Args:
        result: SearchResult from HybridSearch

    Returns:
        SearchResultMetadata with file info but no content

    Example:
        >>> result = SearchResult(...)
        >>> metadata = format_metadata(result)
        >>> assert metadata.source_file == "docs/auth.md"
        >>> assert not hasattr(metadata, "chunk_text")
    """
    return SearchResultMetadata(
        chunk_id=result.chunk_id,
        source_file=result.source_file,
        source_category=result.source_category,
        hybrid_score=result.hybrid_score,
        rank=result.rank,
        chunk_index=result.chunk_index,
        total_chunks=result.total_chunks,
    )


def format_preview(result: SearchResult) -> SearchResultPreview:
    """Convert SearchResult to preview format (Level 2).

    Token Budget: ~500-1000 tokens per result

    Args:
        result: SearchResult from HybridSearch

    Returns:
        SearchResultPreview with metadata + 200-char snippet

    Example:
        >>> result = SearchResult(chunk_text="Long content here...", ...)
        >>> preview = format_preview(result)
        >>> assert len(preview.chunk_snippet) <= 203  # 200 + "..."
    """
    # Create snippet (first 200 chars)
    snippet = (
        result.chunk_text[:200] + "..."
        if len(result.chunk_text) > 200
        else result.chunk_text
    )

    return SearchResultPreview(
        chunk_id=result.chunk_id,
        source_file=result.source_file,
        source_category=result.source_category,
        hybrid_score=result.hybrid_score,
        rank=result.rank,
        chunk_index=result.chunk_index,
        total_chunks=result.total_chunks,
        chunk_snippet=snippet,
        context_header=result.context_header,
    )


def format_full(result: SearchResult) -> SearchResultFull:
    """Convert SearchResult to full-content format (Level 3).

    Token Budget: ~1500+ tokens per result

    Args:
        result: SearchResult from HybridSearch

    Returns:
        SearchResultFull with all fields including complete chunk text

    Example:
        >>> result = SearchResult(chunk_text="Full content...", chunk_token_count=512, ...)
        >>> full = format_full(result)
        >>> assert full.chunk_token_count == 512
        >>> assert len(full.chunk_text) > 200
    """
    return SearchResultFull(
        chunk_id=result.chunk_id,
        chunk_text=result.chunk_text,
        similarity_score=result.similarity_score,
        bm25_score=result.bm25_score,
        hybrid_score=result.hybrid_score,
        rank=result.rank,
        score_type=result.score_type,
        source_file=result.source_file,
        source_category=result.source_category,
        context_header=result.context_header,
        chunk_index=result.chunk_index,
        total_chunks=result.total_chunks,
        chunk_token_count=result.chunk_token_count,
    )


@mcp.tool()  # type: ignore[misc]
def semantic_search(
    query: str,
    top_k: int | None = None,
    page_size: int = 10,
    cursor: str | None = None,
    fields: list[str] | None = None,
    response_mode: str = "metadata",
) -> SemanticSearchResponse:
    """Hybrid semantic search with caching, pagination, and field filtering.

    Searches the knowledge base using hybrid search (vector similarity + BM25 full-text)
    with Reciprocal Rank Fusion merging. Supports caching for repeated queries,
    cursor-based pagination for large result sets, and field-level filtering.

    Args:
        query: Search query (natural language or keywords). Max 500 chars.
        top_k: DEPRECATED - Use page_size instead. If provided, overrides page_size.
        page_size: Number of results per page (1-50). Default: 10.
        cursor: Pagination cursor from previous response (base64-encoded JSON).
        fields: Optional list of fields to include in response (whitelist filtering).
        response_mode: Response detail level. Default: "metadata".
            - "ids_only": Chunk IDs + scores only (~100 tokens for 10 results)
            - "metadata": IDs + file info + scores (~2-4K tokens for 10 results)
            - "preview": metadata + 200-char snippet (~5-10K tokens for 10 results)
            - "full": Complete chunk content (~15K+ tokens for 10 results)

    Returns:
        SemanticSearchResponse with:
        - results: List of search results (type depends on response_mode)
        - total_found: Total matching results
        - strategy_used: Search strategy (always "hybrid")
        - execution_time_ms: Execution time in milliseconds
        - pagination: Pagination metadata (cursor, page_size, has_more) or None

    Raises:
        ValueError: If query is invalid or parameters out of range
        RuntimeError: If search execution fails

    Performance:
        - Cached results: <100ms P95
        - Fresh search (metadata): <200ms P50, <500ms P95
        - Fresh search (full): <300ms P50, <800ms P95

    Examples:
        # Basic search (cached for 30s)
        >>> response = semantic_search("JWT authentication")
        >>> assert response.total_found > 0

        # Pagination
        >>> page1 = semantic_search("auth", page_size=10)
        >>> if page1.pagination and page1.pagination.has_more:
        ...     page2 = semantic_search("auth", cursor=page1.pagination.cursor)

        # Field filtering
        >>> response = semantic_search(
        ...     "auth",
        ...     fields=["chunk_id", "hybrid_score", "source_file"],
        ...     response_mode="metadata"
        ... )
    """
    # Validate request using Pydantic
    try:
        request = SemanticSearchRequest(
            query=query,
            top_k=top_k,
            page_size=page_size,
            cursor=cursor,
            fields=fields,
            response_mode=response_mode,  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.error(f"Request validation failed: {e}")
        raise ValueError(f"Invalid request parameters: {e}") from e

    # Handle top_k vs page_size precedence (backward compatibility)
    effective_top_k = request.top_k if request.top_k is not None else request.page_size

    # Parse cursor if provided
    offset = 0
    if request.cursor:
        try:
            cursor_data = parse_cursor(request.cursor)
            offset = cursor_data["offset"]
            # Validate cursor matches current query
            expected_hash = hash_query({
                "query": request.query,
                "top_k": effective_top_k,
                "response_mode": request.response_mode,
            })
            if cursor_data["query_hash"] != expected_hash:
                logger.warning("Cursor query_hash mismatch - treating as new query")
                offset = 0
        except ValueError as e:
            logger.warning(f"Invalid cursor: {e} - treating as new query")
            offset = 0

    # Compute cache key
    cache_key = compute_search_cache_key(
        request.query, effective_top_k, request.response_mode
    )
    cache_layer = get_cache_layer()

    # Check cache first
    start_time = time.time()
    cached_results = cache_layer.get(cache_key)
    cache_hit = cached_results is not None

    if cache_hit:
        # Use cached results
        all_results: list[SearchResult] = cached_results
        logger.debug(f"Cache hit: {cache_key}")
    else:
        # Execute fresh search
        hybrid_search = get_hybrid_search()
        try:
            all_results = hybrid_search.search(
                query=request.query,
                top_k=effective_top_k,
                strategy="hybrid",
                min_score=0.0,
            )
            # Cache results for 30 seconds (queries change frequently)
            cache_layer.set(cache_key, all_results, ttl_seconds=30)
            logger.debug(f"Cache miss: {cache_key} - stored {len(all_results)} results")
        except Exception as e:
            logger.error(f"Search execution failed: {e}", extra={"query": query})
            raise RuntimeError(f"Search failed: {e}") from e

    execution_time_ms = (time.time() - start_time) * 1000

    # Apply pagination (extract page from all_results)
    total_available = len(all_results)
    page_end = min(offset + request.page_size, total_available)
    page_results = all_results[offset:page_end]
    has_more = page_end < total_available

    # Generate next cursor if has_more
    next_cursor = None
    if has_more:
        next_cursor = generate_next_cursor(
            request.query,
            effective_top_k,
            request.response_mode,
            page_end,  # offset for next page
        )

    # Format results based on response_mode
    formatted_results: list[SearchResultIDs] | list[SearchResultMetadata] | list[SearchResultPreview] | list[SearchResultFull]

    if request.response_mode == "ids_only":
        formatted_results = [format_ids_only(r) for r in page_results]
    elif request.response_mode == "metadata":
        formatted_results = [format_metadata(r) for r in page_results]
    elif request.response_mode == "preview":
        formatted_results = [format_preview(r) for r in page_results]
    else:  # full
        formatted_results = [format_full(r) for r in page_results]

    # Apply field filtering if requested
    if request.fields:
        # Convert to dicts, filter, then convert back
        filtered_dicts = [
            filter_response_fields(r.model_dump(), request.fields)
            for r in formatted_results
        ]
        # Note: We return dicts here, not Pydantic models, for field filtering
        # This is acceptable since the response validator allows Any in results
        formatted_results = filtered_dicts  # type: ignore[assignment]

    # Create pagination metadata
    pagination_metadata = PaginationMetadata(
        cursor=next_cursor,
        page_size=len(page_results),
        has_more=has_more,
        total_available=total_available,
    )

    logger.info(
        f"Search completed: {len(page_results)}/{total_available} results, "
        f"{execution_time_ms:.1f}ms, cache_hit={cache_hit}",
        extra={
            "query": query,
            "page_size": request.page_size,
            "offset": offset,
            "response_mode": response_mode,
            "results_count": len(page_results),
            "total_available": total_available,
            "execution_time_ms": execution_time_ms,
            "cache_hit": cache_hit,
        },
    )

    return SemanticSearchResponse(
        results=formatted_results,
        total_found=total_available,
        strategy_used="hybrid",
        execution_time_ms=execution_time_ms,
        pagination=pagination_metadata,
    )
