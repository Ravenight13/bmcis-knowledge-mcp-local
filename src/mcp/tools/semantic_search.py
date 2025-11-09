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

import logging
import time

from src.core.logging import StructuredLogger
from src.mcp.models import (
    SearchResultFull,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from src.mcp.server import get_hybrid_search, mcp
from src.search.results import SearchResult

logger: logging.Logger = StructuredLogger.get_logger(__name__)


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
    top_k: int = 10,
    response_mode: str = "metadata",
) -> SemanticSearchResponse:
    """Hybrid semantic search combining vector similarity and BM25 keyword matching.

    Searches the knowledge base using hybrid search (vector similarity + BM25 full-text)
    with Reciprocal Rank Fusion merging. Returns results in one of 4 progressive disclosure
    formats to optimize token usage.

    Args:
        query: Search query (natural language or keywords). Max 500 chars.
        top_k: Number of results to return (1-50). Default: 10.
        response_mode: Response detail level. Default: "metadata".
            - "ids_only": Chunk IDs + scores only (~100 tokens for 10 results)
            - "metadata": IDs + file info + scores (~2-4K tokens for 10 results)
            - "preview": metadata + 200-char snippet (~5-10K tokens for 10 results)
            - "full": Complete chunk content (~15K+ tokens for 10 results)

    Returns:
        SemanticSearchResponse with results list, total count, strategy, and timing.
        Result type depends on response_mode:
        - ids_only -> List[SearchResultIDs]
        - metadata -> List[SearchResultMetadata]
        - preview -> List[SearchResultPreview]
        - full -> List[SearchResultFull]

    Raises:
        ValueError: If query is invalid or parameters out of range
        RuntimeError: If search execution fails

    Performance:
        - Metadata mode: <200ms P50, <500ms P95
        - Full mode: <300ms P50, <800ms P95
        - Uses existing HybridSearch cache for performance

    Examples:
        # Metadata-only (default - fast, token-efficient)
        >>> response = semantic_search("JWT authentication")
        >>> assert response.total_found > 0
        >>> assert len(response.results) <= 10
        >>> # Results are SearchResultMetadata objects

        # Full content for deep analysis
        >>> response = semantic_search("OAuth2 flow", top_k=5, response_mode="full")
        >>> for result in response.results:
        ...     print(result.chunk_text)  # Full content available

        # IDs only for quick relevance check
        >>> response = semantic_search("authentication", top_k=20, response_mode="ids_only")
        >>> chunk_ids = [r.chunk_id for r in response.results]

        # Preview with snippets
        >>> response = semantic_search("security patterns", response_mode="preview")
        >>> for result in response.results:
        ...     print(result.chunk_snippet)  # First 200 chars
    """
    # Validate request using Pydantic
    try:
        request = SemanticSearchRequest(
            query=query,
            top_k=top_k,
            response_mode=response_mode,  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.error(f"Request validation failed: {e}")
        raise ValueError(f"Invalid request parameters: {e}") from e

    # Execute hybrid search
    start_time = time.time()
    hybrid_search = get_hybrid_search()

    try:
        results: list[SearchResult] = hybrid_search.search(
            query=request.query,
            top_k=request.top_k,
            strategy="hybrid",  # Always use hybrid for MCP (best quality)
            min_score=0.0,  # No filtering (let user decide)
        )
    except Exception as e:
        logger.error(f"Search execution failed: {e}", extra={"query": query, "error": str(e)})
        raise RuntimeError(f"Search failed: {e}") from e

    execution_time_ms = (time.time() - start_time) * 1000

    # Format results based on response_mode
    formatted_results: list[SearchResultIDs] | list[SearchResultMetadata] | list[SearchResultPreview] | list[SearchResultFull]

    if request.response_mode == "ids_only":
        formatted_results = [format_ids_only(r) for r in results]
    elif request.response_mode == "metadata":
        formatted_results = [format_metadata(r) for r in results]
    elif request.response_mode == "preview":
        formatted_results = [format_preview(r) for r in results]
    else:  # full
        formatted_results = [format_full(r) for r in results]

    logger.info(
        f"Search completed: {len(formatted_results)} results in {execution_time_ms:.1f}ms",
        extra={
            "query": query,
            "top_k": top_k,
            "response_mode": response_mode,
            "results_count": len(formatted_results),
            "execution_time_ms": execution_time_ms,
        },
    )

    return SemanticSearchResponse(
        results=formatted_results,
        total_found=len(results),
        strategy_used="hybrid",
        execution_time_ms=execution_time_ms,
    )
