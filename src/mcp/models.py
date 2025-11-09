"""Pydantic models for FastMCP server request/response schemas.

This module defines type-safe schemas for the semantic_search MCP tool,
implementing progressive disclosure with 4 response modes:
- ids_only: Chunk IDs + scores only (~100 tokens for 10 results)
- metadata: IDs + file info + scores (~2-4K tokens for 10 results)
- preview: metadata + 200-char snippet (~5-10K tokens for 10 results)
- full: Complete chunk content (~15K+ tokens for 10 results)

All models use Pydantic v2 for validation and are mypy-strict compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SemanticSearchRequest(BaseModel):
    """Request schema for semantic_search tool.

    MCP Tool Parameters:
    - query: Search query string (required)
    - top_k: Number of results (default: 10, max: 50)
    - response_mode: Progressive disclosure level (ids_only/metadata/preview/full)

    Example:
        >>> request = SemanticSearchRequest(
        ...     query="JWT authentication best practices",
        ...     top_k=5,
        ...     response_mode="metadata"
        ... )
    """

    query: str = Field(
        ...,
        description="Search query (natural language or keywords)",
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(
        default=10, description="Number of results to return", ge=1, le=50
    )
    response_mode: Literal["ids_only", "metadata", "preview", "full"] = Field(
        default="metadata",
        description=(
            "Response detail level: "
            "ids_only (~100 tokens), "
            "metadata (~2-4K tokens), "
            "preview (~5-10K tokens), "
            "full (~10-50K+ tokens)"
        ),
    )

    @field_validator('query', mode='before')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty or whitespace-only.

        Args:
            v: Query string (raw input, before other validators)

        Returns:
            Stripped query string

        Raises:
            ValueError: If query is empty or whitespace-only
        """
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError('Query cannot be empty or whitespace-only')
        return v


class SearchResultIDs(BaseModel):
    """Minimal search result with IDs and scores only (Level 0).

    Token Budget: ~10 tokens per result (~100 tokens for 10 results)
    Use Case: Quick relevance check, result counting

    Example:
        >>> result = SearchResultIDs(chunk_id=1, hybrid_score=0.85, rank=1)
    """

    chunk_id: int = Field(..., description="Unique chunk identifier")
    hybrid_score: float = Field(..., description="Combined relevance score (0.0-1.0)", ge=0.0, le=1.0)
    rank: int = Field(..., description="Result rank (1-based)", ge=1)


class SearchResultMetadata(BaseModel):
    """Search result with metadata but no content (Level 1).

    Token Budget: ~100-200 tokens per result (~2-4K tokens for 10 results)
    Use Case: File identification, source browsing, quick filtering

    Example:
        >>> result = SearchResultMetadata(
        ...     chunk_id=1,
        ...     source_file="docs/auth.md",
        ...     source_category="security",
        ...     hybrid_score=0.85,
        ...     rank=1,
        ...     chunk_index=0,
        ...     total_chunks=10
        ... )
    """

    chunk_id: int = Field(..., description="Unique chunk identifier")
    source_file: str = Field(..., description="Source file path")
    source_category: str | None = Field(None, description="Document category")
    hybrid_score: float = Field(..., description="Combined relevance score (0.0-1.0)", ge=0.0, le=1.0)
    rank: int = Field(..., description="Result rank (1-based)", ge=1)
    chunk_index: int = Field(..., description="Chunk position in document", ge=0)
    total_chunks: int = Field(..., description="Total chunks in document", ge=1)


class SearchResultPreview(BaseModel):
    """Search result with metadata + snippet (Level 2).

    Token Budget: ~500-1000 tokens per result (~5-10K tokens for 10 results)
    Use Case: Content preview, quick relevance assessment

    Example:
        >>> result = SearchResultPreview(
        ...     chunk_id=1,
        ...     source_file="docs/auth.md",
        ...     source_category="security",
        ...     hybrid_score=0.85,
        ...     rank=1,
        ...     chunk_index=0,
        ...     total_chunks=10,
        ...     chunk_snippet="JWT authentication provides...",
        ...     context_header="auth.md > Security > Authentication"
        ... )
    """

    chunk_id: int = Field(..., description="Unique chunk identifier")
    source_file: str = Field(..., description="Source file path")
    source_category: str | None = Field(None, description="Document category")
    hybrid_score: float = Field(..., description="Combined relevance score (0.0-1.0)", ge=0.0, le=1.0)
    rank: int = Field(..., description="Result rank (1-based)", ge=1)
    chunk_index: int = Field(..., description="Chunk position in document", ge=0)
    total_chunks: int = Field(..., description="Total chunks in document", ge=1)
    chunk_snippet: str = Field(..., description="First 200 chars of chunk text")
    context_header: str = Field(..., description="Hierarchical context path")


class SearchResultFull(BaseModel):
    """Full search result with all fields (Level 3).

    Token Budget: ~1500+ tokens per result (~15K+ tokens for 10 results)
    Use Case: Deep analysis, complete context, implementation details

    Example:
        >>> result = SearchResultFull(
        ...     chunk_id=1,
        ...     chunk_text="Full content here...",
        ...     similarity_score=0.80,
        ...     bm25_score=0.70,
        ...     hybrid_score=0.85,
        ...     rank=1,
        ...     score_type="hybrid",
        ...     source_file="docs/auth.md",
        ...     source_category="security",
        ...     context_header="auth.md > Security",
        ...     chunk_index=0,
        ...     total_chunks=10,
        ...     chunk_token_count=512
        ... )
    """

    chunk_id: int = Field(..., description="Unique chunk identifier")
    chunk_text: str = Field(..., description="Full chunk content (can be 1000+ tokens)")
    similarity_score: float = Field(..., description="Vector similarity score (0.0-1.0)", ge=0.0, le=1.0)
    bm25_score: float = Field(..., description="BM25 relevance score (0.0-1.0)", ge=0.0, le=1.0)
    hybrid_score: float = Field(..., description="Combined score (0.0-1.0)", ge=0.0, le=1.0)
    rank: int = Field(..., description="Result rank (1-based)", ge=1)
    score_type: str = Field(..., description="Score type (vector/bm25/hybrid)")
    source_file: str = Field(..., description="Source file path")
    source_category: str | None = Field(None, description="Document category")
    context_header: str = Field(..., description="Hierarchical context path")
    chunk_index: int = Field(..., description="Chunk position in document", ge=0)
    total_chunks: int = Field(..., description="Total chunks in document", ge=1)
    chunk_token_count: int = Field(..., description="Number of tokens in chunk", ge=0)


class SemanticSearchResponse(BaseModel):
    """Response schema for semantic_search tool.

    Supports 4 progressive disclosure levels via response_mode parameter:
    - ids_only: List[SearchResultIDs] (~100 tokens for 10 results)
    - metadata: List[SearchResultMetadata] (~2-4K tokens for 10 results)
    - preview: List[SearchResultPreview] (~5-10K tokens for 10 results)
    - full: List[SearchResultFull] (~15K+ tokens for 10 results)

    Token Reduction Example:
    - Traditional (full, 10 results): ~15,000 tokens
    - Progressive (metadata, 10 results): ~2,500 tokens (83% reduction)
    - Selective (metadata 10 + full 3): ~6,500 tokens (57% reduction)

    Example:
        >>> response = SemanticSearchResponse(
        ...     results=[SearchResultMetadata(...)],
        ...     total_found=42,
        ...     strategy_used="hybrid",
        ...     execution_time_ms=245.3
        ... )
    """

    results: list[SearchResultIDs] | list[SearchResultMetadata] | list[SearchResultPreview] | list[SearchResultFull] = Field(..., description="Search results (type depends on response_mode)")
    total_found: int = Field(..., description="Total matching results before top_k limit", ge=0)
    strategy_used: str = Field(..., description="Search strategy used (vector/bm25/hybrid)")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds", ge=0.0)
