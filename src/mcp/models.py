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

from typing import Any, Literal

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


# ==============================================================================
# PHASE A: VendorInfo Models (Task 10.2)
# ==============================================================================
# Request/response models for find_vendor_info tool with progressive disclosure
# ==============================================================================


class FindVendorInfoRequest(BaseModel):
    """Request schema for find_vendor_info tool.

    Validates vendor info search requests with vendor name, response mode,
    and relationship inclusion preferences.

    Example:
        >>> request = FindVendorInfoRequest(
        ...     vendor_name="Acme Corp",
        ...     response_mode="metadata",
        ...     include_relationships=True
        ... )
    """

    vendor_name: str = Field(
        ...,
        description="Vendor name to search (1-200 characters, whitespace stripped)",
        min_length=1,
        max_length=200,
    )
    response_mode: Literal["ids_only", "metadata", "preview", "full"] = Field(
        default="metadata",
        description=(
            "Response detail level: "
            "ids_only (~100-500 tokens), "
            "metadata (~2-4K tokens), "
            "preview (~5-10K tokens), "
            "full (~10-50K+ tokens)"
        ),
    )
    include_relationships: bool = Field(
        default=False,
        description="Include relationship data in response",
    )

    @field_validator("vendor_name", mode="before")
    @classmethod
    def validate_vendor_name(cls, v: str) -> str:
        """Validate that vendor_name is not empty or whitespace-only.

        Args:
            v: Vendor name (raw input, before other validators)

        Returns:
            Stripped vendor name

        Raises:
            ValueError: If vendor_name is empty or whitespace-only
        """
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("Vendor name cannot be empty or whitespace-only")
        return v


class VendorEntity(BaseModel):
    """Single entity in vendor graph.

    Represents a single entity node in the vendor knowledge graph with
    confidence scores and optional snippet preview.

    Example:
        >>> entity = VendorEntity(
        ...     entity_id="vendor_123",
        ...     name="Acme Corporation",
        ...     entity_type="COMPANY",
        ...     confidence=0.95,
        ...     snippet="Acme Corp is a leading provider..."
        ... )
    """

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (e.g., PRODUCT, PERSON, GPE)")
    confidence: float = Field(
        ..., description="Match confidence (0.0-1.0)", ge=0.0, le=1.0
    )
    snippet: str | None = Field(
        default=None, description="Preview text (max 200 characters)", max_length=200
    )


class VendorRelationship(BaseModel):
    """Relationship between entities in vendor graph.

    Represents a directed edge between two entities with optional metadata.

    Example:
        >>> rel = VendorRelationship(
        ...     source_id="vendor_1",
        ...     target_id="vendor_2",
        ...     relationship_type="PARTNER",
        ...     metadata={"strength": 0.9}
        ... )
    """

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Relationship type (e.g., PARTNER, COMPETITOR)")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata about the relationship"
    )


class VendorStatistics(BaseModel):
    """Statistics about vendor graph.

    Aggregated counts and type distributions for entities and relationships.

    Example:
        >>> stats = VendorStatistics(
        ...     entity_count=85,
        ...     relationship_count=25,
        ...     entity_type_distribution={"COMPANY": 50, "PERSON": 25},
        ...     relationship_type_distribution={"PARTNER": 15, "COMPETITOR": 10}
        ... )
    """

    entity_count: int = Field(
        ..., description="Total entities (≥ 0)", ge=0
    )
    relationship_count: int = Field(
        ..., description="Total relationships (≥ 0)", ge=0
    )
    entity_type_distribution: dict[str, int] | None = Field(
        default=None, description="Distribution of entities by type"
    )
    relationship_type_distribution: dict[str, int] | None = Field(
        default=None, description="Distribution of relationships by type"
    )


class VendorInfoIDs(BaseModel):
    """Minimal vendor info response (Level 0).

    Token Budget: ~100-500 tokens
    Use Case: ID verification, count retrieval

    Example:
        >>> response = VendorInfoIDs(
        ...     vendor_name="Acme Corp",
        ...     entity_ids=["vendor_1", "vendor_2"]
        ... )
    """

    vendor_name: str = Field(..., description="Vendor name")
    entity_ids: list[str] = Field(..., description="List of entity IDs")
    relationship_ids: list[str] = Field(
        default_factory=list, description="List of relationship IDs"
    )


class VendorInfoMetadata(BaseModel):
    """Vendor info response with metadata (Level 1).

    Token Budget: ~2-4K tokens
    Use Case: File identification, source browsing

    Example:
        >>> response = VendorInfoMetadata(
        ...     vendor_name="Acme Corp",
        ...     statistics=VendorStatistics(entity_count=50, relationship_count=25),
        ...     top_entities=[...]
        ... )
    """

    vendor_name: str = Field(..., description="Vendor name")
    statistics: VendorStatistics = Field(..., description="Vendor graph statistics")
    top_entities: list[VendorEntity] | None = Field(
        default=None, description="Top entities (if available)"
    )
    last_updated: str | None = Field(
        default=None, description="ISO timestamp of last update"
    )


class VendorInfoPreview(BaseModel):
    """Vendor info response with preview (Level 2).

    Token Budget: ~5-10K tokens
    Use Case: Content preview, quick relevance assessment
    Constraints: Max 5 entities, max 5 relationships

    Example:
        >>> response = VendorInfoPreview(
        ...     vendor_name="Acme Corp",
        ...     entities=[...],
        ...     relationships=[...],
        ...     statistics=VendorStatistics(...)
        ... )
    """

    vendor_name: str = Field(..., description="Vendor name")
    entities: list[VendorEntity] = Field(
        ..., description="Top entities (max 5)", max_length=5
    )
    relationships: list[VendorRelationship] = Field(
        default_factory=list, description="Top relationships (max 5)", max_length=5
    )
    statistics: VendorStatistics = Field(..., description="Vendor graph statistics")

    @field_validator("entities", mode="after")
    @classmethod
    def validate_entity_count(cls, v: list[VendorEntity]) -> list[VendorEntity]:
        """Validate that entities list has at most 5 items.

        Args:
            v: List of vendor entities

        Returns:
            Validated entities list

        Raises:
            ValueError: If more than 5 entities
        """
        if len(v) > 5:
            raise ValueError("entities must contain at most 5 items")
        return v

    @field_validator("relationships", mode="after")
    @classmethod
    def validate_relationship_count(
        cls, v: list[VendorRelationship]
    ) -> list[VendorRelationship]:
        """Validate that relationships list has at most 5 items.

        Args:
            v: List of vendor relationships

        Returns:
            Validated relationships list

        Raises:
            ValueError: If more than 5 relationships
        """
        if len(v) > 5:
            raise ValueError("relationships must contain at most 5 items")
        return v


class VendorInfoFull(BaseModel):
    """Complete vendor info response (Level 3).

    Token Budget: ~10-50K+ tokens
    Use Case: Deep analysis, complete context
    Constraints: Max 100 entities, max 500 relationships

    Example:
        >>> response = VendorInfoFull(
        ...     vendor_name="Acme Corp",
        ...     entities=[...],
        ...     relationships=[...],
        ...     statistics=VendorStatistics(...)
        ... )
    """

    vendor_name: str = Field(..., description="Vendor name")
    entities: list[VendorEntity] = Field(
        ..., description="All entities (max 100)", max_length=100
    )
    relationships: list[VendorRelationship] = Field(
        default_factory=list, description="All relationships (max 500)", max_length=500
    )
    statistics: VendorStatistics = Field(..., description="Vendor graph statistics")

    @field_validator("entities", mode="after")
    @classmethod
    def validate_entity_count(cls, v: list[VendorEntity]) -> list[VendorEntity]:
        """Validate that entities list has at most 100 items.

        Args:
            v: List of vendor entities

        Returns:
            Validated entities list

        Raises:
            ValueError: If more than 100 entities
        """
        if len(v) > 100:
            raise ValueError("entities must contain at most 100 items")
        return v

    @field_validator("relationships", mode="after")
    @classmethod
    def validate_relationship_count(
        cls, v: list[VendorRelationship]
    ) -> list[VendorRelationship]:
        """Validate that relationships list has at most 500 items.

        Args:
            v: List of vendor relationships

        Returns:
            Validated relationships list

        Raises:
            ValueError: If more than 500 relationships
        """
        if len(v) > 500:
            raise ValueError("relationships must contain at most 500 items")
        return v


class AuthenticationError(BaseModel):
    """Authentication failure response.

    Example:
        >>> error = AuthenticationError(
        ...     error_code="AUTH_INVALID_KEY",
        ...     message="API key is invalid or expired",
        ...     details="Key expired on 2025-11-01"
        ... )
    """

    error_code: str = Field(..., description="Error code (e.g., AUTH_INVALID_KEY)")
    message: str = Field(..., description="Actionable error message")
    details: str | None = Field(default=None, description="Additional error details")


class AuthenticationConfig(BaseModel):
    """Authentication configuration.

    Example:
        >>> config = AuthenticationConfig(
        ...     max_auth_attempts=5,
        ...     token_expiry_seconds=3600,
        ...     rate_limit_per_minute=100
        ... )
    """

    max_auth_attempts: int = Field(
        default=5, description="Max attempts before lockout", ge=1
    )
    token_expiry_seconds: int = Field(
        default=3600, description="Token lifetime in seconds", ge=1
    )
    rate_limit_per_minute: int = Field(
        default=100, description="Request limit per minute", ge=1
    )
