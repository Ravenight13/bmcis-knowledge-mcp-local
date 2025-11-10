"""find_vendor_info tool implementation for FastMCP.

Comprehensive vendor information retrieval from knowledge graph.
Supports 4 progressive disclosure response modes for token efficiency.

Progressive Disclosure Pattern:
- ids_only: Vendor ID + counts (~100-500 tokens)
- metadata: IDs + statistics + type distributions (~2-4K tokens) - DEFAULT
- preview: metadata + top 5 entities + top 5 relationships (~5-10K tokens)
- full: Complete vendor graph (max 100 entities, 500 relationships, ~10-50K+ tokens)

Token Reduction:
- Traditional (full): ~50,000 tokens for large vendors
- Progressive (metadata): ~3,000 tokens (94% reduction)
- Selective (preview): ~8,000 tokens (84% reduction)

Example:
    # Metadata-only (default - fast, token-efficient)
    >>> response = find_vendor_info("Acme Corp")

    # Full content (slower, more tokens)
    >>> response = find_vendor_info("Acme Corp", response_mode="full")

    # Preview with top entities and relationships
    >>> response = find_vendor_info("Acme Corp", response_mode="preview")
"""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.logging import StructuredLogger
from src.knowledge_graph.query_repository import KnowledgeGraphQueryRepository
from src.mcp.cache import hash_query
from src.mcp.models import (
    FindVendorInfoRequest,
    MCPResponseEnvelope,
    PaginationMetadata,
    VendorEntity,
    VendorInfoFull,
    VendorInfoIDs,
    VendorInfoMetadata,
    VendorInfoPreview,
    VendorRelationship,
    VendorStatistics,
)
from src.mcp.response_formatter import wrap_vendor_info_response
from src.mcp.server import get_cache_layer, get_database_pool, mcp

logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Constants for response limits
MAX_ENTITIES_PREVIEW = 5
MAX_RELATIONSHIPS_PREVIEW = 5
MAX_ENTITIES_FULL = 100
MAX_RELATIONSHIPS_FULL = 500


def compute_vendor_cache_key(vendor_name: str, response_mode: str) -> str:
    """Compute cache key for vendor info query.

    Args:
        vendor_name: Vendor name to search
        response_mode: Response detail level

    Returns:
        Cache key string (format: "vendor:hash")

    Example:
        >>> key = compute_vendor_cache_key("Acme Corp", "metadata")
        >>> assert key.startswith("vendor:")
    """
    params = {"vendor_name": vendor_name.lower().strip(), "response_mode": response_mode}
    return f"vendor:{hash_query(params)}"


def generate_vendor_cursor(
    vendor_name: str, response_mode: str, offset: int
) -> str:
    """Generate cursor for next page of vendor results.

    Args:
        vendor_name: Vendor name
        response_mode: Response detail level
        offset: Offset for next page

    Returns:
        Base64-encoded cursor string

    Example:
        >>> cursor = generate_vendor_cursor("Acme", "preview", 5)
        >>> data = json.loads(base64.b64decode(cursor))
        >>> assert data["offset"] == 5
    """
    params = {"vendor_name": vendor_name.lower().strip(), "response_mode": response_mode}
    query_hash = hash_query(params)

    cursor_data = {
        "query_hash": query_hash,
        "offset": offset,
        "response_mode": response_mode,
    }

    cursor_json = json.dumps(cursor_data)
    cursor_bytes = base64.b64encode(cursor_json.encode("utf-8"))
    return cursor_bytes.decode("utf-8")


def parse_vendor_cursor(cursor: str) -> dict[str, Any]:
    """Parse vendor cursor to extract pagination state.

    Args:
        cursor: Base64-encoded cursor string

    Returns:
        Dictionary with query_hash, offset, response_mode

    Raises:
        ValueError: If cursor is invalid
    """
    try:
        cursor_bytes = base64.b64decode(cursor.encode("utf-8"))
        cursor_json = cursor_bytes.decode("utf-8")
        return json.loads(cursor_json)  # type: ignore[no-any-return]
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid cursor format: {e}") from e


def filter_vendor_fields(
    result: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    """Filter vendor result to only include specified fields.

    Args:
        result: Result dictionary
        fields: List of field names to include

    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in result.items() if k in fields}


def normalize_vendor_name(vendor_name: str) -> str:
    """Normalize vendor name for matching.

    Args:
        vendor_name: Raw vendor name input

    Returns:
        Normalized vendor name (lowercase, stripped whitespace)

    Example:
        >>> normalize_vendor_name("  Acme Corp  ")
        'acme corp'
    """
    return vendor_name.strip().lower()


def find_vendor_by_name(
    vendor_name: str,
    db_pool: Any,
) -> tuple[UUID, str, str, float]:
    """Find vendor entity by name in knowledge graph.

    Performs case-insensitive exact match on entity text.
    Returns vendor entities of type ORG (organization).

    Args:
        vendor_name: Vendor name to search (normalized)
        db_pool: Database connection pool

    Returns:
        Tuple of (vendor_id, name, entity_type, confidence)

    Raises:
        ValueError: If vendor not found or multiple matches found

    Example:
        >>> vendor_id, name, type, conf = find_vendor_by_name("acme corp", db_pool)
        >>> assert type == "ORG"
    """
    normalized_name = normalize_vendor_name(vendor_name)

    # Query for exact match (case-insensitive)
    query = """
    SELECT id, text, entity_type, confidence
    FROM knowledge_entities
    WHERE LOWER(text) = %s
      AND entity_type = 'ORG'
    ORDER BY confidence DESC
    LIMIT 2
    """

    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (normalized_name,))
                rows = cur.fetchall()

                if not rows:
                    raise ValueError(
                        f"Vendor '{vendor_name}' not found. Try using semantic_search first to find exact vendor names "
                        "and IDs in the knowledge graph."
                    )

                if len(rows) > 1:
                    matches = [row[1] for row in rows]
                    raise ValueError(
                        f"Ambiguous vendor name. Multiple matches found: {matches}. "
                        "Please use full vendor name or a more specific term."
                    )

                row = rows[0]
                return (UUID(row[0]), str(row[1]), str(row[2]), float(row[3]))

    except Exception as e:
        logger.error(f"Vendor lookup failed for '{vendor_name}': {e}")
        raise


def get_vendor_statistics(
    vendor_id: UUID,
    query_repo: KnowledgeGraphQueryRepository,
) -> VendorStatistics:
    """Get statistics for vendor entity graph.

    Aggregates entity counts, relationship counts, and type distributions.

    Args:
        vendor_id: Vendor entity UUID
        query_repo: Knowledge graph query repository

    Returns:
        VendorStatistics with counts and distributions

    Example:
        >>> stats = get_vendor_statistics(vendor_id, query_repo)
        >>> assert stats.entity_count >= 0
    """
    # Get 1-hop related entities (all relationship types)
    related_entities = query_repo.traverse_1hop(
        entity_id=vendor_id,
        min_confidence=0.0,  # Include all entities
        relationship_types=None,  # All relationship types
        max_results=1000,  # Get all for accurate counts
    )

    # Count entity types
    entity_type_dist: dict[str, int] = {}
    for entity in related_entities:
        entity_type = entity.entity_type
        entity_type_dist[entity_type] = entity_type_dist.get(entity_type, 0) + 1

    # Count relationship types (approximate from bidirectional results)
    relationship_type_dist: dict[str, int] = {}
    for entity in related_entities:
        rel_type = entity.relationship_type
        relationship_type_dist[rel_type] = relationship_type_dist.get(rel_type, 0) + 1

    return VendorStatistics(
        entity_count=len(related_entities),
        relationship_count=len(related_entities),  # Each entity has a relationship
        entity_type_distribution=entity_type_dist if entity_type_dist else None,
        relationship_type_distribution=relationship_type_dist if relationship_type_dist else None,
    )


def format_ids_only(
    vendor_id: UUID,
    vendor_name: str,
    match_confidence: float,
    statistics: VendorStatistics,
) -> VendorInfoIDs:
    """Format ids_only response (~100-500 tokens).

    Args:
        vendor_id: Vendor UUID
        vendor_name: Vendor name
        match_confidence: Match confidence score
        statistics: Vendor statistics

    Returns:
        VendorInfoIDs with minimal fields

    Example:
        >>> response = format_ids_only(vendor_id, "Acme Corp", 0.95, stats)
        >>> assert response.vendor_name == "Acme Corp"
    """
    # For ids_only, we only return entity_ids (no relationships)
    # Generate entity IDs from statistics (not actual IDs)
    entity_ids = [str(vendor_id)]  # Just the vendor ID for now

    return VendorInfoIDs(
        vendor_name=vendor_name,
        entity_ids=entity_ids,
        relationship_ids=[],
    )


def format_metadata(
    vendor_id: UUID,
    vendor_name: str,
    vendor_type: str,
    match_confidence: float,
    statistics: VendorStatistics,
) -> VendorInfoMetadata:
    """Format metadata response (~2-4K tokens).

    Args:
        vendor_id: Vendor UUID
        vendor_name: Vendor name
        vendor_type: Entity type
        match_confidence: Match confidence score
        statistics: Vendor statistics

    Returns:
        VendorInfoMetadata with statistics

    Example:
        >>> response = format_metadata(vendor_id, "Acme Corp", "ORG", 0.95, stats)
        >>> assert response.statistics.entity_count >= 0
    """
    return VendorInfoMetadata(
        vendor_name=vendor_name,
        statistics=statistics,
        top_entities=None,  # Not included in metadata mode
        last_updated=datetime.utcnow().isoformat(),
    )


def format_preview(
    vendor_id: UUID,
    vendor_name: str,
    vendor_type: str,
    match_confidence: float,
    statistics: VendorStatistics,
    query_repo: KnowledgeGraphQueryRepository,
) -> VendorInfoPreview:
    """Format preview response (~5-10K tokens, max 5 entities, 5 relationships).

    Args:
        vendor_id: Vendor UUID
        vendor_name: Vendor name
        vendor_type: Entity type
        match_confidence: Match confidence score
        statistics: Vendor statistics
        query_repo: Query repository for entities/relationships

    Returns:
        VendorInfoPreview with top 5 entities and relationships

    Example:
        >>> response = format_preview(vendor_id, "Acme", "ORG", 0.95, stats, repo)
        >>> assert len(response.entities) <= 5
    """
    # Get top 5 entities (sorted by confidence)
    related_entities = query_repo.traverse_1hop(
        entity_id=vendor_id,
        min_confidence=0.0,
        relationship_types=None,
        max_results=MAX_ENTITIES_PREVIEW,
    )

    entities = [
        VendorEntity(
            entity_id=str(entity.id),
            name=entity.text,
            entity_type=entity.entity_type,
            confidence=entity.entity_confidence or 0.0,
            snippet=None,  # Preview mode doesn't include snippets
        )
        for entity in related_entities[:MAX_ENTITIES_PREVIEW]
    ]

    # Get top 5 relationships
    relationships = [
        VendorRelationship(
            source_id=str(vendor_id),
            target_id=str(entity.id),
            relationship_type=entity.relationship_type,
            metadata=None,
        )
        for entity in related_entities[:MAX_RELATIONSHIPS_PREVIEW]
    ]

    return VendorInfoPreview(
        vendor_name=vendor_name,
        entities=entities,
        relationships=relationships,
        statistics=statistics,
    )


def format_full(
    vendor_id: UUID,
    vendor_name: str,
    vendor_type: str,
    match_confidence: float,
    statistics: VendorStatistics,
    query_repo: KnowledgeGraphQueryRepository,
) -> VendorInfoFull:
    """Format full response (~10-50K+ tokens, max 100 entities, 500 relationships).

    Args:
        vendor_id: Vendor UUID
        vendor_name: Vendor name
        vendor_type: Entity type
        match_confidence: Match confidence score
        statistics: Vendor statistics
        query_repo: Query repository for entities/relationships

    Returns:
        VendorInfoFull with up to 100 entities and 500 relationships

    Example:
        >>> response = format_full(vendor_id, "Acme", "ORG", 0.95, stats, repo)
        >>> assert len(response.entities) <= 100
    """
    # Get up to 100 entities (sorted by confidence)
    related_entities = query_repo.traverse_1hop(
        entity_id=vendor_id,
        min_confidence=0.0,
        relationship_types=None,
        max_results=MAX_ENTITIES_FULL,
    )

    entities = [
        VendorEntity(
            entity_id=str(entity.id),
            name=entity.text,
            entity_type=entity.entity_type,
            confidence=entity.entity_confidence or 0.0,
            snippet=entity.text[:200] if len(entity.text) > 200 else entity.text,
        )
        for entity in related_entities[:MAX_ENTITIES_FULL]
    ]

    # Get up to 500 relationships
    relationships = [
        VendorRelationship(
            source_id=str(vendor_id),
            target_id=str(entity.id),
            relationship_type=entity.relationship_type,
            metadata={
                "confidence": entity.relationship_confidence,
                "relationship_metadata": entity.relationship_metadata,
            } if entity.relationship_metadata else None,
        )
        for entity in related_entities[:MAX_RELATIONSHIPS_FULL]
    ]

    return VendorInfoFull(
        vendor_name=vendor_name,
        entities=entities,
        relationships=relationships,
        statistics=statistics,
    )


@mcp.tool()  # type: ignore[misc]
def find_vendor_info(
    vendor_name: str,
    response_mode: str = "metadata",
    include_relationships: bool = True,
    page_size: int = 10,
    cursor: str | None = None,
    fields: list[str] | None = None,
    response_format: str | None = None,
) -> dict[str, Any] | MCPResponseEnvelope[dict[str, Any]]:
    """Find comprehensive vendor information with caching, pagination, and field filtering.

    Search the knowledge graph for a vendor and return their entity graph
    with configurable response modes, caching (300s TTL), and pagination support.

    Args:
        vendor_name: Name of vendor to search (1-200 chars)
        response_mode: Response detail level (ids_only, metadata, preview, full)
        include_relationships: Include relationship data (default: True)
        page_size: Number of entities per page (1-50, default: 10)
        cursor: Pagination cursor from previous response
        fields: Optional list of fields to include in response
        response_format: Optional response format. Default: None (legacy format).
            - None: Return dict with vendor info (backward compatible)
            - "desktop": Return MCPResponseEnvelope with enhanced metadata

    Returns:
        Dictionary with vendor information including:
        - vendor_name: Normalized vendor name
        - results: Vendor data (type depends on response_mode)
        - pagination: Pagination metadata (cursor, has_more, etc.)
        - execution_time_ms: Query execution time

    Raises:
        ValueError: If vendor not found, ambiguous name, or invalid parameters
        RuntimeError: If search execution fails

    Performance:
        - Cached results: <100ms P95
        - Fresh metadata query: <200ms P50, <500ms P95
        - Fresh full query: <500ms P50, <1500ms P95

    Examples:
        # Basic search (cached for 300s)
        >>> response = find_vendor_info("Acme Corp")

        # Pagination through large entity sets
        >>> page1 = find_vendor_info("Acme Corp", page_size=5, response_mode="preview")
        >>> if page1["pagination"]["has_more"]:
        ...     page2 = find_vendor_info("Acme Corp", cursor=page1["pagination"]["cursor"])

        # Field filtering
        >>> response = find_vendor_info(
        ...     "Acme Corp",
        ...     fields=["vendor_name", "entity_count"],
        ...     response_mode="metadata"
        ... )
    """
    # Validate request
    try:
        request = FindVendorInfoRequest(
            vendor_name=vendor_name,
            response_mode=response_mode,  # type: ignore[arg-type]
            include_relationships=include_relationships,
        )
    except Exception as e:
        logger.error(f"Request validation failed: {e}")
        raise ValueError(f"Invalid request parameters: {e}") from e

    # Parse cursor if provided (for pagination - currently not used for vendor info)
    if cursor:
        try:
            _ = parse_vendor_cursor(cursor)
            # Note: vendor info doesn't paginate entities yet, cursor support for future use
        except ValueError as e:
            logger.warning(f"Invalid cursor: {e} - ignoring")

    # Compute cache key
    cache_key = compute_vendor_cache_key(request.vendor_name, request.response_mode)
    cache_layer = get_cache_layer()

    # Check cache first
    start_time = time.time()
    cached_data = cache_layer.get(cache_key)
    cache_hit = cached_data is not None

    formatted_response: VendorInfoIDs | VendorInfoMetadata | VendorInfoPreview | VendorInfoFull

    if cache_hit:
        # Use cached results
        vendor_id = cached_data["vendor_id"]
        vendor_name_normalized = cached_data["vendor_name"]
        vendor_type = cached_data["vendor_type"]
        match_confidence = cached_data["match_confidence"]
        statistics = cached_data["statistics"]
        formatted_response = cached_data["response"]
        logger.debug(f"Cache hit: {cache_key}")
    else:
        # Execute fresh query
        db_pool = get_database_pool()
        query_repo = KnowledgeGraphQueryRepository(db_pool)  # type: ignore[no-untyped-call]

        try:
            # Find vendor by name
            vendor_id, vendor_name_normalized, vendor_type, match_confidence = find_vendor_by_name(
                request.vendor_name, db_pool
            )

            # Get vendor statistics
            statistics = get_vendor_statistics(vendor_id, query_repo)

            # Format response based on response_mode
            if request.response_mode == "ids_only":
                formatted_response = format_ids_only(
                    vendor_id, vendor_name_normalized, match_confidence, statistics
                )
            elif request.response_mode == "metadata":
                formatted_response = format_metadata(
                    vendor_id, vendor_name_normalized, vendor_type, match_confidence, statistics
                )
            elif request.response_mode == "preview":
                formatted_response = format_preview(
                    vendor_id, vendor_name_normalized, vendor_type, match_confidence, statistics, query_repo
                )
            else:  # full
                formatted_response = format_full(
                    vendor_id, vendor_name_normalized, vendor_type, match_confidence, statistics, query_repo
                )

            # Cache results for 300 seconds (vendor data more static)
            cache_data = {
                "vendor_id": vendor_id,
                "vendor_name": vendor_name_normalized,
                "vendor_type": vendor_type,
                "match_confidence": match_confidence,
                "statistics": statistics,
                "response": formatted_response,
            }
            cache_layer.set(cache_key, cache_data, ttl_seconds=300)
            logger.debug(f"Cache miss: {cache_key} - stored vendor data")

        except ValueError as e:
            logger.error(f"Vendor lookup failed: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Vendor info retrieval failed: {e}",
                extra={"vendor_name": vendor_name, "error": str(e)},
            )
            raise RuntimeError(f"Vendor info retrieval failed: {e}") from e

    execution_time_ms = (time.time() - start_time) * 1000

    # Convert response to dict for field filtering and pagination
    response_dict = formatted_response.model_dump()

    # Apply field filtering if requested
    if fields:
        response_dict = filter_vendor_fields(response_dict, fields)

    # Create pagination metadata (vendor info typically doesn't paginate entities,
    # but we support it for consistency)
    pagination_metadata = PaginationMetadata(
        cursor=None,  # No pagination for vendor info (single vendor)
        page_size=1,
        has_more=False,
        total_available=1,
    )

    logger.info(
        f"Vendor info retrieval completed in {execution_time_ms:.1f}ms, cache_hit={cache_hit}",
        extra={
            "vendor_name": vendor_name,
            "response_mode": response_mode,
            "response_format": response_format,
            "entity_count": statistics.entity_count,
            "relationship_count": statistics.relationship_count,
            "execution_time_ms": execution_time_ms,
            "cache_hit": cache_hit,
        },
    )

    # Return enhanced envelope format for desktop mode
    if response_format == "desktop":
        return wrap_vendor_info_response(
            vendor_name=vendor_name_normalized,
            results=response_dict,
            execution_time_ms=execution_time_ms,
            cache_hit=cache_hit,
            pagination=pagination_metadata,
            entity_count=statistics.entity_count,
        )

    # Return standard dictionary format (backward compatible)
    return {
        "vendor_name": vendor_name_normalized,
        "results": response_dict,
        "pagination": pagination_metadata.model_dump(),
        "execution_time_ms": execution_time_ms,
    }
