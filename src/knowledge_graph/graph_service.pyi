"""Type stubs for KnowledgeGraphService - Graph query service with cache integration."""

from typing import Optional, List, Any, Dict
from uuid import UUID


class KnowledgeGraphService:
    """Graph query service with integrated LRU cache.

    Provides high-level interface for entity and relationship queries:
    - Checks cache first for hot path optimization
    - Falls back to database for cache misses
    - Manages cache invalidation on writes

    Expected performance:
    - Cache hit (1-2 microseconds)
    - Cache miss + DB query (5-20ms for normalized schema)
    """

    def __init__(self, db_pool: Any, cache: Any = None, cache_config: Any = None) -> None:
        """Initialize graph service with database pool and optional cache.

        Args:
            db_pool: PostgreSQL connection pool (from core.database.pool)
            cache: KnowledgeGraphCache instance (optional, will create if not provided)
            cache_config: CacheConfig instance (optional, uses defaults if not provided)
        """
        ...

    def get_entity(self, entity_id: UUID) -> Optional[Any]:
        """Get entity by ID (checks cache first).

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity object if found, None otherwise
        """
        ...

    def traverse_1hop(
        self, entity_id: UUID, rel_type: str, min_confidence: float = 0.7
    ) -> List[Any]:
        """Traverse 1-hop relationships (checks cache first).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse
            min_confidence: Minimum relationship confidence (default: 0.7)

        Returns:
            List of related entities
        """
        ...

    def traverse_2hop(
        self, entity_id: UUID, rel_type: Optional[str] = None, min_confidence: float = 0.7
    ) -> List[Any]:
        """Traverse 2-hop relationships.

        Args:
            entity_id: Source entity UUID
            rel_type: Optional relationship type filter
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of entities reachable in 2 hops
        """
        ...

    def traverse_bidirectional(
        self, entity_id: UUID, min_confidence: float = 0.7, max_depth: int = 1
    ) -> List[Any]:
        """Traverse bidirectional relationships (both incoming + outgoing).

        Args:
            entity_id: Source entity UUID
            min_confidence: Minimum confidence threshold (default: 0.7)
            max_depth: Maximum traversal depth (default: 1)

        Returns:
            List of connected entities
        """
        ...

    def get_mentions(self, entity_id: UUID, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get documents/chunks where entity is mentioned.

        Args:
            entity_id: Entity UUID
            max_results: Maximum results to return (default: 100)

        Returns:
            List of mention dictionaries with document/chunk info
        """
        ...

    def traverse_with_type_filter(
        self,
        entity_id: UUID,
        rel_type: str,
        target_entity_types: List[str],
        min_confidence: float = 0.7,
    ) -> List[Any]:
        """Get related entities filtered by type(s).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse
            target_entity_types: Entity types to include (e.g., ['VENDOR', 'PRODUCT'])
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of related entities of specified types
        """
        ...

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity cache on write (call after entity update).

        Args:
            entity_id: Entity UUID to invalidate
        """
        ...

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate relationship cache on write (call after relationship update).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate
        """
        ...

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with hits, misses, evictions, size, and hit rate
        """
        ...
