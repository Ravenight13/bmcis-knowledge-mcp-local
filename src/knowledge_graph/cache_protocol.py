"""Cache interface protocol for dependency injection.

This module defines the CacheProtocol that all cache implementations must follow.
Using Python's Protocol (structural subtyping), any class implementing these methods
is compatible, enabling flexible dependency injection.

Benefits:
- Swap cache implementations (LRU → Redis) without code changes
- Inject mock caches for testing
- Follows Dependency Inversion Principle (depend on abstractions)
- Type-safe with mypy --strict compliance

Example:
    # Default LRU cache
    service = KnowledgeGraphService(db_pool)

    # Custom cache configuration
    cache = KnowledgeGraphCache(max_entities=10000)
    service = KnowledgeGraphService(db_pool, cache=cache)

    # Mock cache for testing
    mock_cache = MockCache()
    service = KnowledgeGraphService(db_pool, cache=mock_cache)
    assert mock_cache.get_calls  # Verify cache was used
"""

from __future__ import annotations

from typing import Protocol, Optional, List, TYPE_CHECKING
from uuid import UUID

# Import Entity and CacheStats from cache module to avoid duplication
from src.knowledge_graph.cache import Entity, CacheStats


class CacheProtocol(Protocol):
    """Protocol defining cache interface for knowledge graph operations.

    All cache implementations (LRU, Redis, Mock) must implement these methods.
    Python's Protocol uses structural subtyping - no explicit inheritance needed.

    Methods:
        Entity Operations:
            - get_entity: Retrieve cached entity by ID
            - set_entity: Store entity in cache
            - invalidate_entity: Remove entity from cache

        Relationship Operations:
            - get_relationships: Retrieve cached 1-hop relationships
            - set_relationships: Store 1-hop relationships
            - invalidate_relationships: Remove relationship cache

        Management Operations:
            - clear: Clear entire cache
            - stats: Get cache statistics

    Performance Expectations:
        - get_entity/get_relationships: <2μs (in-memory)
        - set_entity/set_relationships: <5μs (in-memory)
        - Thread-safe for concurrent access
    """

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get cached entity by ID.

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity if found in cache, None otherwise

        Thread-safety:
            Must be thread-safe for concurrent reads
        """
        ...

    def set_entity(self, entity: Entity) -> None:
        """Cache entity object.

        Args:
            entity: Entity to store in cache

        Behavior:
            - Overwrites existing entry if present
            - May evict oldest entry if cache is full (LRU)

        Thread-safety:
            Must be thread-safe for concurrent writes
        """
        ...

    def get_relationships(
        self,
        entity_id: UUID,
        rel_type: str
    ) -> Optional[List[Entity]]:
        """Get cached 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type (e.g., 'hierarchical', 'mentions-in-document')

        Returns:
            List of related entities if cached, None otherwise

        Cache key:
            (entity_id, rel_type) uniquely identifies relationship cache
        """
        ...

    def set_relationships(
        self,
        entity_id: UUID,
        rel_type: str,
        entities: List[Entity]
    ) -> None:
        """Cache 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type
            entities: List of related entities

        Behavior:
            - Stores list of entities for (entity_id, rel_type) cache key
            - May evict oldest entry if cache is full
        """
        ...

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity and all associated caches.

        Args:
            entity_id: Entity UUID to invalidate

        Behavior:
            - Removes entity from entity cache
            - Removes all outbound relationship caches for this entity
            - May remove inbound relationship caches (implementation-specific)

        Use case:
            Call after updating entity in database to maintain consistency
        """
        ...

    def invalidate_relationships(
        self,
        entity_id: UUID,
        rel_type: str
    ) -> None:
        """Invalidate specific relationship cache.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate

        Use case:
            Call after modifying relationships in database
        """
        ...

    def clear(self) -> None:
        """Clear entire cache (all entities and relationships).

        Use case:
            - Testing cleanup
            - Emergency cache flush
            - Bulk data updates
        """
        ...

    def stats(self) -> CacheStats:
        """Get cache statistics for monitoring.

        Returns:
            CacheStats with hits, misses, evictions, size, and capacity

        Use case:
            - Performance monitoring
            - Cache tuning
            - Hit rate analysis
        """
        ...
