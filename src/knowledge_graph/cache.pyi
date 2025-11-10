"""Type stubs for KnowledgeGraphCache - LRU cache for entity relationships."""

from typing import Optional, List, Dict, Any
from uuid import UUID
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock


@dataclass
class Entity:
    """Entity object with full metadata."""
    id: UUID
    text: str
    type: str
    confidence: float
    mention_count: int


@dataclass
class CacheStats:
    """Cache statistics tracking hits, misses, and evictions."""
    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int


class KnowledgeGraphCache:
    """LRU cache for knowledge graph entities and relationships.

    Stores:
    1. Entity objects with full metadata
    2. 1-hop relationship traversals (entity_id + relationship_type -> [entities])
    3. Entity mention counts

    Features:
    - LRU eviction using OrderedDict
    - Thread-safe access via Lock
    - Metrics tracking (hits/misses/evictions)
    - Cache invalidation for consistency
    """

    max_entities: int
    max_relationship_caches: int
    _max_entities: int
    _max_relationship_caches: int

    def __init__(self, max_entities: int = 5000, max_relationship_caches: int = 10000) -> None:
        """Initialize cache with size limits.

        Args:
            max_entities: Maximum entity entries in cache (default 5,000)
            max_relationship_caches: Maximum relationship cache entries (default 10,000)
        """
        ...

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get cached entity by ID.

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity if found in cache, None otherwise
        """
        ...

    def set_entity(self, entity: Entity) -> None:
        """Cache entity object.

        Args:
            entity: Entity object to cache
        """
        ...

    def get_relationships(
        self, entity_id: UUID, rel_type: str
    ) -> Optional[List[Entity]]:
        """Get cached 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type (e.g., 'hierarchical', 'mentions-in-document')

        Returns:
            List of related entities if cached, None otherwise
        """
        ...

    def set_relationships(
        self, entity_id: UUID, rel_type: str, entities: List[Entity]
    ) -> None:
        """Cache 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type
            entities: List of related entities
        """
        ...

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity and all outbound 1-hop caches.

        Args:
            entity_id: Entity UUID to invalidate
        """
        ...

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate specific relationship cache.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate
        """
        ...

    def clear(self) -> None:
        """Clear all cache entries (entity and relationship)."""
        ...

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hits, misses, evictions, and current size
        """
        ...
