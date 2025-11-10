"""LRU cache implementation for knowledge graph entities and relationships."""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from uuid import UUID
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


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
    1. Entity objects with full metadata (id, text, type, confidence, mention_count)
    2. 1-hop relationship traversals (entity_id + relationship_type -> [entities])
    3. Entity mention counts

    Features:
    - LRU eviction using OrderedDict (move_to_end for access tracking)
    - Thread-safe access via Lock
    - Metrics tracking (hits/misses/evictions)
    - Cache invalidation for consistency on writes

    Cache invalidation strategy:
    - When entity is written: invalidate entity entry + all outbound 1-hop caches
    - When relationship is written: invalidate outbound cache for source + inbound for target
    - Ensures consistency across normalized schema queries
    """

    def __init__(
        self,
        max_entities: int = 5000,
        max_relationship_caches: int = 10000,
    ) -> None:
        """Initialize cache with size limits.

        Args:
            max_entities: Maximum entity entries in cache (default 5,000)
            max_relationship_caches: Maximum relationship cache entries (default 10,000)
        """
        self.max_entities: int = max_entities
        self.max_relationship_caches: int = max_relationship_caches

        # Keep private versions for backwards compatibility
        self._max_entities: int = max_entities
        self._max_relationship_caches: int = max_relationship_caches

        # LRU ordered dict for entities (UUID -> Entity)
        self._entities: OrderedDict[UUID, Entity] = OrderedDict()

        # LRU ordered dict for relationships (tuple(entity_id, rel_type) -> List[Entity])
        self._relationships: OrderedDict[Tuple[UUID, str], List[Entity]] = OrderedDict()

        # Track reverse relationships for bidirectional invalidation
        # (target_entity_id, rel_type) -> set of source entity IDs
        self._reverse_relationships: Dict[Tuple[UUID, str], set[UUID]] = {}

        # Thread-safe lock for concurrent access
        self._lock: Lock = Lock()

        # Metrics tracking
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get cached entity by ID.

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity if found in cache, None otherwise
        """
        with self._lock:
            if entity_id in self._entities:
                # Move to end for LRU tracking
                self._entities.move_to_end(entity_id)
                self._hits += 1
                return self._entities[entity_id]
            else:
                self._misses += 1
                return None

    def set_entity(self, entity: Entity) -> None:
        """Cache entity object.

        Args:
            entity: Entity object to cache
        """
        with self._lock:
            # Remove if exists to reset position
            if entity.id in self._entities:
                del self._entities[entity.id]

            # Check if we need to evict
            if len(self._entities) >= self._max_entities:
                # Remove oldest (first) item
                oldest_id, _ = self._entities.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Evicted entity {oldest_id} from cache (size limit reached)")

            # Add to cache (will be at end, most recently used)
            self._entities[entity.id] = entity

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
        cache_key: Tuple[UUID, str] = (entity_id, rel_type)

        with self._lock:
            if cache_key in self._relationships:
                # Move to end for LRU tracking
                self._relationships.move_to_end(cache_key)
                self._hits += 1
                return self._relationships[cache_key]
            else:
                self._misses += 1
                return None

    def set_relationships(
        self, entity_id: UUID, rel_type: str, entities: List[Entity]
    ) -> None:
        """Cache 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type
            entities: List of related entities
        """
        cache_key: Tuple[UUID, str] = (entity_id, rel_type)

        with self._lock:
            # Remove if exists to reset position
            if cache_key in self._relationships:
                del self._relationships[cache_key]

            # Check if we need to evict
            if len(self._relationships) >= self._max_relationship_caches:
                # Remove oldest (first) item
                oldest_key, _ = self._relationships.popitem(last=False)
                oldest_id, oldest_type = oldest_key
                self._evictions += 1
                logger.debug(
                    f"Evicted relationship cache {oldest_id}/{oldest_type} "
                    f"(size limit reached)"
                )

                # Clean up reverse relationship tracking
                self._cleanup_reverse_relationships(oldest_key)

            # Add to cache (will be at end, most recently used)
            self._relationships[cache_key] = entities

            # Track reverse relationships for bidirectional invalidation
            for target_entity in entities:
                rev_key: Tuple[UUID, str] = (target_entity.id, rel_type)
                if rev_key not in self._reverse_relationships:
                    self._reverse_relationships[rev_key] = set()
                self._reverse_relationships[rev_key].add(entity_id)

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity and all outbound 1-hop caches.

        When an entity is modified, we invalidate:
        1. The entity itself from the entity cache
        2. All outbound 1-hop caches for that entity
        3. All inbound 1-hop caches (reverse relationships)

        Args:
            entity_id: Entity UUID to invalidate
        """
        with self._lock:
            # Invalidate entity entry
            if entity_id in self._entities:
                del self._entities[entity_id]

            # Invalidate all outbound 1-hop caches for this entity
            # (any cache key that starts with this entity_id)
            keys_to_delete: List[Tuple[UUID, str]] = [
                key for key in self._relationships.keys()
                if key[0] == entity_id
            ]
            for key in keys_to_delete:
                del self._relationships[key]
                self._cleanup_reverse_relationships(key)

            # Invalidate all inbound 1-hop caches (where this entity is target)
            inbound_keys: List[Tuple[UUID, str]] = [
                key for key in self._reverse_relationships.keys()
                if key[0] == entity_id
            ]
            for inbound_key in inbound_keys:
                source_ids = self._reverse_relationships.pop(inbound_key, set())
                for source_id in source_ids:
                    cache_key = (source_id, inbound_key[1])
                    if cache_key in self._relationships:
                        del self._relationships[cache_key]

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate specific relationship cache.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate
        """
        cache_key: Tuple[UUID, str] = (entity_id, rel_type)

        with self._lock:
            if cache_key in self._relationships:
                del self._relationships[cache_key]
                self._cleanup_reverse_relationships(cache_key)

    def clear(self) -> None:
        """Clear all cache entries (entity and relationship)."""
        with self._lock:
            self._entities.clear()
            self._relationships.clear()
            self._reverse_relationships.clear()
            logger.debug("Cleared all cache entries")

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hits, misses, evictions, and current size
        """
        with self._lock:
            total_size = len(self._entities) + len(self._relationships)
            max_size = self._max_entities + self._max_relationship_caches

            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=total_size,
                max_size=max_size,
            )

    def _cleanup_reverse_relationships(self, cache_key: Tuple[UUID, str]) -> None:
        """Clean up reverse relationship tracking when entry is evicted.

        Args:
            cache_key: Tuple of (entity_id, rel_type) being evicted
        """
        source_id, rel_type = cache_key
        reverse_keys_to_clean: List[Tuple[UUID, str]] = []

        for rev_key, source_ids in self._reverse_relationships.items():
            if source_id in source_ids:
                source_ids.discard(source_id)
                if not source_ids:
                    reverse_keys_to_clean.append(rev_key)

        for key in reverse_keys_to_clean:
            del self._reverse_relationships[key]
