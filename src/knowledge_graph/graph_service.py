"""Graph query service with integrated LRU cache for knowledge graph."""

from __future__ import annotations

from typing import Optional, List, Any
from uuid import UUID
import logging

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity
from src.knowledge_graph.cache_config import CacheConfig

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Graph query service with integrated LRU cache.

    Provides high-level interface for entity and relationship queries:
    - Checks cache first for hot path optimization
    - Falls back to database for cache misses
    - Manages cache invalidation on writes

    Expected performance:
    - Cache hit: 1-2 microseconds (in-memory OrderedDict lookup)
    - Cache miss + DB query: 5-20ms (normalized schema with indexes)
    - Overall with >80% hit rate: P95 <10ms for 1-hop queries
    """

    def __init__(
        self,
        db_session: Any,
        cache: Optional[KnowledgeGraphCache] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Initialize graph service with database session and optional cache.

        Args:
            db_session: SQLAlchemy session for database access
            cache: KnowledgeGraphCache instance (optional, will create if not provided)
            cache_config: CacheConfig instance (optional, uses defaults if not provided)
        """
        self._db_session: Any = db_session

        # Initialize cache if not provided
        if cache is None:
            config = cache_config if cache_config is not None else CacheConfig()
            self._cache = KnowledgeGraphCache(
                max_entities=config.max_entities,
                max_relationship_caches=config.max_relationship_caches,
            )
        else:
            self._cache = cache

        logger.info(
            f"Initialized KnowledgeGraphService with cache "
            f"(max_entities={self._cache.max_entities}, "
            f"max_relationships={self._cache.max_relationship_caches})"
        )

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity by ID (checks cache first).

        Query flow:
        1. Check entity cache
        2. If hit: return cached entity
        3. If miss: query database, cache result, return
        4. Expected latency: <2us cache hit, 5-10ms cache miss

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity object if found, None otherwise
        """
        # Check cache first
        cached = self._cache.get_entity(entity_id)
        if cached is not None:
            return cached

        # Cache miss: query database (stub for now)
        # In real implementation, would query normalized schema:
        # SELECT id, text, type, confidence, mention_count FROM entities WHERE id = ?
        entity = self._query_entity_from_db(entity_id)

        if entity is not None:
            # Cache result for future queries
            self._cache.set_entity(entity)

        return entity

    def traverse_1hop(
        self, entity_id: UUID, rel_type: str
    ) -> List[Entity]:
        """Traverse 1-hop relationships (checks cache first).

        Query flow:
        1. Check relationship cache (entity_id, rel_type)
        2. If hit: return cached entities
        3. If miss: query database, cache result, return
        4. Expected latency: <2us cache hit, 10-20ms cache miss

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse
                (e.g., 'hierarchical', 'mentions-in-document', 'similar-to')

        Returns:
            List of related entities (empty list if none found)
        """
        # Check cache first
        cached = self._cache.get_relationships(entity_id, rel_type)
        if cached is not None:
            return cached

        # Cache miss: query database (stub for now)
        # In real implementation, would query:
        # SELECT e.* FROM entities e
        # JOIN relationships r ON e.id = r.target_entity_id
        # WHERE r.source_entity_id = ? AND r.relationship_type = ?
        # ORDER BY r.confidence DESC
        entities = self._query_relationships_from_db(entity_id, rel_type)

        # Cache result for future queries
        if entities:
            self._cache.set_relationships(entity_id, rel_type, entities)

        return entities

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity cache on write (call after entity update).

        Args:
            entity_id: Entity UUID to invalidate
        """
        self._cache.invalidate_entity(entity_id)
        logger.debug(f"Invalidated entity {entity_id} from cache")

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate relationship cache on write (call after relationship update).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate
        """
        self._cache.invalidate_relationships(entity_id, rel_type)
        logger.debug(f"Invalidated relationships {entity_id}/{rel_type} from cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with hits, misses, evictions, size, and hit rate
        """
        stats = self._cache.stats()
        total_accesses = stats.hits + stats.misses
        hit_rate = (stats.hits / total_accesses * 100) if total_accesses > 0 else 0.0

        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "evictions": stats.evictions,
            "size": stats.size,
            "max_size": stats.max_size,
            "hit_rate_percent": hit_rate,
        }

    # Private methods (database query stubs)

    def _query_entity_from_db(self, entity_id: UUID) -> Optional[Entity]:
        """Query entity from database.

        Stub implementation - to be replaced with actual database query.
        Expected schema:
        - SELECT id, text, type, confidence, mention_count FROM entities WHERE id = ?

        Args:
            entity_id: Entity UUID to query

        Returns:
            Entity object if found, None otherwise
        """
        # Placeholder: actual implementation would query database
        return None

    def _query_relationships_from_db(
        self, entity_id: UUID, rel_type: str
    ) -> List[Entity]:
        """Query 1-hop relationships from database.

        Stub implementation - to be replaced with actual database query.
        Expected schema:
        - SELECT e.id, e.text, e.type, e.confidence, e.mention_count
          FROM entities e
          JOIN relationships r ON e.id = r.target_entity_id
          WHERE r.source_entity_id = ? AND r.relationship_type = ?
          ORDER BY r.confidence DESC

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse

        Returns:
            List of related entities
        """
        # Placeholder: actual implementation would query database
        return []
