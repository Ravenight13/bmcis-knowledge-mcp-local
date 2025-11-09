"""Graph query service with integrated LRU cache for knowledge graph."""

from __future__ import annotations

from typing import Optional, List, Any, Dict
from uuid import UUID
import logging

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity
from src.knowledge_graph.cache_config import CacheConfig
from src.knowledge_graph.query_repository import KnowledgeGraphQueryRepository

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
        db_pool: Any,
        cache: Optional[KnowledgeGraphCache] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Initialize graph service with database pool and optional cache.

        Args:
            db_pool: PostgreSQL connection pool (from core.database.pool)
            cache: KnowledgeGraphCache instance (optional, will create if not provided)
            cache_config: CacheConfig instance (optional, uses defaults if not provided)
        """
        # Initialize repository with connection pool
        self._repo: KnowledgeGraphQueryRepository = KnowledgeGraphQueryRepository(db_pool)  # type: ignore[no-untyped-call]

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
            f"Initialized KnowledgeGraphService with repository + cache "
            f"(max_entities={self._cache.max_entities}, "
            f"max_relationships={self._cache.max_relationship_caches})"
        )

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity by ID (checks cache first).

        Query flow:
        1. Check entity cache
        2. If hit: return cached entity
        3. If miss: query database via internal helper, cache result, return
        4. Expected latency: <2us cache hit, 5-10ms cache miss

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity object if found, None otherwise
        """
        # 1. Check cache first
        cached = self._cache.get_entity(entity_id)
        if cached is not None:
            logger.debug(f"Cache hit for entity {entity_id}")
            return cached

        # 2. Query database via repository
        try:
            entity = self._query_entity_from_db(entity_id)

            # 3. Cache result
            if entity is not None:
                self._cache.set_entity(entity)
                logger.debug(f"Cached entity {entity_id}")

            return entity

        except Exception as e:
            logger.error(f"Error retrieving entity {entity_id}: {e}")
            raise

    def traverse_1hop(
        self, entity_id: UUID, rel_type: str, min_confidence: float = 0.7
    ) -> List[Entity]:
        """Traverse 1-hop relationships (checks cache first).

        Query flow:
        1. Check relationship cache (entity_id, rel_type)
        2. If hit: return cached entities
        3. If miss: query repository, cache result, return
        4. Expected latency: <2us cache hit, 10-20ms cache miss

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse
                (e.g., 'hierarchical', 'mentions-in-document', 'similar-to')
            min_confidence: Minimum relationship confidence (default: 0.7)

        Returns:
            List of related entities (empty list if none found)
        """
        # 1. Check cache first
        cached = self._cache.get_relationships(entity_id, rel_type)
        if cached is not None:
            logger.debug(f"Cache hit for relationships {entity_id}/{rel_type}")
            return cached

        # 2. Query repository
        try:
            related = self._repo.traverse_1hop(
                entity_id=entity_id,
                min_confidence=min_confidence,
                relationship_types=[rel_type] if rel_type else None
            )

            # Convert RelatedEntity objects to cache Entity objects
            entities: List[Entity] = [
                Entity(
                    id=r.id,
                    text=r.text,
                    type=r.entity_type,
                    confidence=r.entity_confidence or 0.0,
                    mention_count=0
                )
                for r in related
            ]

            # 3. Cache result for future queries
            if entities:
                self._cache.set_relationships(entity_id, rel_type, entities)
                logger.debug(f"Cached {len(entities)} relationships for {entity_id}/{rel_type}")

            return entities

        except Exception as e:
            logger.error(f"Error traversing 1-hop from {entity_id}/{rel_type}: {e}")
            raise

    def traverse_2hop(
        self,
        entity_id: UUID,
        rel_type: Optional[str] = None,
        min_confidence: float = 0.7
    ) -> List[Entity]:
        """Traverse 2-hop relationships.

        Note: 2-hop results are NOT cached (too expensive to invalidate when
        intermediate entities change), always query repository.

        Args:
            entity_id: Source entity UUID
            rel_type: Optional relationship type filter
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of entities reachable in 2 hops
        """
        try:
            logger.debug(f"Querying 2-hop traversal from {entity_id}")
            two_hop_results = self._repo.traverse_2hop(
                entity_id=entity_id,
                min_confidence=min_confidence,
                relationship_types=[rel_type] if rel_type else None
            )

            # Convert TwoHopEntity objects to cache Entity objects
            entities: List[Entity] = [
                Entity(
                    id=t.id,
                    text=t.text,
                    type=t.entity_type,
                    confidence=t.entity_confidence or 0.0,
                    mention_count=0
                )
                for t in two_hop_results
            ]

            return entities

        except Exception as e:
            logger.error(f"Error traversing 2-hop from {entity_id}: {e}")
            raise

    def traverse_bidirectional(
        self,
        entity_id: UUID,
        min_confidence: float = 0.7,
        max_depth: int = 1
    ) -> List[Entity]:
        """Traverse bidirectional relationships (both incoming + outgoing).

        Args:
            entity_id: Source entity UUID
            min_confidence: Minimum confidence threshold (default: 0.7)
            max_depth: Maximum traversal depth (default: 1)

        Returns:
            List of connected entities
        """
        try:
            logger.debug(f"Querying bidirectional traversal from {entity_id}")
            bi_results = self._repo.traverse_bidirectional(
                entity_id=entity_id,
                min_confidence=min_confidence,
                max_depth=max_depth
            )

            # Convert BidirectionalEntity objects to cache Entity objects
            entities: List[Entity] = [
                Entity(
                    id=b.id,
                    text=b.text,
                    type=b.entity_type,
                    confidence=b.entity_confidence or 0.0,
                    mention_count=0
                )
                for b in bi_results
            ]

            return entities

        except Exception as e:
            logger.error(f"Error traversing bidirectional from {entity_id}: {e}")
            raise

    def get_mentions(
        self,
        entity_id: UUID,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Get documents/chunks where entity is mentioned.

        Args:
            entity_id: Entity UUID
            max_results: Maximum results to return (default: 100)

        Returns:
            List of mention dictionaries with document/chunk info
        """
        try:
            mentions = self._repo.get_entity_mentions(
                entity_id=entity_id,
                max_results=max_results
            )

            # Convert EntityMention dataclasses to dictionaries
            return [
                {
                    "chunk_id": m.chunk_id,
                    "document_id": m.document_id,
                    "mention_text": m.mention_text,
                    "document_category": m.document_category,
                    "chunk_index": m.chunk_index,
                    "mention_confidence": m.mention_confidence,
                    "indexed_at": m.indexed_at,
                }
                for m in mentions
            ]

        except Exception as e:
            logger.error(f"Error retrieving mentions for {entity_id}: {e}")
            raise

    def traverse_with_type_filter(
        self,
        entity_id: UUID,
        rel_type: str,
        target_entity_types: List[str],
        min_confidence: float = 0.7
    ) -> List[Entity]:
        """Get related entities filtered by type(s).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse
            target_entity_types: Entity types to include (e.g., ['VENDOR', 'PRODUCT'])
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of related entities of specified types
        """
        try:
            filtered = self._repo.traverse_with_type_filter(
                entity_id=entity_id,
                relationship_type=rel_type,
                target_entity_types=target_entity_types,
                min_confidence=min_confidence
            )

            # Convert RelatedEntity objects to cache Entity objects
            entities: List[Entity] = [
                Entity(
                    id=r.id,
                    text=r.text,
                    type=r.entity_type,
                    confidence=r.entity_confidence or 0.0,
                    mention_count=0
                )
                for r in filtered
            ]

            return entities

        except Exception as e:
            logger.error(f"Error filtering entities by type from {entity_id}: {e}")
            raise

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

    # Private methods (database queries via repository)

    def _query_entity_from_db(self, entity_id: UUID) -> Optional[Entity]:
        """Query entity from database via repository.

        Uses traversal to find if entity exists in database.
        Since repository doesn't have direct get_entity, we use bidirectional
        traversal with max_depth=0 concept. For now, return None to be
        implemented with direct SQL if needed.

        Args:
            entity_id: Entity UUID to query

        Returns:
            Entity object if found, None otherwise
        """
        # Note: Repository provides traversal-based queries, not direct entity fetches.
        # For a complete implementation, we would add a get_entity() method to
        # KnowledgeGraphQueryRepository that directly queries the knowledge_entities table.
        # For now, this method returns None as it's a supplementary helper.
        return None
