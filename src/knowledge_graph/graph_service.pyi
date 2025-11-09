"""Type stubs for KnowledgeGraphService - Graph query service with cache integration."""

from typing import Optional, List, Any
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

    def __init__(self, db_session: Any, cache: Any = None) -> None:
        """Initialize graph service with database session and optional cache.

        Args:
            db_session: SQLAlchemy session for database access
            cache: KnowledgeGraphCache instance (optional, will create if not provided)
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

    def traverse_1hop(self, entity_id: UUID, rel_type: str) -> List[Any]:
        """Traverse 1-hop relationships (checks cache first).

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to traverse

        Returns:
            List of related entities
        """
        ...
