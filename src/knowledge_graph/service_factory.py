"""Factory for creating KnowledgeGraphService with different configurations.

This factory enables environment-based configuration of cache backends without
modifying application code. Supports multiple cache implementations:
- 'memory': In-memory LRU cache (KnowledgeGraphCache)
- 'redis': Redis-backed cache (future implementation)

Benefits:
- Centralized service creation logic
- Easy environment switching (dev → prod)
- Configuration-driven cache selection
- Extensible for future cache backends

Example:
    # Development environment (in-memory LRU)
    service = ServiceFactory.create_service(db_pool, cache_type='memory')

    # Production environment (Redis, when available)
    service = ServiceFactory.create_service(
        db_pool,
        cache_type='redis',
        redis_client=redis_client
    )

Usage Pattern:
    # In application initialization
    import os
    from src.knowledge_graph.service_factory import ServiceFactory

    cache_type = os.getenv('CACHE_TYPE', 'memory')
    service = ServiceFactory.create_service(db_pool, cache_type=cache_type)
"""

from __future__ import annotations

from typing import Any, Optional

from src.knowledge_graph.graph_service import KnowledgeGraphService
from src.knowledge_graph.cache import KnowledgeGraphCache
from src.knowledge_graph.cache_config import CacheConfig


class ServiceFactory:
    """Factory for creating KnowledgeGraphService with different cache backends.

    Centralized factory for instantiating graph services with environment-specific
    cache implementations. Supports multiple cache types and handles configuration.

    Methods:
        create_service: Create service with specified cache backend
        create_with_config: Create service with custom cache configuration

    Cache Types:
        - 'memory': In-memory LRU cache (default)
        - 'redis': Redis-backed distributed cache (future)

    Thread-safety:
        Factory methods are stateless and thread-safe.
    """

    @staticmethod
    def create_service(
        db_pool: Any,
        cache_type: str = 'memory',
        **kwargs: Any
    ) -> KnowledgeGraphService:
        """Create service with specified cache type.

        Args:
            db_pool: Database connection pool
            cache_type: Cache backend type ('memory' or 'redis')
            **kwargs: Additional arguments for cache creation
                - For memory: max_entities, max_relationship_caches
                - For redis: redis_client (future)

        Returns:
            KnowledgeGraphService with configured cache

        Raises:
            ValueError: If cache_type is unknown
            NotImplementedError: If cache_type is not yet implemented

        Example:
            # Default in-memory cache
            service = ServiceFactory.create_service(db_pool)

            # Custom LRU cache size
            service = ServiceFactory.create_service(
                db_pool,
                cache_type='memory',
                max_entities=10000,
                max_relationship_caches=20000
            )

            # Redis cache (when available)
            service = ServiceFactory.create_service(
                db_pool,
                cache_type='redis',
                redis_client=redis_client
            )
        """
        if cache_type == 'memory':
            # Create in-memory LRU cache
            cache_config = CacheConfig(
                max_entities=kwargs.get('max_entities', 5000),
                max_relationship_caches=kwargs.get('max_relationship_caches', 10000)
            )
            cache = KnowledgeGraphCache(
                max_entities=cache_config.max_entities,
                max_relationship_caches=cache_config.max_relationship_caches
            )
            return KnowledgeGraphService(db_pool, cache=cache)

        elif cache_type == 'redis':
            # Future: Redis cache implementation
            # from src.knowledge_graph.redis_cache import RedisCache
            # redis_client = kwargs.get('redis_client')
            # if redis_client is None:
            #     raise ValueError("redis_client required for cache_type='redis'")
            # cache = RedisCache(redis_client)
            # return KnowledgeGraphService(db_pool, cache=cache)
            raise NotImplementedError(
                "Redis cache not yet implemented. "
                "Use cache_type='memory' for in-memory LRU cache."
            )

        else:
            raise ValueError(
                f"Unknown cache_type: {cache_type!r}. "
                f"Supported types: 'memory', 'redis'"
            )

    @staticmethod
    def create_with_config(
        db_pool: Any,
        cache_config: CacheConfig
    ) -> KnowledgeGraphService:
        """Create service with custom cache configuration.

        Args:
            db_pool: Database connection pool
            cache_config: CacheConfig instance with custom settings

        Returns:
            KnowledgeGraphService with configured LRU cache

        Example:
            from src.knowledge_graph.cache_config import CacheConfig

            config = CacheConfig(
                max_entities=20000,
                max_relationship_caches=40000
            )
            service = ServiceFactory.create_with_config(db_pool, config)
        """
        cache = KnowledgeGraphCache(
            max_entities=cache_config.max_entities,
            max_relationship_caches=cache_config.max_relationship_caches
        )
        return KnowledgeGraphService(db_pool, cache=cache)

    @staticmethod
    def create_default(db_pool: Any) -> KnowledgeGraphService:
        """Create service with default configuration.

        Args:
            db_pool: Database connection pool

        Returns:
            KnowledgeGraphService with default LRU cache (5k entities, 10k relationships)

        Example:
            service = ServiceFactory.create_default(db_pool)
        """
        return KnowledgeGraphService(db_pool)


# Environment-based configuration helper
def create_from_environment(db_pool: Any) -> KnowledgeGraphService:
    """Create service based on environment variables.

    Reads cache configuration from environment:
        - CACHE_TYPE: 'memory' or 'redis' (default: 'memory')
        - CACHE_MAX_ENTITIES: Maximum entity cache size (default: 5000)
        - CACHE_MAX_RELATIONSHIPS: Maximum relationship cache size (default: 10000)

    Args:
        db_pool: Database connection pool

    Returns:
        KnowledgeGraphService configured from environment

    Example:
        # In application startup
        import os
        os.environ['CACHE_TYPE'] = 'memory'
        os.environ['CACHE_MAX_ENTITIES'] = '10000'

        service = create_from_environment(db_pool)
    """
    import os

    cache_type = os.getenv('CACHE_TYPE', 'memory')
    max_entities = int(os.getenv('CACHE_MAX_ENTITIES', '5000'))
    max_relationships = int(os.getenv('CACHE_MAX_RELATIONSHIPS', '10000'))

    return ServiceFactory.create_service(
        db_pool,
        cache_type=cache_type,
        max_entities=max_entities,
        max_relationship_caches=max_relationships
    )
