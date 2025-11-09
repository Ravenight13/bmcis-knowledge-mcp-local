"""Type stubs for cache configuration."""

from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for KnowledgeGraphCache.

    Attributes:
        max_entities: Maximum entity entries in cache (default 5,000)
        max_relationship_caches: Maximum relationship cache entries (default 10,000)
        enable_metrics: Whether to track cache metrics (default True)
    """

    max_entities: int = 5000
    max_relationship_caches: int = 10000
    enable_metrics: bool = True
