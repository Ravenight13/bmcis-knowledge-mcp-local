"""Type stubs for MCP cache layer with TTL and LRU eviction."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

@dataclass
class CacheEntry:
    """Single cache entry with TTL metadata."""
    value: Any
    created_at: float
    ttl_seconds: int

    def is_expired(self) -> bool: ...

@dataclass
class CacheStats:
    """Cache statistics for monitoring and debugging."""
    hits: int
    misses: int
    evictions: int
    current_size: int
    memory_usage_bytes: int
    hit_rate: float

    def __str__(self) -> str: ...

class CacheLayer:
    """High-performance in-memory cache with TTL and LRU eviction."""

    _cache: OrderedDict[str, CacheEntry]
    _lock: Lock
    _max_entries: int
    _default_ttl: int
    _enable_metrics: bool
    _hits: int
    _misses: int
    _evictions: int

    def __init__(
        self,
        max_entries: int = 1000,
        enable_metrics: bool = True,
        default_ttl: int = 300
    ) -> None: ...

    def get(self, cache_key: str) -> Optional[Any]: ...

    def set(self, cache_key: str, value: Any, ttl_seconds: int) -> None: ...

    def delete(self, cache_key: str) -> bool: ...

    def clear(self) -> None: ...

    def get_stats(self) -> CacheStats: ...

    def _evict_lru(self) -> None: ...

    def _cleanup_expired(self) -> None: ...

    def _estimate_memory_usage(self) -> int: ...

def hash_query(params: dict[str, Any]) -> str: ...
