"""High-performance in-memory cache layer with TTL and LRU eviction.

This module provides a thread-safe caching layer for MCP tools with:
- Time-To-Live (TTL) expiration
- Least Recently Used (LRU) eviction
- Configurable size limits
- Performance metrics tracking

Example:
    >>> cache = CacheLayer(max_entries=1000, default_ttl=300)
    >>> cache.set("key1", {"data": "value"}, ttl_seconds=60)
    >>> result = cache.get("key1")
    >>> stats = cache.get_stats()
    >>> print(f"Hit rate: {stats.hit_rate:.1%}")
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class CacheEntry:
    """Single cache entry with TTL metadata.

    Attributes:
        value: Cached value (any type)
        created_at: Unix timestamp when entry was created
        ttl_seconds: Time-to-live in seconds
    """

    value: Any
    created_at: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL.

        Returns:
            True if entry has exceeded its TTL, False otherwise
        """
        elapsed = time.time() - self.created_at
        return elapsed >= self.ttl_seconds


@dataclass
class CacheStats:
    """Cache statistics for monitoring and debugging.

    Attributes:
        hits: Number of successful cache lookups
        misses: Number of cache misses
        evictions: Number of entries evicted due to LRU
        current_size: Current number of entries in cache
        memory_usage_bytes: Estimated memory usage in bytes
        hit_rate: Cache hit rate (hits / total requests)
    """

    hits: int
    misses: int
    evictions: int
    current_size: int
    memory_usage_bytes: int
    hit_rate: float

    def __str__(self) -> str:
        """Format cache statistics as human-readable string."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, current_size={self.current_size}, "
            f"memory_usage={self.memory_usage_bytes:,} bytes, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class CacheLayer:
    """High-performance in-memory cache with TTL and LRU eviction.

    Thread-safe cache implementation using OrderedDict for LRU tracking.
    Entries are automatically expired based on TTL and evicted when max
    capacity is reached.

    Example:
        >>> cache = CacheLayer(max_entries=100, default_ttl=60)
        >>> cache.set("search:auth", results, ttl_seconds=30)
        >>> if cached := cache.get("search:auth"):
        ...     print("Cache hit!")
        >>> print(cache.get_stats())
    """

    def __init__(
        self,
        max_entries: int = 1000,
        enable_metrics: bool = True,
        default_ttl: int = 300,
    ) -> None:
        """Initialize cache layer.

        Args:
            max_entries: Maximum number of entries before LRU eviction (default: 1000)
            enable_metrics: Enable hit/miss tracking (default: True)
            default_ttl: Default TTL in seconds (default: 300)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._enable_metrics = enable_metrics

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, cache_key: str) -> Any | None:
        """Retrieve value from cache.

        Checks TTL and removes expired entries. Updates LRU order on access.

        Args:
            cache_key: Unique cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                if self._enable_metrics:
                    self._misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                del self._cache[cache_key]
                if self._enable_metrics:
                    self._misses += 1
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)

            if self._enable_metrics:
                self._hits += 1

            return entry.value

    def set(self, cache_key: str, value: Any, ttl_seconds: int) -> None:
        """Store value in cache with TTL.

        Evicts oldest entries if max capacity reached.

        Args:
            cache_key: Unique cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
        """
        with self._lock:
            # Check if we need to evict
            if cache_key not in self._cache and len(self._cache) >= self._max_entries:
                self._evict_lru()

            # Create new entry
            entry = CacheEntry(
                value=value, created_at=time.time(), ttl_seconds=ttl_seconds
            )

            # Store and move to end (mark as most recently used)
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)

    def delete(self, cache_key: str) -> bool:
        """Remove entry from cache.

        Args:
            cache_key: Cache key to remove

        Returns:
            True if entry was deleted, False if key didn't exist
        """
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current metrics
        """
        with self._lock:
            # Cleanup expired entries before reporting stats
            self._cleanup_expired()

            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                current_size=len(self._cache),
                memory_usage_bytes=self._estimate_memory_usage(),
                hit_rate=hit_rate,
            )

    def _evict_lru(self) -> None:
        """Evict least recently used entry.

        Must be called while holding the lock.
        """
        # Remove first item (least recently used)
        if self._cache:
            self._cache.popitem(last=False)
            if self._enable_metrics:
                self._evictions += 1

    def _cleanup_expired(self) -> None:
        """Remove all expired entries.

        Must be called while holding the lock.
        """
        expired_keys = [
            key for key, entry in self._cache.items() if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes.

        This is a rough estimate based on entry count.

        Returns:
            Estimated memory usage in bytes
        """
        # Rough estimate: each entry ~1KB overhead + value size
        # This is approximate; real memory usage depends on value types
        return len(self._cache) * 1024


def hash_query(params: dict[str, Any]) -> str:
    """Generate stable cache key from query parameters.

    Uses SHA-256 for deterministic hashing of query parameters.

    Args:
        params: Query parameters dictionary

    Returns:
        Hex-encoded SHA-256 hash

    Example:
        >>> key = hash_query({"query": "auth", "top_k": 10})
        >>> print(len(key))
        64
    """
    # Sort keys for deterministic ordering
    sorted_json = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.sha256(sorted_json.encode("utf-8"))
    return hash_obj.hexdigest()
