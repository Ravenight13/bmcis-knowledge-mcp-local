"""Type stubs for query result caching module.

Provides type definitions for in-memory query result caching with TTL support,
LRU eviction, and metrics tracking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar, Optional, Any
from collections.abc import Hashable

# Type variables
ResultType = TypeVar("ResultType")
KeyType = TypeVar("KeyType", bound=Hashable)


@dataclass(frozen=True)
class CacheEntry(Generic[ResultType]):
    """A single cache entry with metadata.

    Attributes:
        result: The cached result object.
        created_at: When the entry was created.
        accessed_at: When the entry was last accessed.
        access_count: Number of times accessed.
        size_bytes: Approximate size in bytes.
    """

    result: ResultType
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int


@dataclass(frozen=True)
class CacheStatistics:
    """Cache performance statistics.

    Attributes:
        total_hits: Total cache hits.
        total_misses: Total cache misses.
        hit_rate_percent: Hit rate as percentage (0-100).
        avg_latency_cached_ms: Average latency for cached results.
        avg_latency_uncached_ms: Average latency for uncached queries.
        total_memory_bytes: Total memory used by cache.
        num_entries: Number of entries in cache.
        eviction_count: Number of entries evicted.
    """

    total_hits: int
    total_misses: int
    hit_rate_percent: float
    avg_latency_cached_ms: float
    avg_latency_uncached_ms: float
    total_memory_bytes: int
    num_entries: int
    eviction_count: int


class SearchQueryCache(Generic[ResultType]):
    """In-memory cache for search results with TTL support.

    Implements LRU eviction, TTL expiration, and comprehensive metrics
    tracking for search result caching.

    Type parameters:
        ResultType: The type of results being cached.

    Example:
        >>> cache = SearchQueryCache[str](max_size=1000, ttl_seconds=3600)
        >>> cache.put("query1", ["result1", "result2"])
        >>> results = cache.get("query1")
        >>> stats = cache.get_statistics()
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
    ) -> None:
        """Initialize query cache.

        Args:
            max_size: Maximum number of entries in cache.
            ttl_seconds: Time-to-live for cache entries (0 = no TTL).
            cleanup_interval_seconds: Interval for expired entry cleanup.
        """
        ...

    def get(
        self, query: str, query_hash: str | None = None
    ) -> ResultType | None:
        """Get cached result for query.

        Args:
            query: The query string.
            query_hash: Optional pre-computed query hash (SHA-256).

        Returns:
            Cached result if found and not expired, None otherwise.
        """
        ...

    def put(
        self,
        query: str,
        result: ResultType,
        query_hash: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        """Cache result for query.

        Args:
            query: The query string.
            result: The result to cache.
            query_hash: Optional pre-computed query hash.
            size_bytes: Optional size estimate for memory tracking.
        """
        ...

    def delete(self, query: str, query_hash: str | None = None) -> bool:
        """Delete specific cache entry.

        Args:
            query: The query string.
            query_hash: Optional pre-computed query hash.

        Returns:
            True if entry was deleted, False if not found.
        """
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...

    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics.

        Returns:
            CacheStatistics with hit rate, memory usage, etc.
        """
        ...

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0-1).

        Returns:
            Hit rate ratio.
        """
        ...

    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB.

        Returns:
            Memory usage estimate.
        """
        ...

    def compute_query_hash(self, query: str) -> str:
        """Compute SHA-256 hash of query string.

        Args:
            query: Query string to hash.

        Returns:
            Hexadecimal hash string.
        """
        ...

    def is_expired(self, query_hash: str) -> bool:
        """Check if cache entry is expired.

        Args:
            query_hash: Hash of query.

        Returns:
            True if entry exists and is expired.
        """
        ...

    def cleanup_expired_entries(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        ...

    def get_entry_metadata(
        self, query: str, query_hash: str | None = None
    ) -> dict[str, Any] | None:
        """Get metadata for cache entry without retrieving result.

        Args:
            query: The query string.
            query_hash: Optional pre-computed query hash.

        Returns:
            Dictionary with entry metadata or None if not found.
        """
        ...

    def set_ttl(self, query: str, ttl_seconds: int) -> bool:
        """Update TTL for existing cache entry.

        Args:
            query: The query string.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if entry was updated, False if not found.
        """
        ...
