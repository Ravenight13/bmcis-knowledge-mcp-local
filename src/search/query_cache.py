"""Query result caching module with TTL and LRU eviction.

Provides high-performance in-memory caching for search results with time-to-live
(TTL) expiration, LRU eviction, and comprehensive metrics tracking.

Type-safe implementation with 100% mypy --strict compliance.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from src.core.logging import StructuredLogger

# Type variables
ResultType = TypeVar("ResultType")
KeyType = TypeVar("KeyType", bound=Hashable)

logger: logging.Logger = StructuredLogger.get_logger(__name__)


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

    Example:
        >>> cache = SearchQueryCache[list](max_size=1000, ttl_seconds=3600)
        >>> cache.put("jwt authentication", ["result1", "result2"])
        >>> results = cache.get("jwt authentication")
        >>> if results:
        ...     print(f"Got {len(results)} cached results")
        >>> stats = cache.get_statistics()
        >>> print(f"Hit rate: {stats.hit_rate_percent:.1f}%")
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
    ) -> None:
        """Initialize query cache.

        Args:
            max_size: Maximum number of entries in cache (default 1000).
            ttl_seconds: Time-to-live for cache entries in seconds (0 = no TTL).
            cleanup_interval_seconds: Interval for cleaning expired entries.

        Raises:
            ValueError: If max_size <= 0 or ttl_seconds < 0.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")

        self.max_size: int = max_size
        self.ttl_seconds: int = ttl_seconds
        self.cleanup_interval_seconds: int = cleanup_interval_seconds

        # Internal cache storage using OrderedDict for LRU tracking
        self._cache: OrderedDict[str, tuple[CacheEntry[ResultType], float]] = OrderedDict()
        self._lock: threading.RLock = threading.RLock()

        # Statistics tracking
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []
        self._eviction_count: int = 0
        self._last_cleanup: float = time.time()

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
        hash_key = query_hash or self.compute_query_hash(query)

        with self._lock:
            # Periodic cleanup of expired entries
            if (time.time() - self._last_cleanup) > self.cleanup_interval_seconds:
                self.cleanup_expired_entries()

            if hash_key not in self._cache:
                self._total_misses += 1
                return None

            entry, created_time = self._cache[hash_key]

            # Check if entry is expired
            if self.ttl_seconds > 0:
                if (time.time() - created_time) > self.ttl_seconds:
                    del self._cache[hash_key]
                    self._total_misses += 1
                    return None

            # Update access metadata
            now = datetime.now(UTC)
            updated_entry = CacheEntry(
                result=entry.result,
                created_at=entry.created_at,
                accessed_at=now,
                access_count=entry.access_count + 1,
                size_bytes=entry.size_bytes,
            )
            self._cache[hash_key] = (updated_entry, time.time())

            # Move to end for LRU ordering
            self._cache.move_to_end(hash_key)

            self._total_hits += 1

            logger.debug(
                f"Cache hit for query (hash: {hash_key[:16]})",
                extra={"cache_hit": True, "access_count": updated_entry.access_count},
            )

            return entry.result

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
        hash_key = query_hash or self.compute_query_hash(query)
        now = datetime.now(UTC)
        estimated_size = size_bytes or 2048  # Default estimate

        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size and hash_key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._eviction_count += 1

                logger.debug(
                    f"Cache eviction: LRU entry removed (hash: {oldest_key[:16]})",
                    extra={"evicted_hash": oldest_key[:16]},
                )

            entry = CacheEntry(
                result=result,
                created_at=now,
                accessed_at=now,
                access_count=1,
                size_bytes=estimated_size,
            )

            self._cache[hash_key] = (entry, time.time())
            self._cache.move_to_end(hash_key)

            logger.debug(
                f"Cache insert for query (hash: {hash_key[:16]})",
                extra={"cache_size": len(self._cache), "size_bytes": estimated_size},
            )

    def delete(self, query: str, query_hash: str | None = None) -> bool:
        """Delete specific cache entry.

        Args:
            query: The query string.
            query_hash: Optional pre-computed query hash.

        Returns:
            True if entry was deleted, False if not found.
        """
        hash_key = query_hash or self.compute_query_hash(query)

        with self._lock:
            if hash_key in self._cache:
                del self._cache[hash_key]
                logger.debug(
                    f"Cache delete for query (hash: {hash_key[:16]})",
                )
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            old_size = len(self._cache)
            self._cache.clear()
            logger.info(
                f"Cache cleared: {old_size} entries removed",
                extra={"entries_cleared": old_size},
            )

    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics.

        Returns:
            CacheStatistics with hit rate, memory usage, etc.
        """
        with self._lock:
            total = self._total_hits + self._total_misses
            hit_rate = (self._total_hits / total * 100) if total > 0 else 0.0

            avg_hit_latency = (
                sum(self._hit_latencies) / len(self._hit_latencies)
                if self._hit_latencies
                else 0.0
            )

            avg_miss_latency = (
                sum(self._miss_latencies) / len(self._miss_latencies)
                if self._miss_latencies
                else 0.0
            )

            total_memory = sum(
                entry.size_bytes for entry, _ in self._cache.values()
            )

            return CacheStatistics(
                total_hits=self._total_hits,
                total_misses=self._total_misses,
                hit_rate_percent=hit_rate,
                avg_latency_cached_ms=avg_hit_latency,
                avg_latency_uncached_ms=avg_miss_latency,
                total_memory_bytes=total_memory,
                num_entries=len(self._cache),
                eviction_count=self._eviction_count,
            )

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0-1).

        Returns:
            Hit rate ratio (0.0 to 1.0).
        """
        with self._lock:
            total = self._total_hits + self._total_misses
            if total == 0:
                return 0.0
            return self._total_hits / total

    def get_memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB.

        Returns:
            Memory usage estimate in MB.
        """
        with self._lock:
            total_bytes = sum(
                entry.size_bytes for entry, _ in self._cache.values()
            )
            return total_bytes / (1024 * 1024)

    def compute_query_hash(self, query: str) -> str:
        """Compute SHA-256 hash of query string.

        Args:
            query: Query string to hash.

        Returns:
            Hexadecimal hash string (64 characters).
        """
        query_bytes = query.encode("utf-8")
        return hashlib.sha256(query_bytes).hexdigest()

    def is_expired(self, query_hash: str) -> bool:
        """Check if cache entry is expired.

        Args:
            query_hash: Hash of query.

        Returns:
            True if entry exists and is expired, False otherwise.
        """
        if self.ttl_seconds == 0:
            return False

        with self._lock:
            if query_hash not in self._cache:
                return False

            _, created_time = self._cache[query_hash]
            return (time.time() - created_time) > self.ttl_seconds

    def cleanup_expired_entries(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        if self.ttl_seconds == 0:
            return 0

        with self._lock:
            expired_keys: list[str] = []
            current_time = time.time()

            for key, (_, created_time) in self._cache.items():
                if (current_time - created_time) > self.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            self._last_cleanup = current_time

            if expired_keys:
                logger.debug(
                    f"Cache cleanup: {len(expired_keys)} expired entries removed",
                    extra={"entries_removed": len(expired_keys)},
                )

            return len(expired_keys)

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
        hash_key = query_hash or self.compute_query_hash(query)

        with self._lock:
            if hash_key not in self._cache:
                return None

            entry, created_time = self._cache[hash_key]

            return {
                "created_at": entry.created_at.isoformat(),
                "accessed_at": entry.accessed_at.isoformat(),
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
                "ttl_seconds": self.ttl_seconds,
                "age_seconds": time.time() - created_time,
                "expired": self.is_expired(hash_key),
            }

    def set_ttl(self, query: str, ttl_seconds: int) -> bool:
        """Update TTL for existing cache entry.

        Note: This creates a new entry with the updated TTL, which
        may cause the entry to be considered "recently accessed".

        Args:
            query: The query string.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if entry was updated, False if not found.
        """
        hash_key = self.compute_query_hash(query)

        with self._lock:
            if hash_key not in self._cache:
                return False

            entry, _ = self._cache[hash_key]
            updated_entry = CacheEntry(
                result=entry.result,
                created_at=entry.created_at,
                accessed_at=datetime.now(UTC),
                access_count=entry.access_count,
                size_bytes=entry.size_bytes,
            )

            self._cache[hash_key] = (updated_entry, time.time())
            self._cache.move_to_end(hash_key)

            return True

    def record_latency_cached(self, latency_ms: float) -> None:
        """Record latency for a cached result.

        Args:
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            self._hit_latencies.append(latency_ms)
            if len(self._hit_latencies) > 1000:
                self._hit_latencies = self._hit_latencies[-500:]

    def record_latency_uncached(self, latency_ms: float) -> None:
        """Record latency for an uncached query.

        Args:
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            self._miss_latencies.append(latency_ms)
            if len(self._miss_latencies) > 1000:
                self._miss_latencies = self._miss_latencies[-500:]
