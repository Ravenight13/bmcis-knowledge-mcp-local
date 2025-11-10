"""Comprehensive test suite for cache layer with TTL and LRU eviction.

Test Coverage:
- Basic operations (5 tests)
- TTL expiration (8 tests)
- LRU eviction (6 tests)
- Metrics & stats (6 tests)
- Thread safety (3 tests)
- Edge cases (2 tests)
- Hash query utility (2 tests)

Total: 32+ tests
"""

import threading
import time
from typing import Any

from src.mcp.cache import CacheEntry, CacheLayer, CacheStats, hash_query

# ============================================================================
# Basic Operations (5 tests)
# ============================================================================


def test_set_and_get() -> None:
    """Test basic set and get operations."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)

    result = cache.get("key1")
    assert result == "value1"


def test_delete() -> None:
    """Test delete operation for existing and non-existing keys."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)

    # Delete existing key
    assert cache.delete("key1") is True
    assert cache.get("key1") is None

    # Delete non-existing key
    assert cache.delete("key2") is False


def test_clear() -> None:
    """Test clear operation removes all entries."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)

    cache.clear()

    # Verify stats reset immediately after clear
    stats = cache.get_stats()
    assert stats.current_size == 0
    assert stats.hits == 0
    assert stats.misses == 0

    # Verify entries are gone
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") is None


def test_get_nonexistent() -> None:
    """Test getting non-existent key returns None."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    assert cache.get("nonexistent") is None


def test_overwrite() -> None:
    """Test overwriting existing key updates value."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key1", "value2", ttl_seconds=60)

    result = cache.get("key1")
    assert result == "value2"


# ============================================================================
# TTL Expiration (8 tests)
# ============================================================================


def test_ttl_expiration() -> None:
    """Test entry expires after TTL."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=1)

    # Should be available immediately
    assert cache.get("key1") == "value1"

    # Wait for expiration
    time.sleep(1.1)

    # Should be expired
    assert cache.get("key1") is None


def test_ttl_not_expired() -> None:
    """Test entry not expired before TTL."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=10)

    # Should be available
    assert cache.get("key1") == "value1"

    # Wait less than TTL
    time.sleep(0.5)

    # Should still be available
    assert cache.get("key1") == "value1"


def test_varying_ttls() -> None:
    """Test different entries with different TTLs."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("short", "value1", ttl_seconds=1)
    cache.set("medium", "value2", ttl_seconds=5)
    cache.set("long", "value3", ttl_seconds=10)

    # All should be available initially
    assert cache.get("short") == "value1"
    assert cache.get("medium") == "value2"
    assert cache.get("long") == "value3"

    # Wait for short to expire
    time.sleep(1.1)

    # Short expired, others still available
    assert cache.get("short") is None
    assert cache.get("medium") == "value2"
    assert cache.get("long") == "value3"


def test_expired_entry_cleanup() -> None:
    """Test expired entries are removed on access."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=1)
    cache.set("key2", "value2", ttl_seconds=10)

    # Wait for key1 to expire
    time.sleep(1.1)

    # Access key1 (triggers cleanup)
    assert cache.get("key1") is None

    # Check stats reflect cleanup
    stats = cache.get_stats()
    assert stats.current_size == 1  # Only key2 remains


def test_expired_entry_not_counted() -> None:
    """Test expired entries not included in stats."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=1)
    cache.set("key2", "value2", ttl_seconds=10)

    # Wait for key1 to expire
    time.sleep(1.1)

    # Get stats (triggers cleanup)
    stats = cache.get_stats()
    assert stats.current_size == 1  # Only non-expired entry


def test_zero_ttl() -> None:
    """Test zero TTL causes immediate expiration."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=0)

    # Should be immediately expired
    assert cache.get("key1") is None


def test_negative_ttl() -> None:
    """Test negative TTL is treated as expired."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=-1)

    # Should be immediately expired
    assert cache.get("key1") is None


def test_long_ttl() -> None:
    """Test very long TTL doesn't interfere with operations."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=86400)  # 1 day

    # Should be available
    assert cache.get("key1") == "value1"

    # Should still be available after short wait
    time.sleep(0.1)
    assert cache.get("key1") == "value1"


# ============================================================================
# LRU Eviction (6 tests)
# ============================================================================


def test_lru_eviction() -> None:
    """Test least recently used entry is evicted when max reached."""
    cache = CacheLayer(max_entries=3, default_ttl=300)

    # Fill cache to capacity
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)

    # Add one more (should evict key1)
    cache.set("key4", "value4", ttl_seconds=60)

    # key1 should be evicted
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_max_entries() -> None:
    """Test cache cannot exceed max_entries."""
    cache = CacheLayer(max_entries=5, default_ttl=300)

    # Add more than max
    for i in range(10):
        cache.set(f"key{i}", f"value{i}", ttl_seconds=60)

    # Size should not exceed max
    stats = cache.get_stats()
    assert stats.current_size <= 5


def test_access_updates_lru() -> None:
    """Test accessing an entry updates its LRU position."""
    cache = CacheLayer(max_entries=3, default_ttl=300)

    # Fill cache
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)

    # Access key1 (should move to end, making key2 least recently used)
    cache.get("key1")

    # Add new entry (should evict key2)
    cache.set("key4", "value4", ttl_seconds=60)

    # key2 should be evicted, key1 should still be there
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_multiple_evictions() -> None:
    """Test multiple entries evicted as needed."""
    cache = CacheLayer(max_entries=2, default_ttl=300)

    # Add entries sequentially
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)  # Evicts key1
    cache.set("key4", "value4", ttl_seconds=60)  # Evicts key2

    # Only last 2 should remain
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_eviction_stat() -> None:
    """Test eviction counter increments correctly."""
    cache = CacheLayer(max_entries=2, default_ttl=300)

    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)  # 1 eviction
    cache.set("key4", "value4", ttl_seconds=60)  # 2 evictions

    stats = cache.get_stats()
    assert stats.evictions == 2


def test_memory_pressure() -> None:
    """Test large values trigger eviction correctly."""
    cache = CacheLayer(max_entries=3, default_ttl=300)

    # Add large values
    large_value = "x" * 10000
    cache.set("key1", large_value, ttl_seconds=60)
    cache.set("key2", large_value, ttl_seconds=60)
    cache.set("key3", large_value, ttl_seconds=60)
    cache.set("key4", large_value, ttl_seconds=60)

    # Should still respect max_entries limit
    stats = cache.get_stats()
    assert stats.current_size == 3


# ============================================================================
# Metrics & Stats (6 tests)
# ============================================================================


def test_hit_rate() -> None:
    """Test hit rate calculation is correct."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)

    # 1 hit
    cache.get("key1")
    # 1 miss
    cache.get("key2")
    # 1 hit
    cache.get("key1")

    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert abs(stats.hit_rate - 0.6667) < 0.01  # 2/3


def test_hit_miss_counts() -> None:
    """Test hits and misses are tracked correctly."""
    cache = CacheLayer(max_entries=10, default_ttl=300)
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)

    # 2 hits
    cache.get("key1")
    cache.get("key2")
    # 3 misses
    cache.get("key3")
    cache.get("key4")
    cache.get("key5")

    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 3


def test_memory_usage() -> None:
    """Test memory usage estimation is reasonable."""
    cache = CacheLayer(max_entries=10, default_ttl=300)

    # Empty cache
    stats = cache.get_stats()
    assert stats.memory_usage_bytes == 0

    # Add some entries
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)

    stats = cache.get_stats()
    # Should estimate roughly 1KB per entry
    assert stats.memory_usage_bytes > 0
    assert stats.memory_usage_bytes >= 2 * 1024  # At least 2KB for 2 entries


def test_current_size() -> None:
    """Test current_size reports correct entry count."""
    cache = CacheLayer(max_entries=10, default_ttl=300)

    # Empty
    stats = cache.get_stats()
    assert stats.current_size == 0

    # Add entries
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)

    stats = cache.get_stats()
    assert stats.current_size == 3

    # Delete one
    cache.delete("key2")

    stats = cache.get_stats()
    assert stats.current_size == 2


def test_stats_after_operations() -> None:
    """Test stats update correctly after various operations."""
    cache = CacheLayer(max_entries=3, default_ttl=300)

    # Add entries
    cache.set("key1", "value1", ttl_seconds=60)
    cache.set("key2", "value2", ttl_seconds=60)
    cache.set("key3", "value3", ttl_seconds=60)

    # Mix of hits and misses
    cache.get("key1")  # hit
    cache.get("key4")  # miss
    cache.get("key2")  # hit

    # Add one more (eviction)
    cache.set("key5", "value5", ttl_seconds=60)

    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.evictions == 1
    assert stats.current_size == 3


def test_no_division_by_zero() -> None:
    """Test hit rate works correctly with zero accesses."""
    cache = CacheLayer(max_entries=10, default_ttl=300)

    # No accesses yet
    stats = cache.get_stats()
    assert stats.hit_rate == 0.0
    assert stats.hits == 0
    assert stats.misses == 0


# ============================================================================
# Thread Safety (3 tests)
# ============================================================================


def test_concurrent_gets() -> None:
    """Test multiple threads can read safely."""
    cache = CacheLayer(max_entries=100, default_ttl=300)

    # Pre-populate
    for i in range(10):
        cache.set(f"key{i}", f"value{i}", ttl_seconds=60)

    results: list[Any] = []

    def reader() -> None:
        for i in range(10):
            value = cache.get(f"key{i}")
            results.append(value)

    # Spawn multiple readers
    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All reads should succeed
    assert len(results) == 50  # 5 threads * 10 reads each
    assert all(r is not None for r in results)


def test_concurrent_sets() -> None:
    """Test multiple threads can write safely."""
    cache = CacheLayer(max_entries=100, default_ttl=300)

    def writer(thread_id: int) -> None:
        for i in range(10):
            cache.set(f"key{thread_id}_{i}", f"value{thread_id}_{i}", ttl_seconds=60)

    # Spawn multiple writers
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All entries should be present
    stats = cache.get_stats()
    assert stats.current_size == 50  # 5 threads * 10 entries each


def test_concurrent_operations() -> None:
    """Test mixed get/set/delete operations from multiple threads."""
    cache = CacheLayer(max_entries=100, default_ttl=300)

    # Pre-populate
    for i in range(20):
        cache.set(f"key{i}", f"value{i}", ttl_seconds=60)

    def mixed_ops(thread_id: int) -> None:
        # Mix of operations
        cache.get(f"key{thread_id}")
        cache.set(f"new_key{thread_id}", f"new_value{thread_id}", ttl_seconds=60)
        cache.delete(f"key{thread_id + 10}")
        cache.get(f"key{thread_id + 5}")

    # Spawn multiple threads with mixed operations
    threads = [threading.Thread(target=mixed_ops, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cache should remain consistent
    stats = cache.get_stats()
    assert stats.current_size > 0
    assert stats.current_size <= 100


# ============================================================================
# Edge Cases (2 tests)
# ============================================================================


def test_unicode_keys() -> None:
    """Test handling of unicode characters in cache keys."""
    cache = CacheLayer(max_entries=10, default_ttl=300)

    # Various unicode keys
    cache.set("你好", "chinese", ttl_seconds=60)
    cache.set("مرحبا", "arabic", ttl_seconds=60)
    cache.set("🚀emoji", "rocket", ttl_seconds=60)
    cache.set("café", "french", ttl_seconds=60)

    # All should work correctly
    assert cache.get("你好") == "chinese"
    assert cache.get("مرحبا") == "arabic"
    assert cache.get("🚀emoji") == "rocket"
    assert cache.get("café") == "french"


def test_large_values() -> None:
    """Test handling of large objects in cache."""
    cache = CacheLayer(max_entries=10, default_ttl=300)

    # Large dictionary
    large_dict = {f"key{i}": f"value{i}" * 100 for i in range(100)}
    cache.set("large_dict", large_dict, ttl_seconds=60)

    # Large list
    large_list = [f"item{i}" * 100 for i in range(100)]
    cache.set("large_list", large_list, ttl_seconds=60)

    # Should retrieve correctly
    assert cache.get("large_dict") == large_dict
    assert cache.get("large_list") == large_list


# ============================================================================
# Hash Query Utility (2 tests)
# ============================================================================


def test_hash_query_deterministic() -> None:
    """Test hash_query produces deterministic results."""
    params1 = {"query": "authentication", "top_k": 10}
    params2 = {"query": "authentication", "top_k": 10}

    hash1 = hash_query(params1)
    hash2 = hash_query(params2)

    # Same params should produce same hash
    assert hash1 == hash2


def test_hash_query_different_params() -> None:
    """Test hash_query produces different hashes for different params."""
    params1 = {"query": "authentication", "top_k": 10}
    params2 = {"query": "authorization", "top_k": 10}
    params3 = {"query": "authentication", "top_k": 20}

    hash1 = hash_query(params1)
    hash2 = hash_query(params2)
    hash3 = hash_query(params3)

    # Different params should produce different hashes
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3


# ============================================================================
# CacheStats String Representation
# ============================================================================


def test_cache_stats_str() -> None:
    """Test CacheStats string formatting."""
    stats = CacheStats(
        hits=100,
        misses=50,
        evictions=10,
        current_size=45,
        memory_usage_bytes=46080,
        hit_rate=0.6667,
    )

    str_repr = str(stats)
    assert "hits=100" in str_repr
    assert "misses=50" in str_repr
    assert "evictions=10" in str_repr
    assert "current_size=45" in str_repr
    assert "46,080 bytes" in str_repr
    assert "66.7%" in str_repr


# ============================================================================
# CacheEntry Tests
# ============================================================================


def test_cache_entry_is_expired() -> None:
    """Test CacheEntry.is_expired() method."""
    # Not expired
    entry1 = CacheEntry(value="test", created_at=time.time(), ttl_seconds=60)
    assert entry1.is_expired() is False

    # Expired
    entry2 = CacheEntry(value="test", created_at=time.time() - 100, ttl_seconds=60)
    assert entry2.is_expired() is True

    # Edge case: exactly at TTL
    entry3 = CacheEntry(value="test", created_at=time.time() - 60, ttl_seconds=60)
    assert entry3.is_expired() is True  # At exactly TTL, should be expired
