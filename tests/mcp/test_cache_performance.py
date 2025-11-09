"""Performance tests for cache layer."""

import time
from src.mcp.cache import CacheLayer


def test_cache_hit_latency() -> None:
    """Measure cache hit latency."""
    cache = CacheLayer(max_entries=1000, default_ttl=300)

    # Pre-populate
    for i in range(100):
        cache.set(f"key{i}", {"data": f"value{i}", "index": i}, ttl_seconds=300)

    # Measure cache hits
    iterations = 10000
    start = time.perf_counter()

    for i in range(iterations):
        key = f"key{i % 100}"
        result = cache.get(key)
        assert result is not None

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    avg_latency_us = (elapsed_ms * 1000) / iterations

    print(f"\nCache Hit Performance:")
    print(f"  Total iterations: {iterations:,}")
    print(f"  Total time: {elapsed_ms:.2f}ms")
    print(f"  Average latency: {avg_latency_us:.2f}µs per hit")
    print(f"  Throughput: {iterations / (elapsed_ms / 1000):,.0f} ops/sec")

    # Should be very fast (< 100µs per hit)
    assert avg_latency_us < 100


def test_cache_miss_latency() -> None:
    """Measure cache miss latency."""
    cache = CacheLayer(max_entries=1000, default_ttl=300)

    # Measure cache misses
    iterations = 10000
    start = time.perf_counter()

    for i in range(iterations):
        result = cache.get(f"nonexistent{i}")
        assert result is None

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    avg_latency_us = (elapsed_ms * 1000) / iterations

    print(f"\nCache Miss Performance:")
    print(f"  Total iterations: {iterations:,}")
    print(f"  Total time: {elapsed_ms:.2f}ms")
    print(f"  Average latency: {avg_latency_us:.2f}µs per miss")
    print(f"  Throughput: {iterations / (elapsed_ms / 1000):,.0f} ops/sec")

    # Should be very fast (< 100µs per miss)
    assert avg_latency_us < 100


def test_cache_set_latency() -> None:
    """Measure cache set operation latency."""
    cache = CacheLayer(max_entries=10000, default_ttl=300)

    iterations = 10000
    start = time.perf_counter()

    for i in range(iterations):
        cache.set(f"key{i}", {"data": f"value{i}", "index": i}, ttl_seconds=300)

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    avg_latency_us = (elapsed_ms * 1000) / iterations

    print(f"\nCache Set Performance:")
    print(f"  Total iterations: {iterations:,}")
    print(f"  Total time: {elapsed_ms:.2f}ms")
    print(f"  Average latency: {avg_latency_us:.2f}µs per set")
    print(f"  Throughput: {iterations / (elapsed_ms / 1000):,.0f} ops/sec")

    # Should be very fast (< 200µs per set)
    assert avg_latency_us < 200
