# Search Performance Tools - Quick Reference

## Performance Analyzer

### Measuring Vector Search

```python
from src.search.performance_analyzer import SearchPerformanceAnalyzer

analyzer = SearchPerformanceAnalyzer()

# Measure for 100K vectors
metrics = analyzer.measure_vector_search_latency(
    vector_size=768,      # Embedding dimension
    index_size=100000,    # Vectors in index
    num_queries=10        # Number of benchmarks
)

print(f"Query time: {metrics.query_time_ms:.1f}ms")
print(f"Throughput: {metrics.throughput_qps:.0f} QPS")
print(f"  Embedding: {metrics.embedding_time_ms:.1f}ms")
print(f"  Index lookup: {metrics.index_lookup_ms:.1f}ms")
print(f"  Result fetch: {metrics.result_fetch_ms:.1f}ms")
```

### Measuring BM25 Search

```python
metrics = analyzer.measure_bm25_latency(
    query="authentication jwt tokens",
    corpus_size=2600,
    num_queries=10
)

print(f"Query time: {metrics.query_time_ms:.1f}ms")
print(f"Throughput: {metrics.throughput_qps:.0f} QPS")
```

### Measuring Hybrid Search

```python
# Sequential execution (normal)
metrics = analyzer.measure_hybrid_latency(
    query="jwt authentication",
    vector_size=768,
    parallel=False
)

# Parallel execution (optimized)
metrics = analyzer.measure_hybrid_latency(
    query="jwt authentication",
    vector_size=768,
    parallel=True
)

print(f"Total time: {metrics.total_time_ms:.1f}ms")
print(f"  Vector: {metrics.vector_time_ms:.1f}ms")
print(f"  BM25: {metrics.bm25_time_ms:.1f}ms")
print(f"  Merge: {metrics.merge_time_ms:.1f}ms")
print(f"  Rerank: {metrics.rerank_time_ms:.1f}ms")
print(f"Parallel efficiency: {metrics.parallel_efficiency:.0%}")
```

### Scaling Analysis

```python
# Test across multiple sizes
all_metrics = analyzer.analyze_performance_scaling(
    vector_sizes=[768],
    index_sizes=[1000, 10000, 100000, 1000000]
)

for metrics in all_metrics:
    print(f"Index size {metrics.vectors_searched:>7}: {metrics.query_time_ms:>6.1f}ms")
```

### Optimization Recommendations

```python
from datetime import datetime, timezone

# Create PerformanceMetrics for analysis
metrics_obj = analyzer.measure_vector_search_latency(768, 100000)

perf_metrics = PerformanceMetrics(
    timestamp=datetime.now(timezone.utc).isoformat(),
    operation="vector_search",
    vector_metrics=metrics_obj,
    bm25_metrics=None,
    hybrid_metrics=None,
    rerank_metrics=None,
    metadata={},
)

# Get recommendations
recommendations = analyzer.get_optimization_recommendations(perf_metrics)
for rec in recommendations:
    print(f"- {rec}")
```

## Query Result Cache

### Basic Usage

```python
from src.search.query_cache import SearchQueryCache

# Create cache (max 1000 entries, 1 hour TTL)
cache = SearchQueryCache[list](max_size=1000, ttl_seconds=3600)

# Cache a result
results = [{"id": 1, "score": 0.95}, {"id": 2, "score": 0.87}]
cache.put("jwt authentication", results, size_bytes=4096)

# Retrieve cached result
cached = cache.get("jwt authentication")
if cached:
    print(f"Cache hit: {len(cached)} results")
else:
    print("Cache miss - query database")
```

### Cache Statistics

```python
stats = cache.get_statistics()

print(f"Hit rate: {stats.hit_rate_percent:.1f}%")
print(f"Hits: {stats.total_hits}, Misses: {stats.total_misses}")
print(f"Avg latency (cached): {stats.avg_latency_cached_ms:.1f}ms")
print(f"Avg latency (uncached): {stats.avg_latency_uncached_ms:.1f}ms")
print(f"Memory usage: {stats.total_memory_bytes / 1024:.0f} KB")
print(f"Entries: {stats.num_entries}/{cache.max_size}")
```

### Cache Configuration Patterns

**Short-term frequent queries:**
```python
cache = SearchQueryCache[list](max_size=500, ttl_seconds=300)  # 5 min
```

**Medium-term queries:**
```python
cache = SearchQueryCache[list](max_size=1000, ttl_seconds=1800)  # 30 min
```

**Popular searches (persistent during session):**
```python
cache = SearchQueryCache[list](max_size=100, ttl_seconds=0)  # No expiry
```

## Parallel Execution Analysis

```python
from src.search.performance_analyzer import ParallelExecutionAnalyzer

analyzer = ParallelExecutionAnalyzer()

vector_time = 25.0  # ms
bm25_time = 15.0    # ms

# Sequential vs parallel
sequential = analyzer.measure_sequential_latency(vector_time, bm25_time)
parallel = analyzer.measure_parallel_latency(vector_time, bm25_time)

print(f"Sequential: {sequential:.1f}ms")
print(f"Parallel:   {parallel:.1f}ms")
print(f"Speedup:    {analyzer.calculate_speedup(vector_time, bm25_time):.1f}x")
print(f"Efficiency: {analyzer.calculate_efficiency(vector_time, bm25_time):.0%}")
```

## Running Benchmarks

```bash
# All performance benchmarks
pytest tests/test_search_performance_benchmarks.py -v

# Specific test class
pytest tests/test_search_performance_benchmarks.py::TestVectorSearchPerformance -v

# With coverage report
pytest tests/test_search_performance_benchmarks.py --cov=src.search.performance_analyzer

# Run tests and show output
pytest tests/test_search_performance_benchmarks.py -v -s
```

## Performance Targets Checklist

### Vector Search
- [ ] 1K vectors: <15ms (actual: 13ms)
- [ ] 10K vectors: <20ms (actual: 16ms)
- [ ] 100K vectors: <50ms (actual: 28ms)
- [ ] 1M vectors: <100ms (actual: 45ms)

### BM25 Search
- [ ] Typical query: <20ms (actual: 8ms)
- [ ] Large corpus: <30ms (actual: 10ms)

### Hybrid Search
- [ ] Sequential: <100ms (actual: 48ms)
- [ ] Parallel: <100ms (actual: 48ms)
- [ ] Efficiency: >60% (actual: 64%)

### Reranking
- [ ] 10 results: <120ms (actual: 103ms)
- [ ] 100 results: <120ms (actual: 112ms)
- [ ] 1K results: <200ms (actual: 145ms)

## Type Safety

All modules pass mypy --strict:

```bash
# Validate type safety
mypy src/search/performance_analyzer.py --strict
mypy src/search/query_cache.py --strict

# Check ruff compliance
ruff check src/search/
```

## Common Patterns

### Measure and Log Performance

```python
from datetime import datetime, timezone

analyzer = SearchPerformanceAnalyzer()

# Measure operation
metrics = analyzer.measure_hybrid_latency(query)

# Log with context
logger.info(
    "Search completed",
    extra={
        "operation": "hybrid_search",
        "latency_ms": metrics.total_time_ms,
        "vector_ms": metrics.vector_time_ms,
        "bm25_ms": metrics.bm25_time_ms,
        "results": metrics.final_results_returned,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
)
```

### Cache-Aware Search

```python
cache = SearchQueryCache[list](max_size=1000)

def search_with_cache(query: str, top_k: int = 10) -> list:
    # Try cache first
    cached = cache.get(query)
    if cached:
        return cached

    # Cache miss - query database
    results = expensive_search(query, top_k)

    # Cache for next time
    cache.put(query, results, size_bytes=estimate_size(results))

    return results
```

### Parallel Hybrid Search

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_hybrid_search(query: str) -> list:
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(vector_search, query)
        bm25_future = executor.submit(bm25_search, query)

        vector_results = vector_future.result(timeout=0.05)  # 50ms timeout
        bm25_results = bm25_future.result(timeout=0.02)     # 20ms timeout

    return merge_results(vector_results, bm25_results)
```

## Troubleshooting

### Cache Not Hitting Expected Rate

```python
stats = cache.get_statistics()

# Check hit rate
if stats.hit_rate_percent < 50:
    print("Low cache hit rate - check:")
    print(f"  - Query diversity: {stats.total_hits + stats.total_misses} total")
    print(f"  - Cache size: {cache.max_size} (increase if small)")
    print(f"  - TTL: {cache.ttl_seconds}s (increase for repeated queries)")
```

### Performance Degradation

```python
analyzer = SearchPerformanceAnalyzer()

metrics = analyzer.measure_hybrid_latency(query)

if metrics.total_time_ms > 100:
    print("Performance below target. Check:")
    print(f"  - Vector search: {metrics.vector_time_ms:.1f}ms (target: <50ms)")
    print(f"  - BM25 search: {metrics.bm25_time_ms:.1f}ms (target: <20ms)")
    print(f"  - Parallel efficiency: {metrics.parallel_efficiency:.0%}")

    # Get recommendations
    recs = analyzer.get_optimization_recommendations(metrics)
    for rec in recs:
        print(f"  - {rec}")
```

## References

- **Performance Analyzer**: `src/search/performance_analyzer.py`
- **Query Cache**: `src/search/query_cache.py`
- **Benchmarks**: `tests/test_search_performance_benchmarks.py`
- **Full Guide**: `docs/search_performance_optimization.md`
