"""Benchmark BM25 full-text search performance.

Measures query latency, throughput, and index effectiveness for BM25 search
with varying result sizes and query patterns.

Usage:
    python scripts/benchmark_bm25_search.py
"""

import logging
import time
from statistics import mean, median, stdev
from typing import Any

import psycopg2

from src.core.config import get_settings
from src.core.logging import StructuredLogger
from src.search.bm25_search import BM25Search, SearchResult

# Configure logging
StructuredLogger.initialize()
logger = StructuredLogger.get_logger(__name__)


def get_table_stats() -> dict[str, Any]:
    """Get knowledge_base table statistics."""
    settings = get_settings()
    db = settings.database

    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        database=db.database,
        user=db.user,
        password=db.password.get_secret_value(),
    )

    try:
        with conn.cursor() as cur:
            # Get row count
            cur.execute("SELECT COUNT(*) FROM knowledge_base")
            row_count = cur.fetchone()[0]

            # Get index size
            cur.execute(
                """
                SELECT pg_size_pretty(pg_relation_size('idx_knowledge_fts'))
                AS index_size
                """
            )
            index_size = cur.fetchone()[0]

            # Get table size
            cur.execute(
                """
                SELECT pg_size_pretty(pg_total_relation_size('knowledge_base'))
                AS table_size
                """
            )
            table_size = cur.fetchone()[0]

            # Check if ts_vector is populated
            cur.execute(
                """
                SELECT COUNT(*) FROM knowledge_base
                WHERE ts_vector IS NOT NULL
                """
            )
            tsvector_count = cur.fetchone()[0]

        return {
            "row_count": row_count,
            "index_size": index_size,
            "table_size": table_size,
            "tsvector_count": tsvector_count,
            "tsvector_populated": tsvector_count == row_count,
        }
    finally:
        conn.close()


def benchmark_search_query(
    search: BM25Search,
    query: str,
    iterations: int = 10,
    top_k: int = 10,
) -> dict[str, Any]:
    """Benchmark a single search query.

    Args:
        search: BM25Search instance.
        query: Search query text.
        iterations: Number of iterations to run.
        top_k: Number of results to return.

    Returns:
        Benchmark statistics including latency metrics.
    """
    latencies: list[float] = []
    result_counts: list[int] = []

    for _ in range(iterations):
        start = time.perf_counter()
        results = search.search(query, top_k=top_k)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        result_counts.append(len(results))

    return {
        "query": query,
        "iterations": iterations,
        "top_k": top_k,
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "stdev_latency_ms": stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_results": mean(result_counts),
        "queries_per_second": 1000 / mean(latencies),
    }


def benchmark_phrase_search(
    search: BM25Search,
    phrase: str,
    iterations: int = 10,
    top_k: int = 10,
) -> dict[str, Any]:
    """Benchmark phrase search query.

    Args:
        search: BM25Search instance.
        phrase: Search phrase.
        iterations: Number of iterations to run.
        top_k: Number of results to return.

    Returns:
        Benchmark statistics.
    """
    latencies: list[float] = []
    result_counts: list[int] = []

    for _ in range(iterations):
        start = time.perf_counter()
        results = search.search_phrase(phrase, top_k=top_k)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        result_counts.append(len(results))

    return {
        "phrase": phrase,
        "iterations": iterations,
        "top_k": top_k,
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "stdev_latency_ms": stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_results": mean(result_counts),
        "queries_per_second": 1000 / mean(latencies),
    }


def print_benchmark_results(results: dict[str, Any], title: str) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

    if "query" in results:
        print(f"Query: {results['query']}")
    elif "phrase" in results:
        print(f"Phrase: {results['phrase']}")

    print(f"Iterations: {results['iterations']}")
    print(f"Top-k: {results['top_k']}")
    print(f"\nLatency Metrics:")
    print(f"  Mean:   {results['mean_latency_ms']:.2f} ms")
    print(f"  Median: {results['median_latency_ms']:.2f} ms")
    print(f"  Min:    {results['min_latency_ms']:.2f} ms")
    print(f"  Max:    {results['max_latency_ms']:.2f} ms")
    print(f"  StdDev: {results['stdev_latency_ms']:.2f} ms")
    print(f"\nThroughput:")
    print(f"  Queries/sec: {results['queries_per_second']:.2f}")
    print(f"  Mean results: {results['mean_results']:.1f}")


def main() -> None:
    """Run BM25 search benchmarks."""
    logger.info("Starting BM25 search benchmarks")

    # Get table statistics
    print("\nDatabase Statistics:")
    print("=" * 80)
    stats = get_table_stats()
    print(f"Total rows: {stats['row_count']:,}")
    print(f"ts_vector populated: {stats['tsvector_count']:,} / {stats['row_count']:,}")
    print(f"Index size: {stats['index_size']}")
    print(f"Table size: {stats['table_size']}")

    if not stats["tsvector_populated"]:
        logger.warning(
            "ts_vector column not fully populated (%d / %d rows)",
            stats["tsvector_count"],
            stats["row_count"],
        )
        print(
            f"\n⚠️  WARNING: ts_vector not fully populated "
            f"({stats['tsvector_count']} / {stats['row_count']} rows)"
        )

    # Initialize search
    search = BM25Search()

    # Test queries with varying complexity
    test_queries = [
        ("authentication", "Single keyword"),
        ("user authentication", "Two keywords"),
        ("user authentication jwt token", "Multiple keywords"),
        ("database connection pool", "Technical terms"),
        ("how to configure settings", "Natural language"),
    ]

    # Benchmark standard search
    for query, description in test_queries:
        results = benchmark_search_query(search, query, iterations=20, top_k=10)
        print_benchmark_results(results, f"Standard Search - {description}")

    # Benchmark phrase search
    phrase_queries = [
        ("authentication token", "Two-word phrase"),
        ("JWT authentication", "Technical phrase"),
    ]

    for phrase, description in phrase_queries:
        results = benchmark_phrase_search(search, phrase, iterations=20, top_k=10)
        print_benchmark_results(results, f"Phrase Search - {description}")

    # Benchmark varying result sizes
    print(f"\n{'=' * 80}")
    print("Result Size Impact on Latency")
    print(f"{'=' * 80}")

    query = "authentication"
    for top_k in [5, 10, 20, 50, 100]:
        results = benchmark_search_query(search, query, iterations=20, top_k=top_k)
        print(
            f"top_k={top_k:3d}: {results['mean_latency_ms']:6.2f} ms "
            f"(median: {results['median_latency_ms']:6.2f} ms)"
        )

    # Test category filtering
    print(f"\n{'=' * 80}")
    print("Category Filtering Impact")
    print(f"{'=' * 80}")

    # Without filter
    start = time.perf_counter()
    results_no_filter = search.search("authentication", top_k=10)
    end = time.perf_counter()
    latency_no_filter = (end - start) * 1000

    print(f"No filter:     {latency_no_filter:.2f} ms ({len(results_no_filter)} results)")

    # With filter (if categories exist)
    categories = ["product_docs", "kb_article", "api_docs"]
    for category in categories:
        start = time.perf_counter()
        results_with_filter = search.search(
            "authentication", top_k=10, category_filter=category
        )
        end = time.perf_counter()
        latency_with_filter = (end - start) * 1000

        print(
            f"Filter '{category}': {latency_with_filter:.2f} ms "
            f"({len(results_with_filter)} results)"
        )

    # Performance targets
    print(f"\n{'=' * 80}")
    print("Performance Target Analysis")
    print(f"{'=' * 80}")

    target_latency_ms = 50.0
    results = benchmark_search_query(search, "authentication", iterations=50, top_k=10)

    mean_latency = results["mean_latency_ms"]
    p95_latency = results["mean_latency_ms"] + 1.645 * results["stdev_latency_ms"]

    print(f"Target latency:   {target_latency_ms:.2f} ms")
    print(f"Mean latency:     {mean_latency:.2f} ms")
    print(f"P95 latency:      {p95_latency:.2f} ms")

    if mean_latency <= target_latency_ms:
        print(f"✓ Mean latency meets target (<{target_latency_ms}ms)")
    else:
        print(f"✗ Mean latency exceeds target (>{target_latency_ms}ms)")

    if p95_latency <= target_latency_ms * 2:
        print(f"✓ P95 latency acceptable (<{target_latency_ms * 2}ms)")
    else:
        print(f"✗ P95 latency too high (>{target_latency_ms * 2}ms)")

    logger.info("BM25 benchmarks completed")


if __name__ == "__main__":
    main()
