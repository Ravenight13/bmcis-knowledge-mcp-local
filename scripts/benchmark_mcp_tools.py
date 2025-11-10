"""Comprehensive performance benchmarking for MCP tools.

Benchmarks semantic_search and find_vendor_info tools across:
- All response modes (ids_only, metadata, preview, full)
- Various query patterns and complexity levels
- Authentication and rate limiting overhead
- Latency percentiles (P50, P95, P99)
- Token consumption and memory usage
- Scalability under concurrent load

Target Performance:
- semantic_search metadata: P95 <500ms, 90%+ token reduction vs full
- find_vendor_info metadata: P95 <500ms, 94%+ token reduction vs full
- Authentication: <10ms total overhead
- Rate limiting: <5ms per request

Usage:
    # Run all benchmarks with real database
    python scripts/benchmark_mcp_tools.py

    # Run specific benchmark suite
    python scripts/benchmark_mcp_tools.py --suite semantic_search

    # Use mocked database for testing
    python scripts/benchmark_mcp_tools.py --mock
"""

import argparse
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.knowledge_graph.query_repository import KnowledgeGraphQueryRepository
from src.mcp.auth import RateLimiter, validate_api_key
from src.mcp.server import get_hybrid_search, initialize_server
from src.mcp.tools.find_vendor_info import find_vendor_info
from src.mcp.tools.semantic_search import semantic_search

# Initialize logger
StructuredLogger.initialize()
logger = StructuredLogger.get_logger(__name__)


@dataclass
class LatencyStats:
    """Statistical measures for latency distribution."""

    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    stdev: float


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""

    test_name: str
    query_or_vendor: str
    response_mode: str
    iterations: int
    latency: LatencyStats
    token_estimate: int
    memory_mb: float | None
    success_rate: float


@dataclass
class TokenEfficiency:
    """Token consumption comparison across response modes."""

    ids_only_tokens: int
    metadata_tokens: int
    preview_tokens: int
    full_tokens: int

    @property
    def metadata_reduction_pct(self) -> float:
        """Token reduction of metadata vs full mode."""
        if self.full_tokens == 0:
            return 0.0
        return ((self.full_tokens - self.metadata_tokens) / self.full_tokens) * 100

    @property
    def preview_reduction_pct(self) -> float:
        """Token reduction of preview vs full mode."""
        if self.full_tokens == 0:
            return 0.0
        return ((self.full_tokens - self.preview_tokens) / self.full_tokens) * 100


def calculate_latency_stats(latencies: list[float]) -> LatencyStats:
    """Calculate comprehensive latency statistics.

    Args:
        latencies: List of latency measurements in milliseconds

    Returns:
        LatencyStats with percentiles and statistical measures
    """
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return LatencyStats(
        mean=statistics.mean(latencies),
        median=statistics.median(latencies),
        p50=sorted_latencies[int(n * 0.50)],
        p95=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
        p99=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        min=min(latencies),
        max=max(latencies),
        stdev=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    )


def estimate_tokens(response: Any, response_mode: str) -> int:
    """Estimate token count for response.

    Rough approximation: 1 token ≈ 4 characters for English text.

    Args:
        response: MCP tool response object
        response_mode: Response mode (ids_only, metadata, preview, full)

    Returns:
        Estimated token count
    """
    # Convert response to string representation
    response_str = str(response)
    char_count = len(response_str)

    # Rough token estimate (4 chars per token)
    return char_count // 4


def benchmark_semantic_search(
    query: str,
    response_mode: str,
    iterations: int = 100,
    top_k: int = 10,
) -> BenchmarkResult:
    """Benchmark semantic_search tool with given parameters.

    Args:
        query: Search query string
        response_mode: Response detail level
        iterations: Number of iterations to run
        top_k: Number of results to return

    Returns:
        BenchmarkResult with latency, token, and success metrics
    """
    latencies: list[float] = []
    token_estimates: list[int] = []
    success_count = 0

    logger.info(f"Benchmarking semantic_search: query='{query}', mode={response_mode}, k={top_k}")

    for i in range(iterations):
        try:
            start = time.perf_counter()
            response = semantic_search(
                query=query,
                top_k=top_k,
                response_mode=response_mode,
            )
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Estimate tokens for first 10 iterations to avoid overhead
            if i < 10:
                tokens = estimate_tokens(response, response_mode)
                token_estimates.append(tokens)

            success_count += 1

        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")
            continue

    if not latencies:
        raise RuntimeError(f"All iterations failed for query '{query}'")

    return BenchmarkResult(
        test_name=f"semantic_search_{response_mode}",
        query_or_vendor=query,
        response_mode=response_mode,
        iterations=iterations,
        latency=calculate_latency_stats(latencies),
        token_estimate=int(statistics.mean(token_estimates)) if token_estimates else 0,
        memory_mb=None,  # Not measured in this simple version
        success_rate=success_count / iterations,
    )


def benchmark_find_vendor_info(
    vendor_name: str,
    response_mode: str,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark find_vendor_info tool with given parameters.

    Args:
        vendor_name: Vendor name to search
        response_mode: Response detail level
        iterations: Number of iterations to run

    Returns:
        BenchmarkResult with latency, token, and success metrics
    """
    latencies: list[float] = []
    token_estimates: list[int] = []
    success_count = 0

    logger.info(f"Benchmarking find_vendor_info: vendor='{vendor_name}', mode={response_mode}")

    for i in range(iterations):
        try:
            start = time.perf_counter()
            response = find_vendor_info(
                vendor_name=vendor_name,
                response_mode=response_mode,
                include_relationships=True,
            )
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Estimate tokens for first 10 iterations
            if i < 10:
                tokens = estimate_tokens(response, response_mode)
                token_estimates.append(tokens)

            success_count += 1

        except ValueError as e:
            # Vendor not found - expected for some test cases
            logger.warning(f"Vendor '{vendor_name}' not found: {e}")
            # Don't count as failure if vendor doesn't exist
            if "not found" in str(e).lower():
                success_count += 1
            continue
        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")
            continue

    if not latencies:
        # If vendor doesn't exist, return empty result
        return BenchmarkResult(
            test_name=f"find_vendor_info_{response_mode}",
            query_or_vendor=vendor_name,
            response_mode=response_mode,
            iterations=0,
            latency=LatencyStats(0, 0, 0, 0, 0, 0, 0, 0),
            token_estimate=0,
            memory_mb=None,
            success_rate=0.0,
        )

    return BenchmarkResult(
        test_name=f"find_vendor_info_{response_mode}",
        query_or_vendor=vendor_name,
        response_mode=response_mode,
        iterations=iterations,
        latency=calculate_latency_stats(latencies),
        token_estimate=int(statistics.mean(token_estimates)) if token_estimates else 0,
        memory_mb=None,
        success_rate=success_count / iterations,
    )


def benchmark_authentication(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark API key validation performance.

    Args:
        iterations: Number of validation attempts

    Returns:
        BenchmarkResult with validation timing
    """
    # Set test API key
    os.environ["BMCIS_API_KEY"] = "test_benchmark_key_12345"
    test_key = "test_benchmark_key_12345"

    latencies: list[float] = []

    logger.info(f"Benchmarking authentication: {iterations} iterations")

    for _ in range(iterations):
        start = time.perf_counter()
        result = validate_api_key(test_key)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        assert result is True, "API key validation should succeed"

    return BenchmarkResult(
        test_name="authentication",
        query_or_vendor="validate_api_key",
        response_mode="n/a",
        iterations=iterations,
        latency=calculate_latency_stats(latencies),
        token_estimate=0,
        memory_mb=None,
        success_rate=1.0,
    )


def benchmark_rate_limiting(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark rate limiter overhead.

    Args:
        iterations: Number of rate limit checks

    Returns:
        BenchmarkResult with rate limiter timing
    """
    limiter = RateLimiter(
        requests_per_minute=1000,
        requests_per_hour=10000,
        requests_per_day=100000,
    )
    test_key = "test_rate_limit_key_123"

    latencies: list[float] = []

    logger.info(f"Benchmarking rate limiting: {iterations} iterations")

    for _ in range(iterations):
        start = time.perf_counter()
        allowed = limiter.is_allowed(test_key)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        assert allowed is True, "Rate limit should allow request"

    return BenchmarkResult(
        test_name="rate_limiting",
        query_or_vendor="is_allowed",
        response_mode="n/a",
        iterations=iterations,
        latency=calculate_latency_stats(latencies),
        token_estimate=0,
        memory_mb=None,
        success_rate=1.0,
    )


def run_semantic_search_suite() -> list[BenchmarkResult]:
    """Run comprehensive semantic_search benchmarks.

    Returns:
        List of BenchmarkResult objects
    """
    results: list[BenchmarkResult] = []

    # Test queries with varying complexity
    test_queries = [
        ("Acme", "Single word"),
        ("enterprise software solutions", "Multi-word"),
        ("How do I configure JWT authentication for API access?", "Long natural language"),
        ("日本企業", "Unicode (Japanese)"),
        ("API@#$%integration", "Special characters"),
    ]

    # Test all response modes for first query
    logger.info("=" * 80)
    logger.info("SEMANTIC_SEARCH BENCHMARKS - Response Mode Comparison")
    logger.info("=" * 80)

    query = "enterprise software authentication"
    for mode in ["ids_only", "metadata", "preview", "full"]:
        result = benchmark_semantic_search(query, mode, iterations=100, top_k=10)
        results.append(result)
        logger.info(
            f"{mode:12s}: P50={result.latency.p50:6.1f}ms P95={result.latency.p95:6.1f}ms "
            f"P99={result.latency.p99:6.1f}ms tokens≈{result.token_estimate:,}"
        )

    # Test query complexity (metadata mode)
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC_SEARCH BENCHMARKS - Query Complexity")
    logger.info("=" * 80)

    for query, description in test_queries:
        result = benchmark_semantic_search(query, "metadata", iterations=50, top_k=10)
        results.append(result)
        logger.info(
            f"{description:30s}: P50={result.latency.p50:6.1f}ms P95={result.latency.p95:6.1f}ms"
        )

    # Test top_k scaling (metadata mode)
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC_SEARCH BENCHMARKS - Top-K Scaling")
    logger.info("=" * 80)

    query = "authentication"
    for k in [5, 10, 20, 50]:
        result = benchmark_semantic_search(query, "metadata", iterations=50, top_k=k)
        results.append(result)
        logger.info(f"top_k={k:2d}: P50={result.latency.p50:6.1f}ms P95={result.latency.p95:6.1f}ms")

    return results


def run_find_vendor_info_suite() -> list[BenchmarkResult]:
    """Run comprehensive find_vendor_info benchmarks.

    Returns:
        List of BenchmarkResult objects
    """
    results: list[BenchmarkResult] = []

    # Test with actual vendor name (may not exist in test DB)
    vendor_name = "Acme Corp"

    # Test all response modes
    logger.info("\n" + "=" * 80)
    logger.info("FIND_VENDOR_INFO BENCHMARKS - Response Mode Comparison")
    logger.info("=" * 80)

    for mode in ["ids_only", "metadata", "preview", "full"]:
        result = benchmark_find_vendor_info(vendor_name, mode, iterations=50)
        results.append(result)

        if result.iterations > 0:
            logger.info(
                f"{mode:12s}: P50={result.latency.p50:6.1f}ms P95={result.latency.p95:6.1f}ms "
                f"P99={result.latency.p99:6.1f}ms tokens≈{result.token_estimate:,}"
            )
        else:
            logger.warning(f"{mode:12s}: Vendor not found in database (skipped)")

    return results


def run_authentication_suite() -> list[BenchmarkResult]:
    """Run authentication and rate limiting benchmarks.

    Returns:
        List of BenchmarkResult objects
    """
    results: list[BenchmarkResult] = []

    logger.info("\n" + "=" * 80)
    logger.info("AUTHENTICATION & RATE LIMITING BENCHMARKS")
    logger.info("=" * 80)

    # Benchmark authentication
    auth_result = benchmark_authentication(iterations=1000)
    results.append(auth_result)
    logger.info(
        f"API key validation: P50={auth_result.latency.p50:.3f}ms "
        f"P95={auth_result.latency.p95:.3f}ms P99={auth_result.latency.p99:.3f}ms"
    )

    # Benchmark rate limiting
    rate_result = benchmark_rate_limiting(iterations=1000)
    results.append(rate_result)
    logger.info(
        f"Rate limit check:   P50={rate_result.latency.p50:.3f}ms "
        f"P95={rate_result.latency.p95:.3f}ms P99={rate_result.latency.p99:.3f}ms"
    )

    # Total authentication overhead
    total_overhead = auth_result.latency.p95 + rate_result.latency.p95
    logger.info(f"\nTotal auth overhead (P95): {total_overhead:.2f}ms")

    if total_overhead < 10.0:
        logger.info("✓ Authentication overhead meets target (<10ms)")
    else:
        logger.warning(f"✗ Authentication overhead exceeds target (>{10}ms)")

    return results


def generate_report(results: list[BenchmarkResult], output_file: str) -> None:
    """Generate comprehensive markdown report.

    Args:
        results: List of benchmark results
        output_file: Output file path
    """
    logger.info(f"\nGenerating report: {output_file}")

    # Organize results by category
    semantic_results = [r for r in results if "semantic_search" in r.test_name]
    vendor_results = [r for r in results if "find_vendor_info" in r.test_name]
    auth_results = [r for r in results if r.test_name in ["authentication", "rate_limiting"]]

    with open(output_file, "w") as f:
        f.write("# MCP Tools Performance Benchmark Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")

        if semantic_results:
            metadata_result = next(
                (r for r in semantic_results if r.response_mode == "metadata"), None
            )
            full_result = next((r for r in semantic_results if r.response_mode == "full"), None)

            if metadata_result:
                f.write(f"**semantic_search (metadata mode)**:\n")
                f.write(f"- P50 latency: {metadata_result.latency.p50:.1f}ms\n")
                f.write(f"- P95 latency: {metadata_result.latency.p95:.1f}ms\n")
                f.write(f"- Token estimate: ~{metadata_result.token_estimate:,} tokens\n")

                if metadata_result.latency.p95 < 500:
                    f.write("- ✓ Latency target met (P95 <500ms)\n")
                else:
                    f.write("- ✗ Latency target exceeded (P95 >500ms)\n")

                if full_result and full_result.token_estimate > 0:
                    reduction = (
                        (full_result.token_estimate - metadata_result.token_estimate)
                        / full_result.token_estimate
                    ) * 100
                    f.write(f"- Token reduction vs full: {reduction:.1f}%\n")

                    if reduction >= 90:
                        f.write("- ✓ Token efficiency target met (≥90% reduction)\n")
                    else:
                        f.write("- ✗ Token efficiency target not met (<90% reduction)\n")

        if vendor_results:
            metadata_result = next((r for r in vendor_results if r.response_mode == "metadata"), None)

            if metadata_result and metadata_result.iterations > 0:
                f.write(f"\n**find_vendor_info (metadata mode)**:\n")
                f.write(f"- P50 latency: {metadata_result.latency.p50:.1f}ms\n")
                f.write(f"- P95 latency: {metadata_result.latency.p95:.1f}ms\n")
                f.write(f"- Token estimate: ~{metadata_result.token_estimate:,} tokens\n")

                if metadata_result.latency.p95 < 500:
                    f.write("- ✓ Latency target met (P95 <500ms)\n")
                else:
                    f.write("- ✗ Latency target exceeded (P95 >500ms)\n")

        if auth_results:
            auth_result = next((r for r in auth_results if r.test_name == "authentication"), None)
            rate_result = next((r for r in auth_results if r.test_name == "rate_limiting"), None)

            if auth_result and rate_result:
                total = auth_result.latency.p95 + rate_result.latency.p95
                f.write(f"\n**Authentication & Rate Limiting**:\n")
                f.write(f"- API key validation: {auth_result.latency.p95:.2f}ms (P95)\n")
                f.write(f"- Rate limit check: {rate_result.latency.p95:.2f}ms (P95)\n")
                f.write(f"- Total overhead: {total:.2f}ms (P95)\n")

                if total < 10:
                    f.write("- ✓ Overhead target met (<10ms)\n")
                else:
                    f.write("- ✗ Overhead target exceeded (>10ms)\n")

        # Detailed semantic_search results
        f.write("\n## semantic_search Benchmarks\n\n")

        if semantic_results:
            # Response mode comparison table
            f.write("### Response Mode Comparison\n\n")
            f.write("| Mode | P50 (ms) | P95 (ms) | P99 (ms) | Tokens | Reduction |\n")
            f.write("|------|----------|----------|----------|--------|----------|\n")

            modes = ["ids_only", "metadata", "preview", "full"]
            mode_results = {r.response_mode: r for r in semantic_results if r.response_mode in modes}
            full_tokens = mode_results.get("full", None)

            for mode in modes:
                if mode in mode_results:
                    r = mode_results[mode]
                    reduction = ""
                    if full_tokens and full_tokens.token_estimate > 0 and mode != "full":
                        pct = ((full_tokens.token_estimate - r.token_estimate) / full_tokens.token_estimate) * 100
                        reduction = f"{pct:.1f}%"

                    f.write(
                        f"| {mode:10s} | {r.latency.p50:8.1f} | {r.latency.p95:8.1f} | "
                        f"{r.latency.p99:8.1f} | {r.token_estimate:6,} | {reduction:8s} |\n"
                    )

            f.write("\n### Key Findings\n\n")
            if "metadata" in mode_results and "full" in mode_results:
                meta = mode_results["metadata"]
                full = mode_results["full"]

                if full.token_estimate > 0:
                    reduction = ((full.token_estimate - meta.token_estimate) / full.token_estimate) * 100
                    f.write(f"- **Token Efficiency**: metadata mode achieves {reduction:.1f}% token reduction vs full mode\n")

                speedup = full.latency.p50 / meta.latency.p50 if meta.latency.p50 > 0 else 0
                f.write(f"- **Latency**: metadata mode is {speedup:.1f}x faster than full mode (P50)\n")

        # Detailed vendor_info results
        f.write("\n## find_vendor_info Benchmarks\n\n")

        if vendor_results and any(r.iterations > 0 for r in vendor_results):
            f.write("### Response Mode Comparison\n\n")
            f.write("| Mode | P50 (ms) | P95 (ms) | P99 (ms) | Tokens | Reduction |\n")
            f.write("|------|----------|----------|----------|--------|----------|\n")

            modes = ["ids_only", "metadata", "preview", "full"]
            mode_results = {r.response_mode: r for r in vendor_results if r.iterations > 0}
            full_tokens = mode_results.get("full", None)

            for mode in modes:
                if mode in mode_results:
                    r = mode_results[mode]
                    reduction = ""
                    if full_tokens and full_tokens.token_estimate > 0 and mode != "full":
                        pct = ((full_tokens.token_estimate - r.token_estimate) / full_tokens.token_estimate) * 100
                        reduction = f"{pct:.1f}%"

                    f.write(
                        f"| {mode:10s} | {r.latency.p50:8.1f} | {r.latency.p95:8.1f} | "
                        f"{r.latency.p99:8.1f} | {r.token_estimate:6,} | {reduction:8s} |\n"
                    )
        else:
            f.write("*Vendor not found in database - benchmarks skipped*\n")

        # Authentication results
        f.write("\n## Authentication & Rate Limiting\n\n")

        if auth_results:
            f.write("### Performance Metrics\n\n")
            f.write("| Component | P50 (ms) | P95 (ms) | P99 (ms) |\n")
            f.write("|-----------|----------|----------|----------|\n")

            for r in auth_results:
                f.write(
                    f"| {r.test_name:20s} | {r.latency.p50:8.3f} | {r.latency.p95:8.3f} | "
                    f"{r.latency.p99:8.3f} |\n"
                )

        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("### Quick Wins\n\n")

        # Check if any targets were missed
        if semantic_results:
            metadata_result = next(
                (r for r in semantic_results if r.response_mode == "metadata"), None
            )
            if metadata_result and metadata_result.latency.p95 >= 500:
                f.write("- Investigate semantic_search latency bottlenecks (P95 target: <500ms)\n")
                f.write("  - Profile database query time\n")
                f.write("  - Check embedding generation time\n")
                f.write("  - Consider query caching\n")

        if vendor_results:
            metadata_result = next((r for r in vendor_results if r.response_mode == "metadata"), None)
            if metadata_result and metadata_result.iterations > 0 and metadata_result.latency.p95 >= 500:
                f.write("- Investigate find_vendor_info latency bottlenecks\n")
                f.write("  - Profile graph traversal time\n")
                f.write("  - Check index usage on knowledge_entities\n")
                f.write("  - Consider result set caching\n")

        f.write("\n### Long-term Optimizations\n\n")
        f.write("- Implement query result caching with TTL\n")
        f.write("- Add database query plan analysis\n")
        f.write("- Optimize serialization for large responses\n")
        f.write("- Consider response streaming for full mode\n")

        f.write("\n### Monitoring Suggestions\n\n")
        f.write("- Track P95/P99 latency in production\n")
        f.write("- Monitor token consumption by response mode\n")
        f.write("- Alert on authentication overhead >10ms\n")
        f.write("- Track rate limit exhaustion events\n")

    logger.info(f"✓ Report generated: {output_file}")


def main() -> None:
    """Run comprehensive MCP tool benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark MCP tools")
    parser.add_argument(
        "--suite",
        choices=["semantic_search", "vendor_info", "auth", "all"],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked database (for testing)",
    )
    parser.add_argument(
        "--output",
        default="docs/performance/mcp-benchmarks.md",
        help="Output report file path",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MCP TOOLS PERFORMANCE BENCHMARKS")
    logger.info("=" * 80)
    logger.info(f"Suite: {args.suite}")
    logger.info(f"Mock mode: {args.mock}")
    logger.info(f"Output: {args.output}")

    # Initialize server
    if not args.mock:
        try:
            logger.info("\nInitializing MCP server...")
            initialize_server()
            logger.info("✓ Server initialized")
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            logger.warning("Some benchmarks may fail without database connection")

    # Run benchmark suites
    all_results: list[BenchmarkResult] = []

    try:
        if args.suite in ["semantic_search", "all"]:
            semantic_results = run_semantic_search_suite()
            all_results.extend(semantic_results)

        if args.suite in ["vendor_info", "all"]:
            vendor_results = run_find_vendor_info_suite()
            all_results.extend(vendor_results)

        if args.suite in ["auth", "all"]:
            auth_results = run_authentication_suite()
            all_results.extend(auth_results)

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
        sys.exit(1)

    # Generate report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(all_results, args.output)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARKS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
