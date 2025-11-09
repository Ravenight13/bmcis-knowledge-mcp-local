"""Type stubs for SearchProfiler - query performance measurement and optimization.

Provides comprehensive performance profiling system for search queries including:
- Query timing breakdown (planning, execution, result parsing)
- Index usage analysis via EXPLAIN ANALYZE integration
- Performance metrics collection and reporting
- Query caching analysis
- Automatic slow query detection (>100ms)
"""

from dataclasses import dataclass
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import time
from typing import Any, TypeVar, Generic

# Type variables for generic query results
QueryResultType = TypeVar("QueryResultType")
ProfileKeyType = TypeVar("ProfileKeyType", bound=str)

@dataclass(frozen=True)
class TimingBreakdown:
    """Timing measurements for different query stages.

    Attributes:
        planning_ms: Time spent on query planning (EXPLAIN step).
        execution_ms: Time spent on query execution (index scan, joins).
        fetch_ms: Time spent fetching and parsing results.
        total_ms: Total query time (planning + execution + fetch).
    """
    planning_ms: float
    execution_ms: float
    fetch_ms: float
    total_ms: float

@dataclass(frozen=True)
class ExplainAnalyzePlan:
    """Query execution plan from EXPLAIN ANALYZE output.

    Attributes:
        plan_json: Raw JSON representation of execution plan.
        nodes_with_children: List of plan nodes with their children.
        index_scans: Count of index scan operations.
        sequential_scans: Count of sequential (full table) scans.
        planning_time_ms: Server-side query planning time.
        execution_time_ms: Server-side query execution time.
        rows_returned: Total rows returned by query.
        total_cost_estimate: Planner's estimated cost.
        actual_cost: Actual execution cost.
    """
    plan_json: dict[str, Any]
    nodes_with_children: list[dict[str, Any]]
    index_scans: int
    sequential_scans: int
    planning_time_ms: float
    execution_time_ms: float
    rows_returned: int
    total_cost_estimate: float
    actual_cost: float

@dataclass(frozen=True)
class CacheMetrics:
    """Cache performance metrics for query results.

    Attributes:
        cache_hits: Number of cache hits for this query pattern.
        cache_misses: Number of cache misses.
        hit_rate_percent: Cache hit rate as percentage (0-100).
        avg_cache_latency_ms: Average latency for cached results.
        avg_db_latency_ms: Average latency for database queries.
        memory_saved_bytes: Approximate memory saved via caching.
    """
    cache_hits: int
    cache_misses: int
    hit_rate_percent: float
    avg_cache_latency_ms: float
    avg_db_latency_ms: float
    memory_saved_bytes: int

@dataclass(frozen=True)
class ProfileResult:
    """Complete performance profile for a single query.

    Attributes:
        query_name: Identifier for the query (e.g., 'vector_search').
        query_text: The actual SQL query string.
        timing: TimingBreakdown with stage-specific measurements.
        result_count: Number of results returned.
        result_size_bytes: Approximate size of result set in bytes.
        is_slow_query: Whether query exceeded 100ms threshold.
        explain_plan: EXPLAIN ANALYZE output and metrics.
        cache_metrics: Cache performance data (if caching enabled).
        index_hit_rate: Percentage of results from index (vs sequential scan).
        memory_peak_bytes: Peak memory usage during query.
        execution_timestamp: When query was executed (ISO 8601).
        metadata: Custom metadata dictionary for additional context.
    """
    query_name: str
    query_text: str
    timing: TimingBreakdown
    result_count: int
    result_size_bytes: int
    is_slow_query: bool
    explain_plan: ExplainAnalyzePlan | None
    cache_metrics: CacheMetrics | None
    index_hit_rate: float
    memory_peak_bytes: int
    execution_timestamp: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class BenchmarkResult:
    """Results from performance benchmarking across multiple runs.

    Attributes:
        test_name: Name of the benchmark test.
        runs: Number of benchmark iterations.
        min_ms: Minimum query time across all runs.
        max_ms: Maximum query time across all runs.
        mean_ms: Average query time.
        median_ms: Median query time.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        std_dev_ms: Standard deviation of latency.
        total_runs_ms: Total time for all benchmark runs.
        avg_result_count: Average number of results returned.
        index_usage_percent: Percentage of queries using indexes.
    """
    test_name: str
    runs: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    total_runs_ms: float
    avg_result_count: float
    index_usage_percent: float

class SearchProfiler(Generic[ProfileKeyType]):
    """Comprehensive performance profiling for search queries.

    Provides context manager-based profiling, automatic slow query detection,
    EXPLAIN ANALYZE integration, and detailed metrics collection.

    Example:
        >>> profiler = SearchProfiler()
        >>> with profiler.profile("vector_search", explain=True):
        ...     results = search_vector(embedding)
        >>> profile = profiler.get_profile("vector_search")
        >>> print(f"Query time: {profile.timing.total_ms}ms")
        >>> if profile.is_slow_query:
        ...     print("Slow query detected!")
    """

    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        enable_explain_analyze: bool = True,
        enable_caching: bool = False,
        profile_memory: bool = False,
    ) -> None:
        """Initialize profiler with configuration.

        Args:
            slow_query_threshold_ms: Queries exceeding this time are marked slow.
            enable_explain_analyze: Automatically run EXPLAIN ANALYZE on slow queries.
            enable_caching: Track cache metrics if query result caching enabled.
            profile_memory: Track peak memory usage during queries.
        """
        ...

    @contextmanager
    def profile(
        self,
        query_name: ProfileKeyType,
        query_text: str = "",
        explain: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Context manager for profiling a single query execution.

        Usage:
            with profiler.profile("search_query"):
                results = execute_search()

        Args:
            query_name: Unique identifier for this query.
            query_text: Optional SQL text for EXPLAIN ANALYZE.
            explain: Whether to run EXPLAIN ANALYZE (auto-runs if slow).
            metadata: Custom metadata to include in profile result.

        Yields:
            None (use profiling context block for query execution).

        Raises:
            ValueError: If query_name already being profiled (nested contexts).
        """
        ...

    def get_profile(self, query_name: ProfileKeyType) -> ProfileResult | None:
        """Retrieve profile result for a query.

        Returns:
            ProfileResult if query has been profiled, None otherwise.
        """
        ...

    def get_all_profiles(self) -> dict[ProfileKeyType, ProfileResult]:
        """Get all collected profile results.

        Returns:
            Dictionary mapping query names to ProfileResult objects.
        """
        ...

    def clear_profile(self, query_name: ProfileKeyType | None = None) -> None:
        """Clear profile data (single query or all).

        Args:
            query_name: Specific query to clear. If None, clears all profiles.
        """
        ...

    def benchmark(
        self,
        query_name: ProfileKeyType,
        query_fn: Callable[[], QueryResultType],
        runs: int = 10,
        explain: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run performance benchmark for a query function.

        Executes query_fn multiple times and collects timing statistics.

        Args:
            query_name: Identifier for benchmark.
            query_fn: Callable that executes the query.
            runs: Number of benchmark iterations (default 10).
            explain: Whether to collect EXPLAIN ANALYZE for each run.
            metadata: Custom metadata for benchmark context.

        Returns:
            BenchmarkResult with statistics across all runs.
        """
        ...

    def compare_queries(
        self,
        baseline_fn: Callable[[], QueryResultType],
        optimized_fn: Callable[[], QueryResultType],
        baseline_name: str = "baseline",
        optimized_name: str = "optimized",
        runs: int = 10,
    ) -> dict[str, Any]:
        """Compare performance of two query implementations.

        Benchmarks both functions and calculates improvement metrics.

        Args:
            baseline_fn: Baseline query function.
            optimized_fn: Optimized query function.
            baseline_name: Name for baseline in results.
            optimized_name: Name for optimized in results.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with baseline/optimized results and improvement metrics:
            - 'baseline': BenchmarkResult for baseline
            - 'optimized': BenchmarkResult for optimized
            - 'improvement_percent': % improvement in mean latency
            - 'speedup': Multiple of improvement (e.g., 2.5x faster)
        """
        ...

    def get_slow_queries(
        self, threshold_ms: float | None = None
    ) -> list[ProfileResult]:
        """Get all queries exceeding slow threshold.

        Args:
            threshold_ms: Custom threshold (uses instance default if None).

        Returns:
            List of ProfileResult objects for slow queries.
        """
        ...

    def export_profiles_json(self) -> dict[str, Any]:
        """Export all profiles as JSON-serializable dictionary.

        Returns:
            Dictionary with all profiles in JSON format.
        """
        ...

    def print_summary(
        self,
        top_n: int = 5,
        include_slow_queries: bool = True,
    ) -> None:
        """Print human-readable summary of profiling results.

        Args:
            top_n: Number of slowest queries to display.
            include_slow_queries: Whether to highlight slow query details.
        """
        ...


class IndexAnalyzer:
    """Analyzes index usage and provides optimization recommendations.

    Examines EXPLAIN ANALYZE output to determine if indexes are being used
    effectively and provides actionable recommendations for optimization.
    """

    @staticmethod
    def analyze_index_usage(explain_plan: ExplainAnalyzePlan) -> dict[str, Any]:
        """Analyze index usage in query plan.

        Args:
            explain_plan: EXPLAIN ANALYZE plan to analyze.

        Returns:
            Dictionary with analysis results:
            - 'indexes_used': List of index names found in plan
            - 'sequential_scans': Count of full table scans
            - 'bitmap_scans': Count of bitmap index scans
            - 'index_scan_count': Count of index scans
            - 'efficiency_score': 0-100 score of index efficiency
            - 'recommendations': List of optimization recommendations
        """
        ...

    @staticmethod
    def recommend_indexes(
        explain_plan: ExplainAnalyzePlan,
        table_name: str = "",
    ) -> list[str]:
        """Generate index recommendations based on query plan.

        Args:
            explain_plan: EXPLAIN ANALYZE plan to analyze.
            table_name: Table name for context (optional).

        Returns:
            List of recommended CREATE INDEX statements.
        """
        ...

    @staticmethod
    def get_index_efficiency_score(explain_plan: ExplainAnalyzePlan) -> float:
        """Calculate efficiency score for index usage (0-100).

        Higher scores indicate better index utilization.

        Args:
            explain_plan: EXPLAIN ANALYZE plan to score.

        Returns:
            Float between 0-100 representing efficiency score.
        """
        ...


class PerformanceOptimizer:
    """Suggests and implements performance optimizations.

    Analyzes profile data and provides concrete optimization strategies
    with estimated impact.
    """

    @staticmethod
    def suggest_optimizations(
        profile: ProfileResult,
    ) -> list[dict[str, Any]]:
        """Suggest optimizations for a profiled query.

        Args:
            profile: ProfileResult to analyze.

        Returns:
            List of optimization suggestions, each containing:
            - 'category': Type of optimization
            - 'description': Human-readable suggestion
            - 'estimated_speedup_percent': Expected improvement
            - 'implementation_complexity': 'low'|'medium'|'high'
            - 'priority': 'high'|'medium'|'low'
        """
        ...

    @staticmethod
    def calculate_hnsw_impact(
        ef_search: int,
        current_latency_ms: float,
    ) -> dict[str, Any]:
        """Estimate latency impact of changing HNSW ef_search parameter.

        Args:
            ef_search: New ef_search value to test.
            current_latency_ms: Baseline query latency.

        Returns:
            Dictionary with estimated metrics:
            - 'estimated_latency_ms': Predicted new latency
            - 'speedup_percent': Expected improvement percentage
            - 'accuracy_impact': Estimated effect on search accuracy
        """
        ...

    @staticmethod
    def calculate_result_limit_impact(
        result_count: int,
        top_k: int,
    ) -> dict[str, float]:
        """Estimate impact of limiting result set size.

        Args:
            result_count: Current result set size.
            top_k: Proposed limit on results.

        Returns:
            Dictionary with impact estimates:
            - 'latency_reduction_percent': Expected latency improvement
            - 'memory_reduction_percent': Expected memory savings
            - 'accuracy_loss_risk': Risk of missing relevant results (0-1)
        """
        ...
