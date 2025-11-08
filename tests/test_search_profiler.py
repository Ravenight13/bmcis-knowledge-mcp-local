"""Comprehensive tests for SearchProfiler performance profiling system.

Tests cover:
- Query profiling with timing measurements
- Automatic slow query detection
- Benchmark execution and statistics calculation
- Performance comparison between implementations
- Profile export and reporting
- Index analysis and optimization suggestions
"""

import time
import pytest
from typing import Any

from src.search.profiler import (
    SearchProfiler,
    ProfileResult,
    TimingBreakdown,
    BenchmarkResult,
    ExplainAnalyzePlan,
    CacheMetrics,
    IndexAnalyzer,
    PerformanceOptimizer,
)


class TestTimingBreakdown:
    """Tests for TimingBreakdown dataclass."""

    def test_timing_breakdown_creation(self) -> None:
        """Test creating TimingBreakdown with all fields."""
        timing: TimingBreakdown = TimingBreakdown(
            planning_ms=1.5,
            execution_ms=50.0,
            fetch_ms=20.0,
            total_ms=71.5,
        )
        assert timing.planning_ms == 1.5
        assert timing.execution_ms == 50.0
        assert timing.fetch_ms == 20.0
        assert timing.total_ms == 71.5

    def test_timing_breakdown_immutable(self) -> None:
        """Test that TimingBreakdown is immutable (frozen dataclass)."""
        timing: TimingBreakdown = TimingBreakdown(
            planning_ms=1.0,
            execution_ms=50.0,
            fetch_ms=20.0,
            total_ms=71.0,
        )
        with pytest.raises(AttributeError):
            timing.planning_ms = 2.0  # type: ignore


class TestExplainAnalyzePlan:
    """Tests for ExplainAnalyzePlan dataclass."""

    def test_explain_plan_creation(self) -> None:
        """Test creating ExplainAnalyzePlan with plan data."""
        plan: ExplainAnalyzePlan = ExplainAnalyzePlan(
            plan_json={"Node Type": "Index Scan"},
            nodes_with_children=[{"type": "Index", "name": "idx_test"}],
            index_scans=5,
            sequential_scans=2,
            planning_time_ms=0.5,
            execution_time_ms=15.2,
            rows_returned=100,
            total_cost_estimate=50.0,
            actual_cost=45.0,
        )
        assert plan.index_scans == 5
        assert plan.sequential_scans == 2
        assert plan.rows_returned == 100


class TestSearchProfiler:
    """Tests for SearchProfiler main functionality."""

    def test_profiler_initialization(self) -> None:
        """Test creating SearchProfiler with custom settings."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=50.0,
            enable_explain_analyze=True,
            enable_caching=True,
            profile_memory=True,
        )
        assert profiler.slow_query_threshold_ms == 50.0
        assert profiler.enable_explain_analyze is True
        assert profiler.enable_caching is True
        assert profiler.profile_memory is True

    def test_simple_profile_context_manager(self) -> None:
        """Test profiling a simple operation using context manager."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with profiler.profile("test_query"):
            time.sleep(0.05)  # 50ms sleep

        profile: ProfileResult | None = profiler.get_profile("test_query")
        assert profile is not None
        assert profile.query_name == "test_query"
        assert profile.timing.total_ms >= 45.0  # Allow some timing variance

    def test_profile_with_slow_query_detection(self) -> None:
        """Test that slow queries are detected automatically."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=50.0
        )

        with profiler.profile("slow_query"):
            time.sleep(0.06)  # 60ms sleep, exceeds 50ms threshold

        profile: ProfileResult | None = profiler.get_profile("slow_query")
        assert profile is not None
        assert profile.is_slow_query is True

    def test_profile_normal_query(self) -> None:
        """Test that normal queries are not marked slow."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=100.0
        )

        with profiler.profile("fast_query"):
            time.sleep(0.02)  # 20ms sleep, under 100ms threshold

        profile: ProfileResult | None = profiler.get_profile("fast_query")
        assert profile is not None
        assert profile.is_slow_query is False

    def test_profile_with_metadata(self) -> None:
        """Test profiling with custom metadata."""
        profiler: SearchProfiler[str] = SearchProfiler()
        metadata: dict[str, Any] = {"query_type": "vector_search", "results": 10}

        with profiler.profile("query_with_meta", metadata=metadata):
            time.sleep(0.01)

        profile: ProfileResult | None = profiler.get_profile("query_with_meta")
        assert profile is not None
        assert profile.metadata == metadata
        assert profile.metadata["query_type"] == "vector_search"

    def test_nested_profile_raises_error(self) -> None:
        """Test that nested profiling contexts raise ValueError."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with pytest.raises(ValueError):
            with profiler.profile("outer"):
                with profiler.profile("inner"):
                    pass

    def test_get_nonexistent_profile(self) -> None:
        """Test getting a profile that doesn't exist."""
        profiler: SearchProfiler[str] = SearchProfiler()
        profile: ProfileResult | None = profiler.get_profile("nonexistent")
        assert profile is None

    def test_get_all_profiles(self) -> None:
        """Test retrieving all profiles at once."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with profiler.profile("query_1"):
            time.sleep(0.01)
        with profiler.profile("query_2"):
            time.sleep(0.01)

        all_profiles: dict[str, ProfileResult] = profiler.get_all_profiles()
        assert len(all_profiles) == 2
        assert "query_1" in all_profiles
        assert "query_2" in all_profiles

    def test_clear_single_profile(self) -> None:
        """Test clearing a single profile."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with profiler.profile("query_1"):
            pass
        with profiler.profile("query_2"):
            pass

        profiler.clear_profile("query_1")
        all_profiles: dict[str, ProfileResult] = profiler.get_all_profiles()
        assert len(all_profiles) == 1
        assert "query_2" in all_profiles

    def test_clear_all_profiles(self) -> None:
        """Test clearing all profiles."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with profiler.profile("query_1"):
            pass
        with profiler.profile("query_2"):
            pass

        profiler.clear_profile()  # None clears all
        all_profiles: dict[str, ProfileResult] = profiler.get_all_profiles()
        assert len(all_profiles) == 0


class TestBenchmarking:
    """Tests for benchmarking functionality."""

    def test_benchmark_single_query(self) -> None:
        """Test benchmarking a single query function."""
        profiler: SearchProfiler[str] = SearchProfiler()

        def dummy_query() -> list[int]:
            time.sleep(0.01)
            return [1, 2, 3]

        result: BenchmarkResult = profiler.benchmark(
            "dummy_test",  # type: ignore
            dummy_query,
            runs=5,
        )

        assert result.test_name == "dummy_test"
        assert result.runs == 5
        assert result.mean_ms >= 8.0  # At least 8ms (5 * 0.01s / 5 = 0.01 but overhead)
        assert result.min_ms <= result.mean_ms
        assert result.max_ms >= result.mean_ms
        assert result.std_dev_ms >= 0.0

    def test_benchmark_percentiles(self) -> None:
        """Test that percentile calculations are reasonable."""
        profiler: SearchProfiler[str] = SearchProfiler()

        def variable_query() -> int:
            # Return variable sleep times to create distribution
            import random

            time.sleep(0.005 + random.random() * 0.01)
            return 42

        result: BenchmarkResult = profiler.benchmark(
            "variable_test",  # type: ignore
            variable_query,
            runs=20,
        )

        assert result.p95_ms >= result.median_ms
        assert result.p99_ms >= result.p95_ms
        assert result.min_ms <= result.median_ms <= result.max_ms

    def test_benchmark_result_count(self) -> None:
        """Test that benchmark tracks result counts."""
        profiler: SearchProfiler[str] = SearchProfiler()

        def return_items() -> list[int]:
            return [1, 2, 3, 4, 5]

        result: BenchmarkResult = profiler.benchmark(
            "items_test",  # type: ignore
            return_items,
            runs=3,
        )

        assert result.avg_result_count == 5.0

    def test_compare_queries(self) -> None:
        """Test comparing two query implementations."""
        profiler: SearchProfiler[str] = SearchProfiler()

        def baseline_query() -> list[int]:
            time.sleep(0.02)
            return [1, 2, 3]

        def optimized_query() -> list[int]:
            time.sleep(0.01)  # Faster
            return [1, 2, 3]

        comparison: dict[str, Any] = profiler.compare_queries(
            baseline_query,
            optimized_query,
            baseline_name="baseline",  # type: ignore
            optimized_name="optimized",  # type: ignore
            runs=5,
        )

        assert "baseline" in comparison
        assert "optimized" in comparison
        assert "improvement_percent" in comparison
        assert "speedup" in comparison
        assert comparison["speedup"] > 1.0  # Optimized should be faster
        assert comparison["improvement_percent"] > 0.0


class TestSlowQueries:
    """Tests for slow query detection and reporting."""

    def test_get_slow_queries(self) -> None:
        """Test retrieving slow queries."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=25.0
        )

        with profiler.profile("fast"):
            time.sleep(0.01)
        with profiler.profile("slow"):
            time.sleep(0.04)

        slow_queries: list[ProfileResult] = profiler.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0].query_name == "slow"

    def test_get_slow_queries_custom_threshold(self) -> None:
        """Test slow queries with custom threshold."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=100.0
        )

        with profiler.profile("moderate"):
            time.sleep(0.05)

        # Use higher threshold
        slow_queries: list[ProfileResult] = profiler.get_slow_queries(
            threshold_ms=30.0
        )
        assert len(slow_queries) == 1


class TestProfileExport:
    """Tests for profile export functionality."""

    def test_export_profiles_json(self) -> None:
        """Test exporting profiles as JSON."""
        profiler: SearchProfiler[str] = SearchProfiler()

        with profiler.profile("query_1"):
            time.sleep(0.01)

        exported: dict[str, Any] = profiler.export_profiles_json()

        assert "query_1" in exported
        assert "timing" in exported["query_1"]
        assert "total_ms" in exported["query_1"]["timing"]

    def test_export_empty_profiles(self) -> None:
        """Test exporting when no profiles exist."""
        profiler: SearchProfiler[str] = SearchProfiler()
        exported: dict[str, Any] = profiler.export_profiles_json()
        assert exported == {}


class TestIndexAnalyzer:
    """Tests for IndexAnalyzer utility class."""

    def test_analyze_good_index_usage(self) -> None:
        """Test analyzing a query with good index usage."""
        plan: ExplainAnalyzePlan = ExplainAnalyzePlan(
            plan_json={},
            nodes_with_children=[],
            index_scans=10,
            sequential_scans=0,
            planning_time_ms=0.5,
            execution_time_ms=5.0,
            rows_returned=100,
            total_cost_estimate=50.0,
            actual_cost=45.0,
        )

        analysis: dict[str, Any] = IndexAnalyzer.analyze_index_usage(plan)
        assert analysis["efficiency_score"] == 100.0
        assert analysis["index_scan_count"] == 10

    def test_analyze_poor_index_usage(self) -> None:
        """Test analyzing a query with poor index usage."""
        plan: ExplainAnalyzePlan = ExplainAnalyzePlan(
            plan_json={},
            nodes_with_children=[],
            index_scans=0,
            sequential_scans=5,
            planning_time_ms=2.0,
            execution_time_ms=50.0,
            rows_returned=1000,
            total_cost_estimate=500.0,
            actual_cost=480.0,
        )

        analysis: dict[str, Any] = IndexAnalyzer.analyze_index_usage(plan)
        assert analysis["efficiency_score"] == 0.0
        assert analysis["sequential_scans"] == 5
        assert len(analysis["recommendations"]) > 0

    def test_index_efficiency_score(self) -> None:
        """Test index efficiency score calculation."""
        plan: ExplainAnalyzePlan = ExplainAnalyzePlan(
            plan_json={},
            nodes_with_children=[],
            index_scans=7,
            sequential_scans=3,
            planning_time_ms=0.5,
            execution_time_ms=10.0,
            rows_returned=100,
            total_cost_estimate=50.0,
            actual_cost=45.0,
        )

        score: float = IndexAnalyzer.get_index_efficiency_score(plan)
        assert 70.0 <= score <= 71.0  # 7/10 * 100

    def test_recommend_indexes(self) -> None:
        """Test index recommendation generation."""
        plan: ExplainAnalyzePlan = ExplainAnalyzePlan(
            plan_json={},
            nodes_with_children=[],
            index_scans=0,
            sequential_scans=3,
            planning_time_ms=0.5,
            execution_time_ms=10.0,
            rows_returned=100,
            total_cost_estimate=50.0,
            actual_cost=45.0,
        )

        recommendations: list[str] = IndexAnalyzer.recommend_indexes(
            plan, "test_table"
        )
        assert len(recommendations) > 0


class TestPerformanceOptimizer:
    """Tests for PerformanceOptimizer utility class."""

    def test_suggest_optimizations_for_slow_query(self) -> None:
        """Test optimization suggestions for slow query."""
        profile: ProfileResult = ProfileResult(
            query_name="slow_query",
            query_text="SELECT * FROM large_table",
            timing=TimingBreakdown(
                planning_ms=1.0,
                execution_ms=150.0,
                fetch_ms=50.0,
                total_ms=201.0,
            ),
            result_count=10000,
            result_size_bytes=10_000_000,
            is_slow_query=True,
            explain_plan=None,
            cache_metrics=None,
            index_hit_rate=0.5,
            memory_peak_bytes=100_000_000,
            execution_timestamp="2025-11-08T12:00:00Z",
            metadata={},
        )

        suggestions: list[dict[str, Any]] = PerformanceOptimizer.suggest_optimizations(
            profile
        )
        assert len(suggestions) > 0
        assert any("Index" in s["category"] for s in suggestions)

    def test_suggest_optimizations_large_result_set(self) -> None:
        """Test optimization suggestions for large result sets."""
        profile: ProfileResult = ProfileResult(
            query_name="large_results",
            query_text="SELECT * FROM table",
            timing=TimingBreakdown(
                planning_ms=1.0, execution_ms=50.0, fetch_ms=25.0, total_ms=76.0
            ),
            result_count=5000,  # Large result set
            result_size_bytes=50_000_000,
            is_slow_query=False,
            explain_plan=None,
            cache_metrics=None,
            index_hit_rate=0.8,
            memory_peak_bytes=50_000_000,
            execution_timestamp="2025-11-08T12:00:00Z",
            metadata={},
        )

        suggestions: list[dict[str, Any]] = PerformanceOptimizer.suggest_optimizations(
            profile
        )
        assert any("Result Limiting" in s["category"] for s in suggestions)

    def test_calculate_hnsw_impact(self) -> None:
        """Test HNSW parameter impact calculation."""
        impact_low: dict[str, Any] = PerformanceOptimizer.calculate_hnsw_impact(
            ef_search=50, current_latency_ms=100.0
        )

        impact_high: dict[str, Any] = PerformanceOptimizer.calculate_hnsw_impact(
            ef_search=200, current_latency_ms=100.0
        )

        assert "estimated_latency_ms" in impact_low
        assert "speedup_percent" in impact_low
        assert "accuracy_impact" in impact_low

        # Higher ef_search should have better accuracy but slower latency
        assert impact_high["accuracy_impact"] > impact_low["accuracy_impact"]

    def test_calculate_result_limit_impact(self) -> None:
        """Test result limiting impact calculation."""
        impact: dict[str, float] = PerformanceOptimizer.calculate_result_limit_impact(
            result_count=1000, top_k=10
        )

        assert "latency_reduction_percent" in impact
        assert "memory_reduction_percent" in impact
        assert "accuracy_loss_risk" in impact

        # Limiting to 10 from 1000 should save 99% memory
        assert impact["memory_reduction_percent"] > 98.0
        assert impact["latency_reduction_percent"] > 98.0


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_cache_metrics_creation(self) -> None:
        """Test creating CacheMetrics."""
        metrics: CacheMetrics = CacheMetrics(
            cache_hits=100,
            cache_misses=20,
            hit_rate_percent=83.3,
            avg_cache_latency_ms=2.0,
            avg_db_latency_ms=50.0,
            memory_saved_bytes=1_000_000,
        )
        assert metrics.hit_rate_percent == 83.3
        assert metrics.cache_hits == 100

    def test_cache_metrics_high_hit_rate(self) -> None:
        """Test cache metrics with high hit rate."""
        metrics: CacheMetrics = CacheMetrics(
            cache_hits=950,
            cache_misses=50,
            hit_rate_percent=95.0,
            avg_cache_latency_ms=1.5,
            avg_db_latency_ms=75.0,
            memory_saved_bytes=10_000_000,
        )
        assert metrics.hit_rate_percent > 90.0


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_profile_result_creation(self) -> None:
        """Test creating a complete ProfileResult."""
        profile: ProfileResult = ProfileResult(
            query_name="test_query",
            query_text="SELECT * FROM knowledge_base",
            timing=TimingBreakdown(
                planning_ms=0.5,
                execution_ms=25.0,
                fetch_ms=10.0,
                total_ms=35.5,
            ),
            result_count=50,
            result_size_bytes=500_000,
            is_slow_query=False,
            explain_plan=None,
            cache_metrics=None,
            index_hit_rate=0.85,
            memory_peak_bytes=5_000_000,
            execution_timestamp="2025-11-08T12:00:00Z",
            metadata={"search_type": "vector"},
        )

        assert profile.query_name == "test_query"
        assert profile.timing.total_ms == 35.5
        assert profile.index_hit_rate == 0.85


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self) -> None:
        """Test creating a BenchmarkResult."""
        result: BenchmarkResult = BenchmarkResult(
            test_name="vector_search",
            runs=100,
            min_ms=10.0,
            max_ms=50.0,
            mean_ms=25.0,
            median_ms=24.0,
            p95_ms=45.0,
            p99_ms=48.0,
            std_dev_ms=8.0,
            total_runs_ms=2500.0,
            avg_result_count=10.5,
            index_usage_percent=95.0,
        )

        assert result.test_name == "vector_search"
        assert result.runs == 100
        assert result.p95_ms > result.median_ms
        assert result.p99_ms > result.p95_ms


class TestProfilerIntegration:
    """Integration tests for complete profiling workflows."""

    def test_complete_profiling_workflow(self) -> None:
        """Test a complete profiling workflow."""
        profiler: SearchProfiler[str] = SearchProfiler(
            slow_query_threshold_ms=50.0
        )

        # Profile multiple queries
        for i in range(3):
            with profiler.profile(f"query_{i}"):
                time.sleep(0.01 * (i + 1))

        # Verify all profiles created
        all_profiles: dict[str, ProfileResult] = profiler.get_all_profiles()
        assert len(all_profiles) == 3

        # Verify timing increases with sleep time
        query_0: ProfileResult | None = profiler.get_profile("query_0")
        query_2: ProfileResult | None = profiler.get_profile("query_2")
        assert query_0 is not None
        assert query_2 is not None
        assert query_2.timing.total_ms > query_0.timing.total_ms

    def test_profiling_with_benchmarking(self) -> None:
        """Test combining profiling and benchmarking."""
        profiler: SearchProfiler[str] = SearchProfiler()

        def sample_query() -> list[int]:
            time.sleep(0.01)
            return [1, 2, 3, 4, 5]

        # Run benchmark
        result: BenchmarkResult = profiler.benchmark(
            "sample_bench",  # type: ignore
            sample_query,
            runs=5,
        )

        # Verify benchmark result
        assert result.runs == 5
        assert result.mean_ms > 8.0

        # Verify profiles created for each run
        all_profiles: dict[str, ProfileResult] = profiler.get_all_profiles()
        assert len(all_profiles) >= 5
