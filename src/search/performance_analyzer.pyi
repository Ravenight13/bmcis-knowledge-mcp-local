"""Type stubs for search performance analyzer module.

Provides type definitions for performance measurement and analysis across all
search components including vector search, BM25, hybrid search, and reranking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Optional
from collections.abc import Iterator

# Type variables
T = TypeVar("T")
MetricKeyType = TypeVar("MetricKeyType", bound=str)


@dataclass(frozen=True)
class VectorSearchMetrics:
    """Performance metrics for vector similarity search.

    Attributes:
        query_time_ms: Total vector search query time.
        embedding_time_ms: Time to prepare embedding.
        index_lookup_ms: Time for HNSW index lookup.
        result_fetch_ms: Time to fetch and parse results.
        vectors_searched: Number of vectors examined.
        results_returned: Number of results returned.
        throughput_qps: Queries per second throughput.
    """

    query_time_ms: float
    embedding_time_ms: float
    index_lookup_ms: float
    result_fetch_ms: float
    vectors_searched: int
    results_returned: int
    throughput_qps: float


@dataclass(frozen=True)
class BM25Metrics:
    """Performance metrics for BM25 full-text search.

    Attributes:
        query_time_ms: Total BM25 query time.
        tokenization_ms: Time to tokenize query.
        gin_lookup_ms: Time for GIN index lookup.
        result_fetch_ms: Time to fetch and parse results.
        documents_searched: Number of documents examined.
        results_returned: Number of results returned.
        throughput_qps: Queries per second throughput.
    """

    query_time_ms: float
    tokenization_ms: float
    gin_lookup_ms: float
    result_fetch_ms: float
    documents_searched: int
    results_returned: int
    throughput_qps: float


@dataclass(frozen=True)
class HybridSearchMetrics:
    """Performance metrics for hybrid search (vector + BM25).

    Attributes:
        total_time_ms: End-to-end hybrid search time.
        vector_time_ms: Vector search component time.
        bm25_time_ms: BM25 search component time.
        merge_time_ms: RRF merge and scoring time.
        rerank_time_ms: Cross-encoder reranking time.
        parallel_efficiency: Efficiency of parallel execution (0-1).
        total_results_merged: Total results from both searches.
        final_results_returned: Results after merging and filtering.
    """

    total_time_ms: float
    vector_time_ms: float
    bm25_time_ms: float
    merge_time_ms: float
    rerank_time_ms: float
    parallel_efficiency: float
    total_results_merged: int
    final_results_returned: int


@dataclass(frozen=True)
class RerankingMetrics:
    """Performance metrics for cross-encoder reranking.

    Attributes:
        total_time_ms: Total reranking time.
        model_load_ms: Time to load model.
        batch_encode_ms: Time to encode batch.
        scoring_ms: Time to compute scores.
        results_count: Number of results reranked.
        batch_size: Reranking batch size.
        throughput_results_per_sec: Reranking throughput.
    """

    total_time_ms: float
    model_load_ms: float
    batch_encode_ms: float
    scoring_ms: float
    results_count: int
    batch_size: int
    throughput_results_per_sec: float


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance metrics for search operations.

    Attributes:
        timestamp: When metrics were recorded.
        operation: Name of operation (vector_search, bm25_search, etc).
        vector_metrics: Metrics for vector search (if applicable).
        bm25_metrics: Metrics for BM25 search (if applicable).
        hybrid_metrics: Metrics for hybrid search (if applicable).
        rerank_metrics: Metrics for reranking (if applicable).
        metadata: Additional context and tags.
    """

    timestamp: str
    operation: str
    vector_metrics: VectorSearchMetrics | None
    bm25_metrics: BM25Metrics | None
    hybrid_metrics: HybridSearchMetrics | None
    rerank_metrics: RerankingMetrics | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PerformanceBaseline:
    """Baseline performance targets and thresholds.

    Attributes:
        vector_search_target_ms: Target vector search latency.
        bm25_target_ms: Target BM25 search latency.
        hybrid_target_ms: Target hybrid search latency.
        rerank_target_ms: Target reranking latency.
        throughput_min_qps: Minimum throughput requirement.
        vector_recall_target: Target recall at vector search threshold.
    """

    vector_search_target_ms: float
    bm25_target_ms: float
    hybrid_target_ms: float
    rerank_target_ms: float
    throughput_min_qps: float
    vector_recall_target: float


class SearchPerformanceAnalyzer:
    """Comprehensive performance analyzer for search components.

    Measures and analyzes latency, throughput, and resource usage across all
    search operations. Provides optimization recommendations based on baseline
    targets.
    """

    def __init__(self, baseline: PerformanceBaseline | None = None) -> None:
        """Initialize performance analyzer.

        Args:
            baseline: Optional performance baseline for comparison.
        """
        ...

    def measure_vector_search_latency(
        self,
        vector_size: int,
        index_size: int,
        num_queries: int = 10,
        embedding_time_ms: float = 5.0,
    ) -> VectorSearchMetrics:
        """Measure vector search latency across different scales.

        Args:
            vector_size: Dimension of vectors (typically 768).
            index_size: Number of vectors in index (e.g., 1K, 100K, 1M).
            num_queries: Number of queries to benchmark.
            embedding_time_ms: Estimated embedding generation time.

        Returns:
            VectorSearchMetrics with latency breakdown.
        """
        ...

    def measure_bm25_latency(
        self,
        query: str,
        corpus_size: int,
        num_queries: int = 10,
    ) -> BM25Metrics:
        """Measure BM25 search latency.

        Args:
            query: Sample query to use for measurement.
            corpus_size: Number of documents in corpus.
            num_queries: Number of queries to benchmark.

        Returns:
            BM25Metrics with latency breakdown.
        """
        ...

    def measure_hybrid_latency(
        self,
        query: str,
        vector_size: int = 768,
        parallel: bool = True,
        num_queries: int = 10,
    ) -> HybridSearchMetrics:
        """Measure end-to-end hybrid search latency.

        Args:
            query: Sample query.
            vector_size: Vector dimension.
            parallel: Whether to measure parallel execution.
            num_queries: Number of queries to benchmark.

        Returns:
            HybridSearchMetrics with latency breakdown.
        """
        ...

    def measure_reranking_latency(
        self,
        results_count: int,
        batch_size: int = 32,
        num_runs: int = 10,
    ) -> RerankingMetrics:
        """Measure cross-encoder reranking latency.

        Args:
            results_count: Number of results to rerank.
            batch_size: Batch size for encoding.
            num_runs: Number of benchmark runs.

        Returns:
            RerankingMetrics with latency breakdown.
        """
        ...

    def analyze_performance_scaling(
        self,
        vector_sizes: list[int],
        index_sizes: list[int],
    ) -> list[VectorSearchMetrics]:
        """Analyze performance scaling across vector and index sizes.

        Args:
            vector_sizes: List of vector dimensions to test.
            index_sizes: List of index sizes to test (1K, 10K, 100K, 1M).

        Returns:
            List of metrics for each size combination.
        """
        ...

    def compare_against_baseline(
        self, metrics: PerformanceMetrics
    ) -> dict[str, Any]:
        """Compare measured metrics against baseline targets.

        Args:
            metrics: Measured performance metrics.

        Returns:
            Dictionary with comparison results and recommendations.
        """
        ...

    def get_optimization_recommendations(
        self, metrics: PerformanceMetrics
    ) -> list[str]:
        """Generate optimization recommendations based on metrics.

        Args:
            metrics: Measured performance metrics.

        Returns:
            List of actionable optimization recommendations.
        """
        ...

    def profile_memory_usage(self, operation: str) -> dict[str, int]:
        """Profile memory usage for an operation.

        Args:
            operation: Operation name (vector_search, bm25_search, etc).

        Returns:
            Dictionary with memory metrics.
        """
        ...

    def get_throughput_estimate(
        self, latency_ms: float, num_workers: int = 1
    ) -> float:
        """Estimate throughput from latency.

        Args:
            latency_ms: Latency in milliseconds.
            num_workers: Number of concurrent workers.

        Returns:
            Queries per second throughput estimate.
        """
        ...


class CachePerformanceAnalyzer(Generic[MetricKeyType]):
    """Analyzer for query result caching performance.

    Measures cache hit rates, memory usage, and latency improvements from caching.
    """

    def __init__(self, max_cache_size: int = 1000) -> None:
        """Initialize cache performance analyzer.

        Args:
            max_cache_size: Maximum cache entries to track.
        """
        ...

    def record_cache_hit(self, key: MetricKeyType, latency_ms: float) -> None:
        """Record a cache hit with latency.

        Args:
            key: Cache key identifier.
            latency_ms: Latency of cached result retrieval.
        """
        ...

    def record_cache_miss(self, key: MetricKeyType, latency_ms: float) -> None:
        """Record a cache miss with database latency.

        Args:
            key: Cache key identifier.
            latency_ms: Latency of database query.
        """
        ...

    def get_hit_rate(self) -> float:
        """Get overall cache hit rate (0-1).

        Returns:
            Cache hit rate.
        """
        ...

    def get_latency_improvement(self) -> float:
        """Get latency improvement percentage from caching.

        Returns:
            Percentage improvement (0-100).
        """
        ...

    def get_memory_usage_mb(self) -> float:
        """Get estimated cache memory usage in MB.

        Returns:
            Memory usage estimate.
        """
        ...


class ParallelExecutionAnalyzer:
    """Analyzer for parallel execution efficiency.

    Measures the efficiency gains from parallel vector + BM25 search.
    """

    def __init__(self) -> None:
        """Initialize parallel execution analyzer."""
        ...

    def measure_sequential_latency(
        self, vector_time_ms: float, bm25_time_ms: float
    ) -> float:
        """Calculate expected sequential latency.

        Args:
            vector_time_ms: Vector search time.
            bm25_time_ms: BM25 search time.

        Returns:
            Total sequential time.
        """
        ...

    def measure_parallel_latency(
        self, vector_time_ms: float, bm25_time_ms: float
    ) -> float:
        """Calculate expected parallel latency.

        Args:
            vector_time_ms: Vector search time.
            bm25_time_ms: BM25 search time.

        Returns:
            Total parallel time (max of both searches).
        """
        ...

    def calculate_efficiency(
        self, vector_time_ms: float, bm25_time_ms: float
    ) -> float:
        """Calculate parallelization efficiency (0-1).

        Args:
            vector_time_ms: Vector search time.
            bm25_time_ms: BM25 search time.

        Returns:
            Efficiency ratio.
        """
        ...

    def calculate_speedup(
        self, vector_time_ms: float, bm25_time_ms: float
    ) -> float:
        """Calculate speedup from parallelization.

        Args:
            vector_time_ms: Vector search time.
            bm25_time_ms: BM25 search time.

        Returns:
            Speedup factor (e.g., 1.5x).
        """
        ...
