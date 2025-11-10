"""Type stubs for boost weight optimization framework.

Provides type definitions for:
- RelevanceMetrics: NDCG@10, MRR, Precision@K metrics
- BoostWeightConfig: Configuration for individual boost weights
- OptimizationResult: Results from optimization experiments
- BoostWeightOptimizer: Main optimizer class for framework
- RelevanceCalculator: Calculate metrics from ranked results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from src.search.boosting import BoostWeights
from src.search.results import SearchResult

# Type variables
T = TypeVar("T")

@dataclass(frozen=True)
class RelevanceMetrics:
    """Calculated relevance metrics for search results.

    Attributes:
        ndcg_10: Normalized Discounted Cumulative Gain at rank 10
        mrr: Mean Reciprocal Rank
        precision_5: Precision at top 5 results
        precision_10: Precision at top 10 results
        map_score: Mean Average Precision
    """
    ndcg_10: float
    mrr: float
    precision_5: float
    precision_10: float
    map_score: float

@dataclass(frozen=True)
class RelevanceMetricsBundle:
    """Bundle of metrics for comparison and analysis.

    Attributes:
        baseline: Baseline metrics (current/default weights)
        improved: Improved metrics (after optimization)
        improvement_delta: Absolute improvement for each metric
        improvement_percent: Percentage improvement for each metric
    """
    baseline: RelevanceMetrics
    improved: RelevanceMetrics
    improvement_delta: dict[str, float]
    improvement_percent: dict[str, float]

@dataclass(frozen=True)
class OptimizedBoostConfig:
    """Configuration for optimized boost weights.

    Attributes:
        weights: BoostWeights configuration
        metrics: RelevanceMetrics achieved with these weights
        composite_score: Composite score for comparison
        note: Human-readable description of configuration
    """
    weights: BoostWeights
    metrics: RelevanceMetrics
    composite_score: float
    note: str

@dataclass(frozen=True)
class OptimizationResult:
    """Result from individual factor analysis or combined optimization.

    Attributes:
        factor_name: Name of factor being optimized (or "combined")
        results: List of OptimizedBoostConfig sorted by composite_score
        best: Highest scoring configuration
        baseline: Baseline metrics for comparison
        experiment_id: Unique experiment identifier
    """
    factor_name: str
    results: list[OptimizedBoostConfig]
    best: OptimizedBoostConfig
    baseline: OptimizedBoostConfig
    experiment_id: str

@dataclass(frozen=True)
class RelevanceCalculator:
    """Calculator for relevance metrics from ranked search results."""

    def calculate_metrics(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
    ) -> RelevanceMetrics:
        """Calculate NDCG@10, MRR, Precision metrics.

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant

        Returns:
            RelevanceMetrics with calculated values
        """
        ...

    def calculate_ndcg_at_k(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@K metric.

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant
            k: Cutoff for ranking (default 10)

        Returns:
            NDCG@K score in range [0.0, 1.0]
        """
        ...

    def calculate_mrr(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant

        Returns:
            MRR score in range [0.0, 1.0]
        """
        ...

    def calculate_precision_at_k(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
        k: int = 10,
    ) -> float:
        """Calculate Precision@K metric.

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant
            k: Cutoff for ranking (default 10)

        Returns:
            Precision@K score in range [0.0, 1.0]
        """
        ...

class TestQuerySet:
    """Collection of test queries with ground truth relevance judgments.

    Attributes:
        queries: Dict mapping query string to relevant result indices
        name: Name of test set
    """

    queries: dict[str, set[int]]
    name: str

    def __init__(self, name: str, queries: dict[str, set[int]]) -> None:
        """Initialize test query set.

        Args:
            name: Name of test set
            queries: Dict mapping query to relevant result indices
        """
        ...

    def add_query(self, query: str, relevant_indices: set[int]) -> None:
        """Add a test query with ground truth.

        Args:
            query: Search query
            relevant_indices: Set of relevant result indices
        """
        ...

    def __len__(self) -> int:
        """Return number of test queries."""
        ...

class BoostWeightOptimizer:
    """Framework for optimizing boost weights to maximize relevance metrics.

    Supports individual factor analysis and combined optimization.
    """

    def __init__(
        self,
        boosting_system: Any,
        search_system: Any,
        calculator: RelevanceCalculator | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            boosting_system: BoostingSystem instance
            search_system: Hybrid search system
            calculator: Optional RelevanceCalculator (creates default if None)
        """
        ...

    def analyze_individual_factors(
        self,
        test_queries: TestQuerySet,
        baseline_weights: BoostWeights | None = None,
        weight_ranges: dict[str, tuple[float, float]] | None = None,
        step_size: float = 0.05,
    ) -> dict[str, OptimizationResult]:
        """Analyze impact of individual boost factors.

        Tests each factor independently by varying its weight while
        keeping other factors at baseline. Returns impact of each factor
        on relevance metrics.

        Args:
            test_queries: Test query set with ground truth
            baseline_weights: Baseline weights (uses defaults if None)
            weight_ranges: Dict mapping factor to (min, max) range
            step_size: Step size for weight variations

        Returns:
            Dict mapping factor name to OptimizationResult
        """
        ...

    def optimize_combined_weights(
        self,
        test_queries: TestQuerySet,
        weight_ranges: dict[str, tuple[float, float]] | None = None,
        method: str = "grid",
        step_size: float = 0.05,
        sample_size: int | None = None,
    ) -> OptimizationResult:
        """Optimize all boost weights jointly.

        Uses grid search or Bayesian optimization to find optimal
        combination of all boost weights.

        Args:
            test_queries: Test query set with ground truth
            weight_ranges: Dict mapping factor to (min, max) range
            method: "grid" for grid search, "bayesian" for Bayesian optimization
            step_size: Step size for grid search
            sample_size: Sample size for Bayesian optimization

        Returns:
            OptimizationResult with best configuration
        """
        ...

    def measure_baseline(
        self,
        test_queries: TestQuerySet,
        weights: BoostWeights | None = None,
    ) -> OptimizedBoostConfig:
        """Measure baseline metrics for given weights.

        Args:
            test_queries: Test query set with ground truth
            weights: Boost weights to measure (uses defaults if None)

        Returns:
            OptimizedBoostConfig with baseline metrics
        """
        ...

    def generate_report(
        self,
        results: dict[str, OptimizationResult] | OptimizationResult,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate comparison report from optimization results.

        Args:
            results: Results from individual or combined optimization
            output_path: Optional path to write JSON report

        Returns:
            Dict with report data (also writes to file if path provided)
        """
        ...
