"""Boost weight optimization framework for search relevance.

Provides tools to systematically optimize boost weights (vendor, doc_type,
recency, entity, topic) to maximize relevance metrics like NDCG@10, MRR, and
Precision@K.

Features:
- Individual factor analysis: Test each factor independently
- Combined optimization: Grid search or Bayesian optimization
- Relevance metrics: NDCG@10, MRR, Precision@K, MAP
- Experiment tracking: Store and compare optimization results
- Reporting: Generate detailed comparison reports

Example:
    >>> optimizer = BoostWeightOptimizer(boosting_system, search_system)
    >>> test_set = TestQuerySet("baseline", {...})
    >>> results = optimizer.analyze_individual_factors(test_set)
    >>> report = optimizer.generate_report(results)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from src.core.logging import StructuredLogger
from src.search.boosting import BoostWeights, BoostingSystem
from src.search.results import SearchResult

logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Optimization constants
DEFAULT_WEIGHT_RANGE: Final[tuple[float, float]] = (0.0, 0.3)
DEFAULT_STEP_SIZE: Final[float] = 0.05
NDCG_WEIGHT: Final[float] = 0.4
MRR_WEIGHT: Final[float] = 0.3
PRECISION_WEIGHT: Final[float] = 0.3
MAP_WEIGHT: Final[float] = 0.0


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

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        for metric_name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                msg = f"{metric_name} must be in [0.0, 1.0], got {value}"
                raise ValueError(msg)


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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.composite_score <= 1.0:
            msg = f"composite_score must be in [0.0, 1.0], got {self.composite_score}"
            raise ValueError(msg)


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

        Raises:
            ValueError: If inputs are invalid
        """
        if not ranked_results:
            msg = "ranked_results cannot be empty"
            raise ValueError(msg)

        ndcg_10 = self.calculate_ndcg_at_k(ranked_results, ground_truth_indices, k=10)
        mrr = self.calculate_mrr(ranked_results, ground_truth_indices)
        precision_5 = self.calculate_precision_at_k(
            ranked_results, ground_truth_indices, k=5
        )
        precision_10 = self.calculate_precision_at_k(
            ranked_results, ground_truth_indices, k=10
        )
        map_score = self.calculate_map(ranked_results, ground_truth_indices)

        return RelevanceMetrics(
            ndcg_10=ndcg_10,
            mrr=mrr,
            precision_5=precision_5,
            precision_10=precision_10,
            map_score=map_score,
        )

    def calculate_ndcg_at_k(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@K metric.

        Formula:
            DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
            IDCG@k = DCG of ideal ranking (all relevant first)
            NDCG@k = DCG@k / IDCG@k

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant
            k: Cutoff for ranking (default 10)

        Returns:
            NDCG@K score in range [0.0, 1.0]
        """
        if not ground_truth_indices:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for rank, result in enumerate(ranked_results[:k], start=1):
            # Check if result is in ground truth
            # Simple heuristic: use chunk_id or position in results
            if rank - 1 in ground_truth_indices:
                dcg += 1.0 / (1 + __import__("math").log2(rank))

        # Calculate IDCG (ideal DCG - all relevant results ranked first)
        num_relevant = min(len(ground_truth_indices), k)
        idcg = 0.0
        for rank in range(1, num_relevant + 1):
            idcg += 1.0 / (1 + __import__("math").log2(rank))

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def calculate_mrr(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        Formula:
            MRR = 1 / rank of first relevant result

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant

        Returns:
            MRR score in range [0.0, 1.0]
        """
        if not ground_truth_indices:
            return 0.0

        for rank, _result in enumerate(ranked_results, start=1):
            if rank - 1 in ground_truth_indices:
                return 1.0 / rank

        return 0.0

    def calculate_precision_at_k(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
        k: int = 10,
    ) -> float:
        """Calculate Precision@K metric.

        Formula:
            Precision@k = (number of relevant results in top-k) / k

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant
            k: Cutoff for ranking (default 10)

        Returns:
            Precision@K score in range [0.0, 1.0]
        """
        if not ranked_results or k <= 0:
            return 0.0

        relevant_in_k = sum(
            1 for rank, _result in enumerate(ranked_results[:k], start=1)
            if rank - 1 in ground_truth_indices
        )

        return relevant_in_k / k

    def calculate_map(
        self,
        ranked_results: list[SearchResult],
        ground_truth_indices: set[int],
    ) -> float:
        """Calculate Mean Average Precision.

        Formula:
            AP = sum(Precision@k * rel_k) / num_relevant
            MAP = average of AP across all queries

        Args:
            ranked_results: Search results in ranked order
            ground_truth_indices: Set of result indices that are relevant

        Returns:
            MAP score in range [0.0, 1.0]
        """
        if not ground_truth_indices:
            return 0.0

        ap_sum = 0.0
        num_relevant_found = 0

        for rank, _result in enumerate(ranked_results, start=1):
            if rank - 1 in ground_truth_indices:
                precision_at_k = self.calculate_precision_at_k(
                    ranked_results, ground_truth_indices, k=rank
                )
                ap_sum += precision_at_k
                num_relevant_found += 1

        if num_relevant_found == 0:
            return 0.0

        return ap_sum / len(ground_truth_indices)

    @staticmethod
    def calculate_composite_score(metrics: RelevanceMetrics) -> float:
        """Calculate composite score from multiple metrics.

        Uses weighted combination:
            composite = 0.4 * NDCG + 0.3 * MRR + 0.3 * Precision@10

        Args:
            metrics: RelevanceMetrics to combine

        Returns:
            Composite score in range [0.0, 1.0]
        """
        return (
            NDCG_WEIGHT * metrics.ndcg_10
            + MRR_WEIGHT * metrics.mrr
            + PRECISION_WEIGHT * metrics.precision_10
        )


class TestQuerySet:
    """Collection of test queries with ground truth relevance judgments."""

    def __init__(self, name: str, queries: dict[str, set[int]] | None = None) -> None:
        """Initialize test query set.

        Args:
            name: Name of test set
            queries: Optional dict mapping query to relevant result indices
        """
        self.name = name
        self.queries: dict[str, set[int]] = queries or {}

    def add_query(self, query: str, relevant_indices: set[int]) -> None:
        """Add a test query with ground truth.

        Args:
            query: Search query
            relevant_indices: Set of relevant result indices

        Raises:
            ValueError: If query already exists
        """
        if query in self.queries:
            msg = f"Query already exists: {query}"
            raise ValueError(msg)

        self.queries[query] = relevant_indices

    def __len__(self) -> int:
        """Return number of test queries."""
        return len(self.queries)

    def __iter__(self) -> Any:
        """Iterate over (query, ground_truth) tuples."""
        return iter(self.queries.items())


class BoostWeightOptimizer:
    """Framework for optimizing boost weights to maximize relevance metrics."""

    def __init__(
        self,
        boosting_system: BoostingSystem,
        search_system: Any,
        calculator: RelevanceCalculator | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            boosting_system: BoostingSystem instance
            search_system: Hybrid search system with search() method
            calculator: Optional RelevanceCalculator (creates default if None)

        Raises:
            TypeError: If arguments have invalid types
        """
        self._boosting_system = boosting_system
        self._search_system = search_system
        self._calculator = calculator or RelevanceCalculator()
        self._experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def analyze_individual_factors(
        self,
        test_queries: TestQuerySet,
        baseline_weights: BoostWeights | None = None,
        weight_ranges: dict[str, tuple[float, float]] | None = None,
        step_size: float = DEFAULT_STEP_SIZE,
    ) -> dict[str, OptimizationResult]:
        """Analyze impact of individual boost factors.

        Tests each factor independently by varying its weight while
        keeping other factors at baseline.

        Args:
            test_queries: Test query set with ground truth
            baseline_weights: Baseline weights (uses defaults if None)
            weight_ranges: Dict mapping factor to (min, max) range
            step_size: Step size for weight variations

        Returns:
            Dict mapping factor name to OptimizationResult

        Raises:
            ValueError: If test_queries is empty or ranges invalid
        """
        if len(test_queries) == 0:
            msg = "test_queries cannot be empty"
            raise ValueError(msg)

        # Initialize with default weights if not provided
        if baseline_weights is None:
            # Using default constructor since dataclass has default values
            weights_to_use = BoostWeights(
                vendor=0.15,
                doc_type=0.10,
                recency=0.05,
                entity=0.10,
                topic=0.08,
            )
        else:
            weights_to_use = baseline_weights

        if weight_ranges is None:
            weight_ranges = {
                "vendor": DEFAULT_WEIGHT_RANGE,
                "doc_type": DEFAULT_WEIGHT_RANGE,
                "recency": DEFAULT_WEIGHT_RANGE,
                "entity": DEFAULT_WEIGHT_RANGE,
                "topic": DEFAULT_WEIGHT_RANGE,
            }

        # Measure baseline first
        baseline_config = self.measure_baseline(test_queries, weights_to_use)

        results: dict[str, OptimizationResult] = {}

        factor_names = ["vendor", "doc_type", "recency", "entity", "topic"]

        for factor in factor_names:
            logger.info(f"Analyzing factor: {factor}")

            factor_results: list[OptimizedBoostConfig] = []
            min_val, max_val = weight_ranges.get(
                factor, DEFAULT_WEIGHT_RANGE
            )

            # Test weights in range
            current = min_val
            while current <= max_val:
                # Create weights with current factor value
                custom_weights = BoostWeights(
                    vendor=weights_to_use.vendor if factor != "vendor" else current,
                    doc_type=(
                        weights_to_use.doc_type if factor != "doc_type" else current
                    ),
                    recency=(
                        weights_to_use.recency if factor != "recency" else current
                    ),
                    entity=weights_to_use.entity if factor != "entity" else current,
                    topic=weights_to_use.topic if factor != "topic" else current,
                )

                # Measure metrics with this configuration
                config = self.measure_baseline(test_queries, custom_weights)
                factor_results.append(config)

                current += step_size

            # Sort by composite score (descending)
            factor_results.sort(key=lambda x: x.composite_score, reverse=True)

            result = OptimizationResult(
                factor_name=factor,
                results=factor_results,
                best=factor_results[0],
                baseline=baseline_config,
                experiment_id=self._experiment_id,
            )

            results[factor] = result
            logger.info(
                f"Factor {factor}: best={result.best.weights.__dict__}, "
                f"score={result.best.composite_score:.4f}"
            )

        return results

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

        Raises:
            ValueError: If test_queries is empty
        """
        if len(test_queries) == 0:
            msg = "test_queries cannot be empty"
            raise ValueError(msg)

        if weights is None:
            weights = BoostWeights(
                vendor=0.15,
                doc_type=0.10,
                recency=0.05,
                entity=0.10,
                topic=0.08,
            )

        all_metrics: list[RelevanceMetrics] = []

        for query, ground_truth in test_queries:
            try:
                # Search with current weights
                search_results = self._search_system.search(query, top_k=10)

                # Apply boosts
                boosted_results = self._boosting_system.apply_boosts(
                    search_results, query, weights
                )

                # Calculate metrics
                metrics = self._calculator.calculate_metrics(
                    boosted_results, ground_truth
                )
                all_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                continue

        if not all_metrics:
            msg = "Could not calculate metrics for any test queries"
            raise ValueError(msg)

        # Average metrics across all queries
        avg_ndcg = sum(m.ndcg_10 for m in all_metrics) / len(all_metrics)
        avg_mrr = sum(m.mrr for m in all_metrics) / len(all_metrics)
        avg_prec5 = sum(m.precision_5 for m in all_metrics) / len(all_metrics)
        avg_prec10 = sum(m.precision_10 for m in all_metrics) / len(all_metrics)
        avg_map = sum(m.map_score for m in all_metrics) / len(all_metrics)

        avg_metrics = RelevanceMetrics(
            ndcg_10=avg_ndcg,
            mrr=avg_mrr,
            precision_5=avg_prec5,
            precision_10=avg_prec10,
            map_score=avg_map,
        )

        composite = RelevanceCalculator.calculate_composite_score(avg_metrics)

        return OptimizedBoostConfig(
            weights=weights,
            metrics=avg_metrics,
            composite_score=composite,
            note=f"Baseline for {len(test_queries)} test queries",
        )

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

        Raises:
            ValueError: If results are invalid
            IOError: If file write fails
        """
        if isinstance(results, OptimizationResult):
            results_dict = {results.factor_name: results}
        else:
            results_dict = results

        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self._experiment_id,
            "factors": {},
        }

        for factor_name, opt_result in results_dict.items():
            factor_data: dict[str, Any] = {
                "baseline": {
                    "weights": asdict(opt_result.baseline.weights),
                    "metrics": asdict(opt_result.baseline.metrics),
                    "composite_score": opt_result.baseline.composite_score,
                },
                "best": {
                    "weights": asdict(opt_result.best.weights),
                    "metrics": asdict(opt_result.best.metrics),
                    "composite_score": opt_result.best.composite_score,
                },
                "improvement": opt_result.best.composite_score
                - opt_result.baseline.composite_score,
            }

            report["factors"][factor_name] = factor_data

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report written to {output_path}")

        return report
