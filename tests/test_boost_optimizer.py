"""Comprehensive test suite for boost weight optimization framework.

Tests cover:
- RelevanceCalculator: NDCG@10, MRR, Precision@K, MAP metrics
- BoostWeightOptimizer: Individual factor analysis, baseline measurement
- TestQuerySet: Query management
- OptimizationResult: Result handling and comparison
- Integration: End-to-end optimization workflows
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from src.search.boost_optimizer import (
    RelevanceCalculator,
    RelevanceMetrics,
    TestQuerySet,
    BoostWeightOptimizer,
    OptimizedBoostConfig,
    OptimizationResult,
)
from src.search.boosting import BoostingSystem, BoostWeights
from src.search.results import SearchResult


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    today = date.today()
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="OpenAI API authentication guide",
            similarity_score=0.95,
            bm25_score=0.90,
            hybrid_score=0.92,
            rank=1,
            score_type="hybrid",
            source_file="doc1.md",
            source_category="api_docs",
            document_date=today - timedelta(days=5),
            context_header="API > Auth",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
            metadata={"vendor": "OpenAI"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Anthropic Claude API reference documentation",
            similarity_score=0.85,
            bm25_score=0.80,
            hybrid_score=0.82,
            rank=2,
            score_type="hybrid",
            source_file="doc2.md",
            source_category="reference",
            document_date=today - timedelta(days=45),
            context_header="Claude > Reference",
            chunk_index=0,
            total_chunks=3,
            chunk_token_count=384,
            metadata={"vendor": "Anthropic"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Google Cloud deployment and scaling",
            similarity_score=0.75,
            bm25_score=0.70,
            hybrid_score=0.72,
            rank=3,
            score_type="hybrid",
            source_file="doc3.md",
            source_category="guide",
            document_date=today - timedelta(days=60),
            context_header="GCP > Deployment",
            chunk_index=0,
            total_chunks=8,
            chunk_token_count=512,
            metadata={"vendor": "Google"},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="AWS Lambda optimization techniques",
            similarity_score=0.65,
            bm25_score=0.60,
            hybrid_score=0.62,
            rank=4,
            score_type="hybrid",
            source_file="doc4.md",
            source_category="guide",
            document_date=today - timedelta(days=3),
            context_header="AWS > Lambda",
            chunk_index=0,
            total_chunks=6,
            chunk_token_count=256,
            metadata={"vendor": "AWS"},
        ),
    ]


@pytest.fixture
def calculator() -> RelevanceCalculator:
    """Create RelevanceCalculator instance."""
    return RelevanceCalculator()


@pytest.fixture
def test_query_set() -> TestQuerySet:
    """Create test query set with ground truth."""
    queries = {
        "OpenAI authentication": {0},  # Result 0 is relevant
        "Claude API": {1},  # Result 1 is relevant
        "Google deployment": {2},  # Result 2 is relevant
        "AWS optimization": {3},  # Result 3 is relevant
    }
    return TestQuerySet("test_set", queries)


class TestRelevanceMetrics:
    """Test RelevanceMetrics dataclass."""

    def test_metrics_initialization(self) -> None:
        """Test creating RelevanceMetrics with valid values."""
        metrics = RelevanceMetrics(
            ndcg_10=0.8,
            mrr=0.9,
            precision_5=0.6,
            precision_10=0.5,
            map_score=0.7,
        )
        assert metrics.ndcg_10 == 0.8
        assert metrics.mrr == 0.9

    def test_metrics_validation_out_of_range(self) -> None:
        """Test that metrics validate value ranges."""
        with pytest.raises(ValueError):
            RelevanceMetrics(
                ndcg_10=1.5,  # Out of range
                mrr=0.9,
                precision_5=0.6,
                precision_10=0.5,
                map_score=0.7,
            )

    def test_metrics_boundary_values(self) -> None:
        """Test metrics with boundary values (0.0, 1.0)."""
        # All zeros
        metrics_zero = RelevanceMetrics(
            ndcg_10=0.0,
            mrr=0.0,
            precision_5=0.0,
            precision_10=0.0,
            map_score=0.0,
        )
        assert metrics_zero.ndcg_10 == 0.0

        # All ones
        metrics_one = RelevanceMetrics(
            ndcg_10=1.0,
            mrr=1.0,
            precision_5=1.0,
            precision_10=1.0,
            map_score=1.0,
        )
        assert metrics_one.ndcg_10 == 1.0


class TestRelevanceCalculator:
    """Test RelevanceCalculator metrics calculation."""

    def test_ndcg_calculation_perfect_ranking(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test NDCG when relevant result is ranked first."""
        ground_truth = {0}  # First result is relevant
        ndcg = calculator.calculate_ndcg_at_k(sample_results, ground_truth, k=10)
        assert ndcg > 0.9  # Should be very high

    def test_ndcg_calculation_no_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test NDCG when no relevant results present."""
        ground_truth: set[int] = set()  # No relevant results
        ndcg = calculator.calculate_ndcg_at_k(sample_results, ground_truth, k=10)
        assert ndcg == 0.0

    def test_mrr_calculation_first_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test MRR when relevant result is first."""
        ground_truth = {0}
        mrr = calculator.calculate_mrr(sample_results, ground_truth)
        assert mrr == 1.0  # 1 / 1 = 1.0

    def test_mrr_calculation_second_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test MRR when relevant result is second."""
        ground_truth = {1}
        mrr = calculator.calculate_mrr(sample_results, ground_truth)
        assert mrr == 0.5  # 1 / 2 = 0.5

    def test_mrr_no_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test MRR with no relevant results."""
        ground_truth: set[int] = set()
        mrr = calculator.calculate_mrr(sample_results, ground_truth)
        assert mrr == 0.0

    def test_precision_at_k_all_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test Precision@K when all top-K results are relevant."""
        ground_truth = {0, 1, 2, 3}  # All results relevant
        prec = calculator.calculate_precision_at_k(sample_results, ground_truth, k=4)
        assert prec == 1.0  # 4/4 = 1.0

    def test_precision_at_k_partial_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test Precision@K with some relevant results."""
        ground_truth = {0, 2}  # Results 0 and 2 relevant
        prec = calculator.calculate_precision_at_k(sample_results, ground_truth, k=4)
        assert prec == 0.5  # 2/4 = 0.5

    def test_precision_at_k_no_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test Precision@K with no relevant results."""
        ground_truth: set[int] = set()
        prec = calculator.calculate_precision_at_k(sample_results, ground_truth, k=4)
        assert prec == 0.0

    def test_map_calculation_multiple_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test MAP with multiple relevant results."""
        ground_truth = {0, 1}  # Results 0 and 1 relevant
        map_score = calculator.calculate_map(sample_results, ground_truth)
        assert map_score > 0.0

    def test_map_no_relevant(
        self, calculator: RelevanceCalculator, sample_results: list[SearchResult]
    ) -> None:
        """Test MAP with no relevant results."""
        ground_truth: set[int] = set()
        map_score = calculator.calculate_map(sample_results, ground_truth)
        assert map_score == 0.0

    def test_calculate_composite_score(
        self, calculator: RelevanceCalculator
    ) -> None:
        """Test composite score calculation."""
        metrics = RelevanceMetrics(
            ndcg_10=0.8,
            mrr=0.9,
            precision_5=0.6,
            precision_10=0.7,
            map_score=0.75,
        )
        composite = RelevanceCalculator.calculate_composite_score(metrics)
        # 0.4 * 0.8 + 0.3 * 0.9 + 0.3 * 0.7 = 0.32 + 0.27 + 0.21 = 0.80
        assert abs(composite - 0.80) < 0.001

    def test_calculate_metrics_full_workflow(
        self,
        calculator: RelevanceCalculator,
        sample_results: list[SearchResult],
    ) -> None:
        """Test complete metrics calculation workflow."""
        ground_truth = {0, 1}
        metrics = calculator.calculate_metrics(sample_results, ground_truth)
        assert isinstance(metrics, RelevanceMetrics)
        assert 0.0 <= metrics.ndcg_10 <= 1.0
        assert 0.0 <= metrics.mrr <= 1.0


class TestTestQuerySet:
    """Test TestQuerySet query management."""

    def test_query_set_initialization_empty(self) -> None:
        """Test creating empty query set."""
        qs = TestQuerySet("empty")
        assert len(qs) == 0

    def test_query_set_initialization_with_queries(self) -> None:
        """Test creating query set with initial queries."""
        queries = {"query1": {0, 1}, "query2": {2}}
        qs = TestQuerySet("test", queries)
        assert len(qs) == 2

    def test_query_set_add_query(self) -> None:
        """Test adding queries to set."""
        qs = TestQuerySet("test")
        qs.add_query("query1", {0, 1})
        assert len(qs) == 1

    def test_query_set_add_duplicate_raises(self) -> None:
        """Test that adding duplicate query raises error."""
        qs = TestQuerySet("test")
        qs.add_query("query1", {0})
        with pytest.raises(ValueError):
            qs.add_query("query1", {1})

    def test_query_set_iteration(self) -> None:
        """Test iterating over query set."""
        queries = {"q1": {0}, "q2": {1, 2}}
        qs = TestQuerySet("test", queries)
        items = list(qs)
        assert len(items) == 2
        assert items[0] == ("q1", {0}) or items[0] == ("q2", {1, 2})


class TestOptimizedBoostConfig:
    """Test OptimizedBoostConfig."""

    def test_config_initialization(self) -> None:
        """Test creating boost config."""
        weights = BoostWeights(
            vendor=0.15,
            doc_type=0.10,
            recency=0.05,
            entity=0.10,
            topic=0.08,
        )
        metrics = RelevanceMetrics(
            ndcg_10=0.8,
            mrr=0.9,
            precision_5=0.6,
            precision_10=0.7,
            map_score=0.75,
        )
        config = OptimizedBoostConfig(
            weights=weights,
            metrics=metrics,
            composite_score=0.8,
            note="Test config",
        )
        assert config.composite_score == 0.8

    def test_config_validation_composite_out_of_range(self) -> None:
        """Test that composite score is validated."""
        weights = BoostWeights()
        metrics = RelevanceMetrics(
            ndcg_10=0.8,
            mrr=0.9,
            precision_5=0.6,
            precision_10=0.7,
            map_score=0.75,
        )
        with pytest.raises(ValueError):
            OptimizedBoostConfig(
                weights=weights,
                metrics=metrics,
                composite_score=1.5,  # Invalid
                note="Bad config",
            )


class TestBoostWeightOptimizer:
    """Test BoostWeightOptimizer framework."""

    def test_optimizer_initialization(self) -> None:
        """Test creating optimizer instance."""
        boosting = BoostingSystem()
        search_system: Any = None  # Mock for now
        optimizer = BoostWeightOptimizer(boosting, search_system)
        assert optimizer._experiment_id.startswith("exp_")

    def test_measure_baseline_with_mocked_search(
        self,
        test_query_set: TestQuerySet,
    ) -> None:
        """Test baseline measurement with mocked components."""
        # This is a simplified test - full integration requires actual search system
        boosting = BoostingSystem()

        class MockSearchSystem:
            def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
                # Return mock results
                today = date.today()
                return [
                    SearchResult(
                        chunk_id=1,
                        chunk_text=query,
                        similarity_score=0.8,
                        bm25_score=0.7,
                        hybrid_score=0.75,
                        rank=1,
                        score_type="hybrid",
                        source_file="test.md",
                        source_category="api_docs",
                        document_date=today,
                        context_header="Test",
                        chunk_index=0,
                        total_chunks=1,
                        chunk_token_count=100,
                        metadata={},
                    )
                ]

        search_system = MockSearchSystem()
        optimizer = BoostWeightOptimizer(boosting, search_system)

        # Measure baseline
        baseline = optimizer.measure_baseline(test_query_set)
        assert isinstance(baseline, OptimizedBoostConfig)
        assert isinstance(baseline.metrics, RelevanceMetrics)


class TestOptimizationIntegration:
    """Integration tests for optimization workflows."""

    def test_empty_query_set_raises(self) -> None:
        """Test that empty query set raises ValueError."""
        boosting = BoostingSystem()
        search_system: Any = None
        optimizer = BoostWeightOptimizer(boosting, search_system)
        empty_qs = TestQuerySet("empty")

        with pytest.raises(ValueError):
            optimizer.measure_baseline(empty_qs)

    def test_metric_improvement_calculation(self) -> None:
        """Test calculating improvements between metrics."""
        baseline_metrics = RelevanceMetrics(
            ndcg_10=0.7,
            mrr=0.8,
            precision_5=0.5,
            precision_10=0.6,
            map_score=0.65,
        )

        improved_metrics = RelevanceMetrics(
            ndcg_10=0.75,
            mrr=0.85,
            precision_5=0.55,
            precision_10=0.65,
            map_score=0.70,
        )

        baseline_composite = RelevanceCalculator.calculate_composite_score(
            baseline_metrics
        )
        improved_composite = RelevanceCalculator.calculate_composite_score(
            improved_metrics
        )

        improvement = improved_composite - baseline_composite
        assert improvement > 0.0
