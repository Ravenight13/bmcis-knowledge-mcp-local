"""Example: Using the Boost Weight Optimization Framework

Demonstrates how to:
1. Create test query sets with ground truth judgments
2. Initialize the optimizer with search components
3. Run individual factor analysis
4. Interpret results and identify improvements
5. Generate reports

Usage:
    cd /Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local
    python examples/boost_optimization_example.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.search.boost_optimizer import (
    BoostWeightOptimizer,
    RelevanceCalculator,
    TestQuerySet,
)
from src.search.boosting import BoostingSystem, BoostWeights
from src.search.results import SearchResult


class MockHybridSearch:
    """Mock search system for demonstration purposes."""

    def __init__(self, all_results: list[SearchResult]) -> None:
        """Initialize mock search with predefined results."""
        self.all_results = all_results

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return mock results (same set regardless of query)."""
        return self.all_results[:top_k]


def create_mock_search_results() -> list[SearchResult]:
    """Create realistic mock search results for demonstration."""
    today = date.today()
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="OpenAI API: Complete authentication guide with JWT tokens",
            similarity_score=0.95,
            bm25_score=0.90,
            hybrid_score=0.92,
            rank=1,
            score_type="hybrid",
            source_file="openai_api_auth.md",
            source_category="api_docs",
            document_date=today - timedelta(days=5),
            context_header="OpenAI API > Authentication > JWT",
            chunk_index=0,
            total_chunks=8,
            chunk_token_count=512,
            metadata={"vendor": "OpenAI", "updated": "2024-11-04"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Anthropic Claude: Getting started with authentication",
            similarity_score=0.88,
            bm25_score=0.82,
            hybrid_score=0.85,
            rank=2,
            score_type="hybrid",
            source_file="claude_getting_started.md",
            source_category="guide",
            document_date=today - timedelta(days=10),
            context_header="Claude > Getting Started > Auth",
            chunk_index=1,
            total_chunks=6,
            chunk_token_count=384,
            metadata={"vendor": "Anthropic", "updated": "2024-10-30"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="OAuth 2.0 implementation guide for API security",
            similarity_score=0.80,
            bm25_score=0.75,
            hybrid_score=0.77,
            rank=3,
            score_type="hybrid",
            source_file="oauth_security.md",
            source_category="kb_article",
            document_date=today - timedelta(days=45),
            context_header="Security > OAuth > Implementation",
            chunk_index=0,
            total_chunks=12,
            chunk_token_count=768,
            metadata={"tags": ["security", "oauth"]},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="Google Cloud Platform: API authentication and authorization",
            similarity_score=0.75,
            bm25_score=0.70,
            hybrid_score=0.72,
            rank=4,
            score_type="hybrid",
            source_file="gcp_api_auth.md",
            source_category="api_docs",
            document_date=today - timedelta(days=60),
            context_header="GCP > API > Auth",
            chunk_index=2,
            total_chunks=10,
            chunk_token_count=640,
            metadata={"vendor": "Google"},
        ),
        SearchResult(
            chunk_id=5,
            chunk_text="AWS IAM roles and API key management best practices",
            similarity_score=0.70,
            bm25_score=0.65,
            hybrid_score=0.67,
            rank=5,
            score_type="hybrid",
            source_file="aws_iam_best_practices.md",
            source_category="guide",
            document_date=today - timedelta(days=2),
            context_header="AWS > IAM > Best Practices",
            chunk_index=0,
            total_chunks=7,
            chunk_token_count=416,
            metadata={"vendor": "AWS", "updated": "2024-11-07"},
        ),
    ]


def create_test_query_set() -> TestQuerySet:
    """Create test query set with ground truth relevance judgments.

    Ground truth indices represent which results are relevant:
    - Index 0: OpenAI API auth (most relevant)
    - Index 1: Claude getting started
    - Index 2: OAuth guide
    - Index 3: GCP auth
    - Index 4: AWS IAM
    """
    test_set = TestQuerySet("api_authentication")

    # Query 1: User asking about OpenAI authentication
    # Relevant: Results 0 (OpenAI), 1 (Claude also covers auth), 2 (OAuth general)
    test_set.add_query("How do I authenticate with OpenAI API?", {0, 1, 2})

    # Query 2: User asking about general API authentication
    # Relevant: All results (all cover authentication in some form)
    test_set.add_query("API authentication best practices", {0, 1, 2, 3, 4})

    # Query 3: User asking about OAuth specifically
    # Relevant: Results 2 (OAuth), 0 (mentions OAuth), 3 (GCP uses OAuth)
    test_set.add_query("Implement OAuth 2.0 for APIs", {0, 2, 3})

    # Query 4: User asking about Google Cloud auth
    # Relevant: Results 3 (GCP), 2 (general OAuth)
    test_set.add_query("Google Cloud authentication", {2, 3})

    # Query 5: User asking about AWS authentication
    # Relevant: Results 4 (AWS IAM), 2 (general OAuth)
    test_set.add_query("AWS API authentication and IAM", {2, 4})

    return test_set


def print_factor_results(
    factor_name: str, result: Any, baseline: Any
) -> None:
    """Pretty-print factor analysis results."""
    print(f"\n{'=' * 60}")
    print(f"FACTOR: {factor_name.upper()}")
    print(f"{'=' * 60}")

    baseline_score = baseline.composite_score
    best_score = result.best.composite_score
    improvement = best_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

    print(f"\nBaseline Configuration:")
    print(f"  Weights: {result.baseline.weights.__dict__}")
    print(f"  Composite Score: {baseline_score:.4f}")
    print(f"  Metrics:")
    print(f"    NDCG@10:     {result.baseline.metrics.ndcg_10:.4f}")
    print(f"    MRR:         {result.baseline.metrics.mrr:.4f}")
    print(f"    Precision@5: {result.baseline.metrics.precision_5:.4f}")
    print(f"    Precision@10:{result.baseline.metrics.precision_10:.4f}")

    print(f"\nBest Configuration:")
    print(f"  Weights: {result.best.weights.__dict__}")
    print(f"  Composite Score: {best_score:.4f}")
    print(f"  Metrics:")
    print(f"    NDCG@10:     {result.best.metrics.ndcg_10:.4f}")
    print(f"    MRR:         {result.best.metrics.mrr:.4f}")
    print(f"    Precision@5: {result.best.metrics.precision_5:.4f}")
    print(f"    Precision@10:{result.best.metrics.precision_10:.4f}")

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Percentage: {improvement_pct:+.2f}%")

    print(f"\nTop 3 Configurations Tested:")
    for i, config in enumerate(result.results[:3], 1):
        weights = config.weights.__dict__
        # Show only the factor being optimized
        factor_value = weights[factor_name]
        print(f"  {i}. {factor_name}={factor_value:.2f}, composite={config.composite_score:.4f}")


def main() -> None:
    """Run complete boost optimization example."""
    print("\n" + "=" * 60)
    print("BOOST WEIGHT OPTIMIZATION FRAMEWORK - EXAMPLE")
    print("=" * 60)

    # Step 1: Create test data
    print("\nStep 1: Creating test data...")
    search_results = create_mock_search_results()
    test_set = create_test_query_set()
    print(f"  Created {len(search_results)} mock search results")
    print(f"  Created {len(test_set)} test queries with ground truth")

    # Step 2: Initialize components
    print("\nStep 2: Initializing optimization framework...")
    boosting = BoostingSystem()
    search_system = MockHybridSearch(search_results)
    calculator = RelevanceCalculator()
    optimizer = BoostWeightOptimizer(
        boosting_system=boosting,
        search_system=search_system,
        calculator=calculator,
    )
    print(f"  Optimizer ready (experiment_id: {optimizer._experiment_id})")

    # Step 3: Measure baseline
    print("\nStep 3: Measuring baseline metrics...")
    baseline_weights = BoostWeights(
        vendor=0.15,
        doc_type=0.10,
        recency=0.05,
        entity=0.10,
        topic=0.08,
    )
    baseline = optimizer.measure_baseline(test_set, baseline_weights)
    print(f"  Baseline composite score: {baseline.composite_score:.4f}")
    print(f"  Baseline metrics:")
    print(f"    NDCG@10:      {baseline.metrics.ndcg_10:.4f}")
    print(f"    MRR:          {baseline.metrics.mrr:.4f}")
    print(f"    Precision@10: {baseline.metrics.precision_10:.4f}")

    # Step 4: Run individual factor analysis
    print("\nStep 4: Running individual factor analysis...")
    print("  This tests each factor independently...")
    factor_results = optimizer.analyze_individual_factors(
        test_queries=test_set,
        baseline_weights=baseline_weights,
        weight_ranges={
            "vendor": (0.05, 0.25),
            "doc_type": (0.05, 0.20),
            "recency": (0.0, 0.15),
            "entity": (0.05, 0.20),
            "topic": (0.03, 0.15),
        },
        step_size=0.05,
    )

    # Step 5: Present results
    print("\nStep 5: Analyzing results...")
    print("\nFactor Analysis Summary:")

    best_improvements: dict[str, float] = {}
    for factor, result in factor_results.items():
        improvement = result.best.composite_score - baseline.composite_score
        best_improvements[factor] = improvement
        pct = (improvement / baseline.composite_score * 100) if baseline.composite_score > 0 else 0
        print(f"  {factor:12s}: {improvement:+.4f} ({pct:+.2f}%)")

    # Print detailed results for top 2 factors
    print("\n\nDetailed Results for Top Factors:")
    sorted_factors = sorted(best_improvements.items(), key=lambda x: x[1], reverse=True)
    for factor, _ in sorted_factors[:2]:
        print_factor_results(factor, factor_results[factor], baseline)

    # Step 6: Generate report
    print("\n\nStep 6: Generating report...")
    report = optimizer.generate_report(
        results=factor_results,
        output_path="/tmp/boost_optimization_report.json",
    )
    print(f"  Report generated at /tmp/boost_optimization_report.json")
    print(f"  Report contains:")
    print(f"    - Timestamp: {report.get('timestamp')}")
    print(f"    - Experiment ID: {report.get('experiment_id')}")
    print(f"    - Factor results: {len(report.get('factors', {}))}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOptimization Results:")
    print(f"  Baseline composite score: {baseline.composite_score:.4f}")

    best_factor = sorted_factors[0][0]
    best_improvement = sorted_factors[0][1]
    best_config = factor_results[best_factor].best

    print(f"  Best factor: {best_factor}")
    print(f"  Best improvement: {best_improvement:+.4f} ({best_improvement/baseline.composite_score*100:+.2f}%)")
    print(f"  Recommended config: {best_config.weights.__dict__}")

    print(f"\nNext Steps:")
    print(f"  1. Use results to adjust boost weights in production")
    print(f"  2. Run Phase 2: Combined optimization across all factors")
    print(f"  3. Validate on held-out query set")
    print(f"  4. Monitor metrics in A/B test")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
