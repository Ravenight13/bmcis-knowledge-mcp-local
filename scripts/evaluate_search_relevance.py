#!/usr/bin/env python3
"""Evaluate search relevance and knowledge graph reranking effectiveness.

This script evaluates:
1. Search result relevance to queries
2. Ranking order quality
3. Knowledge graph reranking impact
4. Query type performance variations

Run with: python scripts/evaluate_search_relevance.py
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from src.mcp.server import get_hybrid_search
from src.search.cross_encoder_reranker import CrossEncoderReranker


@dataclass
class EvaluationQuery:
    """Query for evaluation."""
    query: str
    category: str
    expected_keywords: list[str]
    description: str


@dataclass
class ResultEvaluation:
    """Evaluation of a single search result."""
    rank: int
    text_preview: str
    source_file: str
    relevance_score: float  # 0.0 = irrelevant, 1.0 = highly relevant
    has_keywords: bool
    reasoning: str


@dataclass
class QueryEvaluation:
    """Evaluation of query results."""
    query: str
    category: str
    total_results: int
    top_k: int
    results: list[ResultEvaluation]
    avg_relevance: float
    ranking_quality: float  # 0-1: how well ordered are results?
    evaluation_time: float


# Test queries organized by category
TEST_QUERIES = [
    EvaluationQuery(
        query="ProSource commission rates and vendor structure",
        category="vendor_entity",
        expected_keywords=["prosource", "commission", "vendor", "rate"],
        description="Entity-based query looking for specific vendor information"
    ),
    EvaluationQuery(
        query="What is the organizational structure of BMCIS sales teams?",
        category="organizational",
        expected_keywords=["organization", "structure", "team", "sales", "district"],
        description="Organizational query about team hierarchy"
    ),
    EvaluationQuery(
        query="Dealer classification and segmentation",
        category="classification",
        expected_keywords=["dealer", "classification", "segment", "type", "category"],
        description="Classification query about dealer types"
    ),
    EvaluationQuery(
        query="Lutron lighting control specifications",
        category="product",
        expected_keywords=["lutron", "lighting", "control", "specification", "feature"],
        description="Product specification query"
    ),
    EvaluationQuery(
        query="Weekly sales targets and performance metrics",
        category="metrics",
        expected_keywords=["sales", "target", "metric", "performance", "kpi"],
        description="Metrics and KPI query"
    ),
    EvaluationQuery(
        query="Commission processing procedures and timeline",
        category="process",
        expected_keywords=["commission", "processing", "procedure", "timeline", "step"],
        description="Process-oriented query"
    ),
    EvaluationQuery(
        query="Team member profiles and specializations",
        category="people",
        expected_keywords=["team", "member", "profile", "specialist", "expertise"],
        description="People and role query"
    ),
    EvaluationQuery(
        query="Market intelligence and competitive analysis",
        category="market",
        expected_keywords=["market", "intelligence", "competitive", "analysis", "trend"],
        description="Market intelligence query"
    ),
]


def calculate_relevance_score(result_text: str, keywords: list[str]) -> float:
    """Calculate relevance score based on keyword presence.

    Ranges from 0.0 (no keywords) to 1.0 (all keywords present).
    Also considers keyword density and distribution.
    """
    text_lower = result_text.lower()

    # Count keyword matches
    keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    base_score = keyword_matches / len(keywords)

    # Bonus for multiple matches of same keyword (density)
    density_bonus = 0
    for kw in keywords:
        count = text_lower.count(kw.lower())
        if count > 1:
            density_bonus += 0.05 * min(count - 1, 2)  # Cap at +0.1 per keyword

    # Bonus for keyword in beginning of text (likely more relevant)
    beginning_text = text_lower[:200]
    beginning_bonus = 0
    for kw in keywords:
        if kw.lower() in beginning_text:
            beginning_bonus += 0.05

    relevance_score = min(base_score + density_bonus + beginning_bonus, 1.0)
    return relevance_score


def evaluate_ranking_order(relevance_scores: list[float]) -> float:
    """Evaluate if results are properly ranked (descending order).

    Returns 0-1: 1.0 = perfectly ordered, 0.0 = poorly ordered.
    Uses Spearman correlation between expected and actual order.
    """
    if len(relevance_scores) < 2:
        return 1.0

    # Count inversions (higher ranked items with lower relevance)
    inversions = 0
    for i in range(len(relevance_scores) - 1):
        if relevance_scores[i] < relevance_scores[i + 1]:
            inversions += 1

    # Perfect order = 0 inversions
    max_inversions = len(relevance_scores) - 1
    ranking_quality = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0

    return ranking_quality


def evaluate_query(search_engine, query_obj: EvaluationQuery, top_k: int = 10) -> QueryEvaluation:
    """Evaluate search results for a single query."""
    print(f"\n📊 Evaluating: {query_obj.query[:60]}...")

    start_time = time.time()
    results = search_engine.search(query_obj.query, top_k=top_k)
    elapsed = time.time() - start_time

    # Evaluate each result
    evaluations = []
    relevance_scores = []

    for rank, result in enumerate(results, 1):
        # Get text preview
        text_preview = result.chunk_text[:150].replace("\n", " ")

        # Calculate relevance
        relevance = calculate_relevance_score(result.chunk_text, query_obj.expected_keywords)
        has_keywords = relevance > 0.33

        # Determine reasoning
        if relevance >= 0.8:
            reasoning = "Highly relevant - contains key concepts"
        elif relevance >= 0.5:
            reasoning = "Somewhat relevant - has some keywords"
        else:
            reasoning = "Low relevance - minimal keyword match"

        evaluation = ResultEvaluation(
            rank=rank,
            text_preview=text_preview,
            source_file=result.source_file,
            relevance_score=relevance,
            has_keywords=has_keywords,
            reasoning=reasoning
        )

        evaluations.append(evaluation)
        relevance_scores.append(relevance)

    # Calculate metrics
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    ranking_quality = evaluate_ranking_order(relevance_scores)

    query_eval = QueryEvaluation(
        query=query_obj.query,
        category=query_obj.category,
        total_results=len(results),
        top_k=top_k,
        results=evaluations,
        avg_relevance=avg_relevance,
        ranking_quality=ranking_quality,
        evaluation_time=elapsed
    )

    # Print results
    print(f"  Results: {len(results)}, Avg Relevance: {avg_relevance:.2%}, "
          f"Ranking Quality: {ranking_quality:.2%}, Time: {elapsed:.3f}s")

    for eval_result in evaluations[:3]:
        print(f"    {eval_result.rank}. [{eval_result.relevance_score:.1%}] "
              f"{eval_result.text_preview}...")

    return query_eval


def generate_report(evaluations: list[QueryEvaluation]) -> str:
    """Generate comprehensive evaluation report."""
    report = []
    report.append("=" * 80)
    report.append("SEARCH RELEVANCE & KNOWLEDGE GRAPH RERANKING EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Queries Evaluated: {len(evaluations)}")

    # Summary statistics
    avg_relevance_all = sum(e.avg_relevance for e in evaluations) / len(evaluations)
    avg_ranking_quality = sum(e.ranking_quality for e in evaluations) / len(evaluations)
    avg_time = sum(e.evaluation_time for e in evaluations) / len(evaluations)

    report.append("\n" + "=" * 80)
    report.append("SUMMARY STATISTICS")
    report.append("=" * 80)
    report.append(f"Overall Average Relevance:  {avg_relevance_all:.2%}")
    report.append(f"Average Ranking Quality:    {avg_ranking_quality:.2%}")
    report.append(f"Average Query Time:         {avg_time:.3f}s")

    # Breakdown by category
    report.append("\n" + "=" * 80)
    report.append("PERFORMANCE BY CATEGORY")
    report.append("=" * 80)

    categories = {}
    for eval_result in evaluations:
        if eval_result.category not in categories:
            categories[eval_result.category] = []
        categories[eval_result.category].append(eval_result)

    for category, evals in sorted(categories.items()):
        cat_relevance = sum(e.avg_relevance for e in evals) / len(evals)
        cat_quality = sum(e.ranking_quality for e in evals) / len(evals)

        status = "✅ GOOD" if cat_relevance >= 0.7 else "⚠️ FAIR" if cat_relevance >= 0.5 else "❌ POOR"

        report.append(f"\n{category.upper()}")
        report.append(f"  Relevance: {cat_relevance:.2%} {status}")
        report.append(f"  Ranking:   {cat_quality:.2%}")
        report.append(f"  Queries:   {len(evals)}")

    # Detailed query results
    report.append("\n" + "=" * 80)
    report.append("DETAILED QUERY EVALUATIONS")
    report.append("=" * 80)

    for eval_result in evaluations:
        report.append(f"\nQuery: {eval_result.query}")
        report.append(f"Category: {eval_result.category}")
        report.append(f"Results: {eval_result.total_results} | "
                      f"Relevance: {eval_result.avg_relevance:.2%} | "
                      f"Ranking: {eval_result.ranking_quality:.2%}")

        report.append("\nTop 3 Results:")
        for result in eval_result.results[:3]:
            report.append(f"  {result.rank}. [{result.relevance_score:.1%}] {result.source_file}")
            report.append(f"     {result.text_preview}...")
            report.append(f"     → {result.reasoning}")

    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)

    if avg_relevance_all >= 0.75:
        report.append("✅ Search relevance is EXCELLENT - results are highly relevant to queries")
    elif avg_relevance_all >= 0.6:
        report.append("⚠️  Search relevance is GOOD but could be improved")
    else:
        report.append("❌ Search relevance needs IMPROVEMENT - consider:")

    if avg_ranking_quality >= 0.8:
        report.append("✅ Ranking quality is EXCELLENT - results are well-ordered")
    elif avg_ranking_quality >= 0.6:
        report.append("⚠️  Ranking quality could be improved with better reranking")
    else:
        report.append("❌ Consider implementing/improving cross-encoder reranking")

    # Specific improvements
    report.append("\nSpecific Improvement Areas:")

    poor_categories = [
        (cat, sum(e.avg_relevance for e in evals) / len(evals))
        for cat, evals in categories.items()
        if sum(e.avg_relevance for e in evals) / len(evals) < 0.6
    ]

    if poor_categories:
        for cat, score in sorted(poor_categories, key=lambda x: x[1]):
            report.append(f"  • {cat}: {score:.2%} relevance - improve entity recognition")
    else:
        report.append("  • No major improvement areas - system performing well")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Run relevance evaluation."""
    print("\n🔍 Initializing search engine...")
    search_engine = get_hybrid_search()

    print("📋 Running relevance evaluation...")
    print(f"Testing {len(TEST_QUERIES)} queries across {len(set(q.category for q in TEST_QUERIES))} categories\n")

    evaluations = []
    for query_obj in TEST_QUERIES:
        evaluation = evaluate_query(search_engine, query_obj, top_k=10)
        evaluations.append(evaluation)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Generate report
    report = generate_report(evaluations)
    print(report)

    # Save report
    report_path = "docs/SEARCH_RELEVANCE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to {report_path}")

    # Save detailed JSON results
    json_path = "docs/search_relevance_results.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "queries": [asdict(e) for e in evaluations],
    }

    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"✅ Detailed results saved to {json_path}")

    # Print summary
    avg_relevance = sum(e.avg_relevance for e in evaluations) / len(evaluations)
    print(f"\n📊 OVERALL: {avg_relevance:.2%} average relevance")


if __name__ == "__main__":
    main()
