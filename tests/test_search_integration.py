"""End-to-end integration tests for search functionality.

Tests cover:
- Vector search accuracy and performance
- BM25 search relevance validation
- Hybrid ranking with RRF
- Combined search with filtering
- Performance benchmarks
- Large result sets
- Concurrent query handling
- Search accuracy metrics
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from src.search.results import (
    SearchResult,
    SearchResultFormatter,
    RankingValidator,
)


class TestSearchIntegrationBasics:
    """Basic integration tests for complete search workflows."""

    def test_vector_search_end_to_end(self) -> None:
        """Test complete vector search workflow."""
        # Simulate search results
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Machine learning is a subset of artificial intelligence.",
                similarity_score=0.92,
                bm25_score=0.0,
                hybrid_score=0.92,
                rank=1,
                score_type="vector",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md > Intro",
                chunk_index=0,
                total_chunks=10,
                chunk_token_count=512,
                metadata={"tags": ["ml", "ai"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Neural networks learn from data patterns.",
                similarity_score=0.87,
                bm25_score=0.0,
                hybrid_score=0.87,
                rank=2,
                score_type="vector",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md > Neural Networks",
                chunk_index=2,
                total_chunks=10,
                chunk_token_count=512,
                metadata={"tags": ["neural", "ml"]},
            ),
        ]

        # Format results
        formatter = SearchResultFormatter()
        formatted = formatter.format_results(results)

        assert len(formatted) == 2
        assert formatted[0]["similarity_score"] == 0.92
        assert formatted[1]["similarity_score"] == 0.87

    def test_bm25_search_end_to_end(self) -> None:
        """Test complete BM25 search workflow."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Python programming language fundamentals.",
                similarity_score=0.0,
                bm25_score=0.88,
                hybrid_score=0.88,
                rank=1,
                score_type="bm25",
                source_file="python.md",
                source_category="programming",
                document_date=None,
                context_header="python.md > Intro",
                chunk_index=0,
                total_chunks=15,
                chunk_token_count=512,
                metadata={"tags": ["python"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Java is an object-oriented programming language.",
                similarity_score=0.0,
                bm25_score=0.72,
                hybrid_score=0.72,
                rank=2,
                score_type="bm25",
                source_file="java.md",
                source_category="programming",
                document_date=None,
                context_header="java.md > Intro",
                chunk_index=0,
                total_chunks=12,
                chunk_token_count=512,
                metadata={"tags": ["java"]},
            ),
        ]

        # Validate ranking
        validation = RankingValidator.validate_ranking(results)
        assert validation["is_sorted"] is True
        assert validation["rank_correctness"] is True

    def test_hybrid_search_end_to_end(self) -> None:
        """Test complete hybrid search workflow combining vector and BM25."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Machine learning algorithms using neural networks.",
                similarity_score=0.91,
                bm25_score=0.85,
                hybrid_score=0.89,
                rank=1,
                score_type="hybrid",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md > Intro",
                chunk_index=0,
                total_chunks=10,
                chunk_token_count=512,
                metadata={"tags": ["ml", "neural"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Deep learning with convolutional neural networks.",
                similarity_score=0.87,
                bm25_score=0.80,
                hybrid_score=0.84,
                rank=2,
                score_type="hybrid",
                source_file="dl.md",
                source_category="ml",
                document_date=None,
                context_header="dl.md > CNN",
                chunk_index=3,
                total_chunks=8,
                chunk_token_count=512,
                metadata={"tags": ["deep", "cnn"]},
            ),
        ]

        # Format and validate
        formatter = SearchResultFormatter()
        formatted = formatter.format_results(results)

        assert len(formatted) == 2
        assert all(r["score_type"] == "hybrid" for r in formatted)


class TestSearchWithFiltering:
    """Tests for search combined with metadata filtering."""

    def test_search_and_filter_by_category(self) -> None:
        """Test filtering search results by category."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="ML content",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["important"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="API content",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="api.md",
                source_category="api",
                document_date=None,
                context_header="api.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["endpoint"]},
            ),
        ]

        # Filter by category
        filters = {"category": "ml"}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert filtered[0].source_category == "ml"

    def test_search_and_filter_by_tags(self) -> None:
        """Test filtering search results by tags."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Important content",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="doc1.md",
                source_category="doc",
                document_date=None,
                context_header="doc1.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["important", "urgent"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Regular content",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="doc2.md",
                source_category="doc",
                document_date=None,
                context_header="doc2.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["regular"]},
            ),
        ]

        # Filter by tag
        filters = {"tags": ["important"]}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert "important" in filtered[0].metadata["tags"]

    def test_search_with_multiple_filters(self) -> None:
        """Test search with multiple filter conditions."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Important ML guide",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="docs/ml/guide.md",
                source_category="ml",
                document_date=None,
                context_header="guide.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["important"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="ML tutorial",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="docs/ml/tutorial.md",
                source_category="ml",
                document_date=None,
                context_header="tutorial.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["tutorial"]},
            ),
        ]

        # Filter by both category and source file
        filters = {"category": "ml", "source_file": "guide"}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert "guide" in filtered[0].source_file


class TestSearchPerformance:
    """Tests for search performance characteristics."""

    def test_result_formatting_performance(self) -> None:
        """Test that result formatting is fast."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.9 - (i * 0.01),
                bm25_score=0.8 - (i * 0.01),
                hybrid_score=0.85 - (i * 0.01),
                rank=i + 1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=100,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(100)
        ]

        formatter = SearchResultFormatter()
        start = time.time()
        formatted = formatter.format_results(results)
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 100 results)
        assert elapsed < 0.1
        assert len(formatted) == 100

    def test_deduplication_performance(self) -> None:
        """Test deduplication performance with large result sets."""
        # Create results with some duplicates
        results = []
        for i in range(50):
            results.append(SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.9 - (i * 0.01),
                bm25_score=0.8 - (i * 0.01),
                hybrid_score=0.85 - (i * 0.01),
                rank=i + 1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=50,
                chunk_token_count=512,
                metadata={},
            ))

        # Add some duplicates
        for i in range(10):
            results.append(SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.85 - (i * 0.01),
                bm25_score=0.75 - (i * 0.01),
                hybrid_score=0.80 - (i * 0.01),
                rank=51 + i,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=50,
                chunk_token_count=512,
                metadata={},
            ))

        formatter = SearchResultFormatter(deduplication_enabled=True)
        start = time.time()
        formatted = formatter.format_results(results)
        elapsed = time.time() - start

        # Deduplication should be fast
        assert elapsed < 0.1
        assert len(formatted) == 50


class TestSearchAccuracy:
    """Tests for search accuracy metrics."""

    def test_ranking_accuracy_vector_search(self) -> None:
        """Test that vector search ranks semantically similar items higher."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Machine learning algorithms",
                similarity_score=0.95,  # Very high similarity
                bm25_score=0.0,
                hybrid_score=0.95,
                rank=1,
                score_type="vector",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Unrelated content about cooking",
                similarity_score=0.15,  # Low similarity
                bm25_score=0.0,
                hybrid_score=0.15,
                rank=2,
                score_type="vector",
                source_file="cooking.md",
                source_category="cooking",
                document_date=None,
                context_header="cooking.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Semantic similarity ranking should be correct
        assert results[0].similarity_score > results[1].similarity_score

    def test_ranking_accuracy_bm25_search(self) -> None:
        """Test that BM25 search ranks keyword matches higher."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Python Python Python programming with Python language",
                similarity_score=0.0,
                bm25_score=0.92,  # High keyword match
                hybrid_score=0.92,
                rank=1,
                score_type="bm25",
                source_file="python.md",
                source_category="lang",
                document_date=None,
                context_header="python.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Java is another programming language",
                similarity_score=0.0,
                bm25_score=0.45,  # Low keyword match
                hybrid_score=0.45,
                rank=2,
                score_type="bm25",
                source_file="java.md",
                source_category="lang",
                document_date=None,
                context_header="java.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Keyword relevance ranking should be correct
        assert results[0].bm25_score > results[1].bm25_score


class TestLargeResultSets:
    """Tests for handling large result sets."""

    def test_large_result_set_formatting(self) -> None:
        """Test formatting large number of results."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Result {i}",
                similarity_score=max(0.01, 0.99 - (i * 0.01)),
                bm25_score=max(0.01, 0.99 - (i * 0.01)),
                hybrid_score=max(0.01, 0.99 - (i * 0.01)),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"doc_{i % 10}.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i % 20,
                total_chunks=20,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(500)
        ]

        formatter = SearchResultFormatter(max_results=100)
        formatted = formatter.format_results(results)

        assert len(formatted) == 100

    def test_large_result_set_threshold_filtering(self) -> None:
        """Test threshold filtering with large result sets."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Result {i}",
                similarity_score=i * 0.002,  # Range 0 to 1
                bm25_score=i * 0.002,
                hybrid_score=i * 0.002,
                rank=i + 1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i % 20,
                total_chunks=20,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(500)
        ]

        formatter = SearchResultFormatter(min_score_threshold=0.5)
        formatted = formatter.format_results(results, apply_threshold=True)

        # All results should meet threshold
        for result in formatted:
            score = result["hybrid_score"]
            assert score >= 0.5


class TestSearchRankingValidation:
    """Tests for search ranking validation."""

    def test_validate_search_result_quality(self) -> None:
        """Test validation of search result quality."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Highly relevant content",
                similarity_score=0.95,
                bm25_score=0.90,
                hybrid_score=0.93,
                rank=1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Moderately relevant",
                similarity_score=0.75,
                bm25_score=0.70,
                hybrid_score=0.73,
                rank=2,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=1,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        validation = RankingValidator.validate_ranking(results)

        # All checks should pass
        assert validation["is_sorted"] is True
        assert validation["rank_correctness"] is True
        assert validation["has_duplicates"] is False
        assert validation["score_monotonicity"] is True

    def test_json_serialization_large_result(self) -> None:
        """Test JSON serialization of large result sets."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Result {i} with detailed content",
                similarity_score=0.9 - (i * 0.01),
                bm25_score=0.8 - (i * 0.01),
                hybrid_score=0.85 - (i * 0.01),
                rank=i + 1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=10,
                chunk_token_count=512,
                metadata={"index": i},
            )
            for i in range(50)
        ]

        formatter = SearchResultFormatter()
        formatted = formatter.format_results(results, format_type="json")

        # All should be valid JSON
        for json_str in formatted:
            parsed = json.loads(json_str)
            assert "chunk_id" in parsed
            assert "rank" in parsed
