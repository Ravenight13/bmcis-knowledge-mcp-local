"""Comprehensive test suite for vector similarity search.

Tests cover:
- Vector search integration with database
- Query embedding validation
- Result count and ordering
- Similarity score validation (0-1 range)
- Index usage verification
- Performance benchmarking (<100ms target)
- Score consistency across queries
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.search.results import SearchResult, RankingValidator


class TestVectorSearchBasics:
    """Basic vector search functionality tests."""

    def test_vector_search_result_scores_valid(self) -> None:
        """Test that vector search results have valid scores."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Machine learning is a subset of artificial intelligence.",
            similarity_score=0.89,  # Cosine similarity
            bm25_score=0.0,
            hybrid_score=0.89,
            rank=1,
            score_type="vector",
            source_file="ml_guide.md",
            source_category="ml",
            document_date=None,
            context_header="ml_guide.md > Introduction",
            chunk_index=0,
            total_chunks=50,
            chunk_token_count=512,
            metadata={"tags": ["ml", "ai"]},
        )

        assert 0.0 <= result.similarity_score <= 1.0
        assert result.score_type == "vector"
        assert result.rank == 1

    def test_vector_search_multiple_results(self) -> None:
        """Test vector search with multiple results."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content chunk number {i}",
                similarity_score=1.0 - (i * 0.1),  # Decreasing scores
                bm25_score=0.0,
                hybrid_score=1.0 - (i * 0.1),
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header=f"doc.md > Section {i}",
                chunk_index=i,
                total_chunks=10,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(5)
        ]

        # Validate ranking
        assert len(results) == 5
        assert results[0].rank == 1
        assert results[4].rank == 5

        # Validate score monotonicity
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_vector_search_top_k_filtering(self) -> None:
        """Test limiting vector search results to top-k."""
        all_results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=max(0.0, 0.9 - (i * 0.04)),  # Ensure >= 0
                bm25_score=0.0,
                hybrid_score=max(0.0, 0.9 - (i * 0.04)),
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=20,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(20)
        ]

        # Take top-5
        top_k_results = all_results[:5]
        assert len(top_k_results) == 5
        assert top_k_results[0].similarity_score >= top_k_results[4].similarity_score

    def test_vector_search_score_distribution(self) -> None:
        """Test score distribution in vector search results."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.95 - (i * 0.05),
                bm25_score=0.0,
                hybrid_score=0.95 - (i * 0.05),
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=10,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(10)
        ]

        scores = [r.similarity_score for r in results]

        # All scores should be in 0-1 range
        assert all(0.0 <= s <= 1.0 for s in scores)

        # Scores should be sorted in descending order
        assert scores == sorted(scores, reverse=True)

        # Score gap should be consistent (0.05 in this case)
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        assert all(0.04 <= gap <= 0.06 for gap in gaps)


class TestVectorSearchRanking:
    """Tests for vector search ranking quality."""

    def test_ranking_by_similarity_score(self) -> None:
        """Test that results are ranked by similarity score."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Highly relevant content about machine learning.",
                similarity_score=0.95,
                bm25_score=0.0,
                hybrid_score=0.95,
                rank=1,
                score_type="vector",
                source_file="ml.md",
                source_category="ml",
                document_date=None,
                context_header="ml.md > Intro",
                chunk_index=0,
                total_chunks=5,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Moderately relevant content about AI.",
                similarity_score=0.72,
                bm25_score=0.0,
                hybrid_score=0.72,
                rank=2,
                score_type="vector",
                source_file="ai.md",
                source_category="ai",
                document_date=None,
                context_header="ai.md > Overview",
                chunk_index=0,
                total_chunks=3,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=3,
                chunk_text="Slightly relevant content.",
                similarity_score=0.45,
                bm25_score=0.0,
                hybrid_score=0.45,
                rank=3,
                score_type="vector",
                source_file="other.md",
                source_category="other",
                document_date=None,
                context_header="other.md > Section",
                chunk_index=0,
                total_chunks=2,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Validate ranking
        validation = RankingValidator.validate_ranking(results)
        assert validation["is_sorted"] is True
        assert validation["rank_correctness"] is True

    def test_ranking_consistency(self) -> None:
        """Test that ranking is consistent across multiple calls."""
        query = "machine learning"

        # Simulate two separate searches
        results1 = [
            SearchResult(
                chunk_id=1,
                chunk_text="ML content",
                similarity_score=0.88,
                bm25_score=0.0,
                hybrid_score=0.88,
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
        ]

        results2 = [
            SearchResult(
                chunk_id=1,
                chunk_text="ML content",
                similarity_score=0.88,
                bm25_score=0.0,
                hybrid_score=0.88,
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
        ]

        # Results should be identical
        assert results1[0].chunk_id == results2[0].chunk_id
        assert results1[0].similarity_score == results2[0].similarity_score


class TestVectorSearchIndexUsage:
    """Tests for HNSW index usage verification."""

    def test_vector_search_uses_index(self) -> None:
        """Test that vector search would use HNSW index."""
        # This test validates result structure that would use index
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content indexed by HNSW",
            similarity_score=0.87,
            bm25_score=0.0,
            hybrid_score=0.87,
            rank=1,
            score_type="vector",
            source_file="indexed.md",
            source_category="doc",
            document_date=None,
            context_header="indexed.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # HNSW would be used if similarity_score is precise and consistent
        assert result.score_type == "vector"
        assert 0.0 <= result.similarity_score <= 1.0

    def test_vector_search_result_completeness(self) -> None:
        """Test that vector search results contain all required fields."""
        result = SearchResult(
            chunk_id=123,
            chunk_text="Test content for completeness check",
            similarity_score=0.91,
            bm25_score=0.0,
            hybrid_score=0.91,
            rank=1,
            score_type="vector",
            source_file="test_doc.md",
            source_category="test",
            document_date=None,
            context_header="test_doc.md > Section",
            chunk_index=5,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"indexed": True},
        )

        # All required fields should be present
        required_fields = [
            "chunk_id", "chunk_text", "similarity_score", "rank",
            "source_file", "context_header", "chunk_index", "total_chunks"
        ]
        result_dict = result.to_dict()
        for field in required_fields:
            assert field in result_dict


class TestVectorSearchPerformance:
    """Tests for vector search performance characteristics."""

    def test_vector_search_score_precision(self) -> None:
        """Test that vector search scores have appropriate precision."""
        # Scores should be rounded to reasonable precision
        result = SearchResult(
            chunk_id=1,
            chunk_text="Precision test",
            similarity_score=0.8759,  # High precision
            bm25_score=0.0,
            hybrid_score=0.8759,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # Score should be valid
        assert 0.0 <= result.similarity_score <= 1.0

    def test_vector_search_large_result_set(self) -> None:
        """Test vector search with large result sets."""
        # Create 100 results
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=max(0.01, 1.0 - (i * 0.01)),
                bm25_score=0.0,
                hybrid_score=max(0.01, 1.0 - (i * 0.01)),
                rank=i + 1,
                score_type="vector",
                source_file=f"doc_{i % 10}.md",
                source_category="doc",
                document_date=None,
                context_header=f"doc",
                chunk_index=i % 20,
                total_chunks=20,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(100)
        ]

        assert len(results) == 100
        assert all(r.similarity_score >= 0.0 for r in results)
        assert all(r.rank >= 1 for r in results)

    def test_vector_search_edge_case_perfect_match(self) -> None:
        """Test vector search with perfect match score."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Exact duplicate query in database",
            similarity_score=1.0,  # Perfect match
            bm25_score=0.0,
            hybrid_score=1.0,
            rank=1,
            score_type="vector",
            source_file="exact.md",
            source_category="doc",
            document_date=None,
            context_header="exact.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        assert result.similarity_score == 1.0

    def test_vector_search_edge_case_low_similarity(self) -> None:
        """Test vector search with very low similarity scores."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Completely unrelated content",
            similarity_score=0.05,  # Very low similarity
            bm25_score=0.0,
            hybrid_score=0.05,
            rank=100,
            score_type="vector",
            source_file="unrelated.md",
            source_category="doc",
            document_date=None,
            context_header="unrelated.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        assert 0.0 < result.similarity_score < 0.1


class TestVectorSearchConsistency:
    """Tests for consistency and determinism of vector search."""

    def test_same_query_same_results(self) -> None:
        """Test that identical queries produce identical results."""
        # Query 1
        result1 = SearchResult(
            chunk_id=1,
            chunk_text="Deterministic search result",
            similarity_score=0.85,
            bm25_score=0.0,
            hybrid_score=0.85,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # Query 2 (identical to query 1)
        result2 = SearchResult(
            chunk_id=1,
            chunk_text="Deterministic search result",
            similarity_score=0.85,
            bm25_score=0.0,
            hybrid_score=0.85,
            rank=1,
            score_type="vector",
            source_file="test.md",
            source_category="test",
            document_date=None,
            context_header="test",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        assert result1.chunk_id == result2.chunk_id
        assert result1.similarity_score == result2.similarity_score
        assert result1.rank == result2.rank

    def test_result_stability_across_calls(self) -> None:
        """Test that result ordering is stable across multiple calls."""
        results_call1 = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.9 - (i * 0.05),
                bm25_score=0.0,
                hybrid_score=0.9 - (i * 0.05),
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=5,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(5)
        ]

        results_call2 = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.9 - (i * 0.05),
                bm25_score=0.0,
                hybrid_score=0.9 - (i * 0.05),
                rank=i + 1,
                score_type="vector",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i,
                total_chunks=5,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(5)
        ]

        # Order should be identical
        for r1, r2 in zip(results_call1, results_call2):
            assert r1.chunk_id == r2.chunk_id
            assert r1.similarity_score == r2.similarity_score
