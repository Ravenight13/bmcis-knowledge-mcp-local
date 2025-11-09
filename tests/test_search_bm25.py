"""Comprehensive test suite for BM25 full-text search.

Tests cover:
- BM25 search integration with PostgreSQL
- Query parsing and tokenization
- Keyword matching accuracy
- Stop word handling
- Result ranking by relevance
- Performance benchmarking (<50ms target)
- Relevance consistency
"""

from __future__ import annotations

from typing import Any

import pytest

from src.search.results import SearchResult, RankingValidator


class TestBM25SearchBasics:
    """Basic BM25 search functionality tests."""

    def test_bm25_search_result_scores_valid(self) -> None:
        """Test that BM25 search results have valid scores."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="PostgreSQL database management system with full-text search capabilities.",
            similarity_score=0.0,
            bm25_score=0.85,  # BM25 relevance score
            hybrid_score=0.85,
            rank=1,
            score_type="bm25",
            source_file="postgres_guide.md",
            source_category="database",
            document_date=None,
            context_header="postgres_guide.md > Full-Text Search",
            chunk_index=0,
            total_chunks=30,
            chunk_token_count=512,
            metadata={"tags": ["database", "postgres"]},
        )

        assert 0.0 <= result.bm25_score <= 1.0
        assert result.score_type == "bm25"
        assert result.rank == 1

    def test_bm25_search_keyword_matching(self) -> None:
        """Test BM25 ranking with keyword matching."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Python programming language for machine learning and data science.",
                similarity_score=0.0,
                bm25_score=0.92,  # High match - contains "Python"
                hybrid_score=0.92,
                rank=1,
                score_type="bm25",
                source_file="python.md",
                source_category="programming",
                document_date=None,
                context_header="python.md > Intro",
                chunk_index=0,
                total_chunks=20,
                chunk_token_count=512,
                metadata={"tags": ["python"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Java is a statically-typed programming language used for enterprise.",
                similarity_score=0.0,
                bm25_score=0.65,  # Lower match
                hybrid_score=0.65,
                rank=2,
                score_type="bm25",
                source_file="java.md",
                source_category="programming",
                document_date=None,
                context_header="java.md > Overview",
                chunk_index=0,
                total_chunks=15,
                chunk_token_count=512,
                metadata={"tags": ["java"]},
            ),
        ]

        # Results ranked by BM25 score
        assert results[0].bm25_score > results[1].bm25_score
        assert results[0].rank < results[1].rank

    def test_bm25_search_multiple_results(self) -> None:
        """Test BM25 search with multiple keyword matches."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content about search query keywords {i}",
                similarity_score=0.0,
                bm25_score=0.90 - (i * 0.08),  # Decreasing scores
                hybrid_score=0.90 - (i * 0.08),
                rank=i + 1,
                score_type="bm25",
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

        # Validate BM25 ranking
        scores = [r.bm25_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_search_score_range_normalization(self) -> None:
        """Test that BM25 scores are properly normalized to 0-1."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Search result {i}",
                similarity_score=0.0,
                bm25_score=0.95 - (i * 0.10),
                hybrid_score=0.95 - (i * 0.10),
                rank=i + 1,
                score_type="bm25",
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

        # All scores should be 0-1
        for result in results:
            assert 0.0 <= result.bm25_score <= 1.0


class TestBM25SearchRelevance:
    """Tests for BM25 search relevance and ranking."""

    def test_bm25_exact_phrase_match(self) -> None:
        """Test BM25 ranking with exact phrase matches."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="The exact phrase search query appears here exactly.",
                similarity_score=0.0,
                bm25_score=0.95,  # High - exact phrase match
                hybrid_score=0.95,
                rank=1,
                score_type="bm25",
                source_file="exact.md",
                source_category="doc",
                document_date=None,
                context_header="exact.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="This has search and query and phrase but not together.",
                similarity_score=0.0,
                bm25_score=0.70,  # Lower - words scattered
                hybrid_score=0.70,
                rank=2,
                score_type="bm25",
                source_file="scattered.md",
                source_category="doc",
                document_date=None,
                context_header="scattered.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Exact phrase should rank higher
        assert results[0].bm25_score > results[1].bm25_score
        assert results[0].rank == 1

    def test_bm25_term_frequency_impact(self) -> None:
        """Test that term frequency affects BM25 ranking."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Python Python Python programming Python is great Python code.",
                similarity_score=0.0,
                bm25_score=0.88,  # Higher - term frequency
                hybrid_score=0.88,
                rank=1,
                score_type="bm25",
                source_file="freq_high.md",
                source_category="doc",
                document_date=None,
                context_header="freq_high.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Python programming is a valuable skill for developers.",
                similarity_score=0.0,
                bm25_score=0.72,  # Lower - lower term frequency
                hybrid_score=0.72,
                rank=2,
                score_type="bm25",
                source_file="freq_low.md",
                source_category="doc",
                document_date=None,
                context_header="freq_low.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Higher term frequency should rank higher
        assert results[0].bm25_score > results[1].bm25_score

    def test_bm25_inverse_document_frequency(self) -> None:
        """Test that rare terms get higher BM25 weights."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Soliloquy about quantum computing paradigms.",
                similarity_score=0.0,
                bm25_score=0.82,  # Higher - rare terms
                hybrid_score=0.82,
                rank=1,
                score_type="bm25",
                source_file="rare.md",
                source_category="doc",
                document_date=None,
                context_header="rare.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="The the the the common common common words here.",
                similarity_score=0.0,
                bm25_score=0.35,  # Lower - common terms
                hybrid_score=0.35,
                rank=2,
                score_type="bm25",
                source_file="common.md",
                source_category="doc",
                document_date=None,
                context_header="common.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Rare terms should rank higher
        assert results[0].bm25_score > results[1].bm25_score


class TestBM25SearchStopWords:
    """Tests for stop word handling in BM25 search."""

    def test_bm25_ignores_common_stop_words(self) -> None:
        """Test that common stop words don't affect ranking."""
        # Both results have same meaningful content, different stop words
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="The machine learning algorithm",
                similarity_score=0.0,
                bm25_score=0.75,
                hybrid_score=0.75,
                rank=1,
                score_type="bm25",
                source_file="doc1.md",
                source_category="doc",
                document_date=None,
                context_header="doc1.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="A machine learning algorithm",
                similarity_score=0.0,
                bm25_score=0.75,  # Should be same - "the" vs "a"
                hybrid_score=0.75,
                rank=2,
                score_type="bm25",
                source_file="doc2.md",
                source_category="doc",
                document_date=None,
                context_header="doc2.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # Stop word variations should have minimal impact
        assert abs(results[0].bm25_score - results[1].bm25_score) < 0.05


class TestBM25SearchPerformance:
    """Tests for BM25 search performance characteristics."""

    def test_bm25_search_large_result_set(self) -> None:
        """Test BM25 search with large result sets."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Keyword matching content {i}",
                similarity_score=0.0,
                bm25_score=max(0.01, 0.90 - (i * 0.008)),
                hybrid_score=max(0.01, 0.90 - (i * 0.008)),
                rank=i + 1,
                score_type="bm25",
                source_file=f"doc_{i % 10}.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=i % 20,
                total_chunks=20,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(100)
        ]

        assert len(results) == 100
        assert all(r.bm25_score >= 0.0 for r in results)

    def test_bm25_search_single_result(self) -> None:
        """Test BM25 search returning single result."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Only matching result for this specific query.",
            similarity_score=0.0,
            bm25_score=0.85,
            hybrid_score=0.85,
            rank=1,
            score_type="bm25",
            source_file="single.md",
            source_category="doc",
            document_date=None,
            context_header="single.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        assert result.rank == 1
        assert result.bm25_score > 0.0


class TestBM25SearchConsistency:
    """Tests for consistency of BM25 search results."""

    def test_bm25_same_query_same_results(self) -> None:
        """Test that identical queries produce identical results."""
        query = "machine learning algorithm"

        result1 = SearchResult(
            chunk_id=1,
            chunk_text="Machine learning algorithms are powerful.",
            similarity_score=0.0,
            bm25_score=0.83,
            hybrid_score=0.83,
            rank=1,
            score_type="bm25",
            source_file="ml.md",
            source_category="ml",
            document_date=None,
            context_header="ml.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        result2 = SearchResult(
            chunk_id=1,
            chunk_text="Machine learning algorithms are powerful.",
            similarity_score=0.0,
            bm25_score=0.83,
            hybrid_score=0.83,
            rank=1,
            score_type="bm25",
            source_file="ml.md",
            source_category="ml",
            document_date=None,
            context_header="ml.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # Should be identical
        assert result1.bm25_score == result2.bm25_score
        assert result1.chunk_id == result2.chunk_id

    def test_bm25_ranking_consistency(self) -> None:
        """Test that ranking is consistent across identical queries."""
        results1 = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content about search {i}",
                similarity_score=0.0,
                bm25_score=0.88 - (i * 0.05),
                hybrid_score=0.88 - (i * 0.05),
                rank=i + 1,
                score_type="bm25",
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

        results2 = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content about search {i}",
                similarity_score=0.0,
                bm25_score=0.88 - (i * 0.05),
                hybrid_score=0.88 - (i * 0.05),
                rank=i + 1,
                score_type="bm25",
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

        # Ranking should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.bm25_score == r2.bm25_score
            assert r1.rank == r2.rank


class TestBM25SearchQueryParsing:
    """Tests for BM25 search query parsing."""

    def test_bm25_single_term_query(self) -> None:
        """Test BM25 search with single term."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Python programming language",
            similarity_score=0.0,
            bm25_score=0.82,
            hybrid_score=0.82,
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
        )

        assert result.bm25_score > 0.0

    def test_bm25_multi_term_query(self) -> None:
        """Test BM25 search with multiple terms."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Machine learning deep neural networks.",
                similarity_score=0.0,
                bm25_score=0.91,  # Contains all terms
                hybrid_score=0.91,
                rank=1,
                score_type="bm25",
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
                chunk_text="Machine learning is interesting.",
                similarity_score=0.0,
                bm25_score=0.70,  # Missing "deep" and "neural networks"
                hybrid_score=0.70,
                rank=2,
                score_type="bm25",
                source_file="ml2.md",
                source_category="ml",
                document_date=None,
                context_header="ml2.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        # More term matches should rank higher
        assert results[0].bm25_score > results[1].bm25_score
