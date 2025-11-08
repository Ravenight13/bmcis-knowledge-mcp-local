"""Unit tests for RRF (Reciprocal Rank Fusion) algorithm.

Tests cover:
- RRF score calculation correctness
- Two-source merging with deduplication
- Multi-source fusion
- Weight normalization
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from datetime import datetime

from src.search.rrf import RRFScorer
from src.search.results import SearchResult


# Fixtures for creating test data
@pytest.fixture
def sample_vector_results() -> list[SearchResult]:
    """Create sample vector search results."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="Vector result 1",
            similarity_score=0.95,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section A",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Vector result 2",
            similarity_score=0.87,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=2,
            score_type="vector",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section B",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=256,
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Vector result 3",
            similarity_score=0.76,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=3,
            score_type="vector",
            source_file="doc2.md",
            source_category="guide",
            document_date=None,
            context_header="doc2.md > Section C",
            chunk_index=0,
            total_chunks=3,
            chunk_token_count=128,
        ),
    ]


@pytest.fixture
def sample_bm25_results() -> list[SearchResult]:
    """Create sample BM25 search results."""
    return [
        SearchResult(
            chunk_id=2,
            chunk_text="Vector result 2",
            similarity_score=0.0,
            bm25_score=0.92,
            hybrid_score=0.0,
            rank=1,
            score_type="bm25",
            source_file="doc1.md",
            source_category="guide",
            document_date=None,
            context_header="doc1.md > Section B",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=256,
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="BM25 result 2",
            similarity_score=0.0,
            bm25_score=0.88,
            hybrid_score=0.0,
            rank=2,
            score_type="bm25",
            source_file="doc3.md",
            source_category="api_docs",
            document_date=None,
            context_header="doc3.md > Section D",
            chunk_index=2,
            total_chunks=4,
            chunk_token_count=384,
        ),
        SearchResult(
            chunk_id=5,
            chunk_text="BM25 result 3",
            similarity_score=0.0,
            bm25_score=0.75,
            hybrid_score=0.0,
            rank=3,
            score_type="bm25",
            source_file="doc4.md",
            source_category="kb_article",
            document_date=None,
            context_header="doc4.md > Section E",
            chunk_index=1,
            total_chunks=6,
            chunk_token_count=512,
        ),
    ]


class TestRRFScorerInit:
    """Tests for RRFScorer initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default k value."""
        scorer = RRFScorer()
        assert scorer.k == 60

    def test_custom_k_value(self) -> None:
        """Test initialization with custom k value."""
        scorer = RRFScorer(k=100)
        assert scorer.k == 100

    def test_k_value_too_low(self) -> None:
        """Test that k < MIN_K raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            RRFScorer(k=0)

    def test_k_value_too_high(self) -> None:
        """Test that k > MAX_K raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            RRFScorer(k=1001)

    def test_valid_k_boundaries(self) -> None:
        """Test valid boundary values for k."""
        scorer_min = RRFScorer(k=1)
        assert scorer_min.k == 1

        scorer_max = RRFScorer(k=1000)
        assert scorer_max.k == 1000


class TestRRFScoreCalculation:
    """Tests for RRF score calculation formula."""

    def test_rrf_score_first_rank(self) -> None:
        """Test RRF score for first rank (1)."""
        scorer = RRFScorer(k=60)
        # Formula: 1 / (60 + 1) = 1/61 ≈ 0.01639
        score = scorer._calculate_rrf_score(1)
        assert abs(score - (1.0 / 61)) < 1e-6
        assert 0 < score < 1

    def test_rrf_score_second_rank(self) -> None:
        """Test RRF score for second rank."""
        scorer = RRFScorer(k=60)
        # Formula: 1 / (60 + 2) = 1/62
        score = scorer._calculate_rrf_score(2)
        assert abs(score - (1.0 / 62)) < 1e-6

    def test_rrf_score_decreases_with_rank(self) -> None:
        """Test that RRF score decreases as rank increases."""
        scorer = RRFScorer(k=60)
        scores = [scorer._calculate_rrf_score(i) for i in range(1, 6)]

        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_rrf_score_with_different_k(self) -> None:
        """Test that RRF score varies with k parameter."""
        scorer_small_k = RRFScorer(k=10)
        scorer_large_k = RRFScorer(k=100)

        score_small = scorer_small_k._calculate_rrf_score(1)
        score_large = scorer_large_k._calculate_rrf_score(1)

        # Smaller k gives higher score
        assert score_small > score_large

    def test_rrf_score_invalid_rank(self) -> None:
        """Test that invalid rank raises ValueError."""
        scorer = RRFScorer()

        with pytest.raises(ValueError, match="rank must be >= 1"):
            scorer._calculate_rrf_score(0)

        with pytest.raises(ValueError, match="rank must be >= 1"):
            scorer._calculate_rrf_score(-1)

    def test_rrf_score_range(self) -> None:
        """Test that RRF scores are always in (0, 1) range."""
        scorer = RRFScorer(k=60)

        for rank in range(1, 1000):
            score = scorer._calculate_rrf_score(rank)
            assert 0 < score < 1


class TestWeightNormalization:
    """Tests for weight normalization."""

    def test_normalize_equal_weights(self) -> None:
        """Test normalization of equal weights."""
        scorer = RRFScorer()
        w1, w2 = scorer._normalize_weights((1.0, 1.0))
        assert abs(w1 - 0.5) < 1e-6
        assert abs(w2 - 0.5) < 1e-6
        assert abs((w1 + w2) - 1.0) < 1e-6

    def test_normalize_unequal_weights(self) -> None:
        """Test normalization of unequal weights."""
        scorer = RRFScorer()
        w1, w2 = scorer._normalize_weights((3.0, 1.0))
        assert abs(w1 - 0.75) < 1e-6
        assert abs(w2 - 0.25) < 1e-6
        assert abs((w1 + w2) - 1.0) < 1e-6

    def test_normalize_default_weights(self) -> None:
        """Test normalization of default (0.6, 0.4) weights."""
        scorer = RRFScorer()
        w1, w2 = scorer._normalize_weights((0.6, 0.4))
        assert abs(w1 - 0.6) < 1e-6
        assert abs(w2 - 0.4) < 1e-6

    def test_normalize_zero_first_weight(self) -> None:
        """Test normalization with first weight zero."""
        scorer = RRFScorer()
        w1, w2 = scorer._normalize_weights((0.0, 2.0))
        assert abs(w1 - 0.0) < 1e-6
        assert abs(w2 - 1.0) < 1e-6

    def test_normalize_both_zero(self) -> None:
        """Test that both weights being zero raises ValueError."""
        scorer = RRFScorer()

        with pytest.raises(ValueError, match="At least one weight must be positive"):
            scorer._normalize_weights((0.0, 0.0))

    def test_normalize_negative_weights(self) -> None:
        """Test that negative weights raise ValueError."""
        scorer = RRFScorer()

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            scorer._normalize_weights((-1.0, 2.0))

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            scorer._normalize_weights((1.0, -1.0))


class TestMergeResults:
    """Tests for merge_results method."""

    def test_merge_empty_lists(self) -> None:
        """Test merging empty result lists."""
        scorer = RRFScorer()
        result = scorer.merge_results([], [])
        assert result == []

    def test_merge_only_vector_results(
        self, sample_vector_results: list[SearchResult]
    ) -> None:
        """Test merging with only vector results."""
        scorer = RRFScorer()
        result = scorer.merge_results(sample_vector_results, [])

        assert len(result) == 3
        # Results should maintain order, with weighted scores
        assert all(r.score_type == "hybrid" for r in result)
        assert all(0 < r.hybrid_score <= 1.0 for r in result)

    def test_merge_only_bm25_results(
        self, sample_bm25_results: list[SearchResult]
    ) -> None:
        """Test merging with only BM25 results."""
        scorer = RRFScorer()
        result = scorer.merge_results([], sample_bm25_results)

        assert len(result) == 3
        assert all(r.score_type == "hybrid" for r in result)
        assert all(0 < r.hybrid_score <= 1.0 for r in result)

    def test_merge_both_sources_with_deduplication(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test merging with deduplication of common chunks."""
        scorer = RRFScorer(k=60)
        result = scorer.merge_results(sample_vector_results, sample_bm25_results)

        # chunk_id 2 appears in both sources, so total should be 5 not 6
        assert len(result) == 5
        assert all(r.score_type == "hybrid" for r in result)
        assert all(0 < r.hybrid_score <= 1.0 for r in result)

        # Verify ranks are sequential starting at 1
        for i, r in enumerate(result, start=1):
            assert r.rank == i

    def test_merge_order_by_score(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test that results are ordered by RRF score descending."""
        scorer = RRFScorer()
        result = scorer.merge_results(sample_vector_results, sample_bm25_results)

        # Verify scores are in descending order
        scores = [r.hybrid_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_merge_custom_weights(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test merging with custom weights."""
        scorer = RRFScorer()

        # Test with equal weights
        result_equal = scorer.merge_results(
            sample_vector_results, sample_bm25_results, weights=(0.5, 0.5)
        )

        # Test with vector-biased weights
        result_vector_biased = scorer.merge_results(
            sample_vector_results, sample_bm25_results, weights=(0.9, 0.1)
        )

        # Both should produce valid results
        assert len(result_equal) == 5
        assert len(result_vector_biased) == 5
        # Results should have different score distributions due to different weights
        equal_scores = [r.hybrid_score for r in result_equal]
        biased_scores = [r.hybrid_score for r in result_vector_biased]
        assert equal_scores != biased_scores  # At least some scores differ

    def test_merge_preserves_metadata(
        self, sample_vector_results: list[SearchResult],
    ) -> None:
        """Test that merge preserves document metadata."""
        scorer = RRFScorer()
        result = scorer.merge_results(sample_vector_results, [])

        # Check that all fields are preserved
        for original, merged in zip(sample_vector_results, result):
            assert merged.chunk_id == original.chunk_id
            assert merged.chunk_text == original.chunk_text
            assert merged.source_file == original.source_file
            assert merged.source_category == original.source_category
            assert merged.context_header == original.context_header


class TestFuseMultiple:
    """Tests for fuse_multiple method."""

    def test_fuse_empty_dict(self) -> None:
        """Test fusing empty results dictionary."""
        scorer = RRFScorer()
        result = scorer.fuse_multiple({})
        assert result == []

    def test_fuse_single_source(
        self, sample_vector_results: list[SearchResult]
    ) -> None:
        """Test fusing with single source."""
        scorer = RRFScorer()
        result = scorer.fuse_multiple({"vector": sample_vector_results})

        assert len(result) == len(sample_vector_results)
        assert all(r.score_type == "hybrid" for r in result)

    def test_fuse_two_sources(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test fusing two sources."""
        scorer = RRFScorer()
        result = scorer.fuse_multiple({
            "vector": sample_vector_results,
            "bm25": sample_bm25_results,
        })

        assert len(result) == 5  # Deduplicated
        assert all(r.score_type == "hybrid" for r in result)

    def test_fuse_with_custom_weights(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test fusing with custom weight configuration."""
        scorer = RRFScorer()
        result = scorer.fuse_multiple(
            {
                "vector": sample_vector_results,
                "bm25": sample_bm25_results,
            },
            weights={"vector": 0.7, "bm25": 0.3},
        )

        assert len(result) == 5
        assert all(0 < r.hybrid_score <= 1.0 for r in result)

    def test_fuse_invalid_weights_sum(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test that weights not summing to 1.0 raise ValueError."""
        scorer = RRFScorer()

        with pytest.raises(ValueError, match="sum to 1.0"):
            scorer.fuse_multiple(
                {
                    "vector": sample_vector_results,
                    "bm25": sample_bm25_results,
                },
                weights={"vector": 0.5, "bm25": 0.3},  # Sum = 0.8
            )

    def test_fuse_equal_weights_by_default(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test that default weights are equal for all sources."""
        scorer = RRFScorer()
        result = scorer.fuse_multiple({
            "vector": sample_vector_results,
            "bm25": sample_bm25_results,
        })

        # Should succeed with equal weights distribution
        assert len(result) > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_merge_with_single_result_each(self) -> None:
        """Test merging with minimal input."""
        vector_result = SearchResult(
            chunk_id=1,
            chunk_text="Single vector result",
            similarity_score=0.9,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=1,
            score_type="vector",
            source_file="doc.md",
            source_category="guide",
            document_date=None,
            context_header="doc.md > Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=128,
        )

        bm25_result = SearchResult(
            chunk_id=2,
            chunk_text="Single BM25 result",
            similarity_score=0.0,
            bm25_score=0.85,
            hybrid_score=0.0,
            rank=1,
            score_type="bm25",
            source_file="doc2.md",
            source_category="guide",
            document_date=None,
            context_header="doc2.md > Section",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=128,
        )

        scorer = RRFScorer()
        result = scorer.merge_results([vector_result], [bm25_result])

        assert len(result) == 2
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_merge_high_k_value(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test merging with very high k value."""
        scorer = RRFScorer(k=1000)
        result = scorer.merge_results(sample_vector_results, sample_bm25_results)

        assert len(result) > 0
        # High k makes RRF scores very similar
        assert all(0 < r.hybrid_score < 0.01 for r in result)

    def test_merge_low_k_value(
        self, sample_vector_results: list[SearchResult],
        sample_bm25_results: list[SearchResult],
    ) -> None:
        """Test merging with very low k value."""
        scorer = RRFScorer(k=1)
        result = scorer.merge_results(sample_vector_results, sample_bm25_results)

        assert len(result) > 0
        # Low k makes ranking differences more pronounced
        assert max(r.hybrid_score for r in result) > min(r.hybrid_score for r in result)

    def test_score_clamping(self) -> None:
        """Test that scores are clamped to 0-1 range."""
        scorer = RRFScorer()
        results = []
        for i in range(1, 4):
            result = SearchResult(
                chunk_id=i,
                chunk_text=f"Result {i}",
                similarity_score=0.5,
                bm25_score=0.5,
                hybrid_score=0.0,
                rank=i,
                score_type="vector",
                source_file="doc.md",
                source_category="guide",
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=128,
            )
            results.append(result)

        merged = scorer.merge_results(results, [])
        assert all(0.0 <= r.hybrid_score <= 1.0 for r in merged)
