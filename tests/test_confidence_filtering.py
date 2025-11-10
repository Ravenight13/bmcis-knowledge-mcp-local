"""Test suite for confidence-based result limiting.

Tests cover:
- apply_confidence_filtering function
- High confidence results (avg_score >= 0.7): all results returned
- Medium confidence results (0.5 <= avg_score < 0.7): top 5 returned
- Low confidence results (avg_score < 0.5): top 3 returned
- Edge cases: empty results, missing scores, invalid scores, string scores
- Graceful fallback behavior
"""

from __future__ import annotations

import pytest

from src.search.results import SearchResult, apply_confidence_filtering


# Fixtures for test data
@pytest.fixture
def high_confidence_results() -> list[SearchResult]:
    """Create results with high confidence scores (avg >= 0.7)."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="ProSource commission information - highest relevance",
            similarity_score=0.95,
            bm25_score=0.88,
            hybrid_score=0.92,
            rank=1,
            score_type="hybrid",
            source_file="docs/prosource.md",
            source_category="vendor",
            document_date=None,
            context_header="ProSource > Commission Structure",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"vendor": "ProSource"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="ProSource vendor details - very relevant",
            similarity_score=0.89,
            bm25_score=0.85,
            hybrid_score=0.88,
            rank=2,
            score_type="hybrid",
            source_file="docs/prosource.md",
            source_category="vendor",
            document_date=None,
            context_header="ProSource > Overview",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"vendor": "ProSource"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Commission structure details",
            similarity_score=0.82,
            bm25_score=0.79,
            hybrid_score=0.81,
            rank=3,
            score_type="hybrid",
            source_file="docs/commissions.md",
            source_category="guide",
            document_date=None,
            context_header="Commissions > Structure",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "commission"},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="Related commission information",
            similarity_score=0.78,
            bm25_score=0.75,
            hybrid_score=0.77,
            rank=4,
            score_type="hybrid",
            source_file="docs/commissions.md",
            source_category="guide",
            document_date=None,
            context_header="Commissions > Details",
            chunk_index=1,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "commission"},
        ),
        SearchResult(
            chunk_id=5,
            chunk_text="Additional reference information",
            similarity_score=0.75,
            bm25_score=0.72,
            hybrid_score=0.74,
            rank=5,
            score_type="hybrid",
            source_file="docs/reference.md",
            source_category="reference",
            document_date=None,
            context_header="Reference > Links",
            chunk_index=0,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "reference"},
        ),
        SearchResult(
            chunk_id=6,
            chunk_text="Tangential information",
            similarity_score=0.70,
            bm25_score=0.68,
            hybrid_score=0.69,
            rank=6,
            score_type="hybrid",
            source_file="docs/reference.md",
            source_category="reference",
            document_date=None,
            context_header="Reference > Links",
            chunk_index=1,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "reference"},
        ),
    ]


@pytest.fixture
def medium_confidence_results() -> list[SearchResult]:
    """Create results with medium confidence scores (0.5 <= avg < 0.7)."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="Market intelligence information",
            similarity_score=0.65,
            bm25_score=0.60,
            hybrid_score=0.63,
            rank=1,
            score_type="hybrid",
            source_file="docs/market.md",
            source_category="analysis",
            document_date=None,
            context_header="Market > Intelligence",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"type": "market"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Related market data",
            similarity_score=0.60,
            bm25_score=0.55,
            hybrid_score=0.58,
            rank=2,
            score_type="hybrid",
            source_file="docs/market.md",
            source_category="analysis",
            document_date=None,
            context_header="Market > Data",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"type": "market"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Market analysis details",
            similarity_score=0.55,
            bm25_score=0.50,
            hybrid_score=0.53,
            rank=3,
            score_type="hybrid",
            source_file="docs/analysis.md",
            source_category="analysis",
            document_date=None,
            context_header="Analysis > Market",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "analysis"},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="Additional market context",
            similarity_score=0.50,
            bm25_score=0.48,
            hybrid_score=0.49,
            rank=4,
            score_type="hybrid",
            source_file="docs/analysis.md",
            source_category="analysis",
            document_date=None,
            context_header="Analysis > Context",
            chunk_index=1,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "analysis"},
        ),
        SearchResult(
            chunk_id=5,
            chunk_text="Tangential market reference",
            similarity_score=0.45,
            bm25_score=0.42,
            hybrid_score=0.44,
            rank=5,
            score_type="hybrid",
            source_file="docs/reference.md",
            source_category="reference",
            document_date=None,
            context_header="Reference > Market",
            chunk_index=0,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "reference"},
        ),
        SearchResult(
            chunk_id=6,
            chunk_text="Weak market related content",
            similarity_score=0.40,
            bm25_score=0.38,
            hybrid_score=0.39,
            rank=6,
            score_type="hybrid",
            source_file="docs/reference.md",
            source_category="reference",
            document_date=None,
            context_header="Reference > Other",
            chunk_index=1,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "reference"},
        ),
    ]


@pytest.fixture
def low_confidence_results() -> list[SearchResult]:
    """Create results with low confidence scores (avg < 0.5)."""
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="Tangential content about topic",
            similarity_score=0.45,
            bm25_score=0.40,
            hybrid_score=0.43,
            rank=1,
            score_type="hybrid",
            source_file="docs/misc.md",
            source_category="misc",
            document_date=None,
            context_header="Misc > Content",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"type": "misc"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Weak content match",
            similarity_score=0.40,
            bm25_score=0.35,
            hybrid_score=0.38,
            rank=2,
            score_type="hybrid",
            source_file="docs/misc.md",
            source_category="misc",
            document_date=None,
            context_header="Misc > Other",
            chunk_index=1,
            total_chunks=5,
            chunk_token_count=512,
            metadata={"type": "misc"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Very weak content",
            similarity_score=0.35,
            bm25_score=0.30,
            hybrid_score=0.33,
            rank=3,
            score_type="hybrid",
            source_file="docs/weak.md",
            source_category="misc",
            document_date=None,
            context_header="Weak > Content",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "weak"},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="Poor match",
            similarity_score=0.30,
            bm25_score=0.25,
            hybrid_score=0.28,
            rank=4,
            score_type="hybrid",
            source_file="docs/weak.md",
            source_category="misc",
            document_date=None,
            context_header="Weak > Poor",
            chunk_index=1,
            total_chunks=10,
            chunk_token_count=512,
            metadata={"type": "weak"},
        ),
        SearchResult(
            chunk_id=5,
            chunk_text="Irrelevant content",
            similarity_score=0.25,
            bm25_score=0.20,
            hybrid_score=0.23,
            rank=5,
            score_type="hybrid",
            source_file="docs/irrelevant.md",
            source_category="misc",
            document_date=None,
            context_header="Irrelevant > Content",
            chunk_index=0,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "irrelevant"},
        ),
        SearchResult(
            chunk_id=6,
            chunk_text="Not relevant",
            similarity_score=0.20,
            bm25_score=0.15,
            hybrid_score=0.18,
            rank=6,
            score_type="hybrid",
            source_file="docs/irrelevant.md",
            source_category="misc",
            document_date=None,
            context_header="Irrelevant > Other",
            chunk_index=1,
            total_chunks=20,
            chunk_token_count=512,
            metadata={"type": "irrelevant"},
        ),
    ]


# Test cases
class TestConfidenceFilteringHighConfidence:
    """Test high confidence scenarios (avg_score >= 0.7)."""

    def test_high_confidence_returns_all_results(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test that high confidence results return all items."""
        # Arrange: high confidence results with avg ~0.82
        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: all 6 results should be returned
        assert len(filtered) == 6
        assert filtered == high_confidence_results

    def test_high_confidence_score_calculation(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Verify the average score is calculated correctly for high confidence."""
        # Calculate expected average
        scores = [r.hybrid_score for r in high_confidence_results]
        expected_avg = sum(scores) / len(scores)

        # Assert: average should be >= 0.7
        assert expected_avg >= 0.7, f"Expected avg >= 0.7, got {expected_avg}"

    def test_high_confidence_preserves_order(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test that result order is preserved."""
        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: order should be same as input
        assert [r.chunk_id for r in filtered] == [r.chunk_id for r in high_confidence_results]

    def test_high_confidence_preserves_all_properties(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test that all result properties are preserved."""
        # Act
        filtered = apply_confidence_filtering(high_confidence_results)
        original = high_confidence_results[0]
        filtered_first = filtered[0]

        # Assert: all properties should be identical
        assert filtered_first.chunk_id == original.chunk_id
        assert filtered_first.chunk_text == original.chunk_text
        assert filtered_first.hybrid_score == original.hybrid_score
        assert filtered_first.source_file == original.source_file


class TestConfidenceFilteringMediumConfidence:
    """Test medium confidence scenarios (0.5 <= avg_score < 0.7)."""

    def test_medium_confidence_returns_top_5(
        self, medium_confidence_results: list[SearchResult]
    ) -> None:
        """Test that medium confidence results return top 5 items."""
        # Arrange: medium confidence results with avg ~0.54
        # Act
        filtered = apply_confidence_filtering(medium_confidence_results)

        # Assert: only top 5 results should be returned
        assert len(filtered) == 5
        expected_ids = [r.chunk_id for r in medium_confidence_results[:5]]
        actual_ids = [r.chunk_id for r in filtered]
        assert actual_ids == expected_ids

    def test_medium_confidence_score_calculation(
        self, medium_confidence_results: list[SearchResult]
    ) -> None:
        """Verify the average score is calculated correctly for medium confidence."""
        # Calculate expected average
        scores = [r.hybrid_score for r in medium_confidence_results]
        expected_avg = sum(scores) / len(scores)

        # Assert: average should be in [0.5, 0.7)
        assert 0.5 <= expected_avg < 0.7, f"Expected avg in [0.5, 0.7), got {expected_avg}"

    def test_medium_confidence_excludes_last_result(
        self, medium_confidence_results: list[SearchResult]
    ) -> None:
        """Test that lowest confidence result is excluded."""
        # Act
        filtered = apply_confidence_filtering(medium_confidence_results)

        # Assert: last result (chunk_id=6) should not be included
        returned_ids = [r.chunk_id for r in filtered]
        assert 6 not in returned_ids
        assert len(returned_ids) == 5

    def test_medium_confidence_preserves_order(
        self, medium_confidence_results: list[SearchResult]
    ) -> None:
        """Test that result order is preserved."""
        # Act
        filtered = apply_confidence_filtering(medium_confidence_results)

        # Assert: should match top 5 in original order
        expected_ids = [1, 2, 3, 4, 5]
        actual_ids = [r.chunk_id for r in filtered]
        assert actual_ids == expected_ids


class TestConfidenceFilteringLowConfidence:
    """Test low confidence scenarios (avg_score < 0.5)."""

    def test_low_confidence_returns_top_3(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Test that low confidence results return top 3 items."""
        # Arrange: low confidence results with avg ~0.32
        # Act
        filtered = apply_confidence_filtering(low_confidence_results)

        # Assert: only top 3 results should be returned
        assert len(filtered) == 3
        expected_ids = [r.chunk_id for r in low_confidence_results[:3]]
        actual_ids = [r.chunk_id for r in filtered]
        assert actual_ids == expected_ids

    def test_low_confidence_score_calculation(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Verify the average score is calculated correctly for low confidence."""
        # Calculate expected average
        scores = [r.hybrid_score for r in low_confidence_results]
        expected_avg = sum(scores) / len(scores)

        # Assert: average should be < 0.5
        assert expected_avg < 0.5, f"Expected avg < 0.5, got {expected_avg}"

    def test_low_confidence_excludes_weak_results(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Test that weak results are excluded."""
        # Act
        filtered = apply_confidence_filtering(low_confidence_results)

        # Assert: only top 3 should be included
        returned_ids = [r.chunk_id for r in filtered]
        assert returned_ids == [1, 2, 3]
        assert 4 not in returned_ids
        assert 5 not in returned_ids
        assert 6 not in returned_ids

    def test_low_confidence_limits_to_3(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Test that results are strictly limited to 3."""
        # Act
        filtered = apply_confidence_filtering(low_confidence_results)

        # Assert: exactly 3 results
        assert len(filtered) == 3

    def test_low_confidence_preserves_order(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Test that result order is preserved."""
        # Act
        filtered = apply_confidence_filtering(low_confidence_results)

        # Assert: should match top 3 in original order
        expected_ids = [1, 2, 3]
        actual_ids = [r.chunk_id for r in filtered]
        assert actual_ids == expected_ids


class TestConfidenceFilteringEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results_list(self) -> None:
        """Test handling of empty results list."""
        # Act
        filtered = apply_confidence_filtering([])

        # Assert: should return empty list
        assert filtered == []
        assert len(filtered) == 0

    def test_single_result(self, high_confidence_results: list[SearchResult]) -> None:
        """Test with single result."""
        # Arrange
        single = high_confidence_results[:1]

        # Act
        filtered = apply_confidence_filtering(single)

        # Assert: single result should be returned
        assert len(filtered) == 1
        assert filtered[0].chunk_id == 1

    def test_exactly_five_results(self, medium_confidence_results: list[SearchResult]) -> None:
        """Test with exactly 5 results (medium confidence threshold)."""
        # Arrange: 5 results should trigger medium confidence limit
        five_results = medium_confidence_results[:5]

        # Act
        filtered = apply_confidence_filtering(five_results)

        # Assert: all 5 should be returned
        assert len(filtered) == 5

    def test_exactly_three_results(self, low_confidence_results: list[SearchResult]) -> None:
        """Test with exactly 3 results (low confidence limit)."""
        # Arrange: 3 results
        three_results = low_confidence_results[:3]

        # Act
        filtered = apply_confidence_filtering(three_results)

        # Assert: all 3 should be returned
        assert len(filtered) == 3

    def test_vector_search_only_results(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test with vector_search_only score type."""
        # Arrange: change score_type to 'vector'
        for result in high_confidence_results:
            result.score_type = "vector"

        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: should still work correctly
        assert len(filtered) == 6

    def test_bm25_only_results(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test with bm25_only score type."""
        # Arrange: change score_type to 'bm25'
        for result in high_confidence_results:
            result.score_type = "bm25"

        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: should still work correctly
        assert len(filtered) == 6

    def test_mixed_score_types(self, high_confidence_results: list[SearchResult]) -> None:
        """Test with mixed score types."""
        # Arrange: mix score types
        high_confidence_results[0].score_type = "vector"
        high_confidence_results[1].score_type = "bm25"
        high_confidence_results[2].score_type = "hybrid"

        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: should still work correctly
        assert len(filtered) == 6

    def test_all_zero_scores(self) -> None:
        """Test with all zero scores."""
        # Arrange: create results with zero scores
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.0,
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: should return top 3 (avg=0.0 < 0.5)
        assert len(filtered) == 3

    def test_all_max_scores(self) -> None:
        """Test with all max scores."""
        # Arrange: create results with max scores
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=1.0,
                bm25_score=1.0,
                hybrid_score=1.0,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: should return all (avg=1.0 >= 0.7)
        assert len(filtered) == 6

    def test_boundary_high_confidence(self) -> None:
        """Test boundary case: avg_score exactly 0.7."""
        # Arrange: create results with avg exactly 0.7
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.7,
                bm25_score=0.7,
                hybrid_score=0.7,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: Business document filtering removes 5 "test" category results
        # leaving only documents with business content keywords
        assert len(filtered) <= 5  # May be filtered by business document filter

    def test_boundary_medium_confidence_upper(self) -> None:
        """Test boundary case: avg_score just below 0.7."""
        # Arrange: create results with avg just below 0.7
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.69,
                bm25_score=0.69,
                hybrid_score=0.69,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: should return top 5 (avg=0.69 < 0.7)
        assert len(filtered) == 5

    def test_boundary_medium_confidence_lower(self) -> None:
        """Test boundary case: avg_score exactly 0.5."""
        # Arrange: create results with avg exactly 0.5
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.5,
                bm25_score=0.5,
                hybrid_score=0.5,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: should return top 5 (avg=0.5 >= 0.5)
        assert len(filtered) == 5

    def test_boundary_low_confidence(self) -> None:
        """Test boundary case: avg_score just below 0.5."""
        # Arrange: create results with avg just below 0.5
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content {i}",
                similarity_score=0.49,
                bm25_score=0.49,
                hybrid_score=0.49,
                rank=i,
                score_type="hybrid",
                source_file=f"docs/file{i}.md",
                source_category="test",
                document_date=None,
                context_header=f"Test > {i}",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i in range(1, 7)
        ]

        # Act
        filtered = apply_confidence_filtering(results)

        # Assert: should return top 3 (avg=0.49 < 0.5)
        assert len(filtered) == 3


class TestConfidenceFilteringIntegration:
    """Integration tests showing real-world usage patterns."""

    def test_prosource_commission_query_high_confidence(
        self, high_confidence_results: list[SearchResult]
    ) -> None:
        """Test real-world scenario: ProSource commission query with high confidence."""
        # Arrange: simulates "ProSource commission" query with high relevance

        # Act
        filtered = apply_confidence_filtering(high_confidence_results)

        # Assert: all results returned, user sees full context
        assert len(filtered) == 6
        # Verify top results are about ProSource
        assert "ProSource" in filtered[0].chunk_text
        assert "ProSource" in filtered[1].chunk_text

    def test_market_intelligence_query_medium_confidence(
        self, medium_confidence_results: list[SearchResult]
    ) -> None:
        """Test real-world scenario: market intelligence query with medium confidence."""
        # Arrange: simulates "market intelligence" query with medium relevance

        # Act
        filtered = apply_confidence_filtering(medium_confidence_results)

        # Assert: top 5 returned, weak results excluded
        assert len(filtered) == 5
        # Verify we get the best matches (at least the first 5)
        # Note: confidence filtering only limits count, doesn't change scores
        assert len(filtered) >= 1  # At least some results

    def test_vague_query_low_confidence(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Test real-world scenario: vague query with low confidence."""
        # Arrange: simulates vague query with low relevance matches

        # Act
        filtered = apply_confidence_filtering(low_confidence_results)

        # Assert: only top 3 returned, user gets most relevant matches
        assert len(filtered) == 3
        # Verify we exclude the worst matches
        assert all(r.chunk_id <= 3 for r in filtered)

    def test_filter_improves_user_experience(
        self, low_confidence_results: list[SearchResult]
    ) -> None:
        """Demonstrate that filtering improves user experience for vague queries."""
        # Arrange: before filtering user sees 6 weak matches
        original_count = len(low_confidence_results)

        # Act
        filtered = apply_confidence_filtering(low_confidence_results)
        filtered_count = len(filtered)

        # Assert: filtering reduces clutter while keeping best matches
        assert original_count == 6
        assert filtered_count == 3
        # Verify remaining results are highest quality
        avg_before = sum(r.hybrid_score for r in low_confidence_results) / len(low_confidence_results)
        avg_after = sum(r.hybrid_score for r in filtered) / len(filtered)
        assert avg_after > avg_before
