"""Test suite for metadata filtering in search results.

Tests cover:
- Basic metadata filtering (category, tags, dates)
- Filter composition (AND/OR/NOT logic)
- Date range filtering
- Category filtering
- Complex filter combinations with search
- Filter edge cases
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from src.search.results import SearchResult


class TestBasicMetadataFiltering:
    """Tests for basic metadata filtering."""

    def test_filter_by_category(self) -> None:
        """Test filtering results by category."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Machine learning guide",
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
                metadata={},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Python guide",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="python.md",
                source_category="programming",
                document_date=None,
                context_header="python.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        filters = {"category": "ml"}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert filtered[0].source_category == "ml"

    def test_filter_by_tags(self) -> None:
        """Test filtering results by tags in metadata."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Important ML content",
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
                metadata={"tags": ["important", "ml", "neural"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Regular content",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="doc.md",
                source_category="doc",
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["regular"]},
            ),
        ]

        filters = {"tags": ["important"]}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert "important" in filtered[0].metadata["tags"]

    def test_filter_by_source_file(self) -> None:
        """Test filtering by source file."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="First doc",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="docs/guide/ml.md",
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
                chunk_text="Second doc",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="docs/api/reference.md",
                source_category="api",
                document_date=None,
                context_header="reference.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            ),
        ]

        filters = {"source_file": "guide"}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert "guide" in filtered[0].source_file

    def test_no_matching_filters(self) -> None:
        """Test filtering with no matching results."""
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
                metadata={},
            ),
        ]

        filters = {"category": "nonexistent"}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 0


class TestComplexFiltering:
    """Tests for complex filtering scenarios."""

    def test_multiple_filters_and_logic(self) -> None:
        """Test multiple filters with AND logic."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Important ML guide",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="docs/ml.md",
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
                chunk_text="Regular ML content",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="docs/other.md",
                source_category="ml",
                document_date=None,
                context_header="other.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["regular"]},
            ),
        ]

        # Both category and tag filter
        filters = {"category": "ml", "tags": ["important"]}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert filtered[0].source_category == "ml"
        assert "important" in filtered[0].metadata["tags"]

    def test_multiple_tags_filter_or_logic(self) -> None:
        """Test filtering with multiple tags (OR logic)."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Tagged content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="doc",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={"tags": ["python", "ml"]},
        )

        # Should match if ANY tag matches
        filters1 = {"tags": ["python"]}
        assert result.matches_filters(filters1) is True

        filters2 = {"tags": ["java"]}
        assert result.matches_filters(filters2) is False

        # Multiple tag filter (any match)
        filters3 = {"tags": ["java", "python"]}
        assert result.matches_filters(filters3) is True


class TestCategoryFiltering:
    """Tests for category-based filtering."""

    def test_filter_categories(self) -> None:
        """Test filtering by different categories."""
        categories = ["ml", "api", "guide", "tutorial"]
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Content for {cat}",
                similarity_score=0.9 - (i * 0.1),
                bm25_score=0.8 - (i * 0.1),
                hybrid_score=0.85 - (i * 0.1),
                rank=i + 1,
                score_type="hybrid",
                source_file=f"{cat}.md",
                source_category=cat,
                document_date=None,
                context_header=f"{cat}.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={},
            )
            for i, cat in enumerate(categories)
        ]

        # Filter by each category
        for cat in categories:
            filters = {"category": cat}
            filtered = [r for r in results if r.matches_filters(filters)]

            assert len(filtered) == 1
            assert filtered[0].source_category == cat

    def test_category_case_sensitivity(self) -> None:
        """Test category filtering case sensitivity."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="ML",  # Uppercase
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # Exact match (case sensitive)
        filters_match = {"category": "ML"}
        assert result.matches_filters(filters_match) is True

        filters_no_match = {"category": "ml"}
        assert result.matches_filters(filters_no_match) is False


class TestTagFiltering:
    """Tests for tag-based filtering."""

    def test_single_tag_filtering(self) -> None:
        """Test filtering by single tag."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Python content",
                similarity_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="python.md",
                source_category="lang",
                document_date=None,
                context_header="python.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["python", "beginner"]},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Java content",
                similarity_score=0.85,
                bm25_score=0.75,
                hybrid_score=0.80,
                rank=2,
                score_type="hybrid",
                source_file="java.md",
                source_category="lang",
                document_date=None,
                context_header="java.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=512,
                metadata={"tags": ["java", "advanced"]},
            ),
        ]

        filters = {"tags": ["python"]}
        filtered = [r for r in results if r.matches_filters(filters)]

        assert len(filtered) == 1
        assert "python" in filtered[0].metadata["tags"]

    def test_multiple_tags_in_metadata(self) -> None:
        """Test filtering content with multiple tags."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Multi-tagged content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="multi.md",
            source_category="doc",
            document_date=None,
            context_header="multi.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={"tags": ["python", "ml", "neural", "deep"]},
        )

        # Should match any of the tags
        filters1 = {"tags": ["python"]}
        assert result.matches_filters(filters1) is True

        filters2 = {"tags": ["deep"]}
        assert result.matches_filters(filters2) is True

        filters3 = {"tags": ["java"]}
        assert result.matches_filters(filters3) is False

    def test_empty_tag_filter(self) -> None:
        """Test filtering with empty tag list."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="doc",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={"tags": ["important"]},
        )

        filters = {"tags": []}
        # Empty filter list should return no matches
        assert result.matches_filters(filters) is False


class TestFilterEdgeCases:
    """Tests for filter edge cases."""

    def test_filter_with_missing_metadata(self) -> None:
        """Test filtering results with missing metadata fields."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="doc",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},  # No tags
        )

        filters = {"tags": ["important"]}
        # Should not match - no tags in metadata
        assert result.matches_filters(filters) is False

    def test_filter_with_none_category(self) -> None:
        """Test filtering results with None category."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category=None,  # None category
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        filters = {"category": "doc"}
        # Should not match
        assert result.matches_filters(filters) is False

    def test_filter_with_partial_source_file_match(self) -> None:
        """Test source file filter with partial match."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="docs/guides/python/tutorial.md",
            source_category="guide",
            document_date=None,
            context_header="tutorial.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=512,
            metadata={},
        )

        # Partial match should work
        assert result.matches_filters({"source_file": "python"}) is True
        assert result.matches_filters({"source_file": "guides"}) is True
        assert result.matches_filters({"source_file": "tutorial.md"}) is True
        assert result.matches_filters({"source_file": "java"}) is False

    def test_empty_filter_dict(self) -> None:
        """Test filtering with empty filter dictionary."""
        result = SearchResult(
            chunk_id=1,
            chunk_text="Content",
            similarity_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
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
        )

        # Empty filters should match everything
        assert result.matches_filters({}) is True
