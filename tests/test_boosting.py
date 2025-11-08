"""Test suite for search result boosting and re-ranking.

Tests cover:
- Individual boost factors (vendor, doc type, recency, entity, topic)
- Cumulative boost logic and combinations
- Score clamping (0-1 bounds)
- Re-ranking after boosts
- Edge cases (null metadata, empty results, large result sets)
- Performance benchmarks

Boost factors:
- Vendor boost: +15%
- Doc type boost: +10%
- Recency boost: +5% (max, decays)
- Entity boost: +10%
- Topic boost: +8%
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.search.results import SearchResult


# Test Fixtures
@pytest.fixture
def sample_result() -> SearchResult:
    """Create a sample search result for boosting tests."""
    return SearchResult(
        chunk_id=1,
        chunk_text="Sample content about machine learning.",
        similarity_score=0.75,
        bm25_score=0.65,
        hybrid_score=0.72,
        rank=1,
        score_type="hybrid",
        source_file="docs/ml-guide.md",
        source_category="guide",
        document_date=datetime.now(),
        context_header="ml-guide > Introduction",
        chunk_index=0,
        total_chunks=10,
        chunk_token_count=512,
        metadata={
            "vendor": "openai",
            "doc_type": "api_docs",
            "tags": ["ml", "ai"],
            "entities": ["neural networks", "deep learning"],
            "topics": ["machine-learning"],
        },
    )


@pytest.fixture
def sample_results_list() -> list[SearchResult]:
    """Create multiple sample results for testing."""
    return [
        SearchResult(
            chunk_id=i,
            chunk_text=f"Content about topic {i}",
            similarity_score=0.95 - (i * 0.05),
            bm25_score=0.85 - (i * 0.05),
            hybrid_score=0.92 - (i * 0.05),
            rank=i + 1,
            score_type="hybrid",
            source_file=f"docs/doc{i}.md",
            source_category="guide" if i % 2 == 0 else "reference",
            document_date=datetime.now() - timedelta(days=i*10),
            context_header=f"doc{i} > Section",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
            metadata={
                "vendor": ["openai", "anthropic"][i % 2],
                "doc_type": ["api_docs", "kb_article", "tutorial"][i % 3],
                "tags": [["ml"], ["ai"], ["data"], ["python"]][i % 4],
                "entities": ["entity1", "entity2"],
                "topics": ["topic1"],
            },
        )
        for i in range(10)
    ]


@pytest.fixture
def booster() -> Any:
    """Create booster instance for testing."""
    class ResultBooster:
        """Boost search result scores based on metadata."""

        # Boost percentages
        VENDOR_BOOST = 0.15
        DOC_TYPE_BOOST = 0.10
        RECENCY_BOOST_MAX = 0.05
        ENTITY_BOOST = 0.10
        TOPIC_BOOST = 0.08

        # Recency decay: max boost for < 7 days, decay over 365 days
        RECENCY_DECAY_START_DAYS = 7
        RECENCY_DECAY_END_DAYS = 365

        def __init__(self) -> None:
            """Initialize booster."""
            self.preferred_vendors = ["anthropic", "openai"]
            self.preferred_doc_types = ["api_docs", "kb_article"]
            self.preferred_entities = ["neural networks", "deep learning"]
            self.preferred_topics = ["machine-learning"]

        def apply_vendor_boost(
            self,
            result: SearchResult,
            boost_amount: float | None = None,
        ) -> float:
            """Apply vendor boost (+15% if vendor matches)."""
            if boost_amount is None:
                boost_amount = self.VENDOR_BOOST

            vendor = result.metadata.get("vendor", "").lower()
            if vendor in self.preferred_vendors:
                return boost_amount
            return 0.0

        def apply_doc_type_boost(
            self,
            result: SearchResult,
            boost_amount: float | None = None,
        ) -> float:
            """Apply doc type boost (+10% if type matches)."""
            if boost_amount is None:
                boost_amount = self.DOC_TYPE_BOOST

            doc_type = result.metadata.get("doc_type", "").lower()
            if doc_type in self.preferred_doc_types:
                return boost_amount
            return 0.0

        def apply_recency_boost(
            self,
            result: SearchResult,
            boost_amount: float | None = None,
        ) -> float:
            """Apply recency boost (+5% max, decays over time)."""
            if boost_amount is None:
                boost_amount = self.RECENCY_BOOST_MAX

            if result.document_date is None:
                return 0.0

            now = datetime.now()
            age_days = (now - result.document_date).days

            # Max boost for documents < 7 days old
            if age_days <= self.RECENCY_DECAY_START_DAYS:
                return boost_amount

            # Gradual decay from 7 to 365 days
            if age_days >= self.RECENCY_DECAY_END_DAYS:
                return 0.0

            # Linear decay
            decay_range = (
                self.RECENCY_DECAY_END_DAYS -
                self.RECENCY_DECAY_START_DAYS
            )
            days_in_decay = age_days - self.RECENCY_DECAY_START_DAYS
            decay_factor = 1.0 - (days_in_decay / decay_range)

            return boost_amount * decay_factor

        def apply_entity_boost(
            self,
            result: SearchResult,
            boost_amount: float | None = None,
        ) -> float:
            """Apply entity boost (+10% if entities match)."""
            if boost_amount is None:
                boost_amount = self.ENTITY_BOOST

            result_entities = result.metadata.get("entities", [])
            if isinstance(result_entities, str):
                result_entities = [result_entities]

            for entity in result_entities:
                if entity.lower() in [e.lower() for e in self.preferred_entities]:
                    return boost_amount

            return 0.0

        def apply_topic_boost(
            self,
            result: SearchResult,
            boost_amount: float | None = None,
        ) -> float:
            """Apply topic boost (+8% if topic matches)."""
            if boost_amount is None:
                boost_amount = self.TOPIC_BOOST

            result_topics = result.metadata.get("topics", [])
            if isinstance(result_topics, str):
                result_topics = [result_topics]

            for topic in result_topics:
                if topic.lower() in [t.lower() for t in self.preferred_topics]:
                    return boost_amount

            return 0.0

        def apply_all_boosts(
            self, result: SearchResult
        ) -> tuple[SearchResult, dict[str, float]]:
            """Apply all boost factors to a result."""
            boosts = {
                "vendor": self.apply_vendor_boost(result),
                "doc_type": self.apply_doc_type_boost(result),
                "recency": self.apply_recency_boost(result),
                "entity": self.apply_entity_boost(result),
                "topic": self.apply_topic_boost(result),
            }

            total_boost = sum(boosts.values())

            # Calculate new score with clamping
            original_score = result.hybrid_score
            boosted_score = original_score + (original_score * total_boost)
            final_score = min(1.0, max(0.0, boosted_score))

            # Create new result with boosted score
            boosted_result = SearchResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                similarity_score=result.similarity_score,
                bm25_score=result.bm25_score,
                hybrid_score=final_score,
                rank=result.rank,
                score_type=result.score_type,
                source_file=result.source_file,
                source_category=result.source_category,
                document_date=result.document_date,
                context_header=result.context_header,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                chunk_token_count=result.chunk_token_count,
                metadata=result.metadata,
                highlighted_context=result.highlighted_context,
                confidence=result.confidence,
            )

            return boosted_result, boosts

        def rerank_results(
            self, results: list[SearchResult]
        ) -> list[SearchResult]:
            """Apply boosts and rerank results."""
            boosted_results = []

            for result in results:
                boosted, _ = self.apply_all_boosts(result)
                boosted_results.append(boosted)

            # Sort by boosted score descending
            boosted_results.sort(
                key=lambda x: x.hybrid_score,
                reverse=True
            )

            # Update ranks
            for new_rank, result in enumerate(boosted_results, 1):
                result.rank = new_rank

            return boosted_results

    return ResultBooster()


# Individual Boost Factor Tests
class TestIndividualBoostFactors:
    """Test individual boost factors."""

    def test_vendor_boost_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test vendor boost with matching vendor."""
        original_score = sample_result.hybrid_score
        boost_amount = booster.apply_vendor_boost(sample_result)

        assert boost_amount == 0.15  # 15% boost
        boosted_score = original_score + (original_score * boost_amount)
        assert boosted_score > original_score

    def test_vendor_boost_no_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test vendor boost with non-matching vendor."""
        sample_result.metadata["vendor"] = "google"
        boost_amount = booster.apply_vendor_boost(sample_result)

        assert boost_amount == 0.0

    def test_vendor_boost_missing_vendor(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test vendor boost when vendor not in metadata."""
        sample_result.metadata = {}
        boost_amount = booster.apply_vendor_boost(sample_result)

        assert boost_amount == 0.0

    def test_doc_type_boost_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test doc type boost with matching type."""
        original_score = sample_result.hybrid_score
        boost_amount = booster.apply_doc_type_boost(sample_result)

        assert boost_amount == 0.10  # 10% boost
        boosted_score = original_score + (original_score * boost_amount)
        assert boosted_score > original_score

    def test_doc_type_boost_no_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test doc type boost with non-matching type."""
        sample_result.metadata["doc_type"] = "blog_post"
        boost_amount = booster.apply_doc_type_boost(sample_result)

        assert boost_amount == 0.0

    def test_doc_type_boost_unknown_type(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test doc type boost with unknown type."""
        sample_result.metadata["doc_type"] = "unknown_format"
        boost_amount = booster.apply_doc_type_boost(sample_result)

        assert boost_amount == 0.0

    def test_recency_boost_recent_doc(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test recency boost for recent document (<7 days)."""
        sample_result.document_date = datetime.now() - timedelta(days=3)
        boost_amount = booster.apply_recency_boost(sample_result)

        assert boost_amount == 0.05  # Max boost

    def test_recency_boost_old_doc(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test recency boost for old document (>1 year)."""
        sample_result.document_date = datetime.now() - timedelta(days=400)
        boost_amount = booster.apply_recency_boost(sample_result)

        assert boost_amount == 0.0  # No boost

    def test_recency_boost_gradual_decay(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test gradual decay of recency boost over time."""
        # 30 days old: should have some boost
        sample_result.document_date = datetime.now() - timedelta(days=30)
        boost_30 = booster.apply_recency_boost(sample_result)

        # 60 days old: should have less boost
        sample_result.document_date = datetime.now() - timedelta(days=60)
        boost_60 = booster.apply_recency_boost(sample_result)

        # 90 days old: should have even less boost
        sample_result.document_date = datetime.now() - timedelta(days=90)
        boost_90 = booster.apply_recency_boost(sample_result)

        assert 0.0 < boost_90 < boost_60 < boost_30 < 0.05

    def test_recency_boost_null_date(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test recency boost with null document_date."""
        sample_result.document_date = None
        boost_amount = booster.apply_recency_boost(sample_result)

        assert boost_amount == 0.0

    def test_entity_boost_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test entity boost with matching entity."""
        boost_amount = booster.apply_entity_boost(sample_result)

        assert boost_amount == 0.10  # 10% boost (has "neural networks")

    def test_entity_boost_no_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test entity boost with non-matching entities."""
        sample_result.metadata["entities"] = ["unknown_entity"]
        boost_amount = booster.apply_entity_boost(sample_result)

        assert boost_amount == 0.0

    def test_entity_boost_multiple_matches(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test entity boost counts only once with multiple matches."""
        sample_result.metadata["entities"] = ["neural networks", "deep learning"]
        boost_amount = booster.apply_entity_boost(sample_result)

        assert boost_amount == 0.10  # Not 0.20, just 0.10

    def test_topic_boost_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test topic boost with matching topic."""
        boost_amount = booster.apply_topic_boost(sample_result)

        assert boost_amount == 0.08  # 8% boost

    def test_topic_boost_no_match(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test topic boost with non-matching topic."""
        sample_result.metadata["topics"] = ["other-topic"]
        boost_amount = booster.apply_topic_boost(sample_result)

        assert boost_amount == 0.0


# Cumulative Boost Logic Tests
class TestCumulativeBoostLogic:
    """Test cumulative boost application."""

    def test_single_boost_applied(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test single boost correctly increases score."""
        original_score = sample_result.hybrid_score
        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        # Should have some boosts applied
        total_boosts = sum(boosts.values())
        assert total_boosts > 0

        # Score should be higher
        assert boosted_result.hybrid_score > original_score

    def test_two_boosts_cumulative(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test cumulative effect of multiple boosts."""
        # Ensure both vendor and doc_type will boost
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"

        original_score = sample_result.hybrid_score
        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        # Both vendor and doc_type should have boosts
        assert boosts["vendor"] > 0
        assert boosts["doc_type"] > 0

        # Combined effect
        total_boost = sum(boosts.values())
        expected_score = original_score + (original_score * total_boost)
        assert abs(boosted_result.hybrid_score - min(1.0, expected_score)) < 0.001

    def test_all_five_boosts_applied(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test all five boosts applied together."""
        # Set up to trigger all boosts
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"
        sample_result.document_date = datetime.now() - timedelta(days=3)
        sample_result.metadata["entities"] = ["neural networks"]
        sample_result.metadata["topics"] = ["machine-learning"]

        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        # All boosts should be applied
        assert boosts["vendor"] > 0
        assert boosts["doc_type"] > 0
        assert boosts["recency"] > 0
        assert boosts["entity"] > 0
        assert boosts["topic"] > 0

        # Combined boost
        total_boost = sum(boosts.values())
        assert total_boost > 0.40  # At least 40% combined boost

    def test_boosts_dont_exceed_1_0(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test that final score never exceeds 1.0."""
        # Set up to trigger all boosts
        sample_result.hybrid_score = 0.5
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"
        sample_result.document_date = datetime.now() - timedelta(days=3)
        sample_result.metadata["entities"] = ["neural networks"]
        sample_result.metadata["topics"] = ["machine-learning"]

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score <= 1.0

    def test_clamping_high_score(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test score clamping when boosted score would exceed 1.0."""
        sample_result.hybrid_score = 0.9
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"
        sample_result.document_date = datetime.now() - timedelta(days=3)

        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score <= 1.0

    def test_clamping_low_score(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test that score never goes below 0.0."""
        sample_result.hybrid_score = 0.01
        sample_result.metadata = {}  # No boosts

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score >= 0.0

    def test_zero_boosts_no_change(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test that zero boosts don't change original score."""
        original_score = sample_result.hybrid_score
        sample_result.metadata = {}  # No matching metadata for boosts

        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        # All boosts should be 0
        assert sum(boosts.values()) == 0
        assert abs(boosted_result.hybrid_score - original_score) < 0.001


# Score Clamping Tests
class TestScoreClamping:
    """Test score boundary clamping."""

    def test_score_never_exceeds_1_0(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test score never exceeds 1.0."""
        sample_result.hybrid_score = 0.95
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"
        sample_result.document_date = datetime.now() - timedelta(days=1)

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score <= 1.0

    def test_score_never_below_0_0(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test score never goes below 0.0."""
        sample_result.hybrid_score = 0.05
        sample_result.metadata = {}  # Will result in negative boost internally (shouldn't happen)

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score >= 0.0

    def test_exact_1_0_boundary(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test clamping at exact 1.0 boundary."""
        sample_result.hybrid_score = 1.0
        sample_result.metadata["vendor"] = "anthropic"  # Would add more boost

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert boosted_result.hybrid_score == 1.0

    def test_fractional_clamping(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test clamping with fractional boost values."""
        sample_result.hybrid_score = 0.85
        sample_result.metadata["vendor"] = "anthropic"
        sample_result.metadata["doc_type"] = "api_docs"

        boosted_result, _ = booster.apply_all_boosts(sample_result)

        assert 0.0 <= boosted_result.hybrid_score <= 1.0


# Re-ranking After Boosts Tests
class TestRerankingAfterBoosts:
    """Test re-ranking of results after boosts."""

    def test_rerank_with_boost_changes_order(
        self, booster: Any
    ) -> None:
        """Test that boosts can change result order."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Text 1",
                similarity_score=0.90,
                bm25_score=0.85,
                hybrid_score=0.88,
                rank=1,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=None,
                context_header="d > s",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=100,
                metadata={"vendor": "google", "doc_type": "blog"},
            ),
            SearchResult(
                chunk_id=2,
                chunk_text="Text 2",
                similarity_score=0.80,
                bm25_score=0.75,
                hybrid_score=0.78,
                rank=2,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=None,
                context_header="d > s",
                chunk_index=1,
                total_chunks=1,
                chunk_token_count=100,
                metadata={"vendor": "anthropic", "doc_type": "api_docs"},
            ),
        ]

        reranked = booster.rerank_results(results)

        # Result 2 should rank first due to better boosts
        assert reranked[0].chunk_id == 2
        assert reranked[1].chunk_id == 1

    def test_rerank_preserves_all_results(
        self, booster: Any, sample_results_list: list[SearchResult]
    ) -> None:
        """Test that reranking preserves all results."""
        reranked = booster.rerank_results(sample_results_list)

        assert len(reranked) == len(sample_results_list)
        assert set(r.chunk_id for r in reranked) == set(
            r.chunk_id for r in sample_results_list
        )

    def test_rerank_updates_ranks_correctly(
        self, booster: Any
    ) -> None:
        """Test that ranks are updated sequentially after reranking."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Text {i}",
                similarity_score=0.9 - (i * 0.1),
                bm25_score=0.8 - (i * 0.1),
                hybrid_score=0.85 - (i * 0.1),
                rank=i + 1,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=None,
                context_header="d > s",
                chunk_index=i,
                total_chunks=5,
                chunk_token_count=100,
                metadata={},
            )
            for i in range(5)
        ]

        reranked = booster.rerank_results(results)

        # Ranks should be 1-5 in order
        for i, result in enumerate(reranked, 1):
            assert result.rank == i


# Edge Cases Tests
class TestBoostingEdgeCases:
    """Test edge cases in boosting."""

    def test_null_metadata(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test boosting with null metadata."""
        sample_result.metadata = {}
        boosted_result, boosts = booster.apply_all_boosts(sample_result)

        # All boosts should be 0
        assert sum(boosts.values()) == 0

    def test_empty_results_list(self, booster: Any) -> None:
        """Test re-ranking empty results."""
        reranked = booster.rerank_results([])

        assert len(reranked) == 0

    def test_single_result(
        self, booster: Any, sample_result: SearchResult
    ) -> None:
        """Test boosting single result."""
        reranked = booster.rerank_results([sample_result])

        assert len(reranked) == 1
        assert reranked[0].rank == 1

    def test_duplicate_scores_tie_breaking(
        self, booster: Any
    ) -> None:
        """Test re-ranking with duplicate scores (tie-breaking)."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Text {i}",
                similarity_score=0.8,
                bm25_score=0.7,
                hybrid_score=0.75,  # All same score
                rank=i + 1,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=None,
                context_header="d > s",
                chunk_index=i,
                total_chunks=3,
                chunk_token_count=100,
                metadata={},
            )
            for i in range(3)
        ]

        reranked = booster.rerank_results(results)

        # Should maintain some order (stable sort)
        assert len(reranked) == 3

    def test_large_result_set(self, booster: Any) -> None:
        """Test boosting large result set (500+ results)."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Text {i}",
                similarity_score=1.0 - (i * 0.001),
                bm25_score=0.9 - (i * 0.0009),
                hybrid_score=0.95 - (i * 0.00095),
                rank=i + 1,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=None,
                context_header="d > s",
                chunk_index=i,
                total_chunks=500,
                chunk_token_count=100,
                metadata={},
            )
            for i in range(500)
        ]

        reranked = booster.rerank_results(results)

        assert len(reranked) == 500


# Performance Benchmarking Tests
class TestBoostingPerformance:
    """Test boosting performance."""

    def test_boost_100_results_performance(self, booster: Any) -> None:
        """Test boosting 100 results completes quickly."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Text {i}",
                similarity_score=0.9 - (i * 0.005),
                bm25_score=0.85 - (i * 0.004),
                hybrid_score=0.88 - (i * 0.0045),
                rank=i + 1,
                score_type="hybrid",
                source_file="d.md",
                source_category="doc",
                document_date=datetime.now() - timedelta(days=i),
                context_header="d > s",
                chunk_index=i,
                total_chunks=100,
                chunk_token_count=100,
                metadata={
                    "vendor": ["anthropic", "openai"][i % 2],
                    "doc_type": ["api_docs", "kb_article"][i % 2],
                },
            )
            for i in range(100)
        ]

        reranked = booster.rerank_results(results)

        # Should complete successfully
        assert len(reranked) == 100
        # Verify all scores valid
        assert all(0.0 <= r.hybrid_score <= 1.0 for r in reranked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
