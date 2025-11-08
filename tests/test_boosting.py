"""Test suite for multi-factor boosting system.

Tests cover:
- Individual boost factors (vendor, doc type, recency, entity, topic)
- Cumulative boost logic and score clamping
- Query analysis and context extraction
- Re-ranking after boosts applied
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from src.search.boosting import BoostingSystem, BoostWeights, KNOWN_VENDORS
from src.search.results import SearchResult


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results with various metadata."""
    today = date.today()
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="This is about OpenAI GPT models and API design patterns",
            similarity_score=0.85,
            bm25_score=0.70,
            hybrid_score=0.78,
            rank=1,
            score_type="hybrid",
            source_file="doc1.md",
            source_category="api_docs",
            document_date=today - timedelta(days=5),
            context_header="doc1.md > Section A",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
            metadata={"vendor": "OpenAI", "tags": ["api", "auth"]},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Anthropic Claude documentation and authentication guide",
            similarity_score=0.80,
            bm25_score=0.75,
            hybrid_score=0.78,
            rank=2,
            score_type="hybrid",
            source_file="doc2.md",
            source_category="guide",
            document_date=today - timedelta(days=45),
            context_header="doc2.md > Section B",
            chunk_index=1,
            total_chunks=3,
            chunk_token_count=384,
            metadata={"vendor": "Anthropic"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="Google Cloud Platform deployment and infrastructure setup",
            similarity_score=0.75,
            bm25_score=0.80,
            hybrid_score=0.77,
            rank=3,
            score_type="hybrid",
            source_file="doc3.md",
            source_category="kb_article",
            document_date=None,
            context_header="doc3.md > Section C",
            chunk_index=0,
            total_chunks=8,
            chunk_token_count=512,
            metadata={},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="AWS S3 data storage and optimization techniques",
            similarity_score=0.70,
            bm25_score=0.85,
            hybrid_score=0.76,
            rank=4,
            score_type="hybrid",
            source_file="doc4.md",
            source_category="reference",
            document_date=today - timedelta(days=3),
            context_header="doc4.md > Section D",
            chunk_index=2,
            total_chunks=6,
            chunk_token_count=256,
            metadata={"vendor": "AWS"},
        ),
    ]


class TestBoostingSystemInit:
    """Tests for BoostingSystem initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with no arguments."""
        system = BoostingSystem()
        assert system is not None

    def test_initialization_with_db_pool(self) -> None:
        """Test initialization with database pool."""
        system = BoostingSystem(db_pool=None)
        assert system is not None


class TestBoostWeightsDefaults:
    """Tests for BoostWeights default configuration."""

    def test_default_weights(self) -> None:
        """Test default boost weight values."""
        weights = BoostWeights()
        assert weights.vendor == 0.15
        assert weights.doc_type == 0.10
        assert weights.recency == 0.05
        assert weights.entity == 0.10
        assert weights.topic == 0.08

    def test_custom_weights(self) -> None:
        """Test custom boost weight configuration."""
        weights = BoostWeights(
            vendor=0.20,
            doc_type=0.15,
            recency=0.10,
            entity=0.05,
            topic=0.05,
        )
        assert weights.vendor == 0.20
        assert weights.doc_type == 0.15


class TestVendorExtraction:
    """Tests for vendor name extraction from queries."""

    def test_extract_single_vendor(self) -> None:
        """Test extraction of single vendor name."""
        system = BoostingSystem()
        vendors = system._extract_vendors("OpenAI API documentation")
        assert "openai" in vendors

    def test_extract_multiple_vendors(self) -> None:
        """Test extraction of multiple vendor names."""
        system = BoostingSystem()
        vendors = system._extract_vendors("Compare OpenAI, Google, and AWS APIs")
        assert "openai" in vendors
        assert "google" in vendors
        assert "aws" in vendors

    def test_extract_no_vendors(self) -> None:
        """Test query with no known vendors."""
        system = BoostingSystem()
        vendors = system._extract_vendors("How to write Python code")
        assert len(vendors) == 0

    def test_case_insensitive_vendor_extraction(self) -> None:
        """Test that vendor extraction is case-insensitive."""
        system = BoostingSystem()
        vendors_lower = system._extract_vendors("openai documentation")
        vendors_upper = system._extract_vendors("OPENAI DOCUMENTATION")
        assert "openai" in vendors_lower
        assert "openai" in vendors_upper


class TestDocTypeDetection:
    """Tests for document type detection from queries."""

    def test_detect_api_docs(self) -> None:
        """Test detection of API documentation intent."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("API documentation for endpoints")
        assert doc_type == "api_docs"

    def test_detect_guide(self) -> None:
        """Test detection of guide/tutorial intent."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("Getting started guide")
        assert doc_type == "guide"

    def test_detect_kb_article(self) -> None:
        """Test detection of KB article intent."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("Troubleshooting FAQ")
        assert doc_type == "kb_article"

    def test_detect_code_sample(self) -> None:
        """Test detection of code sample intent."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("Code snippet implementation")
        assert doc_type == "code_sample"

    def test_detect_reference(self) -> None:
        """Test detection of reference intent."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("Schema reference specification")
        assert doc_type == "reference"

    def test_detect_no_type(self) -> None:
        """Test query with no clear document type."""
        system = BoostingSystem()
        doc_type = system._detect_doc_type("random query text")
        assert doc_type == ""


class TestRecencyBoost:
    """Tests for recency boost calculation."""

    def test_very_recent_boost(self) -> None:
        """Test boost for very recent document (< 7 days)."""
        system = BoostingSystem()
        today = date.today()
        recent_date = today - timedelta(days=3)
        boost = system._calculate_recency_boost(recent_date)
        assert boost == 1.0

    def test_moderate_recency_boost(self) -> None:
        """Test boost for moderately recent document (7-30 days)."""
        system = BoostingSystem()
        today = date.today()
        moderate_date = today - timedelta(days=15)
        boost = system._calculate_recency_boost(moderate_date)
        assert boost == 0.7

    def test_old_document_no_boost(self) -> None:
        """Test that old document (> 30 days) gets no boost."""
        system = BoostingSystem()
        today = date.today()
        old_date = today - timedelta(days=60)
        boost = system._calculate_recency_boost(old_date)
        assert boost == 0.0

    def test_no_date_no_boost(self) -> None:
        """Test that None date produces no boost."""
        system = BoostingSystem()
        boost = system._calculate_recency_boost(None)
        assert boost == 0.0

    def test_datetime_object_handling(self) -> None:
        """Test handling of datetime objects."""
        system = BoostingSystem()
        today = datetime.now()
        recent_dt = today - timedelta(days=2)
        boost = system._calculate_recency_boost(recent_dt)
        assert boost == 1.0

    def test_future_date_handling(self) -> None:
        """Test handling of future dates."""
        system = BoostingSystem()
        today = date.today()
        future_date = today + timedelta(days=10)
        boost = system._calculate_recency_boost(future_date)
        assert boost == 1.0


class TestTopicDetection:
    """Tests for topic detection from queries."""

    def test_detect_authentication_topic(self) -> None:
        """Test detection of authentication topic."""
        system = BoostingSystem()
        topic = system._detect_topic("How to use JWT authentication")
        assert topic == "authentication"

    def test_detect_api_design_topic(self) -> None:
        """Test detection of API design topic."""
        system = BoostingSystem()
        topic = system._detect_topic("REST API design patterns")
        assert topic == "api_design"

    def test_detect_deployment_topic(self) -> None:
        """Test detection of deployment topic."""
        system = BoostingSystem()
        topic = system._detect_topic("Deploy to Kubernetes")
        assert topic == "deployment"

    def test_detect_optimization_topic(self) -> None:
        """Test detection of optimization topic."""
        system = BoostingSystem()
        topic = system._detect_topic("Performance optimization tips")
        assert topic == "optimization"

    def test_detect_no_topic(self) -> None:
        """Test query with no clear topic."""
        system = BoostingSystem()
        topic = system._detect_topic("random text without meaning")
        assert topic == ""


class TestEntityExtraction:
    """Tests for entity extraction and matching."""

    def test_extract_capitalized_entities(self, sample_results: list[SearchResult]) -> None:
        """Test extraction of capitalized entities."""
        system = BoostingSystem()
        entities = system._extract_entities("OpenAI API", sample_results)
        # Should find some entities
        assert isinstance(entities, dict)

    def test_entity_matching_in_results(self, sample_results: list[SearchResult]) -> None:
        """Test matching of entities to results."""
        system = BoostingSystem()
        entities = system._extract_entities("OpenAI GPT", sample_results)
        # First result mentions OpenAI and GPT
        if 0 in entities:
            assert len(entities[0]) > 0


class TestScoreBoosting:
    """Tests for score boosting and clamping."""

    def test_no_boost(self) -> None:
        """Test score with no boost applied."""
        system = BoostingSystem()
        boosted = system._boost_score(0.75, 0.0)
        assert abs(boosted - 0.75) < 1e-6

    def test_single_boost(self) -> None:
        """Test score with single boost factor."""
        system = BoostingSystem()
        # 0.5 * (1 + 0.1) = 0.55
        boosted = system._boost_score(0.5, 0.1)
        assert abs(boosted - 0.55) < 1e-6

    def test_multiple_boosts(self) -> None:
        """Test score with multiple cumulative boosts."""
        system = BoostingSystem()
        # 0.6 * (1 + 0.30) = 0.78
        boosted = system._boost_score(0.6, 0.30)
        assert abs(boosted - 0.78) < 1e-6

    def test_boost_clamping_upper(self) -> None:
        """Test that boosted score doesn't exceed 1.0."""
        system = BoostingSystem()
        boosted = system._boost_score(0.9, 0.5)  # Would be 1.35, clamped to 1.0
        assert boosted == 1.0

    def test_boost_clamping_lower(self) -> None:
        """Test that boosted score doesn't go below 0.0."""
        system = BoostingSystem()
        boosted = system._boost_score(0.0, 1.0)
        assert boosted == 0.0


class TestMetadataExtraction:
    """Tests for metadata extraction from results."""

    def test_extract_vendor_from_metadata(self) -> None:
        """Test extraction of vendor from result metadata."""
        system = BoostingSystem()
        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.7,
            hybrid_score=0.75,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="guide",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=128,
            metadata={"vendor": "OpenAI"},
        )
        vendor = system._get_vendor_from_metadata(result)
        assert vendor == "OpenAI"

    def test_extract_doc_type_from_category(self) -> None:
        """Test extraction of document type from source_category."""
        system = BoostingSystem()
        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.7,
            hybrid_score=0.75,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="api_docs",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=128,
        )
        doc_type = system._get_doc_type_from_result(result)
        assert doc_type == "api_docs"

    def test_missing_vendor_returns_none(self) -> None:
        """Test that missing vendor returns None."""
        system = BoostingSystem()
        result = SearchResult(
            chunk_id=1,
            chunk_text="Test",
            similarity_score=0.8,
            bm25_score=0.7,
            hybrid_score=0.75,
            rank=1,
            score_type="hybrid",
            source_file="doc.md",
            source_category="guide",
            document_date=None,
            context_header="doc.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=128,
            metadata={},
        )
        vendor = system._get_vendor_from_metadata(result)
        assert vendor is None


class TestApplyBoosts:
    """Tests for full boost application workflow."""

    def test_apply_boosts_empty_results(self) -> None:
        """Test applying boosts to empty result list."""
        system = BoostingSystem()
        result = system.apply_boosts([], "test query")
        assert result == []

    def test_apply_boosts_preserves_count(self, sample_results: list[SearchResult]) -> None:
        """Test that boost application preserves result count."""
        system = BoostingSystem()
        boosted = system.apply_boosts(sample_results, "OpenAI API")
        assert len(boosted) == len(sample_results)

    def test_apply_boosts_reranks_results(self, sample_results: list[SearchResult]) -> None:
        """Test that boosts affect ranking."""
        system = BoostingSystem()
        boosted = system.apply_boosts(sample_results, "OpenAI API documentation")

        # Verify results are reranked (scores may differ)
        boosted_scores = [r.hybrid_score for r in boosted]
        assert boosted_scores == sorted(boosted_scores, reverse=True)

    def test_apply_boosts_updates_scores(self, sample_results: list[SearchResult]) -> None:
        """Test that boost application updates scores."""
        system = BoostingSystem()
        original_scores = [r.hybrid_score for r in sample_results]
        boosted = system.apply_boosts(sample_results, "OpenAI recent API guide")

        boosted_scores = [r.hybrid_score for r in boosted]
        # Some scores should be different (boosted)
        assert original_scores != boosted_scores

    def test_apply_boosts_score_type(self, sample_results: list[SearchResult]) -> None:
        """Test that score type is set to 'hybrid' after boosting."""
        system = BoostingSystem()
        boosted = system.apply_boosts(sample_results, "test query")
        assert all(r.score_type == "hybrid" for r in boosted)

    def test_apply_boosts_vendor_matching(self) -> None:
        """Test vendor boost is applied correctly."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="OpenAI API documentation",
                similarity_score=0.8,
                bm25_score=0.7,
                hybrid_score=0.75,
                rank=1,
                score_type="hybrid",
                source_file="doc.md",
                source_category="guide",
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=128,
                metadata={"vendor": "OpenAI"},
            ),
        ]

        system = BoostingSystem()
        boosted = system.apply_boosts(results, "OpenAI documentation")

        # Score should be boosted (vendor match)
        assert boosted[0].hybrid_score > results[0].hybrid_score

    def test_apply_boosts_custom_weights(self, sample_results: list[SearchResult]) -> None:
        """Test applying boosts with custom weight configuration."""
        system = BoostingSystem()
        weights = BoostWeights(vendor=0.5, doc_type=0.0, recency=0.0, entity=0.0, topic=0.0)

        boosted = system.apply_boosts(sample_results, "OpenAI", boosts=weights)

        assert len(boosted) == len(sample_results)
        assert all(r.score_type == "hybrid" for r in boosted)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_boost_zero_original_score(self) -> None:
        """Test boosting when original score is 0.0."""
        system = BoostingSystem()
        boosted = system._boost_score(0.0, 0.5)
        assert boosted == 0.0

    def test_boost_zero_boost_factor(self) -> None:
        """Test boosting when boost factor is 0.0."""
        system = BoostingSystem()
        boosted = system._boost_score(0.5, 0.0)
        assert abs(boosted - 0.5) < 1e-6

    def test_apply_boosts_with_null_metadata(self) -> None:
        """Test boost application with results missing metadata."""
        results = [
            SearchResult(
                chunk_id=1,
                chunk_text="Some text",
                similarity_score=0.8,
                bm25_score=0.7,
                hybrid_score=0.75,
                rank=1,
                score_type="hybrid",
                source_file="doc.md",
                source_category=None,
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=128,
                metadata=None,
            ),
        ]

        system = BoostingSystem()
        boosted = system.apply_boosts(results, "test query")

        assert len(boosted) == 1
        assert boosted[0].hybrid_score >= 0.0

    def test_apply_boosts_large_result_set(self) -> None:
        """Test boost application performance with many results."""
        results = [
            SearchResult(
                chunk_id=i,
                chunk_text=f"Result {i}",
                similarity_score=0.5 + (i * 0.001),
                bm25_score=0.5,
                hybrid_score=0.5 + (i * 0.001),
                rank=i + 1,  # rank must be >= 1
                score_type="hybrid",
                source_file="doc.md",
                source_category="guide",
                document_date=None,
                context_header="doc.md",
                chunk_index=0,
                total_chunks=1,
                chunk_token_count=128,
            )
            for i in range(100)
        ]

        system = BoostingSystem()
        boosted = system.apply_boosts(results, "test query")

        assert len(boosted) == 100
        assert all(0 <= r.hybrid_score <= 1.0 for r in boosted)
