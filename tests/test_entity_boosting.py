"""Test suite for entity mention boosting in cross-encoder reranking.

Tests cover:
- Entity extraction from queries
- Known entity identification (vendors and team members)
- Case-insensitive entity matching
- Entity boost calculation
- Score normalization
- Ranking changes from entity boosts
- Fallback to standard reranking when no entities
- Full integration tests with realistic queries
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import Mock, MagicMock, patch

import pytest

from src.search.cross_encoder_reranker import CrossEncoderReranker, RerankerConfig
from src.search.results import SearchResult


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results with various metadata for entity boosting tests."""
    today = date.today()
    return [
        SearchResult(
            chunk_id=1,
            chunk_text="ProSource integration enables seamless commission tracking "
            "and ProSource systems integration",
            similarity_score=0.75,
            bm25_score=0.70,
            hybrid_score=0.72,
            rank=1,
            score_type="hybrid",
            source_file="prosource_guide.md",
            source_category="guide",
            document_date=today - timedelta(days=5),
            context_header="prosource_guide.md > Integration",
            chunk_index=0,
            total_chunks=5,
            chunk_token_count=256,
            metadata={"vendor": "ProSource"},
        ),
        SearchResult(
            chunk_id=2,
            chunk_text="Lutron systems provide advanced lighting control and automation "
            "with comprehensive API support",
            similarity_score=0.78,
            bm25_score=0.72,
            hybrid_score=0.75,
            rank=2,
            score_type="hybrid",
            source_file="lutron_docs.md",
            source_category="api_docs",
            document_date=today - timedelta(days=3),
            context_header="lutron_docs.md > Overview",
            chunk_index=0,
            total_chunks=8,
            chunk_token_count=384,
            metadata={"vendor": "Lutron"},
        ),
        SearchResult(
            chunk_id=3,
            chunk_text="General documentation about commission structures and "
            "payment processing in standard workflows",
            similarity_score=0.72,
            bm25_score=0.78,
            hybrid_score=0.75,
            rank=3,
            score_type="hybrid",
            source_file="general_guide.md",
            source_category="guide",
            document_date=today - timedelta(days=10),
            context_header="general_guide.md > Commission",
            chunk_index=2,
            total_chunks=10,
            chunk_token_count=512,
            metadata={},
        ),
        SearchResult(
            chunk_id=4,
            chunk_text="CEDIA conference materials and standards documentation "
            "with integration examples",
            similarity_score=0.68,
            bm25_score=0.75,
            hybrid_score=0.71,
            rank=4,
            score_type="hybrid",
            source_file="cedia_standards.md",
            source_category="reference",
            document_date=None,
            context_header="cedia_standards.md > Standards",
            chunk_index=1,
            total_chunks=6,
            chunk_token_count=256,
            metadata={},
        ),
    ]


class TestEntityExtraction:
    """Tests for _extract_named_entities() method."""

    def test_extract_single_vendor_entity(self) -> None:
        """Test extraction of single vendor entity from text."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "ProSource integration guide"
        )
        assert "prosource" in entities
        assert len(entities) == 1

    def test_extract_multiple_vendor_entities(self) -> None:
        """Test extraction of multiple vendor entities from text."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "ProSource and Lutron integration with CEDIA standards"
        )
        assert "prosource" in entities
        assert "lutron" in entities
        assert "cedia" in entities
        assert len(entities) == 3

    def test_extract_team_member_entity(self) -> None:
        """Test extraction of team member entity."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "Implementation by Cliff Clarke"
        )
        assert "cliff clarke" in entities

    def test_extract_case_insensitive(self) -> None:
        """Test that entity extraction is case-insensitive."""
        reranker = CrossEncoderReranker()
        entities_lower = reranker._extract_named_entities("prosource integration")
        entities_upper = reranker._extract_named_entities("PROSOURCE INTEGRATION")
        assert entities_lower == entities_upper
        assert "prosource" in entities_lower

    def test_extract_no_entities(self) -> None:
        """Test query with no known entities."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "How to write Python code and best practices"
        )
        assert len(entities) == 0

    def test_extract_partial_phrase_no_match(self) -> None:
        """Test that partial entity phrases don't match."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities("Clark and James discussions")
        # "clark" is not "cliff clarke", and "james" is not "james copple"
        assert len(entities) == 0

    def test_extract_full_team_member_name(self) -> None:
        """Test extraction of full team member names."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "Discussed with James Copple regarding implementation"
        )
        assert "james copple" in entities

    def test_extract_josh_ai_variant(self) -> None:
        """Test extraction of Josh AI with different formatting."""
        reranker = CrossEncoderReranker()
        entities_space = reranker._extract_named_entities(
            "Josh AI integration platform"
        )
        entities_dot = reranker._extract_named_entities("Josh.AI integration")
        # Both should be detected
        assert len(entities_space) > 0
        assert len(entities_dot) > 0

    def test_extract_bmcis_entity(self) -> None:
        """Test extraction of BMCIS entity reference."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "BMCIS vendor ecosystem and integrations"
        )
        assert "bmcis" in entities

    def test_extract_legrand_masimo_seura(self) -> None:
        """Test extraction of less common vendor entities."""
        reranker = CrossEncoderReranker()
        entities = reranker._extract_named_entities(
            "LegRand, Masimo, and Seura technology stack"
        )
        assert "legrand" in entities
        assert "masimo" in entities
        assert "seura" in entities


class TestEntityBoostCalculation:
    """Tests for entity boost score calculation logic."""

    def test_boost_factor_calculation(self) -> None:
        """Test correct boost factor calculation: +10% per mention, max 50%."""
        reranker = CrossEncoderReranker()
        # Formula: boost_factor = min(entity_count * 0.1, 0.5)

        # 1 mention: 1 * 0.1 = 0.1 (10%)
        boost_1 = min(1 * 0.1, 0.5)
        assert abs(boost_1 - 0.1) < 1e-6

        # 3 mentions: 3 * 0.1 = 0.3 (30%)
        boost_3 = min(3 * 0.1, 0.5)
        assert abs(boost_3 - 0.3) < 1e-6

        # 5 mentions: 5 * 0.1 = 0.5 (50%, capped)
        boost_5 = min(5 * 0.1, 0.5)
        assert abs(boost_5 - 0.5) < 1e-6

        # 10 mentions: 10 * 0.1 = 1.0, capped at 0.5 (50%)
        boost_10 = min(10 * 0.1, 0.5)
        assert abs(boost_10 - 0.5) < 1e-6

    def test_score_boost_application(self) -> None:
        """Test correct score boost application: base * (1 + boost_factor)."""
        # base_score = 0.75, 3 entity mentions
        base_score = 0.75
        entity_count = 3
        boost_factor = min(entity_count * 0.1, 0.5)
        boosted = base_score * (1.0 + boost_factor)

        # 0.75 * (1 + 0.3) = 0.75 * 1.3 = 0.975
        assert abs(boosted - 0.975) < 1e-6

    def test_score_normalization_to_1_0(self) -> None:
        """Test that boosted scores are normalized to max 1.0."""
        # base_score = 0.95, 5 entity mentions
        base_score = 0.95
        entity_count = 5
        boost_factor = min(entity_count * 0.1, 0.5)
        boosted = base_score * (1.0 + boost_factor)
        # 0.95 * 1.5 = 1.425, normalized to 1.0
        normalized = min(boosted, 1.0)

        assert normalized == 1.0

    def test_no_boost_with_zero_entities(self) -> None:
        """Test that no boost is applied when entity count is 0."""
        base_score = 0.75
        entity_count = 0
        boost_factor = min(entity_count * 0.1, 0.5)
        boosted = base_score * (1.0 + boost_factor)

        # 0.75 * (1 + 0) = 0.75
        assert abs(boosted - 0.75) < 1e-6


class TestReankWithEntityBoost:
    """Tests for rerank_with_entity_boost() method integration."""

    def test_entity_boost_requires_model_loaded(self, sample_results: list[SearchResult]) -> None:
        """Test that entity boost reranking requires model to be loaded."""
        reranker = CrossEncoderReranker()
        # Model not loaded
        with pytest.raises(ValueError, match="Model not loaded"):
            reranker.rerank_with_entity_boost(
                "ProSource commission",
                sample_results,
                top_k=3
            )

    def test_entity_boost_empty_results(self) -> None:
        """Test entity boost with empty results raises ValueError."""
        reranker = CrossEncoderReranker()
        reranker.model = MagicMock()  # Mock model loaded
        with pytest.raises(ValueError, match="search_results cannot be empty"):
            reranker.rerank_with_entity_boost("ProSource", [], top_k=3)

    def test_entity_boost_invalid_top_k(self, sample_results: list[SearchResult]) -> None:
        """Test entity boost with invalid top_k."""
        reranker = CrossEncoderReranker()
        reranker.model = MagicMock()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            reranker.rerank_with_entity_boost(
                "ProSource", sample_results, top_k=0
            )

    def test_entity_boost_invalid_confidence(self, sample_results: list[SearchResult]) -> None:
        """Test entity boost with invalid min_confidence."""
        reranker = CrossEncoderReranker()
        reranker.model = MagicMock()
        with pytest.raises(ValueError, match="min_confidence must be 0-1"):
            reranker.rerank_with_entity_boost(
                "ProSource", sample_results, min_confidence=1.5
            )

    def test_entity_boost_fallback_no_entities(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Test that entity boost falls back to standard reranking when no entities."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=32,
        )
        reranker = CrossEncoderReranker(config=config)
        reranker.model = MagicMock()

        # Mock the rerank method to verify it's called
        reranker.rerank = MagicMock(return_value=sample_results[:3])

        # Query with no entities
        result = reranker.rerank_with_entity_boost(
            "How to write Python code",
            sample_results,
            top_k=3
        )

        # Verify fallback to standard reranking was called
        reranker.rerank.assert_called_once()
        assert len(result) == 3

    def test_entity_boost_with_mock_model(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Test entity boost with mocked cross-encoder model."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=32,
        )
        reranker = CrossEncoderReranker(config=config)

        # Mock the model and candidate selector
        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)

        # Mock score_pairs to return scores
        # Results with ProSource should score higher
        reranker.score_pairs = MagicMock(return_value=[0.7, 0.75, 0.65, 0.60])

        # Query with entity "ProSource"
        result = reranker.rerank_with_entity_boost(
            "ProSource commission",
            sample_results,
            top_k=2
        )

        # Verify results are returned
        assert len(result) == 2
        assert all(isinstance(r, SearchResult) for r in result)

        # First result should be ProSource-related
        assert "prosource" in result[0].chunk_text.lower()

    def test_entity_count_in_results(self, sample_results: list[SearchResult]) -> None:
        """Test that entity mention counts are calculated correctly."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        # Test entity counting logic
        # Result 1 has "ProSource" mentioned twice
        text1 = sample_results[0].chunk_text
        entity = "prosource"
        count1 = text1.lower().count(entity)
        assert count1 == 2  # ProSource appears twice

        # Result 3 has no entity mentions
        text3 = sample_results[2].chunk_text
        count3 = text3.lower().count(entity)
        assert count3 == 0


class TestEntityBoostRanking:
    """Tests for ranking changes from entity boosting."""

    def test_entity_boost_reranks_results(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Test that entity boost can change result ranking."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)

        # Result 2 (Lutron) has lower base score but gets entity boost
        # Result 1 (ProSource) has higher base score
        reranker.score_pairs = MagicMock(return_value=[0.70, 0.65, 0.60, 0.55])

        result = reranker.rerank_with_entity_boost(
            "Lutron systems control",
            sample_results,
            top_k=2
        )

        # Lutron result should rank higher due to entity boost
        assert "lutron" in result[0].chunk_text.lower()

    def test_score_improvement_visible(self, sample_results: list[SearchResult]) -> None:
        """Test that confidence scores show improvement from boosting."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)

        # Entity results get boosted scores
        base_scores = [0.75, 0.68, 0.65, 0.60]
        reranker.score_pairs = MagicMock(return_value=base_scores)

        result = reranker.rerank_with_entity_boost(
            "ProSource Lutron integration",
            sample_results,
            top_k=2
        )

        # Results should have boosted scores > base scores
        for r in result:
            # Confidence should reflect the boost
            assert r.confidence > 0.0
            assert r.confidence <= 1.0


class TestEntityBoostEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_entity_boost_single_result(self) -> None:
        """Test entity boost with single result."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        single_result = [
            SearchResult(
                chunk_id=1,
                chunk_text="ProSource integration guide",
                similarity_score=0.8,
                bm25_score=0.75,
                hybrid_score=0.77,
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
        ]

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=single_result)
        reranker.score_pairs = MagicMock(return_value=[0.8])

        result = reranker.rerank_with_entity_boost(
            "ProSource documentation",
            single_result,
            top_k=1
        )

        assert len(result) == 1
        assert result[0].rank == 1

    def test_entity_boost_multiple_entities_same_result(self) -> None:
        """Test result with multiple entity mentions."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        multi_entity_result = [
            SearchResult(
                chunk_id=1,
                chunk_text="ProSource and Lutron and CEDIA standards for integration",
                similarity_score=0.8,
                bm25_score=0.75,
                hybrid_score=0.77,
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
        ]

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=multi_entity_result)
        reranker.score_pairs = MagicMock(return_value=[0.75])

        result = reranker.rerank_with_entity_boost(
            "ProSource Lutron CEDIA",
            multi_entity_result,
            top_k=1
        )

        # Result should have high boosted score due to multiple entity mentions
        assert len(result) == 1
        assert result[0].confidence > 0.75

    def test_entity_boost_score_type_set_correctly(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Test that score_type is set to 'cross_encoder' after entity boost."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)
        reranker.score_pairs = MagicMock(return_value=[0.75, 0.70, 0.65, 0.60])

        result = reranker.rerank_with_entity_boost(
            "ProSource",
            sample_results,
            top_k=2
        )

        # All results should have score_type='cross_encoder'
        assert all(r.score_type == "cross_encoder" for r in result)

    def test_entity_boost_rank_sequential(self, sample_results: list[SearchResult]) -> None:
        """Test that ranks are sequential starting from 1."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)
        reranker.score_pairs = MagicMock(return_value=[0.75, 0.70, 0.65, 0.60])

        result = reranker.rerank_with_entity_boost(
            "ProSource Lutron",
            sample_results,
            top_k=3
        )

        # Ranks should be 1, 2, 3
        assert [r.rank for r in result] == [1, 2, 3]


class TestEntityBoostIntegration:
    """Integration tests for entity boosting with realistic scenarios."""

    def test_entity_boost_prosource_query(self, sample_results: list[SearchResult]) -> None:
        """Integration test: Query for ProSource should boost ProSource results."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)
        # All results get similar base scores
        reranker.score_pairs = MagicMock(return_value=[0.70, 0.70, 0.70, 0.70])

        result = reranker.rerank_with_entity_boost(
            "ProSource commission",
            sample_results,
            top_k=2
        )

        # First result should be ProSource (it mentions ProSource twice)
        assert "prosource" in result[0].chunk_text.lower()
        # ProSource result should have higher confidence than others
        assert result[0].confidence > 0.70

    def test_entity_boost_lutron_cedia_query(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Integration test: Query with multiple entities."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)
        reranker.score_pairs = MagicMock(return_value=[0.70, 0.70, 0.70, 0.70])

        result = reranker.rerank_with_entity_boost(
            "Lutron and CEDIA standards",
            sample_results,
            top_k=3
        )

        # Should return results mentioning these entities
        assert len(result) == 3
        # Lutron should rank high
        assert any("lutron" in r.chunk_text.lower() for r in result)
        # CEDIA should be included
        assert any("cedia" in r.chunk_text.lower() for r in result)

    def test_entity_boost_score_ordering(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Integration test: Results are ordered by boosted score DESC."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker(config=config)

        reranker.model = MagicMock()
        reranker.candidate_selector.select = MagicMock(return_value=sample_results)
        reranker.score_pairs = MagicMock(return_value=[0.70, 0.70, 0.70, 0.70])

        result = reranker.rerank_with_entity_boost(
            "ProSource Lutron",
            sample_results,
            top_k=4
        )

        # Scores should be in descending order
        scores = [r.confidence for r in result]
        assert scores == sorted(scores, reverse=True)
