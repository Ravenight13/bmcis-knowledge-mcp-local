"""Test suite for query routing system.

Tests cover:
- Query type analysis (semantic vs. keyword)
- Strategy selection logic
- Confidence scoring
- Complexity classification
- Edge cases and performance
"""

from __future__ import annotations

import pytest

from src.search.query_router import QueryRouter, RoutingDecision


class TestQueryRouterInit:
    """Tests for QueryRouter initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with no arguments."""
        router = QueryRouter()
        assert router is not None

    def test_initialization_with_settings(self) -> None:
        """Test initialization with settings."""
        router = QueryRouter(settings=None)
        assert router is not None


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self) -> None:
        """Test creating a routing decision."""
        decision = RoutingDecision(
            strategy="hybrid",
            confidence=0.8,
            reason="Test reason",
            keyword_score=0.5,
            complexity="moderate",
        )
        assert decision.strategy == "hybrid"
        assert decision.confidence == 0.8
        assert decision.keyword_score == 0.5

    def test_routing_decision_fields(self) -> None:
        """Test all routing decision fields."""
        decision = RoutingDecision(
            strategy="vector",
            confidence=0.95,
            reason="Semantic query detected",
            keyword_score=0.1,
            complexity="simple",
        )
        assert decision.strategy == "vector"
        assert 0.5 <= decision.confidence <= 1.0
        assert 0.0 <= decision.keyword_score <= 1.0
        assert decision.complexity in ["simple", "moderate", "complex"]


class TestSelectStrategy:
    """Tests for select_strategy method."""

    def test_semantic_query_selection(self) -> None:
        """Test selection of vector search for semantic query."""
        router = QueryRouter()
        decision = router.select_strategy("How do I authenticate users?")
        assert decision.strategy == "vector"
        assert decision.keyword_score < 0.3

    def test_keyword_query_selection(self) -> None:
        """Test selection of BM25 for keyword query."""
        router = QueryRouter()
        decision = router.select_strategy("API endpoint authentication JWT token")
        assert decision.strategy == "bm25"
        assert decision.keyword_score > 0.7

    def test_mixed_query_selection(self) -> None:
        """Test selection of hybrid for mixed query."""
        router = QueryRouter()
        decision = router.select_strategy("How to use the API authentication endpoint")
        assert decision.strategy == "hybrid"
        assert 0.3 <= decision.keyword_score <= 0.7

    def test_empty_query_handling(self) -> None:
        """Test handling of empty query."""
        router = QueryRouter()
        decision = router.select_strategy("")
        assert decision.strategy == "hybrid"
        assert decision.confidence >= 0.5

    def test_whitespace_only_query(self) -> None:
        """Test handling of whitespace-only query."""
        router = QueryRouter()
        decision = router.select_strategy("   ")
        assert decision.strategy == "hybrid"

    def test_confidence_field_valid(self) -> None:
        """Test that confidence is in valid range."""
        router = QueryRouter()
        decision = router.select_strategy("test query")
        assert 0.5 <= decision.confidence <= 1.0

    def test_keyword_score_field_valid(self) -> None:
        """Test that keyword_score is in valid range."""
        router = QueryRouter()
        decision = router.select_strategy("test query")
        assert 0.0 <= decision.keyword_score <= 1.0

    def test_available_strategies_filtering(self) -> None:
        """Test filtering to available strategies."""
        router = QueryRouter()
        # Force vector query but only offer bm25 and hybrid
        decision = router.select_strategy(
            "How do I do something?",
            available_strategies=["bm25", "hybrid"],
        )
        assert decision.strategy in ["bm25", "hybrid"]

    def test_unavailable_strategy_fallback(self) -> None:
        """Test fallback when selected strategy unavailable."""
        router = QueryRouter()
        # Force a keyword query but only offer vector and hybrid
        decision = router.select_strategy(
            "API endpoint documentation reference",
            available_strategies=["vector", "hybrid"],
        )
        assert decision.strategy in ["vector", "hybrid"]


class TestAnalyzeQueryType:
    """Tests for query type analysis."""

    def test_keyword_density_calculation(self) -> None:
        """Test calculation of keyword density."""
        router = QueryRouter()
        analysis = router._analyze_query_type("API endpoint authentication token")
        assert "keyword_density" in analysis
        assert 0 <= analysis["keyword_density"] <= 1.0

    def test_high_keyword_density(self) -> None:
        """Test query with high keyword density."""
        router = QueryRouter()
        analysis = router._analyze_query_type("API method parameter function class")
        assert analysis["keyword_density"] > 0.5

    def test_low_keyword_density(self) -> None:
        """Test query with low keyword density."""
        router = QueryRouter()
        analysis = router._analyze_query_type("How do I do this thing?")
        assert analysis["keyword_density"] < 0.3

    def test_question_word_detection(self) -> None:
        """Test detection of question words."""
        router = QueryRouter()
        analysis = router._analyze_query_type("How do I authenticate?")
        assert analysis["question_words"] > 0

    def test_operator_count(self) -> None:
        """Test counting of boolean operators."""
        router = QueryRouter()
        analysis = router._analyze_query_type("API and authentication and token")
        assert analysis["operator_count"] >= 2

    def test_quote_detection(self) -> None:
        """Test detection of quoted phrases."""
        router = QueryRouter()
        analysis = router._analyze_query_type('Search for "exact phrase" here')
        assert analysis["quote_count"] >= 1

    def test_entity_detection(self) -> None:
        """Test detection of capitalized entities."""
        router = QueryRouter()
        analysis = router._analyze_query_type("OpenAI Claude API")
        assert analysis["entity_count"] >= 2


class TestEstimateComplexity:
    """Tests for complexity estimation."""

    def test_simple_query(self) -> None:
        """Test classification of simple query."""
        router = QueryRouter()
        complexity = router._estimate_complexity("Help please")
        assert complexity == "simple"

    def test_moderate_query(self) -> None:
        """Test classification of moderate query."""
        router = QueryRouter()
        complexity = router._estimate_complexity("How do I use the API authentication?")
        assert complexity == "moderate"

    def test_complex_query(self) -> None:
        """Test classification of complex query."""
        router = QueryRouter()
        complexity = router._estimate_complexity(
            "Advanced API design patterns and optimization techniques for high-performance "
            "microservices architecture with authentication and caching strategies"
        )
        assert complexity == "complex"

    def test_complexity_with_operators(self) -> None:
        """Test that operators increase complexity."""
        router = QueryRouter()
        simple = router._estimate_complexity("API")
        complex_with_ops = router._estimate_complexity("API and authentication and token and deployment")
        # Complex version should be more complex
        complexity_map = {"simple": 0, "moderate": 1, "complex": 2}
        assert complexity_map[complex_with_ops] >= complexity_map[simple]

    def test_complexity_with_technical_terms(self) -> None:
        """Test that technical terms affect complexity."""
        router = QueryRouter()
        non_technical = router._estimate_complexity("How do I do something?")
        technical = router._estimate_complexity("How do I optimize my API endpoint with caching?")
        complexity_map = {"simple": 0, "moderate": 1, "complex": 2}
        # Technical should be at least as complex
        assert complexity_map[technical] >= complexity_map[non_technical]


class TestCalculateConfidence:
    """Tests for confidence calculation."""

    def test_high_confidence_for_clear_signals(self) -> None:
        """Test high confidence with clear keyword signal."""
        router = QueryRouter()
        analysis = {
            "keyword_density": 0.8,
            "semantic_score": 0.1,
            "quote_count": 0.0,
            "operator_count": 0.0,
            "entity_count": 0.0,
            "question_words": 0.0,
        }
        confidence = router._calculate_confidence(analysis)
        assert confidence > 0.8

    def test_moderate_confidence_for_mixed_signals(self) -> None:
        """Test moderate confidence with mixed signals."""
        router = QueryRouter()
        analysis = {
            "keyword_density": 0.5,
            "semantic_score": 0.3,
            "quote_count": 0.0,
            "operator_count": 0.0,
            "entity_count": 0.0,
            "question_words": 0.1,
        }
        confidence = router._calculate_confidence(analysis)
        assert 0.5 <= confidence <= 0.9

    def test_confidence_bounds(self) -> None:
        """Test that confidence is always in valid range."""
        router = QueryRouter()
        analysis = {
            "keyword_density": 0.5,
            "semantic_score": 0.5,
            "quote_count": 0.0,
            "operator_count": 0.0,
            "entity_count": 0.0,
            "question_words": 0.0,
        }
        confidence = router._calculate_confidence(analysis)
        assert 0.5 <= confidence <= 1.0

    def test_question_words_increase_confidence(self) -> None:
        """Test that question words increase semantic confidence."""
        router = QueryRouter()
        analysis_no_questions = {
            "keyword_density": 0.3,
            "semantic_score": 0.0,
            "quote_count": 0.0,
            "operator_count": 0.0,
            "entity_count": 0.0,
            "question_words": 0.0,
        }
        analysis_with_questions = {
            "keyword_density": 0.3,
            "semantic_score": 0.3,
            "quote_count": 0.0,
            "operator_count": 0.0,
            "entity_count": 0.0,
            "question_words": 0.3,
        }
        conf_no_q = router._calculate_confidence(analysis_no_questions)
        conf_with_q = router._calculate_confidence(analysis_with_questions)
        assert conf_with_q >= conf_no_q


class TestCountMethods:
    """Tests for counting methods."""

    def test_count_keywords(self) -> None:
        """Test keyword counting."""
        router = QueryRouter()
        count = router._count_keywords("API endpoint authentication token")
        assert count >= 3

    def test_count_operators(self) -> None:
        """Test operator counting."""
        router = QueryRouter()
        count = router._count_operators("API and authentication and deployment")
        assert count >= 2

    def test_count_operators_with_quotes(self) -> None:
        """Test operator counting with quoted phrases."""
        router = QueryRouter()
        count = router._count_operators('API "exact phrase" and authentication')
        assert count >= 1

    def test_count_entities(self) -> None:
        """Test entity counting."""
        router = QueryRouter()
        count = router._count_entities("OpenAI Claude Azure")
        assert count >= 3

    def test_count_entities_mixed_case(self) -> None:
        """Test entity counting with mixed case."""
        router = QueryRouter()
        count = router._count_entities("The OpenAI API and Google Cloud")
        assert count >= 3


class TestQueryExamples:
    """Tests with realistic query examples."""

    def test_api_documentation_query(self) -> None:
        """Test typical API documentation query."""
        router = QueryRouter()
        decision = router.select_strategy("OpenAI API authentication endpoint")
        assert decision.strategy == "bm25"
        assert decision.keyword_score > 0.6

    def test_conceptual_question(self) -> None:
        """Test conceptual question."""
        router = QueryRouter()
        decision = router.select_strategy("How do I improve API performance?")
        assert decision.strategy in ["vector", "hybrid"]

    def test_debugging_query(self) -> None:
        """Test debugging/troubleshooting query."""
        router = QueryRouter()
        decision = router.select_strategy("Why is my authentication token expired?")
        assert decision.strategy in ["vector", "hybrid"]

    def test_code_snippet_query(self) -> None:
        """Test query for code examples."""
        router = QueryRouter()
        decision = router.select_strategy("Python JWT authentication code example")
        assert decision.keyword_score > 0.5

    def test_general_help_query(self) -> None:
        """Test general help query."""
        router = QueryRouter()
        decision = router.select_strategy("Help with deployment")
        assert decision.strategy in ["vector", "hybrid"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_query(self) -> None:
        """Test very long query."""
        router = QueryRouter()
        long_query = " ".join(["word"] * 100)
        decision = router.select_strategy(long_query)
        assert decision.strategy in ["vector", "bm25", "hybrid"]
        assert 0.5 <= decision.confidence <= 1.0

    def test_special_characters(self) -> None:
        """Test query with special characters."""
        router = QueryRouter()
        decision = router.select_strategy("API@endpoint#123 query&test")
        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_numbers_only(self) -> None:
        """Test query with numbers only."""
        router = QueryRouter()
        decision = router.select_strategy("123 456 789")
        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_single_word_query(self) -> None:
        """Test single word query."""
        router = QueryRouter()
        decision = router.select_strategy("API")
        assert decision.strategy == "bm25"

    def test_single_question_word(self) -> None:
        """Test single question word."""
        router = QueryRouter()
        decision = router.select_strategy("How?")
        assert decision.strategy == "vector"

    def test_all_operators_query(self) -> None:
        """Test query with many operators."""
        router = QueryRouter()
        decision = router.select_strategy('API and authentication and "token" or deployment not "password"')
        assert decision.strategy in ["vector", "bm25", "hybrid"]
        # Should have high confidence due to clear operators
        assert decision.confidence >= 0.7
