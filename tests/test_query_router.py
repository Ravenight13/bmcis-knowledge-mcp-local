"""Test suite for query routing and semantic analysis.

Tests cover:
- Semantic query detection (conceptual, philosophical, abstract)
- Keyword query detection (technical, code, parameters)
- Hybrid query detection (mixed signals)
- Confidence scoring (high, medium, low)
- Query complexity classification (simple, moderate, complex)
- Routing decision validation
- Performance benchmarks

Routing strategies:
- Vector search: Semantic/conceptual queries (high confidence >0.7)
- BM25 search: Keyword/technical queries (high confidence >0.7)
- Hybrid search: Mixed queries (medium confidence 0.4-0.7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pytest


# Data Classes
@dataclass
class RoutingDecision:
    """Result of query routing analysis."""

    strategy: Literal["vector", "bm25", "hybrid"]
    confidence: float
    reason: str
    query_type: str
    complexity: Literal["simple", "moderate", "complex"]


# Query Router Implementation
@pytest.fixture
def query_router() -> Any:
    """Create query router instance."""
    class QueryRouter:
        """Route queries to appropriate search strategy."""

        def __init__(self) -> None:
            """Initialize router with keyword lists."""
            self.semantic_indicators = {
                "how_to": ["how to", "how do i", "how do you"],
                "what_is": ["what is", "what are", "definition of"],
                "why": ["why", "reason for", "purpose of"],
                "explain": ["explain", "describe", "clarify"],
                "understand": ["understand", "learn about", "know about"],
                "concepts": ["concept", "theory", "principle"],
                "abstract": ["abstract", "conceptual", "philosophical"],
            }

            self.keyword_indicators = {
                "technical": [
                    "postgresql", "pgvector", "hnsw", "index",
                    "async", "await", "error handling",
                    "timeout", "concurrency", "parameter",
                ],
                "code": [
                    "code", "function", "method", "class",
                    "implementation", "algorithm", "syntax",
                    "return", "exception", "debug",
                ],
                "boolean": ["and", "or", "not", "&&", "||", "!"],
                "quoted": ['"', "'"],  # Exact phrase matching
                "operators": [":", "=", "<", ">", "*"],
            }

            self.stop_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "is", "are", "was", "were"
            }

        def analyze_query(self, query: str) -> RoutingDecision:
            """Analyze query and determine routing strategy."""
            if not query or not query.strip():
                return RoutingDecision(
                    strategy="hybrid",
                    confidence=0.0,
                    reason="Empty query",
                    query_type="empty",
                    complexity="simple",
                )

            query_lower = query.lower()

            # Calculate semantic and keyword scores
            semantic_score = self._calculate_semantic_score(query_lower)
            keyword_score = self._calculate_keyword_score(query_lower)
            complexity = self._classify_complexity(query)

            # Determine strategy based on scores
            if abs(semantic_score - keyword_score) > 0.3:
                # Clear preference for one strategy
                if semantic_score > keyword_score:
                    return RoutingDecision(
                        strategy="vector",
                        confidence=min(1.0, semantic_score),
                        reason=f"Semantic signals detected: {self._get_semantic_reasons(query_lower)}",
                        query_type="semantic",
                        complexity=complexity,
                    )
                else:
                    return RoutingDecision(
                        strategy="bm25",
                        confidence=min(1.0, keyword_score),
                        reason=f"Keyword signals detected: {self._get_keyword_reasons(query_lower)}",
                        query_type="keyword",
                        complexity=complexity,
                    )
            else:
                # Mixed signals - use hybrid
                avg_score = (semantic_score + keyword_score) / 2
                return RoutingDecision(
                    strategy="hybrid",
                    confidence=min(1.0, avg_score),
                    reason="Mixed semantic and keyword signals",
                    query_type="mixed",
                    complexity=complexity,
                )

        def _calculate_semantic_score(self, query: str) -> float:
            """Calculate semantic signal strength (0-1)."""
            score = 0.0
            word_count = len(query.split())

            # Check for semantic question patterns
            if any(query.startswith(pattern)
                   for pattern in self.semantic_indicators["how_to"]):
                score += 0.3

            if any(query.startswith(pattern)
                   for pattern in self.semantic_indicators["what_is"]):
                score += 0.3

            if query.startswith("why ") or " why " in query:
                score += 0.25

            if any(word in query
                   for word in self.semantic_indicators["explain"]):
                score += 0.2

            if any(word in query
                   for word in self.semantic_indicators["understand"]):
                score += 0.15

            # Bonus for longer, more descriptive queries
            if word_count > 5:
                score += 0.1

            # Natural language patterns
            if "?" in query:
                score += 0.15

            return min(1.0, score)

        def _calculate_keyword_score(self, query: str) -> float:
            """Calculate keyword signal strength (0-1)."""
            score = 0.0

            # Count technical terms
            tech_count = sum(
                1 for term in self.keyword_indicators["technical"]
                if term in query
            )
            score += min(0.4, tech_count * 0.1)

            # Check for code patterns
            if any(word in query for word in self.keyword_indicators["code"]):
                score += 0.2

            # Check for boolean operators
            if any(op in query for op in self.keyword_indicators["boolean"]):
                score += 0.15

            # Check for quoted phrases
            if '"' in query or "'" in query:
                score += 0.2

            # Multiple keywords (not separated by common words)
            words = [w for w in query.split()
                     if w.lower() not in self.stop_words]
            if len(words) >= 3:
                score += 0.1

            return min(1.0, score)

        def _classify_complexity(
            self, query: str
        ) -> Literal["simple", "moderate", "complex"]:
            """Classify query complexity."""
            words = query.split()
            word_count = len(words)
            clause_count = query.count(",") + query.count(";")

            if word_count <= 3 and clause_count == 0:
                return "simple"
            elif word_count <= 15 and clause_count <= 1:
                return "moderate"
            else:
                return "complex"

        def _get_semantic_reasons(self, query: str) -> str:
            """Get reasons for semantic classification."""
            reasons = []

            if query.startswith("how "):
                reasons.append("how-to question")
            if query.startswith("what "):
                reasons.append("definition question")
            if "why " in query:
                reasons.append("why question")
            if "explain" in query:
                reasons.append("explanation request")

            return ", ".join(reasons) if reasons else "semantic patterns"

        def _get_keyword_reasons(self, query: str) -> str:
            """Get reasons for keyword classification."""
            reasons = []

            for term in self.keyword_indicators["technical"]:
                if term in query:
                    reasons.append(f"{term}")
                    break

            if any(op in query for op in self.keyword_indicators["boolean"]):
                reasons.append("boolean operators")

            if '"' in query:
                reasons.append("exact phrase")

            return ", ".join(reasons[:2]) if reasons else "keyword patterns"

    return QueryRouter()


# Semantic Query Detection Tests
class TestSemanticQueryDetection:
    """Test detection of semantic queries."""

    def test_how_to_question_semantic(self, query_router: Any) -> None:
        """Test 'how to' question routes to vector search."""
        query = "How to implement REST API design?"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "vector"
        assert decision.confidence > 0.7
        assert decision.query_type == "semantic"

    def test_how_do_question_semantic(self, query_router: Any) -> None:
        """Test 'how do I' question routes to vector search."""
        query = "How do I implement authentication?"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "vector"
        assert decision.confidence > 0.6
        assert decision.query_type == "semantic"

    def test_what_is_question_semantic(self, query_router: Any) -> None:
        """Test 'what is' question routes to vector search."""
        query = "What is machine learning?"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "vector"
        assert decision.confidence > 0.6

    def test_why_question_semantic(self, query_router: Any) -> None:
        """Test 'why' question routes to vector search."""
        query = "Why use microservices architecture?"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "vector"
        assert decision.query_type == "semantic"

    def test_conceptual_question_semantic(self, query_router: Any) -> None:
        """Test conceptual questions route to vector search."""
        query = "Explain the concept of containerization"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "vector"

    def test_philosophical_question_semantic(self, query_router: Any) -> None:
        """Test philosophical questions route to vector search."""
        query = "Why is code maintainability important?"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]

    def test_natural_language_question(self, query_router: Any) -> None:
        """Test natural language questions."""
        query = "What are the best practices for error handling?"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]

    def test_abstract_concept_query(self, query_router: Any) -> None:
        """Test abstract concept queries."""
        query = "Tell me about API design principles"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]

    def test_semantic_question_mark(self, query_router: Any) -> None:
        """Test that question mark increases semantic score."""
        query_with_mark = "What is machine learning?"
        query_without_mark = "What is machine learning"

        decision_with = query_router.analyze_query(query_with_mark)
        decision_without = query_router.analyze_query(query_without_mark)

        assert decision_with.confidence >= decision_without.confidence


# Keyword Query Detection Tests
class TestKeywordQueryDetection:
    """Test detection of keyword queries."""

    def test_technical_keyword_query(self, query_router: Any) -> None:
        """Test technical keyword query routes to BM25."""
        query = "PostgreSQL pgvector HNSW index"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"
        assert decision.query_type == "keyword"

    def test_code_keyword_query(self, query_router: Any) -> None:
        """Test code-related keyword query."""
        query = "async/await error handling implementation"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"
        assert decision.query_type == "keyword"

    def test_api_parameter_query(self, query_router: Any) -> None:
        """Test API parameter query."""
        query = "max_results timeout concurrency"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"

    def test_boolean_operator_query(self, query_router: Any) -> None:
        """Test query with boolean operators."""
        query = "authentication AND authorization OR RBAC"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"
        assert "boolean" in decision.reason.lower()

    def test_quoted_phrase_query(self, query_router: Any) -> None:
        """Test quoted phrase query."""
        query = '"exact phrase match" OR "another phrase"'
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"

    def test_multiple_keywords_query(self, query_router: Any) -> None:
        """Test multiple unrelated keywords."""
        query = "REST API JSON database schema"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"

    def test_technical_jargon_query(self, query_router: Any) -> None:
        """Test technical jargon."""
        query = "HNSW pgvector cosine similarity"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"


# Hybrid Query Detection Tests
class TestHybridQueryDetection:
    """Test detection of hybrid queries."""

    def test_mixed_semantic_keyword(self, query_router: Any) -> None:
        """Test query with both semantic and keyword signals."""
        query = "How to implement REST API with PostgreSQL?"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]
        assert 0.4 <= decision.confidence <= 0.9

    def test_balanced_mixed_query(self, query_router: Any) -> None:
        """Test balanced semantic/keyword query."""
        query = "Explain how to use pgvector for semantic search"
        decision = query_router.analyze_query(query)

        assert decision.query_type in ["semantic", "mixed"]

    def test_keyword_dominant_conceptual(self, query_router: Any) -> None:
        """Test keyword-dominant but conceptual query."""
        query = "PostgreSQL error handling best practices"
        decision = query_router.analyze_query(query)

        # Mixed signals
        assert decision.confidence > 0.0

    def test_semantic_dominant_keyword(self, query_router: Any) -> None:
        """Test semantic-dominant with keyword."""
        query = "What is the best way to structure an API?"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]

    def test_ambiguous_query(self, query_router: Any) -> None:
        """Test ambiguous query."""
        query = "Search engine architecture implementation"
        decision = query_router.analyze_query(query)

        assert decision.confidence > 0.0

    def test_complex_multiclause_query(self, query_router: Any) -> None:
        """Test complex multi-clause query."""
        query = "How to implement OAuth 2.0 authentication with JWT tokens for REST API"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "hybrid"]
        assert decision.complexity == "complex"


# Confidence Scoring Tests
class TestConfidenceScoring:
    """Test confidence score calculation."""

    def test_high_confidence_semantic(self, query_router: Any) -> None:
        """Test high confidence for clear semantic query."""
        query = "How do I implement user authentication?"
        decision = query_router.analyze_query(query)

        assert decision.confidence > 0.7

    def test_medium_confidence_mixed(self, query_router: Any) -> None:
        """Test medium confidence for mixed signals."""
        query = "PostgreSQL authentication implementation"
        decision = query_router.analyze_query(query)

        assert 0.4 <= decision.confidence <= 0.8

    def test_low_confidence_ambiguous(self, query_router: Any) -> None:
        """Test low confidence for ambiguous query."""
        query = "test"
        decision = query_router.analyze_query(query)

        assert decision.confidence < 0.5

    def test_confidence_correlates_with_clarity(
        self, query_router: Any
    ) -> None:
        """Test confidence correlates with query clarity."""
        clear_query = "How to implement REST API with proper error handling?"
        ambiguous_query = "API implementation"

        clear_decision = query_router.analyze_query(clear_query)
        ambiguous_decision = query_router.analyze_query(ambiguous_query)

        assert clear_decision.confidence > ambiguous_decision.confidence


# Query Complexity Classification Tests
class TestQueryComplexityClassification:
    """Test query complexity classification."""

    def test_simple_query_classification(self, query_router: Any) -> None:
        """Test simple query classification."""
        query = "authentication"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "simple"

    def test_moderate_query_classification(self, query_router: Any) -> None:
        """Test moderate query classification."""
        query = "implement user authentication securely"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "moderate"

    def test_complex_query_classification(self, query_router: Any) -> None:
        """Test complex query classification."""
        query = "How to implement OAuth 2.0 authentication with JWT tokens for REST API"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "complex"

    def test_simple_single_word(self, query_router: Any) -> None:
        """Test single word query."""
        query = "REST"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "simple"

    def test_moderate_few_words(self, query_router: Any) -> None:
        """Test query with few words."""
        query = "API authentication patterns"
        decision = query_router.analyze_query(query)

        assert decision.complexity in ["simple", "moderate"]

    def test_complex_many_clauses(self, query_router: Any) -> None:
        """Test query with many clauses."""
        query = "Implement authentication; handle errors; validate tokens; manage sessions"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "complex"


# Routing Decision Validation Tests
class TestRoutingDecisionValidation:
    """Test validity of routing decisions."""

    def test_routing_strategy_valid(self, query_router: Any) -> None:
        """Test that routing strategy is valid."""
        query = "How to implement REST API?"
        decision = query_router.analyze_query(query)

        assert decision.strategy in ["vector", "bm25", "hybrid"]

    def test_confidence_in_valid_range(self, query_router: Any) -> None:
        """Test confidence is in 0-1 range."""
        queries = [
            "How to implement API?",
            "PostgreSQL pgvector",
            "REST API implementation",
        ]

        for query in queries:
            decision = query_router.analyze_query(query)
            assert 0.0 <= decision.confidence <= 1.0

    def test_reason_provided(self, query_router: Any) -> None:
        """Test that routing decision includes reason."""
        query = "How to implement authentication?"
        decision = query_router.analyze_query(query)

        assert len(decision.reason) > 0

    def test_query_type_valid(self, query_router: Any) -> None:
        """Test query type is valid."""
        queries = [
            "How to implement API?",
            "PostgreSQL pgvector",
            "REST API implementation",
        ]

        for query in queries:
            decision = query_router.analyze_query(query)
            assert decision.query_type in [
                "semantic", "keyword", "mixed", "empty"
            ]

    def test_empty_query_handling(self, query_router: Any) -> None:
        """Test handling of empty query."""
        decision = query_router.analyze_query("")

        assert decision.strategy == "hybrid"
        assert decision.confidence == 0.0


# Edge Cases Tests
class TestQueryRouterEdgeCases:
    """Test edge cases in query routing."""

    def test_very_long_query(self, query_router: Any) -> None:
        """Test very long query."""
        query = " ".join(["word"] * 100)
        decision = query_router.analyze_query(query)

        assert decision.complexity == "complex"
        assert 0.0 <= decision.confidence <= 1.0

    def test_special_characters_query(self, query_router: Any) -> None:
        """Test query with special characters."""
        query = "REST-API JSON::schema SQL<query>"
        decision = query_router.analyze_query(query)

        assert 0.0 <= decision.confidence <= 1.0

    def test_mixed_case_query(self, query_router: Any) -> None:
        """Test mixed case query."""
        query = "PostgreSQL PGVECTOR Index"
        decision = query_router.analyze_query(query)

        assert decision.strategy == "bm25"

    def test_query_with_numbers(self, query_router: Any) -> None:
        """Test query with numbers."""
        query = "OAuth 2.0 REST API v3"
        decision = query_router.analyze_query(query)

        assert 0.0 <= decision.confidence <= 1.0

    def test_single_character_query(self, query_router: Any) -> None:
        """Test single character query."""
        query = "a"
        decision = query_router.analyze_query(query)

        assert decision.complexity == "simple"


# Performance Tests
class TestQueryRouterPerformance:
    """Test query routing performance."""

    def test_analyze_100_queries_performance(self, query_router: Any) -> None:
        """Test analyzing 100 queries."""
        queries = [
            "How to implement authentication?",
            "PostgreSQL pgvector HNSW",
            "REST API design patterns",
        ] * 33 + ["How to implement"]

        decisions = [query_router.analyze_query(q) for q in queries]

        # All should complete
        assert len(decisions) == 100
        assert all(0.0 <= d.confidence <= 1.0 for d in decisions)

    def test_consistent_routing(self, query_router: Any) -> None:
        """Test that same query routes consistently."""
        query = "How to implement REST API?"

        decisions = [
            query_router.analyze_query(query)
            for _ in range(10)
        ]

        # All decisions should be identical
        assert all(d.strategy == decisions[0].strategy for d in decisions)
        assert all(d.confidence == decisions[0].confidence for d in decisions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
