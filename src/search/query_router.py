"""Query routing system for selecting optimal search strategy.

Analyzes search queries to classify them and select the optimal search strategy:
- Vector search: Semantic, conceptual, or NLP-style queries
- BM25 search: Keyword-heavy, technical, or code queries
- Hybrid search: Mixed queries with balanced semantic and keyword signals

The router uses heuristics based on:
- Keyword density and frequency
- Query structure (operators, quotes, entities)
- Language patterns (questions vs. technical terms)
- Query complexity and ambiguity

Performance target: <100ms for query analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from src.core.config import Settings
from src.core.logging import StructuredLogger

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Technical keywords that signal BM25/keyword search
TECHNICAL_KEYWORDS: Final[set[str]] = {
    # API/Web
    "api", "endpoint", "request", "response", "http", "rest", "graphql",
    "webhook", "authentication", "jwt", "oauth", "token", "rate limit",
    # Programming
    "function", "class", "method", "variable", "parameter", "argument",
    "return", "import", "library", "module", "package", "dependency",
    # Databases
    "database", "query", "sql", "nosql", "mongodb", "postgres", "redis",
    "cache", "index", "schema", "table", "column", "join",
    # DevOps/Infrastructure
    "deploy", "docker", "kubernetes", "ci/cd", "pipeline", "container",
    "infrastructure", "cloud", "aws", "azure", "gcp",
    # Code-related
    "code", "github", "git", "commit", "branch", "merge", "pull request",
    "bug", "issue", "feature", "refactor", "optimization",
}

# Question words that signal semantic search
QUESTION_WORDS: Final[set[str]] = {
    "what", "how", "why", "when", "where", "who", "which",
    "explain", "describe", "tell me", "help", "show", "find",
}

# Boolean operators
BOOLEAN_OPERATORS: Final[set[str]] = {"and", "or", "not", "and not"}

# Complexity indicators
COMPLEX_KEYWORDS: Final[set[str]] = {
    "multi", "complex", "advanced", "expert", "edge case",
    "optimization", "architecture", "design pattern",
}


@dataclass
class RoutingDecision:
    """Result of query routing analysis.

    Attributes:
        strategy: Selected search strategy ("vector", "bm25", or "hybrid").
        confidence: Confidence in decision (0.5-1.0).
        reason: Explanation for the routing decision.
        keyword_score: How keyword-heavy the query is (0=semantic, 1=keyword).
        complexity: Query complexity ("simple", "moderate", "complex").
    """

    strategy: str
    confidence: float
    reason: str
    keyword_score: float
    complexity: str


class QueryRouter:
    """Determine optimal search strategy based on query characteristics.

    Analyzes queries using heuristics to classify them and select the best
    search strategy. Uses simple string-based analysis without ML models.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize query router.

        Args:
            settings: Optional settings for configuration.
            logger: Optional logger for performance metrics.
        """
        self._settings = settings
        self._logger = logger

    def select_strategy(
        self,
        query: str,
        available_strategies: list[str] | None = None,
    ) -> RoutingDecision:
        """Analyze query and select optimal search strategy.

        Rules:
        - High keyword score (>0.7) -> BM25 search
        - Low keyword score (<0.3) -> Vector search
        - Medium (0.3-0.7) -> Hybrid search

        Args:
            query: Search query to analyze.
            available_strategies: Available strategies. Defaults to all three.

        Returns:
            RoutingDecision with selected strategy and analysis.
        """
        if not query or not query.strip():
            # Empty query - default to hybrid
            return RoutingDecision(
                strategy="hybrid",
                confidence=0.5,
                reason="Empty or whitespace-only query, using hybrid fallback",
                keyword_score=0.5,
                complexity="simple",
            )

        if available_strategies is None:
            available_strategies = ["vector", "bm25", "hybrid"]

        # Analyze query
        analysis = self._analyze_query_type(query)
        keyword_score = analysis["keyword_density"]
        complexity = self._estimate_complexity(query)
        confidence = self._calculate_confidence(analysis)

        # Select strategy based on keyword score
        if keyword_score > 0.7:
            strategy = "bm25"
            reason = f"High keyword density ({keyword_score:.2f}) detected, using BM25 full-text search"
        elif keyword_score < 0.3:
            strategy = "vector"
            reason = f"Low keyword density ({keyword_score:.2f}), semantic query detected, using vector search"
        else:
            strategy = "hybrid"
            reason = f"Balanced query ({keyword_score:.2f}), using hybrid search combining both approaches"

        # Adjust strategy if not available
        if strategy not in available_strategies:
            if "hybrid" in available_strategies:
                strategy = "hybrid"
            else:
                strategy = available_strategies[0]

        return RoutingDecision(
            strategy=strategy,
            confidence=confidence,
            reason=reason,
            keyword_score=keyword_score,
            complexity=complexity,
        )

    def _analyze_query_type(self, query: str) -> dict[str, float]:
        """Analyze query characteristics.

        Returns:
            Dict with: keyword_density, semantic_score, quote_count,
            operator_count, entity_count, question_words
        """
        query_lower = query.lower()
        words = query_lower.split()

        # Count technical keywords
        keyword_count = sum(1 for word in words if word in TECHNICAL_KEYWORDS)
        keyword_density = keyword_count / max(len(words), 1)

        # Count question words
        question_count = sum(1 for word in words if word in QUESTION_WORDS)

        # Count operators
        operator_count = 0
        if " and " in query_lower:
            operator_count += query_lower.count(" and ")
        if " or " in query_lower:
            operator_count += query_lower.count(" or ")
        if " not " in query_lower:
            operator_count += query_lower.count(" not ")
        operator_count += query.count('"')  # Count quoted phrases

        # Count entities (capitalized words)
        entity_count = sum(1 for word in query.split() if word and word[0].isupper())

        # Semantic score: higher for questions and common language
        semantic_score = (question_count / max(len(words), 1)) + \
                        (1.0 if any(q in query_lower for q in ["how", "what", "explain"]) else 0.0)

        return {
            "keyword_density": min(keyword_density, 1.0),
            "semantic_score": min(semantic_score, 1.0),
            "quote_count": float(query.count('"')),
            "operator_count": float(operator_count),
            "entity_count": float(entity_count),
            "question_words": float(question_count),
        }

    def _estimate_complexity(self, query: str) -> str:
        """Classify query complexity.

        Simple: Short, direct questions or keywords
        Moderate: Medium length, mixed signals
        Complex: Long, multiple operators, technical terms

        Args:
            query: Query to classify.

        Returns:
            "simple", "moderate", or "complex".
        """
        query_lower = query.lower()
        words = query.split()

        # Complexity scoring
        complexity_score = 0.0

        # Length factor
        if len(words) < 3:
            complexity_score += 0.2
        elif len(words) < 8:
            complexity_score += 0.5
        elif len(words) < 15:
            complexity_score += 1.0
        else:
            complexity_score += 1.5

        # Operator factor
        operators_count = query_lower.count(" and ") + \
                         query_lower.count(" or ") + \
                         query_lower.count(" not ")
        complexity_score += operators_count * 0.3

        # Technical term factor
        technical_count = sum(1 for word in words if word.lower() in TECHNICAL_KEYWORDS)
        complexity_score += (technical_count / max(len(words), 1)) * 0.3

        # Complex keyword factor
        complex_count = sum(1 for word in words if word.lower() in COMPLEX_KEYWORDS)
        complexity_score += complex_count * 0.5

        # Classify
        if complexity_score < 0.5:
            return "simple"
        elif complexity_score < 1.5:
            return "moderate"
        else:
            return "complex"

    def _calculate_confidence(self, analysis: dict[str, float]) -> float:
        """Calculate confidence in routing decision.

        Higher confidence when keyword/semantic split is clear.
        Lower confidence for ambiguous queries.

        Args:
            analysis: Query analysis results.

        Returns:
            Confidence score in range [0.5, 1.0].
        """
        keyword_density = analysis.get("keyword_density", 0.5)
        question_words = analysis.get("question_words", 0.0)

        # Clear signals boost confidence
        if keyword_density > 0.7 or keyword_density < 0.2:
            # Strong keyword or semantic signal
            confidence = 0.9
        elif question_words > 0.3:
            # Strong semantic signal
            confidence = 0.85
        else:
            # Ambiguous, mixed signals
            confidence = 0.6

        # Adjust based on operator count (more operators = more clarity about intent)
        operator_count = analysis.get("operator_count", 0.0)
        if operator_count > 2:
            confidence = min(confidence + 0.1, 1.0)
        elif operator_count == 0 and keyword_density < 0.3:
            confidence = max(confidence - 0.05, 0.5)

        return confidence

    def _count_keywords(self, query: str) -> int:
        """Count technical keywords in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of detected technical keywords.
        """
        query_lower = query.lower()
        words = query_lower.split()
        return sum(1 for word in words if word in TECHNICAL_KEYWORDS)

    def _count_operators(self, query: str) -> int:
        """Count boolean operators in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of AND, OR, NOT, quotes, etc.
        """
        query_lower = query.lower()
        count = 0

        # Count each type of operator
        count += query_lower.count(" and ")
        count += query_lower.count(" or ")
        count += query_lower.count(" not ")
        count += query.count('"') // 2  # Paired quotes

        return count

    def _count_entities(self, query: str) -> int:
        """Count capitalized entities in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of potential entities (capitalized words).
        """
        words = query.split()
        return sum(1 for word in words if word and word[0].isupper())
