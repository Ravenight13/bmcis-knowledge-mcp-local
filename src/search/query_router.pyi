"""Type stubs for query routing system.

Provides type definitions for analyzing queries and selecting optimal search
strategy (vector, BM25, or hybrid).
"""

from dataclasses import dataclass

from src.core.config import Settings
from src.core.logging import StructuredLogger


@dataclass
class RoutingDecision:
    """Result of query routing analysis."""

    strategy: str
    confidence: float
    reason: str
    keyword_score: float
    complexity: str


class QueryRouter:
    """Determine optimal search strategy based on query characteristics.

    Analyzes queries to classify them as:
    - Semantic: Conceptual, NLP-style questions -> vector search
    - Keyword: Technical terms, code, APIs -> BM25 search
    - Mixed: Balanced semantic + keyword -> hybrid search
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
        ...

    def select_strategy(
        self,
        query: str,
        available_strategies: list[str] | None = None,
    ) -> RoutingDecision:
        """Analyze query and select optimal search strategy.

        Rules:
        - Semantic queries (conceptual, NLP): vector search
        - Keyword queries (technical terms, code): BM25 search
        - Mixed queries: hybrid search
        - Confidence: 0.5-1.0 based on clarity

        Args:
            query: Search query to analyze.
            available_strategies: List of available search strategies.
                                 Defaults to ["vector", "bm25", "hybrid"].

        Returns:
            RoutingDecision with strategy and reasoning.
        """
        ...

    def _analyze_query_type(self, query: str) -> dict[str, float]:
        """Analyze query characteristics.

        Returns dict with:
        - keyword_density: 0-1 (high = many technical terms)
        - semantic_score: 0-1 (high = conceptual language)
        - quote_count: 0+ (quoted phrases)
        - operator_count: 0+ (boolean operators)
        - entity_count: 0+ (proper nouns)

        Args:
            query: Query to analyze.

        Returns:
            Dictionary of analysis metrics.
        """
        ...

    def _estimate_complexity(self, query: str) -> str:
        """Classify query complexity.

        Args:
            query: Query to classify.

        Returns:
            "simple", "moderate", or "complex".
        """
        ...

    def _calculate_confidence(self, analysis: dict[str, float]) -> float:
        """Calculate confidence in routing decision (0-1).

        Higher confidence when clear keyword/semantic split.

        Args:
            analysis: Query analysis results.

        Returns:
            Confidence score in range [0.5, 1.0].
        """
        ...

    def _count_keywords(self, query: str) -> int:
        """Count technical keywords in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of detected technical keywords.
        """
        ...

    def _count_operators(self, query: str) -> int:
        """Count boolean operators in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of AND, OR, NOT, quotes, etc.
        """
        ...

    def _count_entities(self, query: str) -> int:
        """Count capitalized entities in query.

        Args:
            query: Query to analyze.

        Returns:
            Count of potential entities.
        """
        ...
