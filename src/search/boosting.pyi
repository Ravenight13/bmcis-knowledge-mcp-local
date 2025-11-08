"""Type stubs for multi-factor boosting system.

Provides type definitions for applying content-aware boosts to search results
based on vendor matching, document type, recency, entity matches, and topics.
"""

from dataclasses import dataclass
from datetime import date, datetime

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.results import SearchResult

@dataclass
class BoostWeights:
    """Configuration for boost factors with default values."""

    vendor: float
    doc_type: float
    recency: float
    entity: float
    topic: float

class BoostingSystem:
    """Apply multi-factor boosts to search results.

    Applies content-aware boosts based on:
    - Vendor matching: +15% if document vendor matches query context
    - Document type: +10% if document type matches query intent
    - Recency: +5% if document is recent (< 30 days old)
    - Entity matching: +10% if query entities found in document
    - Topic matching: +8% if document topic matches query topic

    All boosts are cumulative but clamped to maximum 1.0.
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize boosting system.

        Args:
            db_pool: Optional database pool for vendor/entity lookups.
            settings: Optional settings for configuration.
            logger: Optional logger for performance metrics.
        """
        ...

    def apply_boosts(
        self,
        results: list[SearchResult],
        query: str,
        boosts: BoostWeights | None = None,
    ) -> list[SearchResult]:
        """Apply multi-factor boosts to results.

        Factors:
        1. Vendor match: +15% if document vendor matches query context
        2. Doc type: +10% if document type matches query intent
        3. Recency: +5% if document is recent (< 30 days old)
        4. Entity match: +10% if document mentions query entities
        5. Topic match: +8% if document topic matches query topic

        Args:
            results: Search results to boost.
            query: Search query for context analysis.
            boosts: Optional custom boost weights. If None, uses defaults.

        Returns:
            Results with boosted scores (max 1.0), reranked by new scores.
        """
        ...

    def _extract_vendors(self, query: str) -> list[str]:
        """Extract vendor names from query.

        Common vendor names: OpenAI, Anthropic, Google, AWS, Azure, etc.

        Args:
            query: Search query to analyze.

        Returns:
            List of detected vendor names (lowercase normalized).
        """
        ...

    def _detect_doc_type(self, query: str) -> str:
        """Detect document type intent from query.

        Types: api_docs, guide, kb_article, code_sample, tutorial, reference, etc.

        Args:
            query: Search query to analyze.

        Returns:
            Detected document type string.
        """
        ...

    def _calculate_recency_boost(self, document_date: date | datetime | None) -> float:
        """Calculate recency boost based on document age.

        - Recent (< 7 days): 1.0 (100% boost)
        - Moderate (7-30 days): 0.7 (70% boost)
        - Old (> 30 days): 0.0 (no boost)

        Args:
            document_date: Document creation/update date.

        Returns:
            Recency boost factor in range [0.0, 1.0].
        """
        ...

    def _extract_entities(
        self,
        query: str,
        results: list[SearchResult],
    ) -> dict[int, list[str]]:
        """Extract named entities from query and match to results.

        Identifies proper nouns, product names, technical terms, etc.

        Args:
            query: Search query to extract entities from.
            results: Results to check for entity mentions.

        Returns:
            Dict mapping result index to list of matched entities.
        """
        ...

    def _detect_topic(self, query: str) -> str:
        """Detect primary topic from query.

        Topics: authentication, api_design, data_handling, deployment,
                optimization, error_handling, testing, etc.

        Args:
            query: Search query to analyze.

        Returns:
            Detected topic string (or empty string if no match).
        """
        ...

    def _get_vendor_from_metadata(self, result: SearchResult) -> str | None:
        """Extract vendor from search result metadata.

        Args:
            result: Search result to extract vendor from.

        Returns:
            Vendor name or None if not found.
        """
        ...

    def _get_doc_type_from_result(self, result: SearchResult) -> str:
        """Extract document type from search result.

        Args:
            result: Search result to extract type from.

        Returns:
            Document type string.
        """
        ...

    def _boost_score(
        self,
        original_score: float,
        boost_factor: float,
    ) -> float:
        """Apply boost factor to score with clamping.

        Formula: score * (1 + boost_factor), clamped to [0.0, 1.0]

        Args:
            original_score: Original score (should be in 0-1 range).
            boost_factor: Boost multiplier (cumulative of all factors).

        Returns:
            Boosted score clamped to [0.0, 1.0].
        """
        ...
