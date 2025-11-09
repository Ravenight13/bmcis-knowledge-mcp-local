"""Multi-factor boosting system for search results.

Applies content-aware boosts to search results based on multiple factors:
1. Vendor matching: +15% if document vendor matches query context
2. Document type: +10% if document type matches query intent
3. Recency: +5% if document is recent (< 30 days old)
4. Entity matching: +10% if query entities found in document
5. Topic matching: +8% if document topic matches query topic

All boosts are cumulative but clamped to maximum 1.0 to preserve relative
ranking while amplifying relevant results.

Performance target: <10ms for boosting 100 results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.results import SearchResult

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Boost weight defaults
DEFAULT_VENDOR_BOOST: Final[float] = 0.15  # +15%
DEFAULT_DOC_TYPE_BOOST: Final[float] = 0.10  # +10%
DEFAULT_RECENCY_BOOST: Final[float] = 0.05  # +5%
DEFAULT_ENTITY_BOOST: Final[float] = 0.10  # +10%
DEFAULT_TOPIC_BOOST: Final[float] = 0.08  # +8%

# Recency thresholds (days)
VERY_RECENT_DAYS: Final[int] = 7
RECENT_DAYS: Final[int] = 30

# Known vendors (case-insensitive)
KNOWN_VENDORS: Final[set[str]] = {
    "openai", "anthropic", "google", "aws", "azure", "meta", "xai",
    "mistral", "huggingface", "cohere", "perplexity", "claude", "gpt",
    "gemini", "llama", "deepseek", "databricks", "nvidia", "together",
    "antml "
}

# Document type keywords mapping
DOC_TYPE_KEYWORDS: Final[dict[str, list[str]]] = {
    "api_docs": [
        "api", "endpoint", "request", "response", "authentication",
        "authorization", "rate limit", "parameter", "method", "http"
    ],
    "guide": [
        "guide", "tutorial", "getting started", "introduction", "how to",
        "step by step", "walkthrough", "best practices", "example"
    ],
    "kb_article": [
        "kb", "knowledge base", "article", "faq", "frequently asked",
        "question", "answer", "troubleshooting", "common issue"
    ],
    "code_sample": [
        "code", "example", "sample", "implementation", "snippet",
        "github", "repository", "source code", "library"
    ],
    "reference": [
        "reference", "specification", "spec", "schema", "format",
        "documentation", "standard", "protocol"
    ],
}

# Topic keywords
TOPIC_KEYWORDS: Final[dict[str, list[str]]] = {
    "authentication": [
        "auth", "authentication", "login", "jwt", "token", "password",
        "oauth", "saml", "mfa", "two-factor", "identity"
    ],
    "api_design": [
        "api design", "rest", "graphql", "webhook", "rate limiting",
        "versioning", "deprecation", "sdks", "compatibility"
    ],
    "data_handling": [
        "data", "database", "storage", "query", "caching", "indexing",
        "migration", "backup", "export", "import"
    ],
    "deployment": [
        "deploy", "production", "staging", "devops", "ci/cd", "docker",
        "kubernetes", "scaling", "infrastructure"
    ],
    "optimization": [
        "optimization", "performance", "latency", "throughput", "efficient",
        "fast", "slow", "benchmark", "profiling"
    ],
    "error_handling": [
        "error", "exception", "failure", "retry", "timeout", "circuit breaker",
        "fallback", "debugging", "troubleshoot"
    ],
}


@dataclass
class BoostWeights:
    """Configuration for boost factors.

    Attributes:
        vendor: Boost weight for vendor matching (default +15%).
        doc_type: Boost weight for document type matching (default +10%).
        recency: Boost weight for document recency (default +5%).
        entity: Boost weight for entity matching (default +10%).
        topic: Boost weight for topic matching (default +8%).
    """

    vendor: float = DEFAULT_VENDOR_BOOST
    doc_type: float = DEFAULT_DOC_TYPE_BOOST
    recency: float = DEFAULT_RECENCY_BOOST
    entity: float = DEFAULT_ENTITY_BOOST
    topic: float = DEFAULT_TOPIC_BOOST


class BoostingSystem:
    """Apply multi-factor boosts to search results.

    Analyzes search queries and document metadata to apply cumulative
    boosts that amplify relevant results while preserving overall ranking.
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize boosting system.

        Args:
            db_pool: Optional database pool for lookups.
            settings: Optional settings for configuration.
            logger: Optional logger for performance metrics.
        """
        self._db_pool = db_pool
        self._settings = settings
        self._logger = logger

    def apply_boosts(
        self,
        results: list[SearchResult],
        query: str,
        boosts: BoostWeights | None = None,
    ) -> list[SearchResult]:
        """Apply multi-factor boosts to results.

        Analyzes query and document metadata to apply cumulative boosts.
        Results are reranked by boosted scores.

        Args:
            results: Search results to boost.
            query: Search query for context analysis.
            boosts: Optional custom boost weights. If None, uses defaults.

        Returns:
            Results with boosted scores (max 1.0), reranked by new scores.
        """
        if not results:
            return []

        if boosts is None:
            boosts = BoostWeights()

        # Extract context from query
        vendors = self._extract_vendors(query)
        doc_type_intent = self._detect_doc_type(query)
        query_topic = self._detect_topic(query)
        entities = self._extract_entities(query, results)

        # Apply boosts to each result
        boosted_results: list[tuple[SearchResult, float]] = []

        for idx, result in enumerate(results):
            # Start with original score
            base_score = result.hybrid_score or result.similarity_score

            # Calculate cumulative boost (sum of factors)
            total_boost = 0.0

            # Vendor boost
            if boosts.vendor > 0:
                result_vendor = self._get_vendor_from_metadata(result)
                if result_vendor and result_vendor.lower() in [v.lower() for v in vendors]:
                    total_boost += boosts.vendor

            # Doc type boost
            if boosts.doc_type > 0:
                result_doc_type = self._get_doc_type_from_result(result)
                if result_doc_type and result_doc_type == doc_type_intent:
                    total_boost += boosts.doc_type

            # Recency boost
            if boosts.recency > 0:
                recency_factor = self._calculate_recency_boost(result.document_date)
                if recency_factor > 0:
                    total_boost += boosts.recency * recency_factor

            # Entity boost
            if boosts.entity > 0 and idx in entities and entities[idx]:
                total_boost += boosts.entity

            # Topic boost
            if boosts.topic > 0 and query_topic:
                # Simple heuristic: check if topic keywords appear in text
                topic_keywords = TOPIC_KEYWORDS.get(query_topic, [])
                text_lower = result.chunk_text.lower()
                if any(keyword in text_lower for keyword in topic_keywords):
                    total_boost += boosts.topic

            # Apply boost with clamping
            boosted_score = self._boost_score(base_score, total_boost)
            boosted_results.append((result, boosted_score))

        # Sort by boosted score (descending) and update ranks
        boosted_results.sort(key=lambda x: x[1], reverse=True)

        output: list[SearchResult] = []
        for new_rank, (result, boosted_score) in enumerate(boosted_results, start=1):
            updated_result = SearchResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                similarity_score=result.similarity_score,
                bm25_score=result.bm25_score,
                hybrid_score=boosted_score,
                rank=new_rank,
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
            output.append(updated_result)

        return output

    def _extract_vendors(self, query: str) -> list[str]:
        """Extract vendor names from query.

        Common vendor names: OpenAI, Anthropic, Google, AWS, Azure, etc.

        Args:
            query: Search query to analyze.

        Returns:
            List of detected vendor names (lowercase normalized).
        """
        detected_vendors: list[str] = []
        query_lower = query.lower()

        for vendor in KNOWN_VENDORS:
            if vendor in query_lower:
                detected_vendors.append(vendor)

        return detected_vendors

    def _detect_doc_type(self, query: str) -> str:
        """Detect document type intent from query.

        Args:
            query: Search query to analyze.

        Returns:
            Detected document type string, or empty string if no match.
        """
        query_lower = query.lower()

        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return doc_type

        return ""

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
        if document_date is None:
            return 0.0

        # Normalize to date if datetime
        if isinstance(document_date, datetime):
            doc_date = document_date.date()
        else:
            doc_date = document_date

        # Calculate age
        today = date.today()
        age_days = (today - doc_date).days

        if age_days < 0:
            # Future date, treat as very recent
            return 1.0
        elif age_days < VERY_RECENT_DAYS:
            return 1.0  # Very recent: full boost
        elif age_days < RECENT_DAYS:
            # Moderate recency: 70% boost
            return 0.7
        else:
            return 0.0  # Too old: no boost

    def _extract_entities(
        self,
        query: str,
        results: list[SearchResult],
    ) -> dict[int, list[str]]:
        """Extract named entities from query and match to results.

        Simple heuristic-based entity extraction looking for:
        - Capitalized words (proper nouns)
        - Product/service names from known vendors
        - Technical terms

        Args:
            query: Search query to extract entities from.
            results: Results to check for entity mentions.

        Returns:
            Dict mapping result index to list of matched entities.
        """
        entity_matches: dict[int, list[str]] = {}

        # Simple entity extraction: words with capitals
        words = query.split()
        potential_entities = [w for w in words if w and w[0].isupper()]

        # Also add multi-word phrases
        query_parts = query.split()
        for i in range(len(query_parts) - 1):
            if query_parts[i][0].isupper() and query_parts[i + 1][0].isupper():
                potential_entities.append(f"{query_parts[i]} {query_parts[i+1]}")

        # Match entities to results
        for idx, result in enumerate(results):
            matched: list[str] = []
            text_lower = result.chunk_text.lower()

            for entity in potential_entities:
                if entity.lower() in text_lower:
                    matched.append(entity)

            if matched:
                entity_matches[idx] = matched

        return entity_matches

    def _detect_topic(self, query: str) -> str:
        """Detect primary topic from query.

        Args:
            query: Search query to analyze.

        Returns:
            Detected topic string (or empty string if no match).
        """
        query_lower = query.lower()

        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return topic

        return ""

    def _get_vendor_from_metadata(self, result: SearchResult) -> str | None:
        """Extract vendor from search result metadata.

        Checks metadata dict for vendor field.

        Args:
            result: Search result to extract vendor from.

        Returns:
            Vendor name or None if not found.
        """
        if not result.metadata:
            return None

        vendor = result.metadata.get("vendor")
        if isinstance(vendor, str):
            return vendor

        # Also check common field names
        for field_name in ["vendor_name", "company", "organization", "author"]:
            value = result.metadata.get(field_name)
            if isinstance(value, str):
                return value

        return None

    def _get_doc_type_from_result(self, result: SearchResult) -> str:
        """Extract document type from search result.

        Uses source_category field as primary source.

        Args:
            result: Search result to extract type from.

        Returns:
            Document type string (or empty string if unknown).
        """
        if result.source_category:
            # Map source_category to doc type
            category_lower = result.source_category.lower()

            if "api" in category_lower:
                return "api_docs"
            elif "guide" in category_lower or "tutorial" in category_lower:
                return "guide"
            elif "kb" in category_lower or "article" in category_lower:
                return "kb_article"
            elif "code" in category_lower or "sample" in category_lower:
                return "code_sample"
            elif "ref" in category_lower:
                return "reference"

        return ""

    def _boost_score(
        self,
        original_score: float,
        boost_factor: float,
    ) -> float:
        """Apply boost factor to score with clamping.

        Formula: score * (1 + boost_factor), clamped to [0.0, 1.0]

        Args:
            original_score: Original score (should be in 0-1 range).
            boost_factor: Cumulative boost (sum of all applied factors).

        Returns:
            Boosted score clamped to [0.0, 1.0].
        """
        boosted = original_score * (1.0 + boost_factor)
        return min(max(boosted, 0.0), 1.0)  # Clamp to [0, 1]
