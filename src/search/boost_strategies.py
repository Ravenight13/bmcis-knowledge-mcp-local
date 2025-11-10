"""Extensible boost strategy system for search result ranking.

Provides an abstract base class for boost strategies and concrete implementations
for common boosting scenarios (vendor matching, document type, recency, entities, topics).

Strategies can be composed, extended with custom implementations, and registered
with the BoostStrategyFactory for dynamic creation and management.

Strategy Interface:
- should_boost(): Determine if document should receive boost
- calculate_boost(): Calculate boost multiplier for document

All strategies return boost values in range [0.0, 1.0] to maintain compatibility
with score normalization (final score clamped to [0.0, 1.0]).

Example:
    >>> strategy = VendorBoostStrategy(boost_factor=0.15)
    >>> if strategy.should_boost(query, result):
    ...     boost = strategy.calculate_boost(query, result)
    ...
    >>> # Register custom strategy
    >>> class CustomBoostStrategy(BoostStrategy):
    ...     def should_boost(self, query: str, result) -> bool:
    ...         return True
    ...     def calculate_boost(self, query: str, result) -> float:
    ...         return 0.1
    >>> BoostStrategyFactory.register_strategy("custom", CustomBoostStrategy)
    >>> strategy = BoostStrategyFactory.create_strategy("custom")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Final

from src.core.logging import StructuredLogger
from src.search.results import SearchResult

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Boost factor constraints
MIN_BOOST_FACTOR: Final[float] = 0.0
MAX_BOOST_FACTOR: Final[float] = 1.0

# Recency thresholds (days)
VERY_RECENT_DAYS: Final[int] = 7
RECENT_DAYS: Final[int] = 30

# Known vendors (case-insensitive)
KNOWN_VENDORS: Final[set[str]] = {
    "openai", "anthropic", "google", "aws", "azure", "meta", "xai",
    "mistral", "huggingface", "cohere", "perplexity", "claude", "gpt",
    "gemini", "llama", "deepseek", "databricks", "nvidia", "together",
    "antml"
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


class BoostStrategy(ABC):
    """Abstract base class for boost strategies.

    Defines the interface for content-aware boost strategies that can be applied
    to search results. Implementations determine eligibility (should_boost) and
    calculate boost multipliers (calculate_boost).

    Attributes:
        boost_factor: Base boost multiplier [0.0, 1.0] (default 0.0).
        logger: Optional logger for strategy diagnostics.
    """

    def __init__(
        self,
        boost_factor: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize boost strategy.

        Args:
            boost_factor: Base boost multiplier in range [0.0, 1.0].
            logger: Optional logger for diagnostics.

        Raises:
            ValueError: If boost_factor not in valid range.
        """
        if not MIN_BOOST_FACTOR <= boost_factor <= MAX_BOOST_FACTOR:
            msg = f"boost_factor must be in [{MIN_BOOST_FACTOR}, {MAX_BOOST_FACTOR}], got {boost_factor}"
            raise ValueError(msg)

        self.boost_factor = boost_factor
        self.logger = logger or StructuredLogger.get_logger(__name__)

    @abstractmethod
    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if document should receive boost.

        Implementations should check query context, document metadata,
        and content to determine eligibility for boosting.

        Args:
            query: Search query for context analysis.
            result: Search result to evaluate.

        Returns:
            True if document qualifies for boost, False otherwise.
        """
        ...

    @abstractmethod
    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate boost multiplier for document.

        Returns boost value in range [0.0, 1.0]. Caller is responsible
        for cumulative boost handling and score clamping.

        Args:
            query: Search query for context analysis.
            result: Search result to boost.

        Returns:
            Boost multiplier in range [0.0, 1.0].
        """
        ...


class VendorBoostStrategy(BoostStrategy):
    """Boost documents from vendors mentioned in query.

    Detects vendor names in query and applies boost if result metadata
    indicates vendor match (e.g., boost OpenAI docs when "OpenAI" in query).

    Default boost: +15% (0.15)
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if result matches vendors from query.

        Args:
            query: Search query to extract vendors from.
            result: Search result to check.

        Returns:
            True if result vendor matches query vendor, False otherwise.
        """
        result_vendor = self._get_vendor_from_metadata(result)
        query_vendors = self._extract_vendors_from_query(query)

        if not query_vendors or not result_vendor:
            return False

        return result_vendor.lower() in [v.lower() for v in query_vendors]

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate vendor boost.

        Returns boost_factor if should_boost returns True, else 0.0.

        Args:
            query: Search query.
            result: Search result to boost.

        Returns:
            Boost multiplier (boost_factor or 0.0).
        """
        if self.should_boost(query, result):
            return self.boost_factor
        return 0.0

    @staticmethod
    def _extract_vendors_from_query(query: str) -> list[str]:
        """Extract vendor names from query.

        Args:
            query: Search query to analyze.

        Returns:
            List of detected vendor names (lowercase normalized).
        """
        detected: list[str] = []
        query_lower = query.lower()

        for vendor in KNOWN_VENDORS:
            if vendor in query_lower:
                detected.append(vendor)

        return detected

    @staticmethod
    def _get_vendor_from_metadata(result: SearchResult) -> str | None:
        """Extract vendor from search result metadata.

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

        # Check common field names
        for field_name in ["vendor_name", "company", "organization", "author"]:
            value = result.metadata.get(field_name)
            if isinstance(value, str):
                return value

        return None


class DocumentTypeBoostStrategy(BoostStrategy):
    """Boost documents matching query's document type intent.

    Detects document type keywords in query and applies boost if result
    category matches (e.g., boost "api_docs" when "api" in query).

    Default boost: +10% (0.10)
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if result type matches query intent.

        Args:
            query: Search query to analyze.
            result: Search result to check.

        Returns:
            True if result type matches query document type intent, False otherwise.
        """
        query_doc_type = self._detect_doc_type(query)
        result_doc_type = self._get_doc_type_from_result(result)

        return bool(query_doc_type and result_doc_type and
                   query_doc_type == result_doc_type)

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate document type boost.

        Returns boost_factor if should_boost returns True, else 0.0.

        Args:
            query: Search query.
            result: Search result to boost.

        Returns:
            Boost multiplier (boost_factor or 0.0).
        """
        if self.should_boost(query, result):
            return self.boost_factor
        return 0.0

    @staticmethod
    def _detect_doc_type(query: str) -> str:
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

    @staticmethod
    def _get_doc_type_from_result(result: SearchResult) -> str:
        """Extract document type from search result.

        Uses source_category field as primary source.

        Args:
            result: Search result to extract type from.

        Returns:
            Document type string (or empty string if unknown).
        """
        if result.source_category:
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


class RecencyBoostStrategy(BoostStrategy):
    """Boost recently updated documents.

    Applies graduated boost based on document age:
    - Very recent (< 7 days): 100% of boost_factor
    - Moderate (7-30 days): 70% of boost_factor
    - Old (> 30 days): 0% of boost_factor

    Default boost: +5% (0.05)
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if document is recent enough to boost.

        Args:
            query: Search query (not used for recency, kept for interface).
            result: Search result to check.

        Returns:
            True if document is recent (< 30 days old), False otherwise.
        """
        recency_factor = self._calculate_recency_factor(result.document_date)
        return recency_factor > 0.0

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate recency boost with graduated scaling.

        Returns scaled boost_factor based on age:
        - < 7 days: 100% boost
        - 7-30 days: 70% boost
        - > 30 days: 0% boost

        Args:
            query: Search query (not used).
            result: Search result to boost.

        Returns:
            Scaled boost multiplier.
        """
        recency_factor = self._calculate_recency_factor(result.document_date)
        return self.boost_factor * recency_factor

    @staticmethod
    def _calculate_recency_factor(document_date: date | datetime | None) -> float:
        """Calculate recency boost factor based on document age.

        Args:
            document_date: Document creation/update date.

        Returns:
            Recency factor in range [0.0, 1.0].
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
            return 1.0  # Very recent: 100% boost
        elif age_days < RECENT_DAYS:
            return 0.7  # Moderate recency: 70% boost
        else:
            return 0.0  # Too old: no boost


class EntityBoostStrategy(BoostStrategy):
    """Boost documents containing entities mentioned in query.

    Detects named entities (proper nouns, product names) in query and
    applies boost if entities appear in result text.

    Default boost: +10% (0.10)
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if result contains entities from query.

        Args:
            query: Search query to extract entities from.
            result: Search result to check.

        Returns:
            True if result contains matched entities, False otherwise.
        """
        entities = self._extract_entities(query)
        return any(entity.lower() in result.chunk_text.lower()
                  for entity in entities)

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate entity boost.

        Returns boost_factor if should_boost returns True, else 0.0.

        Args:
            query: Search query.
            result: Search result to boost.

        Returns:
            Boost multiplier (boost_factor or 0.0).
        """
        if self.should_boost(query, result):
            return self.boost_factor
        return 0.0

    @staticmethod
    def _extract_entities(query: str) -> list[str]:
        """Extract named entities from query.

        Looks for capitalized words (proper nouns) and multi-word phrases.

        Args:
            query: Search query to extract entities from.

        Returns:
            List of potential entities.
        """
        entities: list[str] = []

        # Extract capitalized words (proper nouns)
        words = query.split()
        entities.extend([w for w in words if w and w[0].isupper()])

        # Extract multi-word phrases where both words are capitalized
        query_parts = query.split()
        for i in range(len(query_parts) - 1):
            if query_parts[i] and query_parts[i + 1]:
                if query_parts[i][0].isupper() and query_parts[i + 1][0].isupper():
                    entities.append(f"{query_parts[i]} {query_parts[i+1]}")

        return entities


class TopicBoostStrategy(BoostStrategy):
    """Boost documents matching query's primary topic.

    Detects topic keywords in query and applies boost if result contains
    topic-related keywords (e.g., boost deployment docs when "kubernetes" in query).

    Default boost: +8% (0.08)
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Determine if result contains topic keywords from query.

        Args:
            query: Search query to analyze.
            result: Search result to check.

        Returns:
            True if result contains topic keywords, False otherwise.
        """
        query_topic = self._detect_topic(query)
        if not query_topic:
            return False

        topic_keywords = TOPIC_KEYWORDS.get(query_topic, [])
        text_lower = result.chunk_text.lower()

        return any(keyword in text_lower for keyword in topic_keywords)

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate topic boost.

        Returns boost_factor if should_boost returns True, else 0.0.

        Args:
            query: Search query.
            result: Search result to boost.

        Returns:
            Boost multiplier (boost_factor or 0.0).
        """
        if self.should_boost(query, result):
            return self.boost_factor
        return 0.0

    @staticmethod
    def _detect_topic(query: str) -> str:
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


class BoostStrategyFactory:
    """Factory for creating and managing boost strategies.

    Provides registration, creation, and management of boost strategies.
    Supports both built-in strategies and custom user-defined strategies.

    Usage:
        >>> # Create a strategy
        >>> strategy = BoostStrategyFactory.create_strategy("vendor", boost_factor=0.15)
        >>>
        >>> # Register custom strategy
        >>> class CustomBoost(BoostStrategy):
        ...     def should_boost(self, q, r): return True
        ...     def calculate_boost(self, q, r): return 0.1
        >>> BoostStrategyFactory.register_strategy("custom", CustomBoost)
        >>>
        >>> # Create all default strategies
        >>> strategies = BoostStrategyFactory.create_all_strategies()
    """

    _strategies: dict[str, type[BoostStrategy]] = {
        "vendor": VendorBoostStrategy,
        "doc_type": DocumentTypeBoostStrategy,
        "recency": RecencyBoostStrategy,
        "entity": EntityBoostStrategy,
        "topic": TopicBoostStrategy,
    }

    _default_factors: dict[str, float] = {
        "vendor": 0.15,
        "doc_type": 0.10,
        "recency": 0.05,
        "entity": 0.10,
        "topic": 0.08,
    }

    @classmethod
    def register_strategy(
        cls,
        strategy_name: str,
        strategy_class: type[BoostStrategy],
        default_factor: float | None = None,
    ) -> None:
        """Register a custom boost strategy.

        Allows users to create and register custom strategies for use
        with the factory.

        Args:
            strategy_name: Name to register strategy under.
            strategy_class: BoostStrategy subclass to register.
            default_factor: Optional default boost factor [0.0, 1.0].

        Raises:
            TypeError: If strategy_class not a BoostStrategy subclass.
            ValueError: If default_factor not in valid range.
        """
        if not issubclass(strategy_class, BoostStrategy):
            msg = f"{strategy_class} must be a BoostStrategy subclass"
            raise TypeError(msg)

        if default_factor is not None:
            if not MIN_BOOST_FACTOR <= default_factor <= MAX_BOOST_FACTOR:
                msg = f"default_factor must be in [{MIN_BOOST_FACTOR}, {MAX_BOOST_FACTOR}]"
                raise ValueError(msg)
            cls._default_factors[strategy_name] = default_factor

        cls._strategies[strategy_name] = strategy_class
        logger.info(f"Registered boost strategy: {strategy_name} ({strategy_class.__name__})")

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        boost_factor: float | None = None,
    ) -> BoostStrategy:
        """Create a boost strategy instance.

        Args:
            strategy_name: Name of strategy to create.
            boost_factor: Optional custom boost factor. If None, uses default.

        Returns:
            Instantiated BoostStrategy.

        Raises:
            KeyError: If strategy_name not registered.
            ValueError: If boost_factor not in valid range.
        """
        if strategy_name not in cls._strategies:
            msg = f"Unknown boost strategy: {strategy_name}"
            raise KeyError(msg)

        strategy_class = cls._strategies[strategy_name]

        # Use provided factor or default
        if boost_factor is None:
            boost_factor = cls._default_factors.get(strategy_name, 0.0)

        return strategy_class(boost_factor=boost_factor)

    @classmethod
    def create_all_strategies(
        cls,
        custom_factors: dict[str, float] | None = None,
    ) -> list[BoostStrategy]:
        """Create all registered boost strategies.

        Creates one instance of each registered strategy with default or
        custom boost factors.

        Args:
            custom_factors: Optional dict mapping strategy names to custom boost factors.

        Returns:
            List of instantiated strategies.

        Raises:
            ValueError: If custom factors not in valid range.
        """
        strategies: list[BoostStrategy] = []
        custom_factors = custom_factors or {}

        for strategy_name, strategy_class in cls._strategies.items():
            if strategy_name in custom_factors:
                factor = custom_factors[strategy_name]
            else:
                factor = cls._default_factors.get(strategy_name, 0.0)

            strategies.append(strategy_class(boost_factor=factor))

        return strategies

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of all registered strategy names.

        Returns:
            List of strategy names available for creation.
        """
        return list(cls._strategies.keys())

    @classmethod
    def get_default_factors(cls) -> dict[str, float]:
        """Get default boost factors for all strategies.

        Returns:
            Dict mapping strategy names to default boost factors.
        """
        return cls._default_factors.copy()
