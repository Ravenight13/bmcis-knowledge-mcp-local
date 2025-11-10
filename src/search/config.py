"""Configuration management system for hybrid search with RRF.

Centralizes all configuration parameters extracted from magic numbers:
- RRF k parameter (default 60)
- Boost weights: vendor (+15%), doc_type (+10%), recency (+5%), entity (+10%), topic (+8%)
- Recency thresholds: very_recent (7 days), recent (30 days)
- Search defaults: top_k (10), min_score (0.0)

Supports environment variable overrides:
- SEARCH_RRF_K: RRF k constant (1-1000)
- SEARCH_VECTOR_WEIGHT: Vector search weight (0.0-1.0)
- SEARCH_BM25_WEIGHT: BM25 search weight (0.0-1.0)
- SEARCH_BOOST_VENDOR: Vendor boost factor
- SEARCH_BOOST_DOC_TYPE: Document type boost factor
- SEARCH_BOOST_RECENCY: Recency boost factor
- SEARCH_BOOST_ENTITY: Entity boost factor
- SEARCH_BOOST_TOPIC: Topic boost factor
- SEARCH_RECENCY_VERY_RECENT: Very recent days threshold (1-365)
- SEARCH_RECENCY_RECENT: Recent days threshold (1-365)
- SEARCH_TOP_K_DEFAULT: Default top_k value (1-1000)
- SEARCH_MIN_SCORE_DEFAULT: Default min_score threshold (0.0-1.0)

All configurations are immutable (frozen dataclasses) and validated on creation.
Uses singleton pattern for global access via get_search_config().

Example:
    >>> from src.search.config import get_search_config
    >>> config = get_search_config()
    >>> print(f"RRF k: {config.rrf.k}")
    >>> print(f"Vendor boost: {config.boosts.vendor}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion algorithm.

    Attributes:
        k: Constant for RRF formula. Higher values reduce ranking impact.
           Range: 1-1000. Default: 60.
        vector_weight: Weight for vector search results in merging.
           Range: 0.0-1.0. Default: 0.6.
        bm25_weight: Weight for BM25 search results in merging.
           Range: 0.0-1.0. Default: 0.4.
    """

    k: int
    vector_weight: float
    bm25_weight: float

    def validate(self) -> None:
        """Validate RRF configuration parameters.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if not (1 <= self.k <= 1000):
            raise ValueError(f"RRF k must be 1-1000, got {self.k}")

        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValueError(
                f"vector_weight must be 0.0-1.0, got {self.vector_weight}"
            )

        if not (0.0 <= self.bm25_weight <= 1.0):
            raise ValueError(
                f"bm25_weight must be 0.0-1.0, got {self.bm25_weight}"
            )

        # At least one weight should be positive
        if self.vector_weight + self.bm25_weight <= 0.0:
            raise ValueError(
                "At least one weight must be positive (sum > 0)"
            )


@dataclass(frozen=True)
class BoostConfig:
    """Configuration for all boost factors.

    All boost factors are cumulative but clamped to max 1.0 to preserve
    relative ranking while amplifying relevant results.

    Attributes:
        vendor: Boost factor for vendor matching (+15% default, 0.15).
                Range: 0.0-1.0.
        doc_type: Boost factor for document type matching (+10% default, 0.10).
                  Range: 0.0-1.0.
        recency: Boost factor for recent documents (+5% default, 0.05).
                Range: 0.0-1.0.
        entity: Boost factor for entity matching (+10% default, 0.10).
               Range: 0.0-1.0.
        topic: Boost factor for topic matching (+8% default, 0.08).
              Range: 0.0-1.0.
    """

    vendor: float
    doc_type: float
    recency: float
    entity: float
    topic: float

    def validate(self) -> None:
        """Validate boost configuration parameters.

        Raises:
            ValueError: If any boost factor is outside valid range.
        """
        for name, value in [
            ("vendor", self.vendor),
            ("doc_type", self.doc_type),
            ("recency", self.recency),
            ("entity", self.entity),
            ("topic", self.topic),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Boost factor '{name}' must be 0.0-1.0, got {value}"
                )


@dataclass(frozen=True)
class RecencyConfig:
    """Configuration for recency-based boosting thresholds.

    Attributes:
        very_recent_days: Days threshold for "very recent" documents (default 7).
                         Range: 1-365.
        recent_days: Days threshold for "recent" documents (default 30).
                    Range: 1-365. Must be >= very_recent_days.
    """

    very_recent_days: int
    recent_days: int

    def validate(self) -> None:
        """Validate recency configuration parameters.

        Raises:
            ValueError: If parameters are invalid or inconsistent.
        """
        if not (1 <= self.very_recent_days <= 365):
            raise ValueError(
                f"very_recent_days must be 1-365, got {self.very_recent_days}"
            )

        if not (1 <= self.recent_days <= 365):
            raise ValueError(
                f"recent_days must be 1-365, got {self.recent_days}"
            )

        if self.recent_days < self.very_recent_days:
            raise ValueError(
                f"recent_days ({self.recent_days}) must be >= very_recent_days "
                f"({self.very_recent_days})"
            )


@dataclass(frozen=True)
class SearchConfig:
    """Master configuration for hybrid search system.

    Centralizes all configuration with environment variable support.
    Uses singleton pattern for global access.

    Attributes:
        rrf: RRF algorithm configuration.
        boosts: Boost factors configuration.
        recency: Recency thresholds configuration.
        top_k_default: Default top_k for searches (1-1000). Default: 10.
        min_score_default: Default minimum score threshold (0.0-1.0). Default: 0.0.
    """

    rrf: RRFConfig
    boosts: BoostConfig
    recency: RecencyConfig
    top_k_default: int
    min_score_default: float

    _instance: ClassVar[SearchConfig | None] = None

    def validate(self) -> None:
        """Validate all configuration components.

        Raises:
            ValueError: If any configuration is invalid.
        """
        self.rrf.validate()
        self.boosts.validate()
        self.recency.validate()

        if not (1 <= self.top_k_default <= 1000):
            raise ValueError(
                f"top_k_default must be 1-1000, got {self.top_k_default}"
            )

        if not (0.0 <= self.min_score_default <= 1.0):
            raise ValueError(
                f"min_score_default must be 0.0-1.0, got {self.min_score_default}"
            )

    @classmethod
    def from_env(cls) -> SearchConfig:
        """Create SearchConfig from environment variables.

        Reads the following environment variables (falls back to defaults):
        - SEARCH_RRF_K (default: 60)
        - SEARCH_VECTOR_WEIGHT (default: 0.6)
        - SEARCH_BM25_WEIGHT (default: 0.4)
        - SEARCH_BOOST_VENDOR (default: 0.15)
        - SEARCH_BOOST_DOC_TYPE (default: 0.10)
        - SEARCH_BOOST_RECENCY (default: 0.05)
        - SEARCH_BOOST_ENTITY (default: 0.10)
        - SEARCH_BOOST_TOPIC (default: 0.08)
        - SEARCH_RECENCY_VERY_RECENT (default: 7)
        - SEARCH_RECENCY_RECENT (default: 30)
        - SEARCH_TOP_K_DEFAULT (default: 10)
        - SEARCH_MIN_SCORE_DEFAULT (default: 0.0)

        Returns:
            SearchConfig with values from environment variables.

        Raises:
            ValueError: If any environment variable value is invalid.
        """
        # RRF configuration
        rrf_k: int = int(os.getenv("SEARCH_RRF_K", "60"))
        vector_weight: float = float(os.getenv("SEARCH_VECTOR_WEIGHT", "0.6"))
        bm25_weight: float = float(os.getenv("SEARCH_BM25_WEIGHT", "0.4"))

        # Boost configuration
        boost_vendor: float = float(os.getenv("SEARCH_BOOST_VENDOR", "0.15"))
        boost_doc_type: float = float(os.getenv("SEARCH_BOOST_DOC_TYPE", "0.10"))
        boost_recency: float = float(os.getenv("SEARCH_BOOST_RECENCY", "0.05"))
        boost_entity: float = float(os.getenv("SEARCH_BOOST_ENTITY", "0.10"))
        boost_topic: float = float(os.getenv("SEARCH_BOOST_TOPIC", "0.08"))

        # Recency configuration
        recency_very_recent: int = int(
            os.getenv("SEARCH_RECENCY_VERY_RECENT", "7")
        )
        recency_recent: int = int(os.getenv("SEARCH_RECENCY_RECENT", "30"))

        # Search defaults
        top_k_default: int = int(os.getenv("SEARCH_TOP_K_DEFAULT", "10"))
        min_score_default: float = float(
            os.getenv("SEARCH_MIN_SCORE_DEFAULT", "0.0")
        )

        # Create and validate configuration
        config = cls(
            rrf=RRFConfig(
                k=rrf_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            ),
            boosts=BoostConfig(
                vendor=boost_vendor,
                doc_type=boost_doc_type,
                recency=boost_recency,
                entity=boost_entity,
                topic=boost_topic,
            ),
            recency=RecencyConfig(
                very_recent_days=recency_very_recent,
                recent_days=recency_recent,
            ),
            top_k_default=top_k_default,
            min_score_default=min_score_default,
        )

        config.validate()
        return config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SearchConfig:
        """Create SearchConfig from dictionary.

        Expected dict structure:
        {
            "rrf": {"k": 60, "vector_weight": 0.6, "bm25_weight": 0.4},
            "boosts": {
                "vendor": 0.15,
                "doc_type": 0.10,
                "recency": 0.05,
                "entity": 0.10,
                "topic": 0.08,
            },
            "recency": {"very_recent_days": 7, "recent_days": 30},
            "top_k_default": 10,
            "min_score_default": 0.0,
        }

        Args:
            config_dict: Configuration dictionary.

        Returns:
            SearchConfig created from dictionary values.

        Raises:
            ValueError: If required keys missing or values invalid.
            KeyError: If required keys are missing.
        """
        rrf_data = config_dict.get("rrf", {})
        boosts_data = config_dict.get("boosts", {})
        recency_data = config_dict.get("recency", {})

        config = cls(
            rrf=RRFConfig(
                k=rrf_data.get("k", 60),
                vector_weight=rrf_data.get("vector_weight", 0.6),
                bm25_weight=rrf_data.get("bm25_weight", 0.4),
            ),
            boosts=BoostConfig(
                vendor=boosts_data.get("vendor", 0.15),
                doc_type=boosts_data.get("doc_type", 0.10),
                recency=boosts_data.get("recency", 0.05),
                entity=boosts_data.get("entity", 0.10),
                topic=boosts_data.get("topic", 0.08),
            ),
            recency=RecencyConfig(
                very_recent_days=recency_data.get("very_recent_days", 7),
                recent_days=recency_data.get("recent_days", 30),
            ),
            top_k_default=config_dict.get("top_k_default", 10),
            min_score_default=config_dict.get("min_score_default", 0.0),
        )

        config.validate()
        return config

    @classmethod
    def get_instance(cls) -> SearchConfig:
        """Get or create singleton instance of SearchConfig.

        Creates SearchConfig from environment variables on first access.
        Subsequent calls return the same instance.

        Returns:
            SearchConfig singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls.from_env()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (primarily for testing).

        After reset, next call to get_instance() will create a new instance.
        """
        cls._instance = None

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of all configuration values.
        """
        return {
            "rrf": {
                "k": self.rrf.k,
                "vector_weight": self.rrf.vector_weight,
                "bm25_weight": self.rrf.bm25_weight,
            },
            "boosts": {
                "vendor": self.boosts.vendor,
                "doc_type": self.boosts.doc_type,
                "recency": self.boosts.recency,
                "entity": self.boosts.entity,
                "topic": self.boosts.topic,
            },
            "recency": {
                "very_recent_days": self.recency.very_recent_days,
                "recent_days": self.recency.recent_days,
            },
            "top_k_default": self.top_k_default,
            "min_score_default": self.min_score_default,
        }


def get_search_config() -> SearchConfig:
    """Get singleton SearchConfig instance.

    Convenience function equivalent to SearchConfig.get_instance().

    Returns:
        SearchConfig singleton instance.
    """
    return SearchConfig.get_instance()
