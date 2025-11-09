"""Tests for SearchConfig system.

Tests configuration loading, validation, environment variables, and singleton pattern.
"""

from __future__ import annotations

import os
import re

import pytest

from src.search.config import (
    BoostConfig,
    RecencyConfig,
    RRFConfig,
    SearchConfig,
    get_search_config,
)


class TestRRFConfig:
    """Tests for RRFConfig validation."""

    def test_valid_rrf_config(self) -> None:
        """Test creating valid RRF configuration."""
        config = RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4)
        config.validate()  # Should not raise

    def test_rrf_k_boundary_values(self) -> None:
        """Test RRF k parameter boundary values."""
        # Valid min
        RRFConfig(k=1, vector_weight=0.5, bm25_weight=0.5).validate()
        # Valid max
        RRFConfig(k=1000, vector_weight=0.5, bm25_weight=0.5).validate()

    def test_rrf_k_too_low(self) -> None:
        """Test RRF k parameter below minimum."""
        config = RRFConfig(k=0, vector_weight=0.6, bm25_weight=0.4)
        with pytest.raises(ValueError, match="RRF k must be 1-1000"):
            config.validate()

    def test_rrf_k_too_high(self) -> None:
        """Test RRF k parameter above maximum."""
        config = RRFConfig(k=1001, vector_weight=0.6, bm25_weight=0.4)
        with pytest.raises(ValueError, match="RRF k must be 1-1000"):
            config.validate()

    def test_rrf_vector_weight_invalid(self) -> None:
        """Test invalid vector weight."""
        config = RRFConfig(k=60, vector_weight=1.5, bm25_weight=0.4)
        with pytest.raises(ValueError, match=re.escape("vector_weight must be 0.0-1.0")):
            config.validate()

    def test_rrf_bm25_weight_invalid(self) -> None:
        """Test invalid BM25 weight."""
        config = RRFConfig(k=60, vector_weight=0.6, bm25_weight=-0.1)
        with pytest.raises(ValueError, match=re.escape("bm25_weight must be 0.0-1.0")):
            config.validate()

    def test_rrf_both_weights_zero(self) -> None:
        """Test both weights being zero."""
        config = RRFConfig(k=60, vector_weight=0.0, bm25_weight=0.0)
        with pytest.raises(ValueError, match="At least one weight must be positive"):
            config.validate()


class TestBoostConfig:
    """Tests for BoostConfig validation."""

    def test_valid_boost_config(self) -> None:
        """Test creating valid boost configuration."""
        config = BoostConfig(
            vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
        )
        config.validate()  # Should not raise

    def test_boost_config_zero_values(self) -> None:
        """Test boost config with all zeros."""
        config = BoostConfig(
            vendor=0.0, doc_type=0.0, recency=0.0, entity=0.0, topic=0.0
        )
        config.validate()  # Should not raise (all boosting disabled)

    def test_boost_config_max_values(self) -> None:
        """Test boost config with maximum values."""
        config = BoostConfig(
            vendor=1.0, doc_type=1.0, recency=1.0, entity=1.0, topic=1.0
        )
        config.validate()  # Should not raise

    def test_boost_vendor_invalid(self) -> None:
        """Test invalid vendor boost."""
        config = BoostConfig(
            vendor=1.5, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
        )
        with pytest.raises(ValueError, match=re.escape("Boost factor 'vendor' must be 0.0-1.0")):
            config.validate()

    def test_boost_recency_invalid(self) -> None:
        """Test invalid recency boost."""
        config = BoostConfig(
            vendor=0.15, doc_type=0.10, recency=-0.05, entity=0.10, topic=0.08
        )
        with pytest.raises(ValueError, match=re.escape("Boost factor 'recency' must be 0.0-1.0")):
            config.validate()


class TestRecencyConfig:
    """Tests for RecencyConfig validation."""

    def test_valid_recency_config(self) -> None:
        """Test creating valid recency configuration."""
        config = RecencyConfig(very_recent_days=7, recent_days=30)
        config.validate()  # Should not raise

    def test_recency_days_boundary(self) -> None:
        """Test recency days boundary values."""
        # Valid min
        RecencyConfig(very_recent_days=1, recent_days=1).validate()
        # Valid max
        RecencyConfig(very_recent_days=364, recent_days=365).validate()

    def test_recency_very_recent_too_low(self) -> None:
        """Test very_recent_days below minimum."""
        config = RecencyConfig(very_recent_days=0, recent_days=30)
        with pytest.raises(ValueError, match="very_recent_days must be 1-365"):
            config.validate()

    def test_recency_very_recent_too_high(self) -> None:
        """Test very_recent_days above maximum."""
        config = RecencyConfig(very_recent_days=366, recent_days=367)
        with pytest.raises(ValueError, match="very_recent_days must be 1-365"):
            config.validate()

    def test_recency_recent_greater_than_very_recent(self) -> None:
        """Test that recent_days must be >= very_recent_days."""
        config = RecencyConfig(very_recent_days=30, recent_days=7)
        with pytest.raises(
            ValueError,
            match="recent_days \\(7\\) must be >= very_recent_days \\(30\\)",
        ):
            config.validate()


class TestSearchConfigDefaults:
    """Tests for SearchConfig default values."""

    def test_search_config_default_values(self) -> None:
        """Test SearchConfig with all default values."""
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=10,
            min_score_default=0.0,
        )

        assert config.rrf.k == 60
        assert config.rrf.vector_weight == 0.6
        assert config.rrf.bm25_weight == 0.4
        assert config.boosts.vendor == 0.15
        assert config.boosts.doc_type == 0.10
        assert config.boosts.recency == 0.05
        assert config.boosts.entity == 0.10
        assert config.boosts.topic == 0.08
        assert config.recency.very_recent_days == 7
        assert config.recency.recent_days == 30
        assert config.top_k_default == 10
        assert config.min_score_default == 0.0

    def test_search_config_top_k_default_boundary(self) -> None:
        """Test top_k_default boundary values."""
        # Valid min
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=1,
            min_score_default=0.0,
        )
        config.validate()

        # Valid max
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=1000,
            min_score_default=0.0,
        )
        config.validate()

    def test_search_config_invalid_top_k(self) -> None:
        """Test invalid top_k_default."""
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=1001,
            min_score_default=0.0,
        )
        with pytest.raises(ValueError, match="top_k_default must be 1-1000"):
            config.validate()

    def test_search_config_invalid_min_score(self) -> None:
        """Test invalid min_score_default."""
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=10,
            min_score_default=1.5,
        )
        with pytest.raises(ValueError, match=re.escape("min_score_default must be 0.0-1.0")):
            config.validate()


class TestSearchConfigFromEnv:
    """Tests for loading SearchConfig from environment variables."""

    def test_search_config_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env with all defaults (no env vars set)."""
        # Clear all SEARCH_ environment variables
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        config = SearchConfig.from_env()

        assert config.rrf.k == 60
        assert config.rrf.vector_weight == 0.6
        assert config.rrf.bm25_weight == 0.4
        assert config.boosts.vendor == 0.15
        assert config.boosts.doc_type == 0.10
        assert config.boosts.recency == 0.05
        assert config.boosts.entity == 0.10
        assert config.boosts.topic == 0.08
        assert config.recency.very_recent_days == 7
        assert config.recency.recent_days == 30
        assert config.top_k_default == 10
        assert config.min_score_default == 0.0

    def test_search_config_from_env_rrf_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test overriding RRF k parameter."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_RRF_K", "100")
        config = SearchConfig.from_env()
        assert config.rrf.k == 100

    def test_search_config_from_env_boost_vendor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test overriding vendor boost."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_BOOST_VENDOR", "0.25")
        config = SearchConfig.from_env()
        assert config.boosts.vendor == 0.25

    def test_search_config_from_env_all_boosts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test overriding all boost factors."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_BOOST_VENDOR", "0.20")
        monkeypatch.setenv("SEARCH_BOOST_DOC_TYPE", "0.15")
        monkeypatch.setenv("SEARCH_BOOST_RECENCY", "0.10")
        monkeypatch.setenv("SEARCH_BOOST_ENTITY", "0.12")
        monkeypatch.setenv("SEARCH_BOOST_TOPIC", "0.10")

        config = SearchConfig.from_env()
        assert config.boosts.vendor == 0.20
        assert config.boosts.doc_type == 0.15
        assert config.boosts.recency == 0.10
        assert config.boosts.entity == 0.12
        assert config.boosts.topic == 0.10

    def test_search_config_from_env_recency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test overriding recency thresholds."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_RECENCY_VERY_RECENT", "5")
        monkeypatch.setenv("SEARCH_RECENCY_RECENT", "60")

        config = SearchConfig.from_env()
        assert config.recency.very_recent_days == 5
        assert config.recency.recent_days == 60

    def test_search_config_from_env_search_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test overriding search defaults."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_TOP_K_DEFAULT", "20")
        monkeypatch.setenv("SEARCH_MIN_SCORE_DEFAULT", "0.5")

        config = SearchConfig.from_env()
        assert config.top_k_default == 20
        assert config.min_score_default == 0.5

    def test_search_config_from_env_invalid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid environment variable values are caught."""
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_BOOST_VENDOR", "invalid")

        with pytest.raises(ValueError):
            SearchConfig.from_env()


class TestSearchConfigFromDict:
    """Tests for loading SearchConfig from dictionary."""

    def test_search_config_from_dict_defaults(self) -> None:
        """Test from_dict with empty dictionary (uses all defaults)."""
        config = SearchConfig.from_dict({})

        assert config.rrf.k == 60
        assert config.rrf.vector_weight == 0.6
        assert config.boosts.vendor == 0.15
        assert config.top_k_default == 10

    def test_search_config_from_dict_custom(self) -> None:
        """Test from_dict with custom values."""
        config_dict = {
            "rrf": {"k": 80, "vector_weight": 0.5, "bm25_weight": 0.5},
            "boosts": {
                "vendor": 0.20,
                "doc_type": 0.12,
                "recency": 0.08,
                "entity": 0.15,
                "topic": 0.10,
            },
            "recency": {"very_recent_days": 5, "recent_days": 45},
            "top_k_default": 20,
            "min_score_default": 0.3,
        }

        config = SearchConfig.from_dict(config_dict)

        assert config.rrf.k == 80
        assert config.rrf.vector_weight == 0.5
        assert config.boosts.vendor == 0.20
        assert config.recency.very_recent_days == 5
        assert config.top_k_default == 20
        assert config.min_score_default == 0.3

    def test_search_config_from_dict_partial(self) -> None:
        """Test from_dict with partial values (mixed defaults and custom)."""
        config_dict = {
            "rrf": {"k": 100},  # Missing other rrf fields
            "boosts": {"vendor": 0.25},  # Missing other boost fields
        }

        config = SearchConfig.from_dict(config_dict)

        assert config.rrf.k == 100
        assert config.rrf.vector_weight == 0.6  # Default
        assert config.boosts.vendor == 0.25
        assert config.boosts.doc_type == 0.10  # Default


class TestSearchConfigSingleton:
    """Tests for SearchConfig singleton pattern."""

    def test_get_instance_creates_instance(self) -> None:
        """Test that get_instance creates instance on first call."""
        SearchConfig.reset_instance()
        config1 = SearchConfig.get_instance()
        assert config1 is not None

    def test_get_instance_returns_same_instance(self) -> None:
        """Test that subsequent calls return same instance."""
        SearchConfig.reset_instance()
        config1 = SearchConfig.get_instance()
        config2 = SearchConfig.get_instance()
        assert config1 is config2

    def test_get_search_config_function(self) -> None:
        """Test get_search_config convenience function."""
        SearchConfig.reset_instance()
        config = get_search_config()
        assert isinstance(config, SearchConfig)
        assert config.rrf.k == 60

    def test_reset_instance(self) -> None:
        """Test resetting singleton instance."""
        SearchConfig.reset_instance()
        config1 = SearchConfig.get_instance()
        SearchConfig.reset_instance()
        config2 = SearchConfig.get_instance()
        # Different instances (not same object)
        assert config1 is not config2

    def test_singleton_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test singleton reads environment on creation."""
        SearchConfig.reset_instance()
        for key in list(os.environ.keys()):
            if key.startswith("SEARCH_"):
                monkeypatch.delenv(key)

        monkeypatch.setenv("SEARCH_RRF_K", "75")

        config = SearchConfig.get_instance()
        assert config.rrf.k == 75


class TestSearchConfigSerialization:
    """Tests for SearchConfig serialization."""

    def test_to_dict_round_trip(self) -> None:
        """Test converting to dict and back preserves values."""
        original = SearchConfig(
            rrf=RRFConfig(k=80, vector_weight=0.5, bm25_weight=0.5),
            boosts=BoostConfig(
                vendor=0.20, doc_type=0.12, recency=0.08, entity=0.15, topic=0.10
            ),
            recency=RecencyConfig(very_recent_days=5, recent_days=45),
            top_k_default=20,
            min_score_default=0.3,
        )

        dict_repr = original.to_dict()
        restored = SearchConfig.from_dict(dict_repr)

        assert restored.rrf.k == original.rrf.k
        assert restored.rrf.vector_weight == original.rrf.vector_weight
        assert restored.boosts.vendor == original.boosts.vendor
        assert restored.recency.very_recent_days == original.recency.very_recent_days
        assert restored.top_k_default == original.top_k_default

    def test_to_dict_structure(self) -> None:
        """Test to_dict returns correct structure."""
        config = SearchConfig(
            rrf=RRFConfig(k=60, vector_weight=0.6, bm25_weight=0.4),
            boosts=BoostConfig(
                vendor=0.15, doc_type=0.10, recency=0.05, entity=0.10, topic=0.08
            ),
            recency=RecencyConfig(very_recent_days=7, recent_days=30),
            top_k_default=10,
            min_score_default=0.0,
        )

        dict_repr = config.to_dict()

        # Verify structure
        assert "rrf" in dict_repr
        assert "boosts" in dict_repr
        assert "recency" in dict_repr
        assert "top_k_default" in dict_repr
        assert "min_score_default" in dict_repr

        assert dict_repr["rrf"]["k"] == 60
        assert dict_repr["boosts"]["vendor"] == 0.15
        assert dict_repr["recency"]["very_recent_days"] == 7
