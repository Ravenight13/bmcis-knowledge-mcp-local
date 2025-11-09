"""Tests for embedding configuration management.

Tests configuration validation, singleton pattern, environment variable overrides,
and all configuration sub-models.
"""

import os
from typing import Any

import pytest

from src.embedding.config import (
    CircuitBreakerConfiguration,
    EmbeddingConfig,
    GeneratorConfiguration,
    HNSWConfiguration,
    InsertionConfiguration,
    ModelConfiguration,
    get_embedding_config,
    reset_config_for_testing,
)


class TestModelConfiguration:
    """Tests for model selection configuration."""

    def test_default_values(self) -> None:
        """Test default model configuration."""
        config = ModelConfiguration()
        assert config.primary_model == "all-MiniLM-L12-v2"
        assert config.fallback_model == "all-MiniLM-L6-v2"
        assert config.enable_cached_fallback is True
        assert config.enable_dummy_mode is False
        assert config.device == "auto"

    def test_custom_values(self) -> None:
        """Test custom model configuration."""
        config = ModelConfiguration(
            primary_model="sentence-transformers/all-mpnet-base-v2",
            fallback_model="sentence-transformers/all-distilroberta-v1",
            enable_cached_fallback=False,
            enable_dummy_mode=True,
            device="cuda",
        )
        assert config.primary_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.fallback_model == "sentence-transformers/all-distilroberta-v1"
        assert config.enable_cached_fallback is False
        assert config.enable_dummy_mode is True
        assert config.device == "cuda"

    def test_device_values(self) -> None:
        """Test valid device values."""
        for device in ["cuda", "cpu", "auto"]:
            config = ModelConfiguration(device=device)
            assert config.device == device

    def test_device_invalid_value(self) -> None:
        """Test invalid device value."""
        with pytest.raises(ValueError):
            ModelConfiguration(device="gpu")  # type: ignore


class TestGeneratorConfiguration:
    """Tests for embedding generation configuration."""

    def test_default_values(self) -> None:
        """Test default generator configuration."""
        config = GeneratorConfiguration()
        assert config.batch_size == 64
        assert config.num_workers == 4
        assert config.use_threading is True

    def test_custom_values(self) -> None:
        """Test custom generator configuration."""
        config = GeneratorConfiguration(
            batch_size=128,
            num_workers=8,
            use_threading=False,
        )
        assert config.batch_size == 128
        assert config.num_workers == 8
        assert config.use_threading is False

    def test_batch_size_validation(self) -> None:
        """Test batch_size validation constraints."""
        # Valid min
        config = GeneratorConfiguration(batch_size=1)
        assert config.batch_size == 1

        # Valid max
        config = GeneratorConfiguration(batch_size=512)
        assert config.batch_size == 512

        # Invalid below min
        with pytest.raises(ValueError):
            GeneratorConfiguration(batch_size=0)

        # Invalid above max
        with pytest.raises(ValueError):
            GeneratorConfiguration(batch_size=513)

    def test_num_workers_validation(self) -> None:
        """Test num_workers validation constraints."""
        # Valid min
        config = GeneratorConfiguration(num_workers=1)
        assert config.num_workers == 1

        # Valid max
        config = GeneratorConfiguration(num_workers=16)
        assert config.num_workers == 16

        # Invalid below min
        with pytest.raises(ValueError):
            GeneratorConfiguration(num_workers=0)

        # Invalid above max
        with pytest.raises(ValueError):
            GeneratorConfiguration(num_workers=17)


class TestInsertionConfiguration:
    """Tests for database insertion configuration."""

    def test_default_values(self) -> None:
        """Test default insertion configuration."""
        config = InsertionConfiguration()
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.create_index is True

    def test_custom_values(self) -> None:
        """Test custom insertion configuration."""
        config = InsertionConfiguration(
            batch_size=200,
            max_retries=5,
            retry_delay_seconds=2.0,
            create_index=False,
        )
        assert config.batch_size == 200
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.create_index is False

    def test_batch_size_validation(self) -> None:
        """Test batch_size validation constraints."""
        # Valid min
        config = InsertionConfiguration(batch_size=1)
        assert config.batch_size == 1

        # Valid max
        config = InsertionConfiguration(batch_size=1000)
        assert config.batch_size == 1000

        # Invalid below min
        with pytest.raises(ValueError):
            InsertionConfiguration(batch_size=0)

        # Invalid above max
        with pytest.raises(ValueError):
            InsertionConfiguration(batch_size=1001)

    def test_max_retries_validation(self) -> None:
        """Test max_retries validation constraints."""
        # Valid min (0 is allowed)
        config = InsertionConfiguration(max_retries=0)
        assert config.max_retries == 0

        # Valid max
        config = InsertionConfiguration(max_retries=10)
        assert config.max_retries == 10

        # Invalid above max
        with pytest.raises(ValueError):
            InsertionConfiguration(max_retries=11)

    def test_retry_delay_validation(self) -> None:
        """Test retry_delay_seconds validation constraints."""
        # Valid min
        config = InsertionConfiguration(retry_delay_seconds=0.1)
        assert config.retry_delay_seconds == 0.1

        # Valid max
        config = InsertionConfiguration(retry_delay_seconds=30.0)
        assert config.retry_delay_seconds == 30.0

        # Invalid below min
        with pytest.raises(ValueError):
            InsertionConfiguration(retry_delay_seconds=0.05)

        # Invalid above max
        with pytest.raises(ValueError):
            InsertionConfiguration(retry_delay_seconds=31.0)


class TestHNSWConfiguration:
    """Tests for HNSW index configuration."""

    def test_default_values(self) -> None:
        """Test default HNSW configuration."""
        config = HNSWConfiguration()
        assert config.m == 16
        assert config.ef_construction == 200
        assert config.ef_search == 64

    def test_custom_values(self) -> None:
        """Test custom HNSW configuration."""
        config = HNSWConfiguration(
            m=32,
            ef_construction=300,
            ef_search=128,
        )
        assert config.m == 32
        assert config.ef_construction == 300
        assert config.ef_search == 128

    def test_m_parameter_validation(self) -> None:
        """Test m parameter validation constraints."""
        # Valid min
        config = HNSWConfiguration(m=4)
        assert config.m == 4

        # Valid max
        config = HNSWConfiguration(m=64)
        assert config.m == 64

        # Invalid below min
        with pytest.raises(ValueError):
            HNSWConfiguration(m=3)

        # Invalid above max
        with pytest.raises(ValueError):
            HNSWConfiguration(m=65)

    def test_ef_construction_validation(self) -> None:
        """Test ef_construction validation constraints."""
        # Valid min
        config = HNSWConfiguration(ef_construction=10)
        assert config.ef_construction == 10

        # Valid max
        config = HNSWConfiguration(ef_construction=500)
        assert config.ef_construction == 500

        # Invalid below min
        with pytest.raises(ValueError):
            HNSWConfiguration(ef_construction=9)

        # Invalid above max
        with pytest.raises(ValueError):
            HNSWConfiguration(ef_construction=501)

    def test_ef_search_validation(self) -> None:
        """Test ef_search validation constraints."""
        # Valid min
        config = HNSWConfiguration(ef_search=10)
        assert config.ef_search == 10

        # Valid max
        config = HNSWConfiguration(ef_search=500)
        assert config.ef_search == 500

        # Invalid below min
        with pytest.raises(ValueError):
            HNSWConfiguration(ef_search=9)

        # Invalid above max
        with pytest.raises(ValueError):
            HNSWConfiguration(ef_search=501)


class TestCircuitBreakerConfiguration:
    """Tests for circuit breaker configuration."""

    def test_default_values(self) -> None:
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfiguration()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfiguration(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=120.0,
            enabled=False,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 120.0
        assert config.enabled is False

    def test_failure_threshold_validation(self) -> None:
        """Test failure_threshold validation constraints."""
        # Valid min
        config = CircuitBreakerConfiguration(failure_threshold=1)
        assert config.failure_threshold == 1

        # Valid max
        config = CircuitBreakerConfiguration(failure_threshold=20)
        assert config.failure_threshold == 20

        # Invalid below min
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(failure_threshold=0)

        # Invalid above max
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(failure_threshold=21)

    def test_success_threshold_validation(self) -> None:
        """Test success_threshold validation constraints."""
        # Valid min
        config = CircuitBreakerConfiguration(success_threshold=1)
        assert config.success_threshold == 1

        # Valid max
        config = CircuitBreakerConfiguration(success_threshold=10)
        assert config.success_threshold == 10

        # Invalid below min
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(success_threshold=0)

        # Invalid above max
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(success_threshold=11)

    def test_timeout_seconds_validation(self) -> None:
        """Test timeout_seconds validation constraints."""
        # Valid min
        config = CircuitBreakerConfiguration(timeout_seconds=1.0)
        assert config.timeout_seconds == 1.0

        # Valid max
        config = CircuitBreakerConfiguration(timeout_seconds=600.0)
        assert config.timeout_seconds == 600.0

        # Invalid below min
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(timeout_seconds=0.5)

        # Invalid above max
        with pytest.raises(ValueError):
            CircuitBreakerConfiguration(timeout_seconds=601.0)


class TestEmbeddingConfig:
    """Tests for root embedding configuration."""

    def test_default_configuration(self) -> None:
        """Test default configuration has all defaults."""
        config = EmbeddingConfig()

        # Check model config
        assert config.model.primary_model == "all-MiniLM-L12-v2"
        assert config.model.fallback_model == "all-MiniLM-L6-v2"

        # Check generator config
        assert config.generator.batch_size == 64
        assert config.generator.num_workers == 4

        # Check insertion config
        assert config.insertion.batch_size == 100
        assert config.insertion.max_retries == 3

        # Check HNSW config
        assert config.hnsw.m == 16
        assert config.hnsw.ef_construction == 200

        # Check circuit breaker config
        assert config.circuit_breaker.failure_threshold == 5
        assert config.circuit_breaker.enabled is True

    def test_custom_configuration(self) -> None:
        """Test configuration with custom sub-configs."""
        model_config = ModelConfiguration(primary_model="custom-model")
        generator_config = GeneratorConfiguration(batch_size=128)

        config = EmbeddingConfig(
            model=model_config,
            generator=generator_config,
        )

        assert config.model.primary_model == "custom-model"
        assert config.generator.batch_size == 128

    def test_invalid_extra_field(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError):
            EmbeddingConfig(invalid_field="value")  # type: ignore


class TestConfigurationSingleton:
    """Tests for singleton factory pattern."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_config_for_testing()

    def test_get_embedding_config_returns_singleton(self) -> None:
        """Test that get_embedding_config returns same instance."""
        config1 = get_embedding_config()
        config2 = get_embedding_config()

        assert config1 is config2

    def test_singleton_persists_across_calls(self) -> None:
        """Test singleton configuration persists."""
        config1 = get_embedding_config()
        config1.generator.batch_size = 256  # type: ignore

        config2 = get_embedding_config()
        assert config2.generator.batch_size == 256

    def test_reset_creates_new_instance(self) -> None:
        """Test that reset creates new instance."""
        config1 = get_embedding_config()
        config1.generator.batch_size = 256  # type: ignore

        reset_config_for_testing()

        config2 = get_embedding_config()
        assert config2.generator.batch_size == 64  # Back to default


class TestEnvironmentVariableOverrides:
    """Tests for environment variable configuration support.

    Note: BaseSettings environment variable loading is tested through
    integration tests. These tests verify that the configuration objects
    can be instantiated programmatically which is how they're typically
    used in practice.
    """

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_config_for_testing()

    def test_custom_model_config_programmatic(self) -> None:
        """Test creating custom model config programmatically."""
        custom_model = ModelConfiguration(
            primary_model="test-model-v1",
            fallback_model="fallback-v1"
        )
        config = EmbeddingConfig(model=custom_model)
        assert config.model.primary_model == "test-model-v1"
        assert config.model.fallback_model == "fallback-v1"

    def test_custom_generator_config_programmatic(self) -> None:
        """Test creating custom generator config programmatically."""
        custom_gen = GeneratorConfiguration(batch_size=128, num_workers=8)
        config = EmbeddingConfig(generator=custom_gen)
        assert config.generator.batch_size == 128
        assert config.generator.num_workers == 8

    def test_custom_insertion_config_programmatic(self) -> None:
        """Test creating custom insertion config programmatically."""
        custom_insert = InsertionConfiguration(
            batch_size=200, max_retries=5
        )
        config = EmbeddingConfig(insertion=custom_insert)
        assert config.insertion.batch_size == 200
        assert config.insertion.max_retries == 5

    def test_custom_hnsw_config_programmatic(self) -> None:
        """Test creating custom HNSW config programmatically."""
        custom_hnsw = HNSWConfiguration(m=32, ef_search=128)
        config = EmbeddingConfig(hnsw=custom_hnsw)
        assert config.hnsw.m == 32
        assert config.hnsw.ef_search == 128

    def test_custom_circuit_breaker_config_programmatic(self) -> None:
        """Test creating custom circuit breaker config programmatically."""
        custom_cb = CircuitBreakerConfiguration(
            failure_threshold=10, enabled=False
        )
        config = EmbeddingConfig(circuit_breaker=custom_cb)
        assert config.circuit_breaker.failure_threshold == 10
        assert config.circuit_breaker.enabled is False

    def test_all_custom_configs_together(self) -> None:
        """Test creating all custom configs together."""
        config = EmbeddingConfig(
            model=ModelConfiguration(enable_dummy_mode=True),
            generator=GeneratorConfiguration(batch_size=256),
            insertion=InsertionConfiguration(max_retries=5),
            hnsw=HNSWConfiguration(m=32),
            circuit_breaker=CircuitBreakerConfiguration(enabled=False),
        )
        assert config.model.enable_dummy_mode is True
        assert config.generator.batch_size == 256
        assert config.insertion.max_retries == 5
        assert config.hnsw.m == 32
        assert config.circuit_breaker.enabled is False


class TestConfigurationIntegration:
    """Integration tests for complete configuration."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_config_for_testing()

    def test_configuration_for_development(self) -> None:
        """Test configuration suitable for development."""
        dev_config = EmbeddingConfig(
            model=ModelConfiguration(enable_dummy_mode=True),
            generator=GeneratorConfiguration(batch_size=32, num_workers=2),
        )
        assert dev_config.model.enable_dummy_mode is True
        assert dev_config.generator.batch_size == 32
        assert dev_config.generator.num_workers == 2

    def test_configuration_for_production(self) -> None:
        """Test configuration suitable for production."""
        prod_config = EmbeddingConfig(
            generator=GeneratorConfiguration(batch_size=256, num_workers=8),
            circuit_breaker=CircuitBreakerConfiguration(
                failure_threshold=3
            ),
            hnsw=HNSWConfiguration(ef_search=128),
        )
        assert prod_config.generator.batch_size == 256
        assert prod_config.generator.num_workers == 8
        assert prod_config.circuit_breaker.failure_threshold == 3
        assert prod_config.hnsw.ef_search == 128

    def test_all_sub_configs_accessible(self) -> None:
        """Test that all sub-configurations are accessible and valid."""
        config = get_embedding_config()

        # Verify all sub-configs exist and have expected types
        assert isinstance(config.model, ModelConfiguration)
        assert isinstance(config.generator, GeneratorConfiguration)
        assert isinstance(config.insertion, InsertionConfiguration)
        assert isinstance(config.hnsw, HNSWConfiguration)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfiguration)

        # Verify all configs have valid values
        assert config.model.primary_model
        assert config.generator.batch_size > 0
        assert config.insertion.max_retries >= 0
        assert config.hnsw.m > 0
        assert config.circuit_breaker.failure_threshold > 0
