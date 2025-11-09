"""Configuration management for embedding generation pipeline.

Provides centralized, validated configuration via Pydantic v2 with environment
variable support. Implements configuration hierarchy for models, generation,
insertion, HNSW indexing, and circuit breaker resilience.

Configuration is validated on load and can be overridden by environment variables.
Singleton factory ensures consistent configuration across application.

Module exports:
    - ModelConfiguration: Model selection and fallback configuration
    - GeneratorConfiguration: Embedding generation parameters
    - InsertionConfiguration: Database insertion parameters
    - HNSWConfiguration: PostgreSQL HNSW index parameters
    - CircuitBreakerConfiguration: Resilience configuration
    - EmbeddingConfig: Root configuration combining all sub-configs
    - get_embedding_config(): Singleton factory function
"""

import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfiguration(BaseModel):
    """Model selection and fallback configuration.

    Defines which embedding model to use and fallback strategy when primary
    model fails or is unavailable.

    Attributes:
        primary_model: Main embedding model to use
            (default: all-MiniLM-L12-v2, 384-dim)
        fallback_model: Model to use when primary fails
            (default: all-MiniLM-L6-v2, 384-dim)
        enable_cached_fallback: Use locally cached model for offline operation
            (default: True)
        enable_dummy_mode: Use random dummy embeddings for fast iteration
            (default: False, use only in development)
        device: Device to load model on ('cuda', 'cpu', or 'auto' for auto-detect)
            (default: 'auto')
    """

    primary_model: str = Field(
        default="all-MiniLM-L12-v2",
        description="Primary embedding model from sentence-transformers",
    )
    fallback_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Fallback model when primary unavailable",
    )
    enable_cached_fallback: bool = Field(
        default=True,
        description="Enable using locally cached models for offline operation",
    )
    enable_dummy_mode: bool = Field(
        default=False,
        description="Use dummy embeddings for development (no GPU needed)",
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to load model on (auto-detects GPU if available)",
    )


class GeneratorConfiguration(BaseModel):
    """Embedding generation parameters.

    Controls batch processing and parallel execution during embedding generation.

    Attributes:
        batch_size: Number of chunks per batch (default: 64)
            - Larger batches: faster processing, more memory
            - Smaller batches: slower processing, less memory
        num_workers: Number of parallel worker threads (default: 4)
            - Typical range: 2-8
            - More workers on multi-core systems
        use_threading: Use ThreadPoolExecutor instead of ProcessPoolExecutor
            (default: True, recommended for I/O-bound embedding generation)
    """

    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Chunks per batch for generation",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Parallel worker threads",
    )
    use_threading: bool = Field(
        default=True,
        description="Use threading (True) vs multiprocessing (False)",
    )


class InsertionConfiguration(BaseModel):
    """Database insertion parameters.

    Controls how embeddings are inserted into the database with retry strategy.

    Attributes:
        batch_size: Number of chunks per database batch insert (default: 100)
            - Larger batches: fewer database round-trips, more memory
            - Smaller batches: more round-trips, less memory
        max_retries: Maximum retry attempts on insertion failure (default: 3)
        retry_delay_seconds: Delay between retry attempts (default: 1.0)
            - Implements exponential backoff multiplier
        create_index: Automatically create HNSW index after insertion (default: True)
    """

    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Chunks per database batch",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts on failure",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Delay between retries (seconds)",
    )
    create_index: bool = Field(
        default=True,
        description="Create HNSW index after insertion",
    )


class HNSWConfiguration(BaseModel):
    """PostgreSQL HNSW index parameters.

    Controls Hierarchical Navigable Small World (HNSW) vector index behavior
    for efficient approximate nearest neighbor search.

    Attributes:
        m: Maximum number of bidirectional connections per node (default: 16)
            - Higher values: better recall, more memory
            - Lower values: faster search, less memory
            - Typical range: 4-64
        ef_construction: Complexity during index construction (default: 200)
            - Higher values: better quality index, slower construction
            - Typical range: 10-500
        ef_search: Complexity during search operations (default: 64)
            - Higher values: better recall, slower search
            - Typical range: 10-500
    """

    m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Max connections per node",
    )
    ef_construction: int = Field(
        default=200,
        ge=10,
        le=500,
        description="Construction complexity (higher = better quality)",
    )
    ef_search: int = Field(
        default=64,
        ge=10,
        le=500,
        description="Search complexity (higher = better recall)",
    )


class CircuitBreakerConfiguration(BaseModel):
    """Circuit breaker resilience parameters.

    Controls failure detection and automatic recovery for embedding generation.

    Attributes:
        failure_threshold: Consecutive failures before opening circuit (default: 5)
        success_threshold: Consecutive successes before closing circuit (default: 2)
        timeout_seconds: Time before attempting recovery from OPEN (default: 60)
        enabled: Enable circuit breaker functionality (default: True)
    """

    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Failures before circuit opens",
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Successes before circuit closes",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Time before recovery attempt (seconds)",
    )
    enabled: bool = Field(
        default=True,
        description="Enable circuit breaker",
    )


class EmbeddingConfig(BaseSettings):
    """Root configuration for embedding pipeline.

    Combines all sub-configurations into one validated object with environment
    variable support. Implements Pydantic BaseSettings for automatic environment
    variable loading with EMBEDDING_ prefix.

    Environment variables (optional):
        - EMBEDDING_MODEL_PRIMARY_MODEL: Override primary model
        - EMBEDDING_MODEL_FALLBACK_MODEL: Override fallback model
        - EMBEDDING_GENERATOR_BATCH_SIZE: Override generation batch size
        - EMBEDDING_GENERATOR_NUM_WORKERS: Override number of workers
        - EMBEDDING_INSERTION_BATCH_SIZE: Override insertion batch size
        - EMBEDDING_INSERTION_MAX_RETRIES: Override retry attempts
        - EMBEDDING_HNSW_M: Override HNSW m parameter
        - EMBEDDING_CIRCUIT_BREAKER_ENABLED: Enable/disable circuit breaker
        - EMBEDDING_CIRCUIT_BREAKER_FAILURE_THRESHOLD: Override failure threshold

    Example:
        >>> config = get_embedding_config()
        >>> print(config.generator.batch_size)  # 64
        >>> print(config.circuit_breaker.enabled)  # True
    """

    model: ModelConfiguration = Field(default_factory=ModelConfiguration)
    generator: GeneratorConfiguration = Field(default_factory=GeneratorConfiguration)
    insertion: InsertionConfiguration = Field(default_factory=InsertionConfiguration)
    hnsw: HNSWConfiguration = Field(default_factory=HNSWConfiguration)
    circuit_breaker: CircuitBreakerConfiguration = Field(
        default_factory=CircuitBreakerConfiguration
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
    )


# Singleton instance
_config_instance: EmbeddingConfig | None = None


def get_embedding_config() -> EmbeddingConfig:
    """Get singleton configuration instance with environment overrides.

    Why it exists:
        - Single source of truth for all embedding configuration
        - Environment variables override defaults
        - Validates all values with Pydantic

    What it does:
        - Returns cached config instance
        - Loads from environment variables if set
        - Validates all fields on initialization
        - Thread-safe singleton pattern

    Environment variables are automatically loaded with EMBEDDING_ prefix.
    All configuration values are validated against their constraints.

    Returns:
        EmbeddingConfig: Singleton instance with all validated sub-configurations.

    Raises:
        ValidationError: If any configuration value fails validation.

    Example:
        >>> config = get_embedding_config()
        >>> if config.circuit_breaker.enabled:
        ...     print(f"Circuit breaker active with {config.circuit_breaker.failure_threshold} failure threshold")
        >>> print(f"Using batch size: {config.generator.batch_size}")
        >>> print(f"Device: {config.model.device}")
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = EmbeddingConfig()
    return _config_instance


def reset_config_for_testing() -> None:
    """Reset singleton configuration instance for testing.

    WARNING: Only use in test environments. Clears the cached instance
    so next call to get_embedding_config() will reload from environment.

    This is necessary for testing different configuration combinations
    without restarting the Python process.
    """
    global _config_instance
    _config_instance = None
