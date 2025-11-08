"""Type stubs for cross-encoder reranking system.

Provides complete type definitions for cross-encoder model loading, candidate
selection, pair scoring, and result reranking with strict type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from src.search.results import SearchResult
from src.search.reranker_protocol import Reranker

# Type aliases
RerankerDevice = Literal["auto", "cuda", "cpu"]
QueryType = Literal["short", "medium", "long", "complex"]


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Device for inference ("auto", "cpu", or "cuda").
        batch_size: Batch size for pair scoring.
        min_confidence: Minimum confidence threshold for results.
        top_k: Default number of results to return.
        base_pool_size: Base number of candidates to select.
        max_pool_size: Maximum pool size regardless of query.
        adaptive_sizing: Whether to adapt pool size based on query.
        complexity_constants: Tunable constants for complexity calculation.
    """

    model_name: str
    device: Optional[str]
    batch_size: int
    min_confidence: float
    top_k: int
    base_pool_size: int
    max_pool_size: int
    adaptive_sizing: bool
    complexity_constants: dict[str, float]

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        ...


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for adaptive candidate selection.

    Attributes:
        length: Number of characters in query.
        complexity: Complexity score from 0-1 (higher = more complex).
        query_type: Classification of query type.
        keyword_count: Number of distinct keywords identified.
        has_operators: Whether query contains boolean operators.
        has_quotes: Whether query contains quoted phrases.
    """
    length: int
    complexity: float
    query_type: QueryType
    keyword_count: int
    has_operators: bool
    has_quotes: bool

class CandidateSelector:
    """Adaptive candidate selection for cross-encoder reranking.

    Analyzes query characteristics and adapts pool size for reranking
    based on query complexity and result set size.
    """

    def __init__(
        self,
        base_pool_size: int = 25,
        max_pool_size: int = 100,
        complexity_multiplier: float = 1.2,
    ) -> None:
        """Initialize candidate selector with pool sizing parameters.

        Args:
            base_pool_size: Base number of candidates to select (default: 25).
            max_pool_size: Maximum pool size regardless of query (default: 100).
            complexity_multiplier: Multiplier for complex queries (default: 1.2).
        """
        ...

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query characteristics for adaptive pool sizing.

        Performs heuristic analysis including:
        - Character count for length
        - Keyword extraction for keyword_count
        - Boolean operators detection (AND, OR, NOT)
        - Quoted phrase detection
        - Complexity calculation from keyword density and structure

        Args:
            query: User search query string.

        Returns:
            QueryAnalysis with detected characteristics.
        """
        ...

    def calculate_pool_size(
        self,
        query_analysis: QueryAnalysis,
        available_results: int,
    ) -> int:
        """Calculate adaptive pool size based on query and result characteristics.

        Uses formula: base_pool * (1 + complexity * multiplier)
        Capped by max_pool_size and available_results.

        Args:
            query_analysis: Results from analyze_query().
            available_results: Number of results available for selection.

        Returns:
            Recommended pool size for reranking.
        """
        ...

    def select(
        self,
        results: list[SearchResult],
        pool_size: int | None = None,
        query: str | None = None,
    ) -> list[SearchResult]:
        """Select top candidates for reranking.

        If pool_size not provided, calculates adaptively from query if available.
        Returns top-K results by hybrid_score, preserving all metadata.

        Args:
            results: Results from hybrid search.
            pool_size: Number of candidates to select (optional).
            query: Query for adaptive sizing (optional, used if pool_size None).

        Returns:
            Selected candidates in original ranked order.

        Raises:
            ValueError: If pool_size > available results.
        """
        ...

class CrossEncoderReranker:
    """Cross-encoder reranking system implementing Reranker protocol.

    Loads HuggingFace cross-encoder model for pair-wise relevance scoring,
    implements batch inference for efficiency, and provides top-K selection
    from adaptive candidate pools.

    Implements the Reranker protocol for composability with HybridSearch
    and other search systems.

    Attributes:
        config: RerankerConfig with all settings.
        model: Loaded cross-encoder model instance.
        candidate_selector: CandidateSelector for adaptive pool sizing.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model_factory: Optional[Callable[[str, str], Any]] = None,
        # Backward compatibility parameters (deprecated, use config instead)
        model_name: Optional[str] = None,
        device: Optional[RerankerDevice] = None,
        batch_size: Optional[int] = None,
        max_pool_size: Optional[int] = None,
    ) -> None:
        """Initialize cross-encoder reranker.

        Supports both new config-based initialization and legacy parameter-based
        initialization for backward compatibility.

        Args:
            config: RerankerConfig with all settings (recommended).
                If provided, model_name/device/batch_size ignored.
            model_factory: Optional callable to load model (for testing).
                Signature: (model_name: str, device: str) -> model_instance
                Default: Uses HuggingFace CrossEncoder.
            model_name: HuggingFace model identifier (legacy, use config.model_name).
            device: Device for inference (legacy, use config.device).
            batch_size: Batch size for pair scoring (legacy, use config.batch_size).
            max_pool_size: Maximum candidates (legacy, use config.max_pool_size).

        Raises:
            ValueError: If configuration invalid.
            ImportError: If required dependencies missing.

        Example:
            >>> # New approach (recommended)
            >>> config = RerankerConfig(
            ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ...     device="auto",
            ...     batch_size=32,
            ... )
            >>> reranker = CrossEncoderReranker(config=config)
            >>>
            >>> # Old approach (backward compatible)
            >>> reranker = CrossEncoderReranker(
            ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ...     device="auto",
            ... )
        """
        ...

    def load_model(self) -> None:
        """Load and initialize cross-encoder model from HuggingFace.

        Sets device to GPU if available and CUDA-enabled, otherwise CPU.
        Performs warmup inference to optimize GPU caching.

        Raises:
            ImportError: If transformers not installed.
            RuntimeError: If model download/loading fails.
        """
        ...

    def score_pairs(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[float]:
        """Score query-document pairs using cross-encoder model.

        Creates pairs from query + candidate snippets and performs batch inference.
        Scores normalized to 0-1 range via sigmoid activation.

        Args:
            query: User search query.
            candidates: List of candidate documents to score.

        Returns:
            List of confidence scores (0-1) for each candidate.

        Raises:
            ValueError: If candidates list empty.
            RuntimeError: If model inference fails.
        """
        ...

    def rerank(
        self,
        query: str,
        search_results: list[SearchResult],
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder with adaptive candidate selection.

        Full pipeline:
        1. Analyze query for adaptive pool sizing
        2. Select top-K candidates from search results
        3. Score pairs using cross-encoder
        4. Select top-5 by confidence
        5. Return with updated scores and metadata

        Args:
            query: User search query.
            search_results: Results from hybrid search.
            top_k: Number of results to return (default: 5).
            min_confidence: Minimum confidence threshold (default: 0.0).

        Returns:
            Top-K reranked results with confidence scores, sorted by confidence DESC.

        Raises:
            ValueError: If search_results empty or invalid.
            RuntimeError: If reranking pipeline fails.
        """
        ...

    def get_device(self) -> str:
        """Get current device (cuda or cpu).

        Returns:
            Device identifier string ('cuda', 'cpu', etc).
        """
        ...

    def is_model_loaded(self) -> bool:
        """Check if cross-encoder model is loaded and ready.

        Returns:
            True if model initialized, False otherwise.
        """
        ...
