"""Cross-encoder reranking system for search result refinement.

Implements cross-encoder pair-wise relevance scoring using the ms-marco-MiniLM-L-6-v2
model from HuggingFace transformers library. Provides:

- Model loading and initialization with GPU/CPU device detection
- Adaptive candidate pool sizing based on query characteristics
- Batch pair scoring with efficiency optimizations
- Top-K selection with confidence filtering
- Full integration with HybridSearch results

Performance targets:
- Model loading: <5 seconds
- Batch inference: <100ms for 50 pairs
- Pool size calculation: <1ms
- Overall reranking: <200ms (with batching)

Example:
    >>> from src.search.cross_encoder_reranker import CrossEncoderReranker
    >>> from src.search.hybrid_search import HybridSearch
    >>>
    >>> # Initialize reranker
    >>> reranker = CrossEncoderReranker(device="auto", batch_size=32)
    >>>
    >>> # Get results from hybrid search
    >>> hybrid = HybridSearch(db_pool, settings, logger)
    >>> results = hybrid.search("authentication best practices", top_k=50)
    >>>
    >>> # Rerank to top-5
    >>> reranked = reranker.rerank("authentication best practices", results, top_k=5)
    >>> for result in reranked:
    ...     print(f"{result.rank}. {result.source_file} (confidence: {result.confidence:.3f})")
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from src.core.logging import StructuredLogger
from src.search.results import SearchResult

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Type aliases
RerankerDevice = Literal["auto", "cuda", "cpu"]
QueryType = Literal["short", "medium", "long", "complex"]


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker.

    Provides centralized configuration management for CrossEncoderReranker,
    enabling flexible initialization and testability through parameter tuning.

    Attributes:
        model_name: HuggingFace model identifier.
            Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        device: Device for inference - "auto" (auto-detect), "cpu", or "cuda".
            Default: "auto"
        batch_size: Batch size for pair scoring.
            Default: 32
        min_confidence: Minimum confidence threshold for results (0-1).
            Default: 0.0
        top_k: Default number of results to return.
            Default: 5
        base_pool_size: Base number of candidates to select.
            Default: 50
        max_pool_size: Maximum pool size regardless of query.
            Default: 100
        adaptive_sizing: Whether to adapt pool size based on query complexity.
            Default: True
        complexity_constants: Tunable constants for complexity calculation.
            Allows fine-tuning of query complexity scoring algorithm.
            Default: Standard weights for keyword count, operators, quotes

    Example:
        >>> # Default configuration
        >>> config = RerankerConfig()
        >>>
        >>> # Custom configuration
        >>> config = RerankerConfig(
        ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ...     device="cuda",
        ...     batch_size=64,
        ...     max_pool_size=150,
        ... )
        >>> config.validate()  # Validate before use
    """

    # Model configuration
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = "auto"
    batch_size: int = 32

    # Ranking configuration
    min_confidence: float = 0.0
    top_k: int = 5

    # Pool sizing configuration
    base_pool_size: int = 50
    max_pool_size: int = 100
    adaptive_sizing: bool = True

    # Complexity calculation (tuning)
    complexity_constants: dict[str, float] = field(
        default_factory=lambda: {
            "keyword_normalization": 10.0,
            "keyword_weight": 0.6,
            "operator_bonus": 0.2,
            "quote_bonus": 0.2,
        }
    )

    def validate(self) -> None:
        """Validate configuration values.

        Checks that all configuration values are within valid ranges and
        consistent with each other.

        Raises:
            ValueError: If any configuration value is invalid:
                - batch_size < 1
                - min_confidence not in [0, 1]
                - top_k < 1
                - base_pool_size < top_k
                - max_pool_size < base_pool_size
                - device not in ["auto", "cpu", "cuda", None]
                - Any complexity_constant < 0
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.base_pool_size < self.top_k:
            raise ValueError(
                f"base_pool_size ({self.base_pool_size}) must be >= "
                f"top_k ({self.top_k})"
            )
        if self.max_pool_size < self.base_pool_size:
            raise ValueError(
                f"max_pool_size ({self.max_pool_size}) must be >= "
                f"base_pool_size ({self.base_pool_size})"
            )
        if self.device not in ("auto", "cpu", "cuda", None):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda', or None, "
                f"got {self.device}"
            )
        for key, value in self.complexity_constants.items():
            if value < 0:
                raise ValueError(
                    f"complexity_constants['{key}'] must be >= 0, got {value}"
                )


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for adaptive candidate selection.

    Performs heuristic analysis to understand query complexity and adapt
    the candidate pool size accordingly.

    Attributes:
        length: Number of characters in query.
        complexity: Complexity score from 0-1 (higher = more complex).
        query_type: Classification of query type (short/medium/long/complex).
        keyword_count: Number of distinct keywords identified.
        has_operators: Whether query contains boolean operators (AND, OR, NOT).
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

    Analyzes query characteristics and adapts the pool size for reranking
    based on query complexity and result set size. Uses heuristics to
    estimate optimal candidate count for scoring.

    Complexity Calculation (Magic Numbers Extracted to Constants):
    - Complexity formula: min(MAX_COMPLEXITY, (kw_count / KEYWORD_NORMALIZATION_FACTOR) *
      KEYWORD_COMPLEXITY_WEIGHT + OPERATOR_COMPLEXITY_BONUS + QUOTE_COMPLEXITY_BONUS)
    - These constants are extracted from literal values in analyze_query()
    - Can be customized via RerankerConfig.complexity_constants

    Attributes:
        base_pool_size: Base number of candidates (default: 25).
        max_pool_size: Maximum pool size (default: 100).
        complexity_multiplier: Multiplier for complex queries (default: 1.2).
    """

    # Complexity Calculation Constants
    # These define how query complexity is scored (0-1 range)
    KEYWORD_NORMALIZATION_FACTOR: float = 10.0  # Normalize keyword count
    KEYWORD_COMPLEXITY_WEIGHT: float = 0.6      # Contribution of keywords to score
    OPERATOR_COMPLEXITY_BONUS: float = 0.2      # Bonus for boolean operators
    QUOTE_COMPLEXITY_BONUS: float = 0.2         # Bonus for quoted phrases
    MAX_COMPLEXITY: float = 1.0                  # Clamp complexity to [0, 1]

    # Length thresholds for query type classification
    SHORT_QUERY_THRESHOLD: int = 15     # Queries < 15 chars = "short"
    MEDIUM_QUERY_THRESHOLD: int = 50    # Queries < 50 chars = "medium"
    LONG_QUERY_THRESHOLD: int = 100     # Queries < 100 chars = "long"
    # Queries >= 100 chars = "complex"

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

        Raises:
            ValueError: If parameters out of valid range.
        """
        if base_pool_size < 5:
            raise ValueError(f"base_pool_size must be >= 5, got {base_pool_size}")
        if max_pool_size < base_pool_size:
            raise ValueError(
                f"max_pool_size must be >= base_pool_size, "
                f"got {max_pool_size} < {base_pool_size}"
            )
        if complexity_multiplier < 1.0:
            raise ValueError(
                f"complexity_multiplier must be >= 1.0, got {complexity_multiplier}"
            )

        self.base_pool_size = base_pool_size
        self.max_pool_size = max_pool_size
        self.complexity_multiplier = complexity_multiplier

        logger.info(
            f"CandidateSelector initialized: "
            f"base={base_pool_size}, max={max_pool_size}, "
            f"multiplier={complexity_multiplier:.2f}"
        )

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query characteristics for adaptive pool sizing.

        Performs heuristic analysis including:
        - Character count for length measurement
        - Keyword extraction via whitespace and special characters
        - Boolean operators detection (AND, OR, NOT, AND NOT)
        - Quoted phrase detection using regex
        - Complexity calculation from keyword density

        Args:
            query: User search query string.

        Returns:
            QueryAnalysis with detected characteristics.

        Raises:
            ValueError: If query is empty or None.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Basic metrics
        length: int = len(query)
        words: list[str] = query.lower().split()
        keyword_count: int = len(words)

        # Detect operators
        has_operators: bool = bool(
            re.search(r"\b(and|or|not|and\s+not)\b", query.lower())
        )

        # Detect quoted phrases
        has_quotes: bool = bool(re.search(r'"[^"]*"', query))

        # Complexity calculation: higher keyword count and operators = more complex
        # Range: 0.0-1.0
        # Uses extracted constants for tunability and maintainability
        complexity: float = min(
            self.MAX_COMPLEXITY,
            (keyword_count / self.KEYWORD_NORMALIZATION_FACTOR)
            * self.KEYWORD_COMPLEXITY_WEIGHT
            + (self.OPERATOR_COMPLEXITY_BONUS if has_operators else 0.0)
            + (self.QUOTE_COMPLEXITY_BONUS if has_quotes else 0.0),
        )

        # Query type classification using extracted constants
        if length < self.SHORT_QUERY_THRESHOLD:
            query_type: QueryType = "short"
        elif length < self.MEDIUM_QUERY_THRESHOLD:
            query_type = "medium"
        elif length < self.LONG_QUERY_THRESHOLD:
            query_type = "long"
        else:
            query_type = "complex"

        analysis = QueryAnalysis(
            length=length,
            complexity=complexity,
            query_type=query_type,
            keyword_count=keyword_count,
            has_operators=has_operators,
            has_quotes=has_quotes,
        )

        logger.debug(
            f"Query analysis: type={query_type}, complexity={complexity:.2f}, "
            f"keywords={keyword_count}, operators={has_operators}, quotes={has_quotes}"
        )

        return analysis

    def calculate_pool_size(
        self,
        query_analysis: QueryAnalysis,
        available_results: int,
    ) -> int:
        """Calculate adaptive pool size based on query and result characteristics.

        Uses formula: pool_size = base_pool * (1 + complexity * multiplier)
        - Capped by max_pool_size
        - Capped by available_results
        - Minimum 5 to ensure meaningful scoring

        Args:
            query_analysis: Results from analyze_query().
            available_results: Number of results available for selection.

        Returns:
            Recommended pool size for reranking.

        Raises:
            ValueError: If available_results < 1.
        """
        if available_results < 1:
            raise ValueError(f"available_results must be >= 1, got {available_results}")

        # Base calculation with complexity bonus
        complexity_bonus: float = query_analysis.complexity * self.complexity_multiplier
        pool_size: int = int(self.base_pool_size * (1.0 + complexity_bonus))

        # Apply caps
        pool_size = min(pool_size, self.max_pool_size)
        pool_size = min(pool_size, available_results)
        pool_size = max(pool_size, 5)  # Minimum for meaningful results

        logger.debug(
            f"Pool sizing: complexity={query_analysis.complexity:.2f}, "
            f"available={available_results}, calculated={pool_size}"
        )

        return pool_size

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
            ValueError: If results empty, pool_size > available results, or query invalid.
        """
        if not results:
            raise ValueError("Results list cannot be empty")

        # Determine pool size
        if pool_size is None:
            if query is None:
                # No guidance, use base pool size
                pool_size = self.base_pool_size
            else:
                # Analyze query and calculate adaptively
                analysis = self.analyze_query(query)
                pool_size = self.calculate_pool_size(analysis, len(results))

        # Validate pool size
        if pool_size > len(results):
            raise ValueError(
                f"pool_size ({pool_size}) cannot exceed available results ({len(results)})"
            )

        # Select top-K by hybrid score
        selected = sorted(
            results,
            key=lambda r: r.hybrid_score,
            reverse=True,
        )[:pool_size]

        logger.debug(
            f"Selected {len(selected)} candidates from {len(results)} results "
            f"(pool_size={pool_size})"
        )

        return selected


class CrossEncoderReranker:
    """Cross-encoder reranking system implementing Reranker protocol.

    Loads HuggingFace cross-encoder model for pair-wise relevance scoring,
    implements batch inference for efficiency, and provides top-K selection
    from adaptive candidate pools.

    Implements the Reranker protocol for composability with HybridSearch
    and other search systems.

    The ms-marco-MiniLM-L-6-v2 model is a lightweight cross-encoder trained on
    the MS MARCO dataset with 6 layers and 384 hidden dimensions. It provides
    fast, accurate pair-wise relevance scoring suitable for reranking search results.

    **Configuration**: Two initialization approaches:
    1. **Recommended (New)**: Pass RerankerConfig for all settings
    2. **Legacy (Backward Compatible)**: Pass individual parameters

    **Extensibility**: Supports dependency injection of custom model factory
    for testing with different models or custom loading logic.

    **Thread Safety**: Model loading is not thread-safe. Call load_model()
    before concurrent access in multi-threaded environments.

    Attributes:
        config: RerankerConfig with all settings.
        model: Loaded cross-encoder model instance.
        candidate_selector: CandidateSelector instance for adaptive pool sizing.
        model_factory: Callable to load model (defaults to HuggingFace loader).
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model_factory: Optional[Callable[[str, str], Any]] = None,
        # Backward compatibility parameters (deprecated)
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
                If provided, other parameters are ignored.
            model_factory: Optional callable to load model.
                Signature: (model_name: str, device: str) -> model_instance
                Default: Uses HuggingFace CrossEncoder.
                Useful for testing with mock models.
            model_name: HuggingFace model identifier (legacy, use config.model_name).
                Ignored if config provided.
            device: Device for inference (legacy, use config.device).
                Ignored if config provided.
            batch_size: Batch size for pair scoring (legacy, use config.batch_size).
                Ignored if config provided.
            max_pool_size: Maximum candidates (legacy, use config.max_pool_size).
                Ignored if config provided.

        Raises:
            ValueError: If configuration invalid.

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
            >>>
            >>> # With dependency injection for testing
            >>> def mock_factory(name: str, device: str) -> Any:
            ...     return MockCrossEncoder()
            >>> reranker = CrossEncoderReranker(
            ...     config=config,
            ...     model_factory=mock_factory,
            ... )
        """
        # Build config from provided config or legacy parameters
        if config is None:
            # Check if any legacy parameters provided
            if any(p is not None for p in [model_name, device, batch_size, max_pool_size]):
                # Build config from legacy parameters
                config = RerankerConfig(
                    model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device=device or "auto",
                    batch_size=batch_size or 32,
                    max_pool_size=max_pool_size or 100,
                )
            else:
                # Use default config
                config = RerankerConfig()

        # Validate configuration
        config.validate()
        self.config = config

        # Store model factory (dependency injection)
        self.model_factory = model_factory or self._default_model_factory

        # Initialize model as None (lazy loading)
        self.model: Any = None

        # Initialize device resolution
        self._actual_device: str = ""
        device_to_resolve: RerankerDevice = config.device or "auto"  # type: ignore
        if device_to_resolve not in ("auto", "cuda", "cpu"):
            device_to_resolve = "auto"
        self._resolve_device(device_to_resolve)

        # Create candidate selector from config
        self.candidate_selector = CandidateSelector(
            base_pool_size=config.base_pool_size,
            max_pool_size=config.max_pool_size,
            complexity_multiplier=1.2,
        )

        logger.info(
            f"CrossEncoderReranker initialized: model={config.model_name}, "
            f"device={self._actual_device}, batch_size={config.batch_size}"
        )

    @staticmethod
    def _default_model_factory(model_name: str, device: str) -> Any:
        """Default model loading using HuggingFace CrossEncoder.

        Args:
            model_name: HuggingFace model identifier.
            device: Device for inference ("cuda" or "cpu").

        Returns:
            Loaded CrossEncoder model instance.

        Raises:
            ImportError: If sentence-transformers not installed.
            RuntimeError: If model loading fails.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            logger.error(
                "Failed to import sentence_transformers. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking"
            ) from e

        return CrossEncoder(model_name, device=device)

    def _resolve_device(self, device: RerankerDevice) -> None:
        """Resolve device specification to actual device.

        Maps 'auto' to 'cuda' if available, otherwise 'cpu'.

        Args:
            device: Device specification ('auto', 'cuda', or 'cpu').

        Raises:
            ValueError: If device specification invalid.
        """
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got {device}")

        if device == "auto":
            # Try to detect GPU availability
            try:
                import torch
                self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._actual_device = "cpu"
        else:
            self._actual_device = device

        logger.debug(f"Device resolved: {device} -> {self._actual_device}")

    def load_model(self) -> None:
        """Load and initialize cross-encoder model using configured factory.

        Uses the configured model_factory to load the model (default: HuggingFace).
        Performs warmup inference to optimize GPU caching.

        **Lazy Loading Pattern**: Model is not loaded in __init__(). Call this
        method explicitly when ready to perform inference. This allows creation
        of reranker instances without GPU memory overhead until first use.

        **Dependency Injection**: Custom model factories can be provided for:
        - Testing with mock models
        - Loading from alternative sources
        - Custom model configuration

        Raises:
            ImportError: If dependencies not installed.
            RuntimeError: If model loading fails.

        Example:
            >>> reranker = CrossEncoderReranker()  # No model loaded yet
            >>> # Later, when ready to rerank:
            >>> reranker.load_model()  # Load model into GPU/CPU
            >>> results = reranker.rerank(query, candidates)
        """
        try:
            # Load model using configured factory
            logger.info(f"Loading cross-encoder model: {self.config.model_name}")
            self.model = self.model_factory(
                self.config.model_name, self._actual_device
            )
            logger.info(f"Model loaded successfully on device: {self._actual_device}")

            # Warmup inference for GPU optimization
            if self._actual_device == "cuda":
                logger.debug("Performing warmup inference for GPU optimization")
                try:
                    # Suppress warnings during warmup
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = self.model.predict([["test query", "test document"]])
                    logger.debug("Warmup inference completed")
                except Exception as warmup_err:
                    logger.warning(f"Warmup inference failed (non-fatal): {warmup_err}")

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise RuntimeError(
                f"Failed to load model {self.config.model_name}: {e}"
            ) from e

    def score_pairs(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[float]:
        """Score query-document pairs using cross-encoder model.

        Creates pairs from query + candidate snippets and performs batch inference.
        Uses first 512 characters of each document to stay within model limits.
        Scores normalized to 0-1 range via sigmoid activation.

        Args:
            query: User search query.
            candidates: List of candidate documents to score.

        Returns:
            List of confidence scores (0-1) for each candidate in same order.

        Raises:
            ValueError: If candidates list empty or model not loaded.
            RuntimeError: If model inference fails.
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Prepare pairs: [query, document] for each candidate
            pairs: list[list[str]] = [
                [query, candidate.chunk_text[:512]]
                for candidate in candidates
            ]

            # Perform batch scoring
            logger.debug(
                f"Scoring {len(pairs)} pairs with batch_size={self.config.batch_size}"
            )
            raw_scores: list[float] = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Normalize scores to 0-1 via sigmoid
            # CrossEncoder outputs logits, sigmoid converts to probability
            import numpy as np
            normalized_scores: list[float] = [
                float(1.0 / (1.0 + np.exp(-score)))
                for score in raw_scores
            ]

            logger.debug(
                f"Scoring complete: min={min(normalized_scores):.3f}, "
                f"max={max(normalized_scores):.3f}, "
                f"avg={sum(normalized_scores) / len(normalized_scores):.3f}"
            )

            return normalized_scores

        except Exception as e:
            logger.error(f"Failed to score pairs: {e}")
            raise RuntimeError(f"Cross-encoder scoring failed: {e}") from e

    def rerank(
        self,
        query: str,
        search_results: list[SearchResult],
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder with adaptive candidate selection.

        Full pipeline:
        1. Validate inputs
        2. Select adaptive candidate pool based on query
        3. Score all pairs using cross-encoder
        4. Filter by confidence threshold
        5. Select top-K by confidence
        6. Rerank and update metadata
        7. Return with confidence scores

        Args:
            query: User search query.
            search_results: Results from hybrid search.
            top_k: Number of results to return (default: 5).
            min_confidence: Minimum confidence threshold (default: 0.0).

        Returns:
            Top-K reranked results with confidence scores, sorted by confidence DESC.

        Raises:
            ValueError: If search_results empty, invalid parameters, or model not loaded.
            RuntimeError: If reranking pipeline fails.
        """
        if not search_results:
            raise ValueError("search_results cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be 0-1, got {min_confidence}")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(
            f"Reranking {len(search_results)} results with query: {query[:50]}..."
        )

        try:
            # Step 1: Select candidates adaptively
            candidates = self.candidate_selector.select(
                search_results,
                query=query,
            )
            logger.debug(f"Selected {len(candidates)} candidates for reranking")

            # Step 2: Score pairs
            confidence_scores = self.score_pairs(query, candidates)

            # Step 3: Combine scores with results and filter
            scored_results: list[tuple[SearchResult, float]] = [
                (result, score)
                for result, score in zip(candidates, confidence_scores)
                if score >= min_confidence
            ]

            # Step 4: Sort by confidence descending
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Step 5: Take top-K and rerank
            reranked: list[SearchResult] = []
            for rank, (result, confidence) in enumerate(scored_results[:top_k], 1):
                # Create new result with updated score and confidence
                updated_result = SearchResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    similarity_score=result.similarity_score,
                    bm25_score=result.bm25_score,
                    hybrid_score=confidence,  # Use confidence as hybrid_score
                    rank=rank,
                    score_type="cross_encoder",
                    source_file=result.source_file,
                    source_category=result.source_category,
                    document_date=result.document_date,
                    context_header=result.context_header,
                    chunk_index=result.chunk_index,
                    total_chunks=result.total_chunks,
                    chunk_token_count=result.chunk_token_count,
                    metadata=result.metadata,
                    highlighted_context=result.highlighted_context,
                    confidence=confidence,
                )
                reranked.append(updated_result)

            logger.info(
                f"Reranking complete: returned {len(reranked)} results "
                f"(candidates={len(candidates)}, scored={len(scored_results)}, top_k={top_k})"
            )

            return reranked

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Reranking pipeline failed: {e}")
            raise RuntimeError(f"Cross-encoder reranking failed: {e}") from e

    def get_device(self) -> str:
        """Get current device (cuda or cpu).

        Returns:
            Device identifier string ('cuda', 'cpu', etc).
        """
        return self._actual_device

    def is_model_loaded(self) -> bool:
        """Check if cross-encoder model is loaded and ready.

        Returns:
            True if model initialized, False otherwise.
        """
        return self.model is not None
