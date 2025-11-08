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
from dataclasses import dataclass
from typing import Any, Literal

from src.core.logging import StructuredLogger
from src.search.results import SearchResult

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Type aliases
RerankerDevice = Literal["auto", "cuda", "cpu"]
QueryType = Literal["short", "medium", "long", "complex"]


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

    Attributes:
        base_pool_size: Base number of candidates (default: 25).
        max_pool_size: Maximum pool size (default: 100).
        complexity_multiplier: Multiplier for complex queries (default: 1.2).
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
        complexity: float = min(
            1.0,
            (keyword_count / 10.0) * 0.6 + (0.2 if has_operators else 0.0) +
            (0.2 if has_quotes else 0.0)
        )

        # Query type classification
        if length < 15:
            query_type: QueryType = "short"
        elif length < 50:
            query_type = "medium"
        elif length < 100:
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
    """Cross-encoder reranking system using ms-marco-MiniLM-L-6-v2 model.

    Loads HuggingFace cross-encoder model for pair-wise relevance scoring,
    implements batch inference for efficiency, and provides top-5 selection
    from adaptive candidate pools.

    The ms-marco-MiniLM-L-6-v2 model is a lightweight cross-encoder trained on
    the MS MARCO dataset with 6 layers and 384 hidden dimensions. It provides
    fast, accurate pair-wise relevance scoring suitable for reranking search results.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Current device for inference (cuda or cpu).
        batch_size: Batch size for pair scoring.
        model: Loaded cross-encoder model instance.
        candidate_selector: CandidateSelector instance for adaptive pool sizing.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: RerankerDevice = "auto",
        batch_size: int = 32,
        max_pool_size: int = 100,
    ) -> None:
        """Initialize cross-encoder reranker.

        Validates parameters and creates candidate selector. Model loading
        deferred to load_model() for explicit control.

        Args:
            model_name: HuggingFace model identifier
                (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
            device: Device for inference - 'auto' uses GPU if available
                (default: 'auto').
            batch_size: Batch size for pair scoring (default: 32).
            max_pool_size: Maximum candidates to rerank (default: 100).

        Raises:
            ValueError: If batch_size or max_pool_size invalid.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_pool_size < 5:
            raise ValueError(f"max_pool_size must be >= 5, got {max_pool_size}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model: Any = None
        self.device_name: str = device
        self._actual_device: str = ""

        # Initialize device
        self._resolve_device(device)

        # Create candidate selector
        self.candidate_selector = CandidateSelector(
            base_pool_size=25,
            max_pool_size=max_pool_size,
            complexity_multiplier=1.2,
        )

        logger.info(
            f"CrossEncoderReranker initialized: model={model_name}, "
            f"device={self._actual_device}, batch_size={batch_size}"
        )

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
        """Load and initialize cross-encoder model from HuggingFace.

        Sets device to GPU if available and CUDA-enabled, otherwise CPU.
        Performs warmup inference to optimize GPU caching.

        Raises:
            ImportError: If transformers not installed.
            RuntimeError: If model download/loading fails.
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

        try:
            # Load model
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self._actual_device)
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
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

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
            logger.debug(f"Scoring {len(pairs)} pairs with batch_size={self.batch_size}")
            raw_scores: list[float] = self.model.predict(
                pairs,
                batch_size=self.batch_size,
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
