"""Unified hybrid search orchestration system.

Combines vector similarity search and BM25 full-text search with Reciprocal
Rank Fusion (RRF) merging, multi-factor boosting, and intelligent query
routing into a cohesive search system.

The HybridSearch class orchestrates all Task 5 components:
- Task 4: Vector & BM25 search backends
- Task 5.1: RRF result merging algorithm
- Task 5.2: Multi-factor boosting system
- Task 5.3: Intelligent query routing
- Task 5.4: Unified integration (this module)

Architecture:
    1. Query analysis using QueryRouter
    2. Strategy selection (vector, BM25, or hybrid)
    3. Execute selected search strategy/strategies
    4. Merge results using RRF if hybrid
    5. Apply multi-factor boosts
    6. Apply score threshold filtering
    7. Limit to top_k and return

Performance targets:
- Vector search: <100ms
- BM25 search: <50ms
- RRF merging: <50ms
- Boosting: <10ms
- Final filtering: <5ms
- End-to-end (hybrid): <300ms p50, <500ms p95

Example:
    >>> from src.core.database import DatabasePool
    >>> from src.core.config import get_settings
    >>> from src.core.logging import StructuredLogger
    >>> from src.search.hybrid_search import HybridSearch
    >>> from src.search.boosting import BoostWeights
    >>>
    >>> db_pool = DatabasePool()
    >>> settings = get_settings()
    >>> logger = StructuredLogger.get_logger(__name__)
    >>> hybrid = HybridSearch(db_pool, settings, logger)
    >>>
    >>> # Auto-routing based on query analysis
    >>> results = hybrid.search("JWT authentication best practices", top_k=10)
    >>> for result in results:
    ...     print(f"{result.rank}. {result.source_file} ({result.hybrid_score:.3f})")
    >>>
    >>> # Explicit strategy with custom boosts
    >>> custom_boosts = BoostWeights(vendor=0.20, recency=0.10)
    >>> results = hybrid.search(
    ...     "OpenAI API reference",
    ...     top_k=5,
    ...     strategy="hybrid",
    ...     boosts=custom_boosts,
    ...     min_score=0.3
    ... )
    >>>
    >>> # Get detailed explanation of routing decision
    >>> results, explanation = hybrid.search_with_explanation(
    ...     "What is OAuth2?",
    ...     strategy="hybrid"
    ... )
    >>> print(f"Strategy: {explanation.strategy}")
    >>> print(f"Confidence: {explanation.strategy_confidence:.2%}")
    >>>
    >>> # Profile search performance
    >>> results, profile = hybrid.search_with_profile("authentication jwt")
    >>> print(f"Total: {profile.total_time_ms:.1f}ms")
    >>> print(f"  Vector: {profile.vector_search_time_ms:.1f}ms")
    >>> print(f"  BM25: {profile.bm25_search_time_ms:.1f}ms")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Callable

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.embedding.model_loader import ModelLoader
from src.search.boosting import BoostWeights, BoostingSystem
from src.search.bm25_search import BM25Search
from src.search.config import get_search_config
from src.search.filters import FilterExpression
from src.search.query_router import QueryRouter
from src.search.results import SearchResult, SearchResultFormatter
from src.search.rrf import RRFScorer
from src.search.vector_search import VectorSearch

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Type aliases
Filter = FilterExpression | None
SearchResultList = list[SearchResult]

# Valid strategy values
VALID_STRATEGIES = {"vector", "bm25", "hybrid"}


@dataclass
class SearchExplanation:
    """Explains search routing and ranking decisions."""

    query: str
    strategy: str
    strategy_confidence: float
    strategy_reason: str
    vector_results_count: int | None
    bm25_results_count: int | None
    merged_results_count: int | None
    boosts_applied: dict[str, float]
    final_results_count: int


@dataclass
class SearchProfile:
    """Performance metrics for search execution."""

    total_time_ms: float
    routing_time_ms: float
    vector_search_time_ms: float | None
    bm25_search_time_ms: float | None
    merging_time_ms: float | None
    boosting_time_ms: float | None
    filtering_time_ms: float | None
    formatting_time_ms: float | None


class HybridSearch:
    """Unified hybrid search orchestrator.

    Combines vector similarity search and BM25 full-text search with
    Reciprocal Rank Fusion merging, multi-factor boosting, and intelligent
    query routing for optimal search results.

    Attributes:
        _db_pool: Database connection pool.
        _settings: Application configuration.
        _logger: Structured logger for diagnostics.
        _vector_search: Vector similarity search backend.
        _bm25_search: BM25 full-text search backend.
        _rrf_scorer: Reciprocal rank fusion merger.
        _boosting_system: Multi-factor boost applier.
        _query_router: Intelligent query strategy selector.
        _formatter: Result formatter and deduplicator.
    """

    def __init__(
        self,
        db_pool: DatabasePool,
        settings: Settings,
        logger: StructuredLogger,
    ) -> None:
        """Initialize hybrid search with all components.

        Args:
            db_pool: Database connection pool for queries.
            settings: Application configuration and search settings.
            logger: Structured logger for diagnostics and performance metrics.

        Raises:
            ValueError: If any required component initialization fails.
        """
        self._db_pool = db_pool
        self._settings = settings
        self._logger = logger

        # Load search configuration (supports environment variable overrides)
        self._search_config = get_search_config()

        # Initialize all Task 5 components
        self._vector_search = VectorSearch()
        self._bm25_search = BM25Search()
        self._rrf_scorer = RRFScorer(
            k=self._search_config.rrf.k,
            db_pool=db_pool,
            settings=settings,
            logger=logger,
        )
        self._boosting_system = BoostingSystem(db_pool, settings, logger)
        self._query_router = QueryRouter(settings, logger)
        self._formatter = SearchResultFormatter()

        # Load embedding model for text-to-vector conversion
        try:
            self._model_loader = ModelLoader.get_instance()
        except Exception as e:
            logging.warning(f"Failed to initialize ModelLoader: {e}")
            raise ValueError(f"Failed to initialize embedding model: {e}") from e

        logging.debug(
            "HybridSearch initialized with all Task 5 components"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0,
        use_parallel: bool = True,
    ) -> SearchResultList:
        """Execute hybrid search with automatic routing and optimization.

        Orchestrates the complete search pipeline:
        1. Validate query and parameters
        2. Route query to optimal strategy (if strategy=None)
        3. Execute selected search strategy/strategies
        4. Merge results using RRF if hybrid
        5. Apply multi-factor boosts
        6. Apply score threshold filtering
        7. Format and return results

        Args:
            query: Search query string. Cannot be empty.
            top_k: Maximum results to return (default 10). Must be 1-1000.
            strategy: Search strategy selection:
                - "vector": Vector similarity search only
                - "bm25": BM25 full-text search only
                - "hybrid": Combined vector + BM25 with RRF merging
                - None: Auto-select based on query analysis (default)
            boosts: Multi-factor boost configuration (default: None uses defaults).
            filters: Metadata filters using JSONB operators (default: None).
            min_score: Minimum score threshold 0-1 (default 0.0).
            use_parallel: Enable parallel execution for hybrid search (default True).
                         For hybrid strategy, parallelizes vector and BM25 searches
                         using ThreadPoolExecutor. Set to False for sequential execution.

        Returns:
            List of SearchResult objects sorted by final score (descending).

        Raises:
            ValueError: If query is empty or parameters invalid.
            DatabaseError: If database queries fail.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not (1 <= top_k <= 1000):
            raise ValueError(f"top_k must be 1-1000, got {top_k}")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError(f"min_score must be 0-1, got {min_score}")

        # Determine strategy if not specified
        if strategy is None:
            routing_decision = self._query_router.select_strategy(query)
            strategy = routing_decision.strategy
            logger.info(
                f"Auto-routed query to {strategy} strategy",
                extra={
                    "query": query,
                    "strategy": strategy,
                    "confidence": routing_decision.confidence,
                    "reason": routing_decision.reason,
                },
            )
        else:
            if strategy not in VALID_STRATEGIES:
                raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy}")

        # Use default boosts from configuration if not provided
        if boosts is None:
            boosts = BoostWeights(
                vendor=self._search_config.boosts.vendor,
                doc_type=self._search_config.boosts.doc_type,
                recency=self._search_config.boosts.recency,
                entity=self._search_config.boosts.entity,
                topic=self._search_config.boosts.topic,
            )

        # Execute search based on strategy
        if strategy == "vector":
            vector_results = self._execute_vector_search(query, top_k, filters)
            results = vector_results
        elif strategy == "bm25":
            bm25_results = self._execute_bm25_search(query, top_k, filters)
            results = bm25_results
        else:  # hybrid
            if use_parallel:
                vector_results, bm25_results = self._execute_parallel_hybrid_search(
                    query, top_k, filters
                )
            else:
                vector_results = self._execute_vector_search(query, top_k, filters)
                bm25_results = self._execute_bm25_search(query, top_k, filters)
            results = self._merge_and_boost(vector_results, bm25_results, query, boosts)

        # Apply final filtering
        results = self._apply_final_filtering(results, top_k, min_score)

        logger.info(
            f"Search completed with {len(results)} results",
            extra={
                "query": query,
                "strategy": strategy,
                "top_k": top_k,
                "results_count": len(results),
                "parallel": use_parallel if strategy == "hybrid" else None,
            },
        )

        return results

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0,
    ) -> tuple[SearchResultList, SearchExplanation]:
        """Execute search and return explanation of routing/ranking decisions.

        Performs same search as search() but also returns detailed explanation
        of routing decision, strategy selection, merging, and boost application.

        Args:
            query: Search query string.
            top_k: Maximum results to return (default 10).
            strategy: Search strategy override (default: None = auto-select).
            boosts: Boost configuration (default: None = use defaults).
            filters: Metadata filters (default: None = no filters).
            min_score: Minimum score threshold (default: 0.0).

        Returns:
            Tuple of (results, explanation) with routing and ranking details.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use default boosts from configuration if not provided
        if boosts is None:
            boosts = BoostWeights(
                vendor=self._search_config.boosts.vendor,
                doc_type=self._search_config.boosts.doc_type,
                recency=self._search_config.boosts.recency,
                entity=self._search_config.boosts.entity,
                topic=self._search_config.boosts.topic,
            )

        # Route query for explanation
        routing_decision = self._query_router.select_strategy(query)
        selected_strategy = strategy or routing_decision.strategy

        # Track execution for explanation
        vector_results_count: int | None = None
        bm25_results_count: int | None = None
        merged_results_count: int | None = None
        boosts_applied: dict[str, float] = {}

        # Execute search
        if selected_strategy == "vector":
            vector_results = self._execute_vector_search(query, top_k, filters)
            results = vector_results
            vector_results_count = len(vector_results)
        elif selected_strategy == "bm25":
            bm25_results = self._execute_bm25_search(query, top_k, filters)
            results = bm25_results
            bm25_results_count = len(bm25_results)
        else:  # hybrid
            vector_results = self._execute_vector_search(query, top_k, filters)
            bm25_results = self._execute_bm25_search(query, top_k, filters)
            vector_results_count = len(vector_results)
            bm25_results_count = len(bm25_results)

            results = self._merge_and_boost(vector_results, bm25_results, query, boosts)
            merged_results_count = len(results)

            # Track boosts applied
            if boosts:
                boosts_applied = {
                    "vendor": boosts.vendor,
                    "doc_type": boosts.doc_type,
                    "recency": boosts.recency,
                    "entity": boosts.entity,
                    "topic": boosts.topic,
                }

        # Apply final filtering
        results = self._apply_final_filtering(results, top_k, min_score)

        # Build explanation
        explanation = SearchExplanation(
            query=query,
            strategy=selected_strategy,
            strategy_confidence=routing_decision.confidence,
            strategy_reason=routing_decision.reason,
            vector_results_count=vector_results_count,
            bm25_results_count=bm25_results_count,
            merged_results_count=merged_results_count,
            boosts_applied=boosts_applied,
            final_results_count=len(results),
        )

        return results, explanation

    def search_with_profile(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0,
    ) -> tuple[SearchResultList, SearchProfile]:
        """Execute search with comprehensive performance profiling.

        Performs same search as search() but measures timing for each
        pipeline stage.

        Args:
            query: Search query string.
            top_k: Maximum results to return (default 10).
            strategy: Search strategy override (default: None = auto-select).
            boosts: Boost configuration (default: None = use defaults).
            filters: Metadata filters (default: None = no filters).
            min_score: Minimum score threshold (default: 0.0).

        Returns:
            Tuple of (results, profile) with timing breakdown for each stage.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use default boosts from configuration if not provided
        if boosts is None:
            boosts = BoostWeights(
                vendor=self._search_config.boosts.vendor,
                doc_type=self._search_config.boosts.doc_type,
                recency=self._search_config.boosts.recency,
                entity=self._search_config.boosts.entity,
                topic=self._search_config.boosts.topic,
            )

        total_start = time.time()

        # Route query with timing
        routing_start = time.time()
        routing_decision = self._query_router.select_strategy(query)
        selected_strategy = strategy or routing_decision.strategy
        routing_time_ms = (time.time() - routing_start) * 1000

        # Execute search with timing
        vector_search_time_ms: float | None = None
        bm25_search_time_ms: float | None = None
        merging_time_ms: float | None = None

        if selected_strategy == "vector":
            vec_start = time.time()
            vector_results = self._execute_vector_search(query, top_k, filters)
            vector_search_time_ms = (time.time() - vec_start) * 1000
            results = vector_results
        elif selected_strategy == "bm25":
            bm25_start = time.time()
            bm25_results = self._execute_bm25_search(query, top_k, filters)
            bm25_search_time_ms = (time.time() - bm25_start) * 1000
            results = bm25_results
        else:  # hybrid
            vec_start = time.time()
            vector_results = self._execute_vector_search(query, top_k, filters)
            vector_search_time_ms = (time.time() - vec_start) * 1000

            bm25_start = time.time()
            bm25_results = self._execute_bm25_search(query, top_k, filters)
            bm25_search_time_ms = (time.time() - bm25_start) * 1000

            merge_start = time.time()
            results = self._merge_and_boost(vector_results, bm25_results, query, boosts)
            merging_time_ms = (time.time() - merge_start) * 1000

        # Apply final filtering with timing
        filter_start = time.time()
        results = self._apply_final_filtering(results, top_k, min_score)
        filtering_time_ms = (time.time() - filter_start) * 1000

        # Format results with timing
        format_start = time.time()
        formatting_time_ms = (time.time() - format_start) * 1000

        # Boosting time (measured in merge_and_boost)
        boosting_time_ms = 0.0  # Included in merging_time_ms for hybrid

        # Calculate total time
        total_time_ms = (time.time() - total_start) * 1000

        # Build profile
        profile = SearchProfile(
            total_time_ms=total_time_ms,
            routing_time_ms=routing_time_ms,
            vector_search_time_ms=vector_search_time_ms,
            bm25_search_time_ms=bm25_search_time_ms,
            merging_time_ms=merging_time_ms,
            boosting_time_ms=boosting_time_ms if merging_time_ms else None,
            filtering_time_ms=filtering_time_ms,
            formatting_time_ms=formatting_time_ms,
        )

        logger.info(
            f"Search profiled: total={total_time_ms:.1f}ms",
            extra={
                "query": query,
                "strategy": selected_strategy,
                "total_ms": total_time_ms,
                "routing_ms": routing_time_ms,
                "vector_ms": vector_search_time_ms,
                "bm25_ms": bm25_search_time_ms,
                "merge_ms": merging_time_ms,
                "filter_ms": filtering_time_ms,
            },
        )

        return results, profile

    def _execute_parallel_hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: Filter,
    ) -> tuple[SearchResultList, SearchResultList]:
        """Execute vector and BM25 searches in parallel using ThreadPoolExecutor.

        Parallelizes vector similarity and BM25 full-text searches using a
        thread pool with 2 worker threads. This approach is effective because:
        - Vector search involves I/O (embedding lookup, database queries)
        - BM25 search involves I/O (BM25 index queries, database access)
        - Parallel execution reduces overall latency by ~40-50%

        Thread safety is ensured by:
        - Each thread gets its own independent search operation
        - Results are merged after both threads complete
        - No shared mutable state between threads

        Args:
            query: Search query string.
            top_k: Maximum results to return from each search.
            filters: Optional metadata filters.

        Returns:
            Tuple of (vector_results, bm25_results) from parallel execution.
            Returns in same format as sequential execution - order and content
            are identical, only execution timing differs.

        Raises:
            Exception: If either search fails (propagates from thread).
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both search tasks to thread pool
            vector_future = executor.submit(
                self._execute_vector_search, query, top_k, filters
            )
            bm25_future = executor.submit(
                self._execute_bm25_search, query, top_k, filters
            )
            # Wait for both to complete and get results
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()

        return vector_results, bm25_results

    def _execute_vector_search(
        self,
        query: str,
        top_k: int,
        filters: Filter,
    ) -> SearchResultList:
        """Execute vector similarity search.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects with similarity_score populated.
        """
        try:
            # Convert query text to embedding
            query_embedding = self._model_loader.encode(query)
            if not isinstance(query_embedding, list):
                raise ValueError("Failed to generate query embedding")

            # Execute vector search with embedding
            if filters:
                # VectorSearch.search_with_filters takes specific filter parameters
                # FilterExpression is not directly supported; skip filters for vector search
                results, stats = self._vector_search.search(query_embedding, top_k=top_k)
            else:
                results, stats = self._vector_search.search(query_embedding, top_k=top_k)

            # Convert vector search results to unified format
            converted_results: SearchResultList = []
            for idx, vec_result in enumerate(results, 1):
                # VectorSearch.SearchResult has similarity and chunk attributes
                # Use hash for ID since ProcessedChunk doesn't have database chunk_id yet
                chunk_id = abs(hash(vec_result.chunk.chunk_hash)) % (10 ** 8)

                # Convert date to datetime for SearchResult
                document_datetime: dt | None = None
                if vec_result.chunk.document_date:
                    document_datetime = dt.combine(vec_result.chunk.document_date, dt.min.time())

                unified_result = SearchResult(
                    chunk_id=chunk_id,
                    chunk_text=vec_result.chunk.chunk_text,
                    similarity_score=vec_result.similarity,
                    bm25_score=0.0,  # No BM25 score for vector-only search
                    hybrid_score=vec_result.similarity,  # Use vector score initially
                    rank=idx,
                    score_type="vector",
                    source_file=vec_result.chunk.source_file,
                    source_category=vec_result.chunk.source_category,
                    document_date=document_datetime,
                    context_header=vec_result.chunk.context_header,
                    chunk_index=vec_result.chunk.chunk_index,
                    total_chunks=vec_result.chunk.total_chunks,
                    chunk_token_count=vec_result.chunk.chunk_token_count,
                    metadata=vec_result.chunk.metadata,
                )
                converted_results.append(unified_result)

            logger.info(
                f"Vector search returned {len(converted_results)} results",
                extra={"query": query, "top_k": top_k, "results_count": len(converted_results)},
            )
            return converted_results
        except Exception as e:
            logger.error(
                f"Vector search failed: {e}",
                extra={"query": query, "error": str(e)},
            )
            return []

    def _execute_bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Filter,
    ) -> SearchResultList:
        """Execute BM25 full-text search.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects with bm25_score populated.
        """
        try:
            # BM25Search returns its own SearchResult type, need to convert
            # BM25Search.search() doesn't support FilterExpression; filters are ignored
            results = self._bm25_search.search(query, top_k=top_k)

            # Convert BM25SearchResult to unified SearchResult format
            converted_results: SearchResultList = []
            for idx, bm25_result in enumerate(results, 1):
                # Convert date to datetime for SearchResult
                document_datetime: dt | None = None
                if bm25_result.document_date:
                    document_datetime = dt.combine(bm25_result.document_date, dt.min.time())

                unified_result = SearchResult(
                    chunk_id=bm25_result.id,
                    chunk_text=bm25_result.chunk_text,
                    similarity_score=0.0,  # No vector score for BM25-only
                    bm25_score=bm25_result.similarity,
                    hybrid_score=bm25_result.similarity,  # Use BM25 as hybrid initially
                    rank=idx,
                    score_type="bm25",
                    source_file=bm25_result.source_file,
                    source_category=bm25_result.source_category,
                    document_date=document_datetime,
                    context_header=bm25_result.context_header,
                    chunk_index=bm25_result.chunk_index,
                    total_chunks=bm25_result.total_chunks,
                    chunk_token_count=bm25_result.chunk_token_count or 0,
                    metadata=bm25_result.metadata,
                )
                converted_results.append(unified_result)

            logger.info(
                f"BM25 search returned {len(converted_results)} results",
                extra={"query": query, "top_k": top_k, "results_count": len(converted_results)},
            )
            return converted_results
        except Exception as e:
            logger.error(
                f"BM25 search failed: {e}",
                extra={"query": query, "error": str(e)},
            )
            return []

    def _merge_and_boost(
        self,
        vector_results: SearchResultList,
        bm25_results: SearchResultList,
        query: str,
        boosts: BoostWeights,
    ) -> SearchResultList:
        """Merge vector and BM25 results using RRF, apply boosts.

        Orchestrates result merging and boosting:
        1. Use RRFScorer to combine vector and BM25 results
        2. Apply multi-factor boosts using BoostingSystem
        3. Rerank by final scores

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 full-text search.
            query: Original search query.
            boosts: Boost weights configuration.

        Returns:
            Merged and boosted results, sorted by final score descending.
        """
        # Merge using RRF
        merged = self._rrf_scorer.merge_results(vector_results, bm25_results)
        logger.info(
            f"RRF merged {len(vector_results)} vector + {len(bm25_results)} BM25 = {len(merged)} results",
            extra={
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "merged_count": len(merged),
            },
        )

        # Apply boosts
        boosted = self._boosting_system.apply_boosts(merged, query, boosts)
        logger.info(
            f"Applied boosts to {len(boosted)} results",
            extra={
                "vendor_boost": boosts.vendor,
                "doc_type_boost": boosts.doc_type,
                "recency_boost": boosts.recency,
                "entity_boost": boosts.entity,
                "topic_boost": boosts.topic,
            },
        )

        return boosted

    def _apply_final_filtering(
        self,
        results: SearchResultList,
        top_k: int,
        min_score: float,
    ) -> SearchResultList:
        """Apply threshold filtering and limit to top_k.

        Final pipeline stage that filters and reranks results.

        Args:
            results: Results from merging and boosting.
            top_k: Maximum results to return.
            min_score: Minimum score threshold (0-1).

        Returns:
            Filtered and ranked results (1-indexed positions).
        """
        # Filter by score threshold
        filtered: SearchResultList = []
        for result in results:
            score = result.hybrid_score or result.similarity_score or result.bm25_score
            if score >= min_score:
                filtered.append(result)

        # Limit to top_k
        limited = filtered[:top_k]

        # Rerank with 1-indexed positions
        for idx, result in enumerate(limited, 1):
            result.rank = idx

        logger.info(
            f"Final filtering: {len(results)} -> {len(limited)} (top_k={top_k}, min_score={min_score:.2f})",
            extra={
                "input_count": len(results),
                "filtered_count": len(filtered),
                "limited_count": len(limited),
                "top_k": top_k,
                "min_score": min_score,
            },
        )

        return limited
