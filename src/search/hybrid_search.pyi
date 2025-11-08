"""Type stubs for hybrid search orchestration module.

Provides complete type definitions for HybridSearch unified integration class
that combines vector, BM25, RRF, boosting, and query routing components into
a cohesive search system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.core.config import Settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.search.boosting import BoostWeights, BoostingSystem
from src.search.bm25_search import BM25Search
from src.search.filters import FilterExpression
from src.search.query_router import QueryRouter, RoutingDecision
from src.search.results import SearchResult, SearchResultFormatter
from src.search.rrf import RRFScorer
from src.search.vector_search import VectorSearch

# Type aliases
Filter = FilterExpression | None
SearchResultList = list[SearchResult]

@dataclass
class SearchExplanation:
    """Explains search routing and ranking decisions.

    Attributes:
        query: Original search query string.
        strategy: Selected search strategy ("vector", "bm25", "hybrid").
        strategy_confidence: Confidence in strategy selection (0-1).
        strategy_reason: Human-readable explanation for strategy selection.
        vector_results_count: Number of vector search results (or None if not run).
        bm25_results_count: Number of BM25 search results (or None if not run).
        merged_results_count: Number of results after RRF merging.
        boosts_applied: Dictionary mapping boost type names to applied weights.
        final_results_count: Final number of results after filtering.
    """
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
    """Performance metrics for search execution.

    Attributes:
        total_time_ms: Total end-to-end search time in milliseconds.
        routing_time_ms: Time spent on query routing analysis.
        vector_search_time_ms: Time spent on vector search (None if not executed).
        bm25_search_time_ms: Time spent on BM25 search (None if not executed).
        merging_time_ms: Time spent on RRF merging (None if not executed).
        boosting_time_ms: Time spent applying boosts.
        filtering_time_ms: Time spent on final filtering and threshold application.
        formatting_time_ms: Time spent formatting results for output.
    """
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

    The class coordinates all Task 5 components (Tasks 5.1-5.4):
    - Task 4: Vector & BM25 search backends
    - Task 5.1: RRF result merging algorithm
    - Task 5.2: Multi-factor boosting system
    - Task 5.3: Intelligent query routing
    - Task 5.4: Unified integration (this class)

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
        logger: StructuredLogger
    ) -> None:
        """Initialize hybrid search with all components.

        Initializes all Task 5 components and validates their availability:
        - Creates VectorSearch backend for semantic search
        - Creates BM25Search backend for keyword search
        - Creates RRFScorer for result merging with k=60
        - Creates BoostingSystem for multi-factor ranking
        - Creates QueryRouter for automatic strategy selection
        - Creates SearchResultFormatter for result formatting

        Args:
            db_pool: Database connection pool for queries.
            settings: Application configuration and search settings.
            logger: Structured logger for diagnostics and performance metrics.

        Raises:
            ValueError: If any required component initialization fails.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0
    ) -> SearchResultList:
        """Execute hybrid search with automatic routing and optimization.

        Orchestrates the complete search pipeline:
        1. Route query to optimal strategy (if strategy=None)
        2. Execute vector search, BM25 search, or both
        3. Merge results using RRF if hybrid strategy
        4. Apply multi-factor boosts for ranking
        5. Apply score threshold filtering
        6. Limit to top_k results
        7. Return formatted results

        Args:
            query: Search query string. Cannot be empty.
            top_k: Maximum results to return (default 10). Must be 1-1000.
            strategy: Search strategy selection:
                - "vector": Vector similarity search only
                - "bm25": BM25 full-text search only
                - "hybrid": Combined vector + BM25 with RRF merging
                - None: Auto-select based on query analysis (default)
            boosts: Multi-factor boost configuration. If None, uses defaults:
                - vendor: +15% for matching vendors
                - doc_type: +10% for matching document types
                - recency: +5% for recent documents
                - entity: +10% for entity matching
                - topic: +8% for topic matching
            filters: Metadata filters using JSONB containment operators.
                Examples:
                - FilterExpression.equals("source_category", "vendor")
                - FilterExpression.between("document_date", start, end)
            min_score: Minimum score threshold (0-1, default 0.0).
                Results below this threshold are excluded.

        Returns:
            List of SearchResult objects sorted by final score (descending).
            Empty list if query matches no results or all filtered out.

        Raises:
            ValueError: If query is empty or parameters invalid.
            DatabaseError: If database queries fail.

        Example:
            >>> hybrid = HybridSearch(db_pool, settings, logger)
            >>> results = hybrid.search(
            ...     "JWT authentication best practices",
            ...     top_k=10,
            ...     strategy="hybrid",
            ...     min_score=0.3
            ... )
            >>> for result in results:
            ...     print(f"{result.rank}. {result.source_file} ({result.hybrid_score:.3f})")
        """
        ...

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0
    ) -> tuple[SearchResultList, SearchExplanation]:
        """Execute search and return explanation of routing/ranking decisions.

        Performs same search as search() but also returns detailed explanation
        of routing decision, strategy selection, merging, and boost application.

        Useful for:
        - Debugging search quality issues
        - Understanding why specific results were ranked highly
        - Validating strategy selection for specific queries
        - Performance analysis and optimization

        Args:
            query: Search query string.
            top_k: Maximum results to return (default 10).
            strategy: Search strategy override (default: None = auto-select).
            boosts: Boost configuration (default: None = use defaults).
            filters: Metadata filters (default: None = no filters).
            min_score: Minimum score threshold (default: 0.0).

        Returns:
            Tuple of (results, explanation) where:
            - results: List of SearchResult objects (same as search())
            - explanation: SearchExplanation with routing and ranking details

        Example:
            >>> results, explanation = hybrid.search_with_explanation(
            ...     "How to implement OAuth2?",
            ...     strategy="hybrid"
            ... )
            >>> print(f"Strategy: {explanation.strategy}")
            >>> print(f"Confidence: {explanation.strategy_confidence:.2%}")
            >>> print(f"Reason: {explanation.strategy_reason}")
            >>> print(f"Merged {explanation.merged_results_count} results")
        """
        ...

    def search_with_profile(
        self,
        query: str,
        top_k: int = 10,
        strategy: str | None = None,
        boosts: BoostWeights | None = None,
        filters: Filter = None,
        min_score: float = 0.0
    ) -> tuple[SearchResultList, SearchProfile]:
        """Execute search with comprehensive performance profiling.

        Performs same search as search() but measures timing for each
        pipeline stage (routing, vector search, BM25, merging, boosting,
        filtering, formatting).

        Useful for:
        - Performance benchmarking and optimization
        - Identifying bottlenecks in search pipeline
        - Monitoring production search performance
        - Validating service level objectives (SLOs)

        Performance targets:
        - Vector search: <100ms
        - BM25 search: <50ms
        - RRF merging: <50ms
        - Boosting: <10ms
        - Final filtering: <5ms
        - End-to-end (hybrid): <300ms p50, <500ms p95

        Args:
            query: Search query string.
            top_k: Maximum results to return (default 10).
            strategy: Search strategy override (default: None = auto-select).
            boosts: Boost configuration (default: None = use defaults).
            filters: Metadata filters (default: None = no filters).
            min_score: Minimum score threshold (default: 0.0).

        Returns:
            Tuple of (results, profile) where:
            - results: List of SearchResult objects (same as search())
            - profile: SearchProfile with timing breakdown for each stage

        Example:
            >>> results, profile = hybrid.search_with_profile("authentication jwt")
            >>> print(f"Total: {profile.total_time_ms:.1f}ms")
            >>> print(f"  Routing: {profile.routing_time_ms:.1f}ms")
            >>> print(f"  Vector: {profile.vector_search_time_ms:.1f}ms")
            >>> print(f"  BM25: {profile.bm25_search_time_ms:.1f}ms")
            >>> print(f"  Merging: {profile.merging_time_ms:.1f}ms")
            >>> print(f"  Boosting: {profile.boosting_time_ms:.1f}ms")
        """
        ...

    def _execute_vector_search(
        self,
        query: str,
        top_k: int,
        filters: Filter
    ) -> SearchResultList:
        """Execute vector similarity search.

        Uses the VectorSearch backend to find semantically similar documents
        based on query embeddings and document chunk embeddings in pgvector.

        Args:
            query: Search query string for embedding generation.
            top_k: Maximum results to return.
            filters: Optional metadata filters to apply.

        Returns:
            List of SearchResult objects with similarity_score populated.
            Empty list if query embedding generation fails or no results found.

        Raises:
            DatabaseError: If database queries fail.
        """
        ...

    def _execute_bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Filter
    ) -> SearchResultList:
        """Execute BM25 full-text search.

        Uses the BM25Search backend to find documents matching query keywords
        using PostgreSQL's full-text search with GIN indexing and ts_rank_cd
        for BM25-like ranking.

        Args:
            query: Search query string for keyword matching.
            top_k: Maximum results to return.
            filters: Optional metadata filters to apply.

        Returns:
            List of SearchResult objects with bm25_score populated.
            Empty list if no keyword matches found.

        Raises:
            DatabaseError: If database queries fail.
        """
        ...

    def _merge_and_boost(
        self,
        vector_results: SearchResultList,
        bm25_results: SearchResultList,
        query: str,
        boosts: BoostWeights
    ) -> SearchResultList:
        """Merge vector and BM25 results using RRF, apply boosts.

        Orchestrates the result merging and boosting pipeline:
        1. Use RRFScorer to combine vector and BM25 results
        2. Apply multi-factor boosts using BoostingSystem
        3. Rerank by final (boosted) scores

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 full-text search.
            query: Original search query for boost context analysis.
            boosts: Boost weights configuration.

        Returns:
            Merged and boosted results, sorted by final score (descending).
            May contain results from either or both source searches.
        """
        ...

    def _apply_final_filtering(
        self,
        results: SearchResultList,
        top_k: int,
        min_score: float
    ) -> SearchResultList:
        """Apply threshold filtering and limit to top_k.

        Final pipeline stage that:
        1. Filters out results below min_score threshold
        2. Limits results to top_k
        3. Validates and reranks (1-indexed positions)

        Args:
            results: Results from merging and boosting.
            top_k: Maximum results to return.
            min_score: Minimum score threshold (0-1).

        Returns:
            Filtered and ranked results (1 to top_k).

        Raises:
            ValueError: If top_k or min_score invalid.
        """
        ...
