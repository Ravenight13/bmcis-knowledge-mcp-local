"""Search module for knowledge base queries.

Provides BM25 full-text search using PostgreSQL's ts_vector and GIN indexing
for fast keyword-based retrieval with relevance ranking, plus vector similarity
search using pgvector HNSW index for semantic search, unified hybrid search
orchestration, and comprehensive performance profiling and optimization utilities.

Key components:
- vector_search: Similarity search on 768-dimensional embeddings with HNSW
- bm25_search: Full-text search implementation
- hybrid_search: Unified orchestration combining vector, BM25, RRF, boosting
- profiler: Query performance measurement and optimization
"""

from src.search.bm25_search import BM25Search, SearchResult as BM25SearchResult
# Note: HybridSearch import deferred to avoid torch dependency in test environments
# from src.search.hybrid_search import HybridSearch, SearchExplanation, SearchProfile
from src.search.profiler import (
    SearchProfiler,
    ProfileResult,
    TimingBreakdown,
    BenchmarkResult,
    IndexAnalyzer,
    PerformanceOptimizer,
)
from src.search.vector_search import VectorSearch, SearchResult, SearchStats

__all__ = [
    "BM25Search",
    "BM25SearchResult",
    # "HybridSearch",  # Deferred to avoid torch import
    # "SearchExplanation",  # Deferred
    # "SearchProfile",  # Deferred
    "SearchProfiler",
    "ProfileResult",
    "TimingBreakdown",
    "BenchmarkResult",
    "IndexAnalyzer",
    "PerformanceOptimizer",
    "VectorSearch",
    "SearchResult",
    "SearchStats",
]
