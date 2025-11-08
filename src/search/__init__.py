"""Search module for knowledge base queries.

Provides BM25 full-text search using PostgreSQL's ts_vector and GIN indexing
for fast keyword-based retrieval with relevance ranking, plus vector similarity
search using pgvector HNSW index for semantic search, and comprehensive
performance profiling and optimization utilities.

Key components:
- vector_search: Similarity search on 768-dimensional embeddings with HNSW
- bm25_search: Full-text search implementation
- profiler: Query performance measurement and optimization
"""

from src.search.bm25_search import BM25Search, SearchResult as BM25SearchResult
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
