"""Search module for knowledge base queries.

Provides BM25 full-text search using PostgreSQL's ts_vector and GIN indexing
for fast keyword-based retrieval with relevance ranking, plus vector similarity
search using pgvector HNSW index for semantic search, unified hybrid search
orchestration, cross-encoder reranking for result refinement, and comprehensive
performance profiling and optimization utilities.

Key components:
- vector_search: Similarity search on 768-dimensional embeddings with HNSW
- bm25_search: Full-text search implementation
- hybrid_search: Unified orchestration combining vector, BM25, RRF, boosting
- cross_encoder_reranker: Cross-encoder pair-wise relevance scoring and reranking
  - Reranker: Protocol for pluggable reranking implementations
  - RerankerConfig: Configuration management for rerankers
  - CrossEncoderReranker: Cross-encoder implementation
  - CandidateSelector: Adaptive pool sizing for efficiency
- profiler: Query performance measurement and optimization
"""

from src.search.bm25_search import BM25Search, SearchResult as BM25SearchResult
from src.search.reranker_protocol import Reranker
# Note: HybridSearch import deferred to avoid torch dependency in test environments
# from src.search.hybrid_search import HybridSearch, SearchExplanation, SearchProfile
# Note: CrossEncoderReranker imports deferred to avoid transformers dependency in test environments
# from src.search.cross_encoder_reranker import (
#     CrossEncoderReranker,
#     RerankerConfig,
#     CandidateSelector,
#     QueryAnalysis,
# )
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
    "Reranker",  # Protocol for pluggable reranking implementations
    # "HybridSearch",  # Deferred to avoid torch import
    # "SearchExplanation",  # Deferred
    # "SearchProfile",  # Deferred
    # "CrossEncoderReranker",  # Deferred to avoid transformers import
    # "RerankerConfig",  # Deferred to avoid transformers import
    # "CandidateSelector",  # Deferred
    # "QueryAnalysis",  # Deferred
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
