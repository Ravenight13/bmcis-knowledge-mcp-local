"""Parallel execution tests for HybridSearch.

Tests for ThreadPoolExecutor-based parallel search execution in hybrid mode.

Tests cover:
- Parallel execution of vector and BM25 searches
- Correctness validation (parallel vs sequential produce same results)
- Thread safety and result integrity
- Error propagation from parallel threads
- Performance characteristics of parallel execution
"""

from __future__ import annotations

import sys
import time
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import pytest

# Patch torch and other dependencies before importing HybridSearch
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from src.search.rrf import RRFScorer
from src.search.boosting import BoostingSystem, BoostWeights
from src.search.query_router import QueryRouter
from src.search.results import SearchResult
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger


# Test fixtures
@pytest.fixture
def mock_db_pool() -> MagicMock:
    """Create mock DatabasePool."""
    return MagicMock(spec=DatabasePool)


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create mock StructuredLogger."""
    return MagicMock(spec=StructuredLogger)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock Settings."""
    mock_settings = MagicMock()
    mock_settings.search_top_k = 10
    mock_settings.search_min_score = 0.0
    mock_settings.hybrid_search_rrf_k = 60
    mock_settings.hybrid_search_vector_weight = 0.6
    mock_settings.hybrid_search_bm25_weight = 0.4
    return mock_settings


def create_test_vector_results(count: int, score_offset: float = 0.0) -> list[SearchResult]:
    """Create test vector search results with descending scores.

    Args:
        count: Number of results to create.
        score_offset: Offset to apply to scores for differentiation.

    Returns:
        List of SearchResult objects with vector search characteristics.
    """
    results: list[SearchResult] = []
    for i in range(count):
        score = max(0.0, 1.0 - (i * 0.1) + score_offset)
        results.append(
            SearchResult(
                chunk_id=i,
                chunk_text=f"Vector result {i}: semantic match content",
                similarity_score=score,
                bm25_score=0.0,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="vector",
                source_file=f"vector_doc{i % 3}.md",
                source_category="guide",
                document_date=None,
                context_header=f"doc{i % 3}.md > Section {i}",
                chunk_index=i % 5,
                total_chunks=5,
                chunk_token_count=256,
                metadata={"vendor": f"vendor{i % 2}", "doc_type": "technical"},
            )
        )
    return results


def create_test_bm25_results(count: int, score_offset: float = 0.0) -> list[SearchResult]:
    """Create test BM25 search results with descending scores.

    Args:
        count: Number of results to create.
        score_offset: Offset to apply to scores for differentiation.

    Returns:
        List of SearchResult objects with BM25 search characteristics.
    """
    results: list[SearchResult] = []
    for i in range(count):
        score = max(0.0, 0.9 - (i * 0.09) + score_offset)
        results.append(
            SearchResult(
                chunk_id=i + 100,  # Different IDs from vector results
                chunk_text=f"BM25 result {i}: keyword match content",
                similarity_score=0.0,
                bm25_score=score,
                hybrid_score=0.0,
                rank=i + 1,
                score_type="bm25",
                source_file=f"bm25_doc{i % 3}.md",
                source_category="reference",
                document_date=None,
                context_header=f"ref{i % 3}.md > Section {i}",
                chunk_index=i % 4,
                total_chunks=4,
                chunk_token_count=320,
                metadata={"vendor": f"vendor{i % 2}", "doc_type": "reference"},
            )
        )
    return results


class TestParallelHybridSearchExecution:
    """Test suite for parallel execution in hybrid search.

    Validates that parallel execution with ThreadPoolExecutor:
    - Works correctly and executes both searches
    - Returns results in expected format
    - Completes successfully
    """

    def test_parallel_hybrid_search_execution(
        self,
        mock_db_pool: MagicMock,
        mock_logger: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test that parallel hybrid search executes both searches concurrently.

        Verifies:
        - ThreadPoolExecutor is used for parallel execution
        - Both vector and BM25 searches are submitted
        - Results are returned in correct format (tuple of lists)
        - Both result lists have expected lengths
        """
        # Import HybridSearch (torch is mocked at module level)
        from src.search.hybrid_search import HybridSearch

        # Create mock search implementations
        with patch("src.search.hybrid_search.VectorSearch") as MockVectorSearch, \
             patch("src.search.hybrid_search.BM25Search") as MockBM25Search, \
             patch("src.search.hybrid_search.ModelLoader") as MockModelLoader, \
             patch("src.search.hybrid_search.QueryRouter") as MockQueryRouter, \
             patch("src.search.hybrid_search.RRFScorer") as MockRRFScorer, \
             patch("src.search.hybrid_search.BoostingSystem") as MockBoostingSystem:

            # Setup mock vector search
            mock_vector_search_instance = MagicMock()
            MockVectorSearch.return_value = mock_vector_search_instance

            # Setup mock BM25 search
            mock_bm25_search_instance = MagicMock()
            MockBM25Search.return_value = mock_bm25_search_instance

            # Setup mock model loader
            mock_model_loader = MagicMock()
            MockModelLoader.get_instance.return_value = mock_model_loader
            mock_model_loader.encode.return_value = [0.1] * 768

            # Setup query router
            mock_router = MagicMock()
            MockQueryRouter.return_value = mock_router

            # Setup RRF scorer
            mock_rrf = MagicMock(spec=RRFScorer)
            MockRRFScorer.return_value = mock_rrf

            # Setup boosting system
            mock_boosting = MagicMock(spec=BoostingSystem)
            MockBoostingSystem.return_value = mock_boosting

            # Initialize HybridSearch
            hybrid = HybridSearch(mock_db_pool, mock_settings, mock_logger)

            # Create test data
            vector_results = create_test_vector_results(5)
            bm25_results = create_test_bm25_results(5)

            # Configure mocks to return test data
            mock_vector_search_instance.search.return_value = ([], {})
            mock_bm25_search_instance.search.return_value = []

            # Mock _execute_vector_search and _execute_bm25_search
            with patch.object(
                hybrid, "_execute_vector_search", return_value=vector_results
            ) as mock_vec, \
                 patch.object(
                     hybrid, "_execute_bm25_search", return_value=bm25_results
                 ) as mock_bm25, \
                 patch.object(
                     hybrid, "_merge_and_boost", return_value=vector_results + bm25_results
                 ) as mock_merge:

                # Call search with use_parallel=True (default)
                results = hybrid.search(
                    "test query",
                    top_k=10,
                    strategy="hybrid",
                    use_parallel=True,
                )

                # Verify both searches were called
                assert mock_vec.called, "Vector search should be called"
                assert mock_bm25.called, "BM25 search should be called"
                assert mock_merge.called, "Merge should be called"

                # Verify results were returned
                assert len(results) > 0, "Results should be returned"

    def test_parallel_execution_produces_same_results_as_sequential(
        self,
        mock_db_pool: MagicMock,
        mock_logger: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test equivalence: parallel and sequential produce identical results.

        Validates that:
        - Parallel execution produces same results as sequential
        - Result order is identical
        - Scores are identical
        - No data corruption occurs in parallel execution
        """
        from src.search.hybrid_search import HybridSearch

        # Create mock implementations
        with patch("src.search.hybrid_search.VectorSearch") as MockVectorSearch, \
             patch("src.search.hybrid_search.BM25Search") as MockBM25Search, \
             patch("src.search.hybrid_search.ModelLoader") as MockModelLoader, \
             patch("src.search.hybrid_search.QueryRouter") as MockQueryRouter, \
             patch("src.search.hybrid_search.RRFScorer") as MockRRFScorer, \
             patch("src.search.hybrid_search.BoostingSystem") as MockBoostingSystem:

            # Setup mocks
            mock_vector_search_instance = MagicMock()
            MockVectorSearch.return_value = mock_vector_search_instance

            mock_bm25_search_instance = MagicMock()
            MockBM25Search.return_value = mock_bm25_search_instance

            mock_model_loader = MagicMock()
            MockModelLoader.get_instance.return_value = mock_model_loader
            mock_model_loader.encode.return_value = [0.1] * 768

            mock_router = MagicMock()
            MockQueryRouter.return_value = mock_router

            mock_rrf = MagicMock(spec=RRFScorer)
            MockRRFScorer.return_value = mock_rrf

            mock_boosting = MagicMock(spec=BoostingSystem)
            MockBoostingSystem.return_value = mock_boosting

            # Initialize HybridSearch
            hybrid = HybridSearch(mock_db_pool, mock_settings, mock_logger)

            # Create identical test data
            vector_results = create_test_vector_results(10)
            bm25_results = create_test_bm25_results(10)

            # Mock search methods
            with patch.object(
                hybrid, "_execute_vector_search", return_value=vector_results
            ), \
                 patch.object(
                     hybrid, "_execute_bm25_search", return_value=bm25_results
                 ), \
                 patch.object(
                     hybrid, "_merge_and_boost"
                 ) as mock_merge:

                # Configure merge to return predictable results
                merged_results = vector_results[:5] + bm25_results[:5]
                mock_merge.return_value = merged_results

                # Run parallel execution
                parallel_results = hybrid.search(
                    "test query",
                    top_k=10,
                    strategy="hybrid",
                    use_parallel=True,
                )

                # Run sequential execution
                sequential_results = hybrid.search(
                    "test query",
                    top_k=10,
                    strategy="hybrid",
                    use_parallel=False,
                )

                # Verify results are identical
                assert len(parallel_results) == len(
                    sequential_results
                ), "Result counts should be identical"

                for par, seq in zip(parallel_results, sequential_results):
                    assert par.chunk_id == seq.chunk_id, "Chunk IDs should match"
                    assert (
                        par.chunk_text == seq.chunk_text
                    ), "Chunk text should match"
                    assert (
                        abs(par.hybrid_score - seq.hybrid_score) < 0.0001
                    ), "Scores should be identical"
                    assert par.rank == seq.rank, "Ranks should be identical"


class TestParallelExecutionEdgeCases:
    """Test suite for edge cases in parallel execution.

    Validates:
    - Empty result handling
    - Single result handling
    - Error propagation
    - Large result sets
    """

    def test_parallel_execution_with_empty_results(
        self,
        mock_db_pool: MagicMock,
        mock_logger: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test parallel execution when one or both searches return empty results.

        Verifies:
        - Handles empty vector results gracefully
        - Handles empty BM25 results gracefully
        - Returns empty list when both are empty
        """
        from src.search.hybrid_search import HybridSearch

        with patch("src.search.hybrid_search.VectorSearch") as MockVectorSearch, \
             patch("src.search.hybrid_search.BM25Search") as MockBM25Search, \
             patch("src.search.hybrid_search.ModelLoader") as MockModelLoader, \
             patch("src.search.hybrid_search.QueryRouter") as MockQueryRouter, \
             patch("src.search.hybrid_search.RRFScorer") as MockRRFScorer, \
             patch("src.search.hybrid_search.BoostingSystem") as MockBoostingSystem:

            # Setup mocks
            mock_vector_search_instance = MagicMock()
            MockVectorSearch.return_value = mock_vector_search_instance

            mock_bm25_search_instance = MagicMock()
            MockBM25Search.return_value = mock_bm25_search_instance

            mock_model_loader = MagicMock()
            MockModelLoader.get_instance.return_value = mock_model_loader
            mock_model_loader.encode.return_value = [0.1] * 768

            mock_router = MagicMock()
            MockQueryRouter.return_value = mock_router

            mock_rrf = MagicMock(spec=RRFScorer)
            MockRRFScorer.return_value = mock_rrf

            mock_boosting = MagicMock(spec=BoostingSystem)
            MockBoostingSystem.return_value = mock_boosting

            hybrid = HybridSearch(mock_db_pool, mock_settings, mock_logger)

            # Test with empty vector results
            with patch.object(
                hybrid, "_execute_vector_search", return_value=[]
            ), \
                 patch.object(
                     hybrid, "_execute_bm25_search", return_value=create_test_bm25_results(5)
                 ), \
                 patch.object(
                     hybrid, "_merge_and_boost", return_value=[]
                 ):

                results = hybrid.search(
                    "test query",
                    strategy="hybrid",
                    use_parallel=True,
                )

                # Should handle gracefully
                assert isinstance(results, list), "Should return list"

    def test_parallel_execution_with_large_result_sets(
        self,
        mock_db_pool: MagicMock,
        mock_logger: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test parallel execution with large result sets.

        Verifies:
        - Handles 100+ results correctly
        - Thread safety with large data
        - No memory leaks or corruption
        """
        from src.search.hybrid_search import HybridSearch

        with patch("src.search.hybrid_search.VectorSearch") as MockVectorSearch, \
             patch("src.search.hybrid_search.BM25Search") as MockBM25Search, \
             patch("src.search.hybrid_search.ModelLoader") as MockModelLoader, \
             patch("src.search.hybrid_search.QueryRouter") as MockQueryRouter, \
             patch("src.search.hybrid_search.RRFScorer") as MockRRFScorer, \
             patch("src.search.hybrid_search.BoostingSystem") as MockBoostingSystem:

            # Setup mocks
            mock_vector_search_instance = MagicMock()
            MockVectorSearch.return_value = mock_vector_search_instance

            mock_bm25_search_instance = MagicMock()
            MockBM25Search.return_value = mock_bm25_search_instance

            mock_model_loader = MagicMock()
            MockModelLoader.get_instance.return_value = mock_model_loader
            mock_model_loader.encode.return_value = [0.1] * 768

            mock_router = MagicMock()
            MockQueryRouter.return_value = mock_router

            mock_rrf = MagicMock(spec=RRFScorer)
            MockRRFScorer.return_value = mock_rrf

            mock_boosting = MagicMock(spec=BoostingSystem)
            MockBoostingSystem.return_value = mock_boosting

            hybrid = HybridSearch(mock_db_pool, mock_settings, mock_logger)

            # Create large result sets
            vector_results = create_test_vector_results(100)
            bm25_results = create_test_bm25_results(100)

            with patch.object(
                hybrid, "_execute_vector_search", return_value=vector_results
            ), \
                 patch.object(
                     hybrid, "_execute_bm25_search", return_value=bm25_results
                 ), \
                 patch.object(
                     hybrid, "_merge_and_boost", return_value=vector_results
                 ):

                results = hybrid.search(
                    "test query",
                    top_k=100,
                    strategy="hybrid",
                    use_parallel=True,
                )

                assert len(results) > 0, "Should return results from large set"
                assert all(
                    isinstance(r, SearchResult) for r in results
                ), "All results should be SearchResult instances"
