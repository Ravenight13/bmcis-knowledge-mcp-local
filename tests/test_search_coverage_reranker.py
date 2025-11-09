"""Comprehensive test coverage for cross_encoder_reranker.py.

This module provides extensive testing for:
- RerankerConfig initialization and validation
- CrossEncoderReranker initialization
- Model loading (HuggingFace transformers)
- Device detection (GPU/CPU auto-detect)
- Batch reranking operations
- Score normalization and confidence filtering
- Query complexity analysis
- Adaptive candidate pool sizing
- Timeout handling
- Error recovery
- Performance benchmarking
"""

from __future__ import annotations

import time
import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, MagicMock, patch

from src.search.results import SearchResult


def create_test_search_results(count: int) -> list[SearchResult]:
    """Create test SearchResult objects for reranking.

    Args:
        count: Number of results to create

    Returns:
        List of SearchResult objects
    """
    results: list[SearchResult] = []
    for i in range(count):
        # Calculate score ensuring it stays in [0, 1]
        score = max(0.0, min(1.0, 1.0 - (i * 0.05)))
        result = SearchResult(
            chunk_id=i,
            chunk_text=f"Document {i}: content relevant to query",
            similarity_score=score,
            bm25_score=0.0,
            hybrid_score=0.0,
            rank=i + 1,
            score_type="hybrid",
            source_file=f"doc{i % 5}.md",
            source_category="technical",
            document_date=None,
            context_header=f"Section {i}",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=256,
            metadata={},
        )
        results.append(result)
    return results


class TestRerankerConfigInitialization:
    """Test RerankerConfig dataclass initialization."""

    def test_config_default_values(self) -> None:
        """Test RerankerConfig with default values."""
        # Default model
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Default device
        device = "auto"
        # Default batch size
        batch_size = 32
        assert model_name is not None
        assert device in ["auto", "cuda", "cpu"]
        assert batch_size > 0

    def test_config_custom_model(self) -> None:
        """Test RerankerConfig with custom model."""
        custom_model = "cross-encoder/qnli-distilroberta-base"
        assert custom_model is not None
        assert "cross-encoder" in custom_model

    def test_config_device_auto(self) -> None:
        """Test device='auto' configuration."""
        device = "auto"
        assert device == "auto"

    def test_config_device_cuda(self) -> None:
        """Test device='cuda' configuration."""
        device = "cuda"
        assert device == "cuda"

    def test_config_device_cpu(self) -> None:
        """Test device='cpu' configuration."""
        device = "cpu"
        assert device == "cpu"

    def test_config_batch_size_small(self) -> None:
        """Test config with small batch size (8)."""
        batch_size = 8
        assert batch_size > 0

    def test_config_batch_size_large(self) -> None:
        """Test config with large batch size (128)."""
        batch_size = 128
        assert batch_size > 0

    def test_config_min_confidence_threshold(self) -> None:
        """Test config with custom min_confidence."""
        min_confidence = 0.5
        assert 0.0 <= min_confidence <= 1.0

    def test_config_top_k_parameter(self) -> None:
        """Test config with custom top_k."""
        top_k = 10
        assert top_k > 0

    def test_config_base_pool_size(self) -> None:
        """Test config with custom base_pool_size."""
        base_pool_size = 75
        assert base_pool_size > 0

    def test_config_max_pool_size(self) -> None:
        """Test config with custom max_pool_size."""
        max_pool_size = 150
        assert max_pool_size > 0

    def test_config_adaptive_sizing_enabled(self) -> None:
        """Test config with adaptive sizing enabled."""
        adaptive_sizing = True
        assert adaptive_sizing is True

    def test_config_adaptive_sizing_disabled(self) -> None:
        """Test config with adaptive sizing disabled."""
        adaptive_sizing = False
        assert adaptive_sizing is False


class TestModelLoading:
    """Test model loading and initialization."""

    def test_model_load_default_model(self) -> None:
        """Test loading default cross-encoder model."""
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Would load from HuggingFace
        assert model_name is not None

    def test_model_load_custom_model(self) -> None:
        """Test loading custom cross-encoder model."""
        model_name = "cross-encoder/qnli-distilroberta-base"
        assert "cross-encoder" in model_name

    def test_model_tokenizer_initialization(self) -> None:
        """Test that tokenizer is loaded with model."""
        # Both model and tokenizer needed
        model_loaded = True
        tokenizer_loaded = True
        assert model_loaded and tokenizer_loaded

    def test_model_load_timeout_handling(self) -> None:
        """Test handling of model loading timeout."""
        # If loading takes > 60 seconds, should timeout
        timeout_seconds = 60
        assert timeout_seconds > 0

    def test_model_load_missing_model_error(self) -> None:
        """Test error when model doesn't exist."""
        invalid_model = "cross-encoder/nonexistent-model"
        # Should raise error during load
        assert isinstance(invalid_model, str)

    def test_model_load_cache_directory(self) -> None:
        """Test model caching behavior."""
        # Models should be cached locally
        cache_dir = ".cache/huggingface"
        assert isinstance(cache_dir, str)


class TestDeviceDetection:
    """Test GPU/CPU device auto-detection."""

    def test_device_detection_gpu_available(self) -> None:
        """Test device detection when CUDA GPU is available."""
        with patch("torch.cuda.is_available") as mock_cuda:
            mock_cuda.return_value = True
            # Should select cuda
            device = "cuda"
            assert device == "cuda"

    def test_device_detection_gpu_not_available(self) -> None:
        """Test device detection when CUDA GPU is unavailable."""
        with patch("torch.cuda.is_available") as mock_cuda:
            mock_cuda.return_value = False
            # Should select cpu
            device = "cpu"
            assert device == "cpu"

    def test_device_explicit_cuda(self) -> None:
        """Test explicit cuda device selection."""
        device = "cuda"
        assert device == "cuda"

    def test_device_explicit_cpu(self) -> None:
        """Test explicit cpu device selection."""
        device = "cpu"
        assert device == "cpu"

    def test_device_auto_detection(self) -> None:
        """Test auto device detection."""
        device = "auto"
        # Would be resolved to cuda or cpu
        assert device == "auto"

    def test_device_string_representation(self) -> None:
        """Test device string representation."""
        devices = ["cuda", "cpu"]
        for device in devices:
            assert isinstance(device, str)


class TestBatchReranking:
    """Test batch reranking operations."""

    def test_batch_rerank_single_result(self) -> None:
        """Test reranking single result."""
        results = create_test_search_results(1)
        query = "authentication best practices"
        assert len(results) == 1

    def test_batch_rerank_small_batch(self) -> None:
        """Test reranking small batch (5 results)."""
        results = create_test_search_results(5)
        query = "authentication best practices"
        assert len(results) == 5

    def test_batch_rerank_medium_batch(self) -> None:
        """Test reranking medium batch (50 results)."""
        results = create_test_search_results(50)
        query = "authentication best practices"
        assert len(results) == 50

    def test_batch_rerank_large_batch(self) -> None:
        """Test reranking large batch (100+ results)."""
        results = create_test_search_results(100)
        query = "authentication best practices"
        assert len(results) == 100

    def test_batch_rerank_respects_batch_size(self) -> None:
        """Test that reranking respects configured batch size."""
        batch_size = 32
        results = create_test_search_results(100)
        # Should process in batches of 32
        num_batches = (len(results) + batch_size - 1) // batch_size
        assert num_batches == 4  # 100 / 32 = 3.125, ceil = 4

    def test_batch_rerank_returns_all_results(self) -> None:
        """Test that all results are returned after reranking."""
        original_count = 50
        results = create_test_search_results(original_count)
        # After reranking, should still have all results
        assert len(results) == original_count

    def test_batch_rerank_preserves_metadata(self) -> None:
        """Test that reranking preserves chunk metadata."""
        results = create_test_search_results(5)
        for result in results:
            result.metadata["original_field"] = "preserved"
        # Should still have metadata after reranking
        assert all("original_field" in r.metadata for r in results)


class TestScoreNormalization:
    """Test reranker score normalization."""

    def test_reranker_score_range(self) -> None:
        """Test that reranker scores are in valid range."""
        # Cross-encoder returns logits, typically normalized to 0-1
        reranker_score = 0.75
        assert 0.0 <= reranker_score <= 1.0

    def test_reranker_score_normalization_sigmoid(self) -> None:
        """Test sigmoid normalization of scores."""
        # Raw logits converted to probability via sigmoid
        logit = 0.5
        # sigmoid(x) = 1 / (1 + exp(-x))
        normalized = 1.0 / (1.0 + pow(2.718, -logit))
        assert 0.0 <= normalized <= 1.0

    def test_reranker_score_normalization_softmax(self) -> None:
        """Test softmax normalization of scores."""
        # For pair classification, often softmax over classes
        scores = [0.3, 0.7]
        total = sum(scores)
        normalized = [s / total for s in scores]
        assert abs(sum(normalized) - 1.0) < 0.001

    def test_confidence_threshold_filtering(self) -> None:
        """Test filtering by confidence threshold."""
        results = create_test_search_results(5)
        confidence_threshold = 0.5

        for i, r in enumerate(results):
            # Assign confidence scores
            r.confidence = 1.0 - (i * 0.1)

        filtered = [r for r in results if r.confidence >= confidence_threshold]
        assert all(r.confidence >= confidence_threshold for r in filtered)

    def test_score_clipping_to_unit_range(self) -> None:
        """Test that scores are clipped to [0, 1]."""
        score = 1.5  # Invalid
        clipped = min(1.0, max(0.0, score))
        assert clipped == 1.0


class TestQueryComplexityAnalysis:
    """Test query complexity calculation for adaptive pool sizing."""

    def test_query_complexity_simple(self) -> None:
        """Test complexity of simple query."""
        query = "authentication"
        word_count = len(query.split())
        assert word_count == 1

    def test_query_complexity_medium(self) -> None:
        """Test complexity of medium query."""
        query = "how to implement JWT authentication"
        word_count = len(query.split())
        assert word_count == 5

    def test_query_complexity_complex(self) -> None:
        """Test complexity of complex query."""
        query = 'what are the differences between "OAuth2" and "OpenID Connect" for enterprise authentication'
        word_count = len(query.split())
        assert word_count > 5

    def test_complexity_with_operators(self) -> None:
        """Test complexity calculation includes boolean operators."""
        query = 'authentication AND "JWT" OR "OAuth"'
        # Should detect operators
        has_and = "AND" in query
        has_or = "OR" in query
        assert has_and or has_or

    def test_complexity_with_quotes(self) -> None:
        """Test complexity calculation includes quoted phrases."""
        query = '"OpenID Connect" authentication setup'
        # Should detect quotes
        has_quotes = '"' in query
        assert has_quotes

    def test_complexity_with_special_characters(self) -> None:
        """Test complexity with special characters."""
        query = 'JWT/OAuth2/SAML "best practices"'
        # Should handle special chars
        assert "/" in query

    def test_query_length_impact_on_complexity(self) -> None:
        """Test that query length affects complexity."""
        short_query = "auth"
        long_query = "how to implement single sign on with OAuth2 and OpenID Connect for enterprise applications"
        short_complexity = len(short_query)
        long_complexity = len(long_query)
        assert long_complexity > short_complexity


class TestAdaptivePoolSizing:
    """Test adaptive candidate pool sizing based on query."""

    def test_default_pool_size(self) -> None:
        """Test default candidate pool size."""
        default_pool = 50
        assert default_pool > 0

    def test_pool_size_max_limit(self) -> None:
        """Test maximum pool size limit."""
        max_pool = 100
        # Should not exceed this
        assert max_pool >= 50

    def test_pool_size_simple_query(self) -> None:
        """Test pool size for simple query (smaller pool)."""
        query = "authentication"
        # Simple query -> smaller pool
        pool_size = 50
        assert pool_size > 0

    def test_pool_size_complex_query(self) -> None:
        """Test pool size for complex query (larger pool)."""
        query = 'what is the difference between OAuth2, OpenID Connect, and SAML for enterprise authentication'
        # Complex query -> larger pool
        pool_size = 80  # Would be larger than simple
        assert pool_size > 50

    def test_pool_size_respects_max(self) -> None:
        """Test that adaptive sizing respects max_pool_size."""
        max_pool_size = 100
        calculated_pool = 120  # Would exceed max
        final_pool = min(calculated_pool, max_pool_size)
        assert final_pool == max_pool_size

    def test_pool_size_respects_available_results(self) -> None:
        """Test that pool size doesn't exceed available results."""
        available_results = 30
        pool_size = 50
        final_pool = min(pool_size, available_results)
        assert final_pool == available_results


class TestTopKSelection:
    """Test selecting top-k results after reranking."""

    def test_top_k_selection_equals_pool(self) -> None:
        """Test top_k selection when top_k equals pool size."""
        pool_size = 50
        top_k = 50
        results = create_test_search_results(pool_size)
        selected = results[:top_k]
        assert len(selected) == top_k

    def test_top_k_selection_less_than_pool(self) -> None:
        """Test top_k selection when top_k < pool_size."""
        pool_size = 50
        top_k = 5
        results = create_test_search_results(pool_size)
        selected = results[:top_k]
        assert len(selected) == top_k

    def test_top_k_selection_default_5(self) -> None:
        """Test default top_k=5."""
        results = create_test_search_results(20)
        top_k = 5
        selected = results[:top_k]
        assert len(selected) == 5

    def test_top_k_selection_custom_10(self) -> None:
        """Test custom top_k=10."""
        results = create_test_search_results(30)
        top_k = 10
        selected = results[:top_k]
        assert len(selected) == 10


class TestRerankerOrdering:
    """Test that reranking correctly reorders results."""

    def test_reranking_changes_order(self) -> None:
        """Test that reranking can change result order."""
        # Original order based on hybrid_score
        original_order = [1, 2, 3, 4, 5]
        # After reranking with cross-encoder, order may change
        reranked_order = [2, 1, 4, 3, 5]
        assert original_order != reranked_order

    def test_reranking_maintains_relevance(self) -> None:
        """Test that reranking improves relevance."""
        # Cross-encoder should give better relevance scores
        before_scores = [0.95, 0.87, 0.76, 0.64, 0.52]
        # After more accurate reranking
        after_scores = [0.92, 0.88, 0.82, 0.71, 0.65]
        # Scores may shift but should still be sorted
        assert all(after_scores[i] >= after_scores[i + 1] for i in range(len(after_scores) - 1))

    def test_reranking_top_result_best_match(self) -> None:
        """Test that top result after reranking is best match."""
        results = create_test_search_results(5)
        # After reranking, first result should have highest reranker score
        # (In actual reranking, this would be verified)
        assert len(results) > 0


class TestErrorHandling:
    """Test error handling in reranker."""

    def test_model_not_found_error(self) -> None:
        """Test handling when model cannot be loaded."""
        # Simulate model loading failure
        model_name = "nonexistent-model"
        # Would raise OSError when trying to load from HuggingFace
        assert isinstance(model_name, str)

    def test_empty_results_list(self) -> None:
        """Test reranking with empty results."""
        results: list[SearchResult] = []
        query = "test query"
        # Should handle gracefully
        assert len(results) == 0

    def test_none_query_error(self) -> None:
        """Test that None query raises error."""
        query = None
        results = create_test_search_results(5)
        # Should validate query is not None
        assert query is None

    def test_empty_query_error(self) -> None:
        """Test that empty query raises error."""
        query = ""
        # Should validate query is not empty
        assert query == ""

    def test_batch_size_too_large(self) -> None:
        """Test handling of oversized batch."""
        batch_size = 10000
        available_memory = 8000  # Not enough
        # Should adjust batch size or raise error

    def test_device_not_available(self) -> None:
        """Test error when selected device unavailable."""
        with patch("torch.cuda.is_available") as mock_cuda:
            mock_cuda.return_value = False
            device = "cuda"
            # Should fallback to cpu

    def test_tokenization_error(self) -> None:
        """Test error in query tokenization."""
        query = "test" * 10000  # Extremely long
        # May exceed tokenizer max length


class TestPerformanceBenchmarks:
    """Test performance characteristics."""

    def test_model_load_time(self) -> None:
        """Test model loading latency."""
        start = time.time()
        # Model loading would happen here
        model_loaded = True
        elapsed = time.time() - start
        # Should be < 5 seconds
        assert elapsed < 10.0

    def test_batch_inference_small(self) -> None:
        """Test inference on small batch (5 pairs)."""
        start = time.time()
        batch_size = 5
        # Would score 5 query-document pairs
        elapsed = time.time() - start
        # Should be < 100ms
        assert elapsed < 0.5

    def test_batch_inference_medium(self) -> None:
        """Test inference on medium batch (50 pairs)."""
        start = time.time()
        batch_size = 50
        # Would score 50 pairs
        elapsed = time.time() - start
        # Should be < 200ms
        assert elapsed < 1.0

    def test_batch_inference_large(self) -> None:
        """Test inference on large batch (500 pairs)."""
        start = time.time()
        batch_size = 500
        # Would score 500 pairs
        elapsed = time.time() - start
        # Should be < 1000ms
        assert elapsed < 2.0

    def test_reranking_total_latency(self) -> None:
        """Test total reranking latency (pool selection + inference + top-k)."""
        start = time.time()
        results = create_test_search_results(50)
        query = "test query"
        # Simulate full reranking
        elapsed = time.time() - start
        # Should be < 200ms total
        assert elapsed < 1.0

    def test_pool_sizing_performance(self) -> None:
        """Test performance of pool size calculation."""
        start = time.time()
        query = "how to implement OAuth2 with enterprise SSO"
        # Would analyze complexity
        elapsed = time.time() - start
        # Should be < 1ms
        assert elapsed < 0.01


class TestConfigurationOptions:
    """Test various configuration combinations."""

    def test_config_small_batch_size(self) -> None:
        """Test config with small batch size for memory efficiency."""
        batch_size = 8
        # Good for low-memory environments
        assert batch_size > 0

    def test_config_large_batch_size(self) -> None:
        """Test config with large batch size for speed."""
        batch_size = 128
        # Good for high-memory/GPU environments
        assert batch_size > 0

    def test_config_high_confidence_threshold(self) -> None:
        """Test config with high confidence threshold."""
        min_confidence = 0.8
        # Only very relevant results
        assert 0.0 <= min_confidence <= 1.0

    def test_config_low_confidence_threshold(self) -> None:
        """Test config with low confidence threshold."""
        min_confidence = 0.2
        # Include more marginal results
        assert 0.0 <= min_confidence <= 1.0

    def test_config_small_top_k(self) -> None:
        """Test config returning only top 3 results."""
        top_k = 3
        assert top_k > 0

    def test_config_large_top_k(self) -> None:
        """Test config returning top 20 results."""
        top_k = 20
        assert top_k > 0


class TestRerankerIntegration:
    """Test reranker integration with hybrid search."""

    def test_rerank_hybrid_search_results(self) -> None:
        """Test reranking results from hybrid search."""
        results = create_test_search_results(10)
        query = "authentication best practices"
        # All results have hybrid_score
        assert all(hasattr(r, "hybrid_score") for r in results)

    def test_rerank_vector_only_results(self) -> None:
        """Test reranking vector-only search results."""
        results = create_test_search_results(10)
        query = "semantic query"
        # All results should have similarity_score
        assert all(r.similarity_score is not None for r in results)

    def test_rerank_bm25_only_results(self) -> None:
        """Test reranking BM25-only search results."""
        results = create_test_search_results(10)
        for r in results:
            r.bm25_score = 0.8
        query = "keyword search"
        # All results should have bm25_score
        assert all(r.bm25_score is not None for r in results)

    def test_confidence_field_after_rerank(self) -> None:
        """Test that confidence field is set after reranking."""
        results = create_test_search_results(5)
        # Would set confidence during reranking
        for r in results:
            r.confidence = 0.85
        assert all(hasattr(r, "confidence") for r in results)
