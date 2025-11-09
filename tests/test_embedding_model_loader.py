"""Comprehensive test suite for embedding model loader.

Tests cover:
- Model loading and initialization
- Singleton pattern and caching behavior
- Model validation and dimension checking
- Device placement (CPU/GPU)
- Error handling and recovery
- Thread safety and concurrent access
- Cache reset and reloading
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from src.embedding.model_loader import (
    DEFAULT_MODEL_NAME,
    EXPECTED_EMBEDDING_DIMENSION,
    ModelLoadError,
    ModelLoader,
    ModelValidationError,
)


class TestModelLoaderInitialization:
    """Tests for ModelLoader initialization and configuration."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_model_loader_initializes_with_defaults(self) -> None:
        """Test ModelLoader initializes with default configuration."""
        loader = ModelLoader()
        assert loader._model_name == DEFAULT_MODEL_NAME
        assert loader._device in ("cpu", "cuda")
        assert loader._model is None

    def test_model_loader_initializes_with_custom_device(self) -> None:
        """Test ModelLoader accepts custom device parameter."""
        loader = ModelLoader(device="cpu")
        assert loader._device == "cpu"

    def test_model_loader_raises_on_invalid_device(self) -> None:
        """Test ModelLoader raises ValueError for invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            ModelLoader(device="gpu")

    def test_model_loader_accepts_custom_cache_dir(self) -> None:
        """Test ModelLoader accepts custom cache directory."""
        cache_dir = "/tmp/test_cache"
        loader = ModelLoader(cache_dir=cache_dir)
        assert loader._cache_dir == Path(cache_dir)

    def test_model_loader_accepts_path_object_for_cache_dir(self) -> None:
        """Test ModelLoader accepts Path object for cache directory."""
        cache_dir = Path("/tmp/test_cache")
        loader = ModelLoader(cache_dir=cache_dir)
        assert loader._cache_dir == cache_dir


class TestModelLoaderSingleton:
    """Tests for singleton pattern implementation."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_get_instance_returns_same_object(self) -> None:
        """Test get_instance returns same instance on multiple calls."""
        instance1 = ModelLoader.get_instance()
        instance2 = ModelLoader.get_instance()
        assert instance1 is instance2

    def test_get_instance_ignores_subsequent_parameters(self) -> None:
        """Test get_instance ignores device parameter on subsequent calls."""
        loader1 = ModelLoader.get_instance(device="cpu")
        loader2 = ModelLoader.get_instance(device="cuda")
        # Parameters on second call are ignored
        assert loader1 is loader2
        assert loader1._device == "cpu"

    def test_singleton_pattern_across_threads(self) -> None:
        """Test singleton pattern works correctly across threads."""
        instances: list[ModelLoader] = []

        def get_loader_in_thread() -> None:
            """Get loader instance in separate thread."""
            instance = ModelLoader.get_instance()
            instances.append(instance)

        threads = [threading.Thread(target=get_loader_in_thread) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be identical
        assert len(instances) == 5
        assert all(instance is instances[0] for instance in instances)


class TestModelLoading:
    """Tests for actual model loading functionality."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test - skipped when SKIP_SLOW_TESTS=1",
    )
    def test_load_model_successfully(self) -> None:
        """Test successful model loading."""
        loader = ModelLoader.get_instance(device="cpu")
        model = loader.get_model()
        assert model is not None
        assert isinstance(model, SentenceTransformer)

    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test - skipped when SKIP_SLOW_TESTS=1",
    )
    def test_model_loads_only_once(self) -> None:
        """Test model is loaded only once with caching."""
        loader = ModelLoader.get_instance(device="cpu")
        model1 = loader.get_model()
        model2 = loader.get_model()
        assert model1 is model2

    def test_load_model_with_mocked_sentence_transformers(self) -> None:
        """Test model loading with mocked sentence-transformers."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.randn(1, 768)
        mock_model.encode.return_value = mock_embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            model = loader.get_model()
            assert model is mock_model
            assert loader._model is mock_model


class TestModelValidation:
    """Tests for model validation logic."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_validation_checks_embedding_dimension(self) -> None:
        """Test validation checks for correct embedding dimension."""
        mock_model = MagicMock(spec=SentenceTransformer)
        # Wrong dimension
        mock_model.get_sentence_embedding_dimension.return_value = 512

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelValidationError, match="dimension"):
                loader.get_model()

    def test_validation_checks_embedding_generation(self) -> None:
        """Test validation checks that embeddings can be generated."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        # Return empty list for encode
        mock_model.encode.return_value = []

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelValidationError, match="test embeddings"):
                loader.get_model()

    def test_validation_checks_embedding_dimension_after_generation(self) -> None:
        """Test validation checks embedding dimension matches after generation."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        # Return embedding with wrong dimension
        mock_embeddings = np.random.randn(1, 512)
        mock_model.encode.return_value = mock_embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelValidationError, match="dimension"):
                loader.get_model()

    def test_validation_handles_missing_method(self) -> None:
        """Test validation fails gracefully if model missing required method."""
        mock_model = MagicMock(spec=SentenceTransformer)
        # Remove the method
        del mock_model.get_sentence_embedding_dimension

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelValidationError):
                loader.get_model()


class TestGetModelDimension:
    """Tests for getting model embedding dimension."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_get_model_dimension_returns_768(self) -> None:
        """Test get_model_dimension returns 768 for all-mpnet-base-v2."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.randn(1, 768)
        mock_model.encode.return_value = mock_embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            dimension = loader.get_model_dimension()
            assert dimension == 768


class TestCacheReset:
    """Tests for cache reset functionality."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_reset_cache_clears_model(self) -> None:
        """Test reset_cache clears cached model."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.randn(1, 768)
        mock_model.encode.return_value = mock_embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            model1 = loader.get_model()
            assert loader._model is not None

            loader.reset_cache()
            assert loader._model is None

    def test_reset_cache_allows_reloading(self) -> None:
        """Test reset_cache allows model to be reloaded."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.randn(1, 768)
        mock_model.encode.return_value = mock_embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            model1 = loader.get_model()
            loader.reset_cache()

            # Reset model to test reload
            mock_model.reset_mock()
            mock_model.encode.return_value = mock_embeddings.tolist()

            model2 = loader.get_model()
            # New call to SentenceTransformer should have been made
            assert mock_model.encode.call_count >= 2


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_load_error_on_download_failure(self) -> None:
        """Test ModelLoadError raised on download failure."""
        with patch("src.embedding.model_loader.SentenceTransformer", side_effect=ConnectionError("Network error")):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelLoadError, match="Network error"):
                loader.get_model()

    def test_load_error_on_disk_space_failure(self) -> None:
        """Test ModelLoadError raised on disk space error."""
        with patch("src.embedding.model_loader.SentenceTransformer", side_effect=OSError("No space left")):
            loader = ModelLoader(device="cpu")
            with pytest.raises(ModelLoadError, match="No space left"):
                loader.get_model()

    def test_load_error_categorization(self) -> None:
        """Test error categorization for different error types."""
        errors: dict[str, str] = {
            "connection": "Connection refused",
            "disk_space": "No space on device",
            "permissions": "Permission denied",
        }

        for error_type, error_msg in errors.items():
            ModelLoader._instance = None
            with patch("src.embedding.model_loader.SentenceTransformer", side_effect=OSError(error_msg)):
                loader = ModelLoader(device="cpu")
                with pytest.raises(ModelLoadError):
                    loader.get_model()


class TestDevicePlacement:
    """Tests for device placement and GPU/CPU handling."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_device_placement_cpu(self) -> None:
        """Test model loader respects CPU device setting."""
        loader = ModelLoader(device="cpu")
        assert loader._device == "cpu"

    def test_get_device_returns_valid_device(self) -> None:
        """Test _get_device returns cpu or cuda."""
        device = ModelLoader._get_device()
        assert device in ("cpu", "cuda")


class TestEmbeddingGeneration:
    """Tests for embedding generation through model."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_encode_generates_correct_dimension(self) -> None:
        """Test encode generates embeddings with correct dimensions."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        embeddings = np.random.randn(3, 768)
        mock_model.encode.return_value = embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            loader.get_model()

            # Now test encode
            texts = ["Hello", "World", "Test"]
            result = loader.get_model().encode(texts, convert_to_tensor=False)
            assert len(result) == 3
            assert len(result[0]) == 768

    def test_encode_handles_batch_processing(self) -> None:
        """Test encode handles batch processing correctly."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        embeddings = np.random.randn(100, 768)
        mock_model.encode.return_value = embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader(device="cpu")
            loader.get_model()

            texts = [f"Text {i}" for i in range(100)]
            result = loader.get_model().encode(texts, convert_to_tensor=False)
            assert len(result) == 100


class TestModelLoaderIntegration:
    """Integration tests for complete model loading workflow."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_complete_model_loading_workflow(self) -> None:
        """Test complete workflow from initialization to embedding generation."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        embeddings = np.random.randn(2, 768)
        mock_model.encode.return_value = embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            # Initialize loader
            loader = ModelLoader.get_instance(device="cpu")

            # Get model
            model = loader.get_model()
            assert model is not None

            # Get dimension
            dim = loader.get_model_dimension()
            assert dim == 768

            # Verify singleton
            loader2 = ModelLoader.get_instance()
            assert loader is loader2


class TestModelLoaderTypeAnnotations:
    """Tests to verify type safety of ModelLoader."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_loader_returns_correct_types(self) -> None:
        """Test ModelLoader methods return correct types."""
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        embeddings = np.random.randn(1, 768)
        mock_model.encode.return_value = embeddings.tolist()

        with patch("src.embedding.model_loader.SentenceTransformer", return_value=mock_model):
            loader = ModelLoader.get_instance()

            # Return type checks
            model: SentenceTransformer = loader.get_model()
            assert isinstance(model, MagicMock) or isinstance(model, SentenceTransformer)

            dimension: int = loader.get_model_dimension()
            assert isinstance(dimension, int)
            assert dimension == 768
