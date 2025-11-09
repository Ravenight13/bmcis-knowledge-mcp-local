"""Unit tests for sentence-transformers model loader with caching.

Tests cover:
- Model loading with caching
- Singleton pattern enforcement
- Error handling for download failures
- Model validation
- Device configuration
- Reset and cache management
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.embedding.model_loader import (
    DEFAULT_MODEL_NAME,
    EXPECTED_EMBEDDING_DIMENSION,
    ModelLoadError,
    ModelLoader,
    ModelValidationError,
)


class TestModelLoadError:
    """Tests for ModelLoadError exception."""

    def test_model_load_error_initialization(self) -> None:
        """Test ModelLoadError can be initialized with message."""
        message = "Failed to download model"
        error = ModelLoadError(message)

        assert error.message == message
        assert str(error) == message

    def test_model_load_error_inheritance(self) -> None:
        """Test ModelLoadError inherits from Exception."""
        error = ModelLoadError("test error")
        assert isinstance(error, Exception)

    def test_model_load_error_with_cause(self) -> None:
        """Test ModelLoadError can wrap another exception."""
        original = RuntimeError("Network error")
        error = ModelLoadError("Failed to load") from original

        assert error.__cause__ is original


class TestModelValidationError:
    """Tests for ModelValidationError exception."""

    def test_model_validation_error_initialization(self) -> None:
        """Test ModelValidationError can be initialized with message."""
        message = "Model dimension mismatch"
        error = ModelValidationError(message)

        assert error.message == message
        assert str(error) == message

    def test_model_validation_error_inheritance(self) -> None:
        """Test ModelValidationError inherits from Exception."""
        error = ModelValidationError("test error")
        assert isinstance(error, Exception)


class TestModelLoaderInitialization:
    """Tests for ModelLoader initialization."""

    def test_initialization_with_defaults(self) -> None:
        """Test ModelLoader initialization with default parameters."""
        loader = ModelLoader()

        assert loader._model_name == DEFAULT_MODEL_NAME
        assert loader._device == "cpu"
        assert loader._cache_dir is None
        assert loader._model is None

    def test_initialization_with_custom_model_name(self) -> None:
        """Test ModelLoader initialization with custom model name."""
        custom_name = "sentence-transformers/all-MiniLM-L6-v2"
        loader = ModelLoader(model_name=custom_name)

        assert loader._model_name == custom_name

    def test_initialization_with_cuda_device(self) -> None:
        """Test ModelLoader initialization with CUDA device."""
        loader = ModelLoader(device="cuda")
        assert loader._device == "cuda"

    def test_initialization_with_cpu_device(self) -> None:
        """Test ModelLoader initialization with CPU device."""
        loader = ModelLoader(device="cpu")
        assert loader._device == "cpu"

    def test_initialization_with_invalid_device_raises(self) -> None:
        """Test ModelLoader raises ValueError for invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            ModelLoader(device="gpu")

    def test_initialization_with_cache_dir_string(self) -> None:
        """Test ModelLoader accepts cache directory as string."""
        cache_path = "/tmp/model_cache"
        loader = ModelLoader(cache_dir=cache_path)

        assert loader._cache_dir == Path(cache_path)

    def test_initialization_with_cache_dir_path(self) -> None:
        """Test ModelLoader accepts cache directory as Path object."""
        cache_path = Path("/tmp/model_cache")
        loader = ModelLoader(cache_dir=cache_path)

        assert loader._cache_dir == cache_path

    def test_logger_initialization(self) -> None:
        """Test ModelLoader initializes logger."""
        loader = ModelLoader()
        assert isinstance(loader.logger, logging.Logger)


class TestModelLoaderSingleton:
    """Tests for ModelLoader singleton pattern."""

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    def test_get_instance_creates_singleton(self) -> None:
        """Test get_instance() creates singleton instance."""
        instance1 = ModelLoader.get_instance()
        instance2 = ModelLoader.get_instance()

        assert instance1 is instance2

    def test_get_instance_with_parameters_first_call(self) -> None:
        """Test get_instance() uses parameters on first call."""
        custom_model = "custom-model"
        instance = ModelLoader.get_instance(model_name=custom_model)

        assert instance._model_name == custom_model

    def test_get_instance_ignores_parameters_subsequent_calls(self) -> None:
        """Test get_instance() ignores parameters on subsequent calls."""
        instance1 = ModelLoader.get_instance(
            model_name="model-1",
            device="cpu",
        )
        instance2 = ModelLoader.get_instance(
            model_name="model-2",
            device="cuda",
        )

        assert instance1 is instance2
        assert instance1._model_name == "model-1"
        assert instance1._device == "cpu"

    def test_singleton_isolation(self) -> None:
        """Test singleton doesn't share state with new instances."""
        ModelLoader._instance = None

        singleton = ModelLoader.get_instance()
        new_instance = ModelLoader()

        # They are different objects
        assert singleton is not new_instance

        # Both should work independently
        assert singleton._model_name == new_instance._model_name


class TestModelLoaderGetModel:
    """Tests for model loading and caching."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_returns_cached_instance(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model() returns cached model on second call."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()

        # First call should load model
        model1 = loader.get_model()
        assert mock_st.call_count == 1

        # Second call should use cache
        model2 = loader.get_model()
        assert mock_st.call_count == 1  # Not called again
        assert model1 is model2

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_validates_model(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model() validates model before returning."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()
        model = loader.get_model()

        # Verify validation was called
        mock_model.get_sentence_embedding_dimension.assert_called()
        mock_model.encode.assert_called()
        assert model is mock_model

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_handles_load_error(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model() raises ModelLoadError on load failure."""
        mock_st.side_effect = RuntimeError("Network error")

        loader = ModelLoader()

        with pytest.raises(ModelLoadError):
            loader.get_model()

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_handles_validation_error(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model() raises ModelValidationError on validation failure."""
        mock_model = Mock()
        # Wrong dimension
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        loader = ModelLoader()

        with pytest.raises(ModelValidationError):
            loader.get_model()

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_sets_cache_dir_if_configured(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model() sets HF_HOME environment variable."""
        import os

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        cache_dir = "/tmp/test_cache"
        loader = ModelLoader(cache_dir=cache_dir)

        with patch.dict(os.environ, {}, clear=False) as mock_env:
            loader.get_model()
            # Cache dir should be set
            assert os.environ.get("HF_HOME") == cache_dir or mock_env.get("HF_HOME")


class TestModelLoaderValidation:
    """Tests for model validation logic."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_validate_model_checks_dimension(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test _validate_model() checks embedding dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        loader = ModelLoader()

        with pytest.raises(ModelValidationError, match="dimension"):
            loader._validate_model(mock_model)

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_validate_model_tests_encoding(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test _validate_model() tests model encoding capability."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        # No embeddings returned (failure)
        mock_model.encode.return_value = []

        loader = ModelLoader()

        with pytest.raises(ModelValidationError, match="embeddings"):
            loader._validate_model(mock_model)

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_validate_model_success(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test _validate_model() succeeds with valid model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]

        loader = ModelLoader()
        # Should not raise
        loader._validate_model(mock_model)


class TestModelLoaderDimension:
    """Tests for model dimension retrieval."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_dimension(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model_dimension() returns embedding dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()
        dimension = loader.get_model_dimension()

        assert dimension == EXPECTED_EMBEDDING_DIMENSION

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_get_model_dimension_triggers_load(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test get_model_dimension() loads model if not cached."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()
        assert loader._model is None

        loader.get_model_dimension()

        # Model should be loaded now
        assert loader._model is not None


class TestModelLoaderCacheReset:
    """Tests for cache reset functionality."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_reset_cache_clears_model(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test reset_cache() clears in-memory model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()
        model1 = loader.get_model()

        loader.reset_cache()
        assert loader._model is None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_reset_cache_forces_reload(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test reset_cache() forces reload on next access."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader = ModelLoader()
        loader.get_model()
        assert mock_st.call_count == 1

        loader.reset_cache()
        loader.get_model()
        # Model constructor called again
        assert mock_st.call_count == 2


class TestModelLoaderErrorHandling:
    """Tests for error handling and reporting."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    def test_handle_download_error_network(self) -> None:
        """Test _handle_download_error() categorizes network errors."""
        loader = ModelLoader()
        error = ConnectionError("Failed to connect to HuggingFace")

        with pytest.raises(ModelLoadError, match="network"):
            loader._handle_download_error(error)

    def test_handle_download_error_disk_space(self) -> None:
        """Test _handle_download_error() categorizes disk errors."""
        loader = ModelLoader()
        error = OSError("No space left on device")

        with pytest.raises(ModelLoadError, match="disk"):
            loader._handle_download_error(error)

    def test_handle_download_error_permissions(self) -> None:
        """Test _handle_download_error() categorizes permission errors."""
        loader = ModelLoader()
        error = PermissionError("Permission denied to write cache")

        with pytest.raises(ModelLoadError, match="permission"):
            loader._handle_download_error(error)

    def test_handle_download_error_wraps_exception(self) -> None:
        """Test _handle_download_error() preserves original exception."""
        loader = ModelLoader()
        original = RuntimeError("Unknown error")

        try:
            loader._handle_download_error(original)
        except ModelLoadError as e:
            assert e.__cause__ is original


class TestModelLoaderIntegration:
    """Integration tests for model loader."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelLoader._instance = None

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_full_load_workflow(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test complete model loading workflow."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        # Initialize loader
        loader = ModelLoader(
            model_name=DEFAULT_MODEL_NAME,
            device="cpu",
        )

        # First load
        model1 = loader.get_model()
        assert model1 is mock_model

        # Check caching works
        model2 = loader.get_model()
        assert model1 is model2
        assert mock_st.call_count == 1

        # Get dimension
        dim = loader.get_model_dimension()
        assert dim == EXPECTED_EMBEDDING_DIMENSION

        # Reset and reload
        loader.reset_cache()
        model3 = loader.get_model()
        assert mock_st.call_count == 2

    @patch("src.embedding.model_loader.SentenceTransformer")
    def test_multiple_loaders_separate_instances(
        self,
        mock_st: MagicMock,
    ) -> None:
        """Test multiple ModelLoader instances are independent."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            EXPECTED_EMBEDDING_DIMENSION
        )
        mock_model.encode.return_value = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        mock_st.return_value = mock_model

        loader1 = ModelLoader(model_name="model-1")
        loader2 = ModelLoader(model_name="model-2")

        assert loader1 is not loader2
        assert loader1._model_name == "model-1"
        assert loader2._model_name == "model-2"


class TestModelLoaderDefaults:
    """Tests for default configuration values."""

    def test_default_model_name(self) -> None:
        """Test DEFAULT_MODEL_NAME constant value."""
        assert DEFAULT_MODEL_NAME == "sentence-transformers/all-mpnet-base-v2"

    def test_expected_embedding_dimension(self) -> None:
        """Test EXPECTED_EMBEDDING_DIMENSION constant value."""
        assert EXPECTED_EMBEDDING_DIMENSION == 768
