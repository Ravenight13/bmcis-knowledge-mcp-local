"""Sentence-transformers model loading with intelligent caching strategy.

Provides type-safe model loading for the all-mpnet-base-v2 embedding model
with singleton pattern, automatic caching, and error handling for production
use. Integrates with the application's logging and configuration systems.

Module exports:
    - ModelLoader: Singleton model loader with caching
    - ModelLoadError: Exception for loading failures
    - ModelValidationError: Exception for validation failures
"""

import logging
import os
from pathlib import Path
from typing import Final, overload

import torch
from sentence_transformers import SentenceTransformer

from src.core.logging import StructuredLogger

# Type aliases for embeddings and model dimensions
EmbeddingVector = list[float]
ModelDimension = int

# Public constants matching test expectations
DEFAULT_MODEL_NAME: Final[str] = "sentence-transformers/all-mpnet-base-v2"
EXPECTED_EMBEDDING_DIMENSION: Final[int] = 768
BATCH_TEST_SIZE: Final[int] = 1  # Size of test batch for validation

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model loading fails."""

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Error description.
        """
        self.message = message
        super().__init__(message)


class ModelValidationError(Exception):
    """Raised when model validation fails."""

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Error description.
        """
        self.message = message
        super().__init__(message)


class ModelLoader:
    """Type-safe singleton model loader for sentence-transformers.

    Manages loading, caching, and validation of embedding models from
    HuggingFace with automatic in-memory caching to avoid expensive
    reloads. Implements singleton pattern for single model instance
    per application.

    Thread-safe for model access but assumes single initialization.
    Uses application configuration for cache directory and device selection.

    Attributes:
        DEFAULT_MODEL_NAME: Default model to load (all-mpnet-base-v2).
        EMBEDDING_DIMENSION: Expected dimension of embeddings (768).
        CACHE_DIR: Directory for model caching.
    """

    DEFAULT_MODEL_NAME: Final[str] = DEFAULT_MODEL_NAME
    EMBEDDING_DIMENSION: Final[int] = EXPECTED_EMBEDDING_DIMENSION
    CACHE_DIR: Final[Path] = Path.home() / ".cache" / "bmcis" / "models"

    # Singleton instance
    _instance: "ModelLoader | None" = None

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize ModelLoader with optional custom model and cache path.

        Configures model name, cache directory, and device placement.
        Does not load model until get_model() is called (lazy loading).

        Args:
            model_name: Name of sentence-transformers model to load.
                       Defaults to all-mpnet-base-v2.
            cache_dir: Directory for caching models. Defaults to ~/.cache/bmcis.
            device: Device to load model on ('cpu', 'cuda', etc).
                   Auto-detects if not specified.

        Raises:
            ModelValidationError: If device is invalid.
        """
        self._model_name: str = model_name or DEFAULT_MODEL_NAME
        self._cache_dir: Path = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self._device: str = device or self.detect_device()
        self._validate_device(self._device)
        self._model: SentenceTransformer | None = None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ModelLoader initialized: model={self._model_name}, "
            f"device={self._device}, cache_dir={self._cache_dir}"
        )

    @classmethod
    def get_instance(
        cls,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
        device: str | None = None,
    ) -> "ModelLoader":
        """Get or create singleton ModelLoader instance.

        Implements singleton pattern ensuring single model instance per
        application. First call creates instance with provided parameters,
        subsequent calls return same instance (ignoring new parameters).

        Args:
            model_name: Model name. Only used on first call.
            cache_dir: Cache directory. Only used on first call.
            device: Device specification. Only used on first call.

        Returns:
            ModelLoader: Singleton instance.

        Example:
            >>> loader = ModelLoader.get_instance()
            >>> model = loader.get_model()
        """
        if cls._instance is None:
            cls._instance = cls(
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
            )
        return cls._instance

    @staticmethod
    def _validate_device(device: str) -> None:
        """Validate device string.

        Args:
            device: Device string to validate.

        Raises:
            ModelValidationError: If device is invalid.
        """
        valid_devices = ["cpu", "cuda"]
        # Support numbered cuda devices (cuda:0, cuda:1, etc)
        if device.startswith("cuda:"):
            if not device[5:].isdigit():
                raise ModelValidationError(f"Invalid device: {device}")
        elif device not in valid_devices:
            raise ModelValidationError(
                f"Invalid device: {device}. Must be one of {valid_devices} or cuda:N"
            )

    def get_model(self) -> SentenceTransformer:
        """Get loaded model instance, loading if necessary.

        Returns:
            Loaded SentenceTransformer model instance.

        Raises:
            ModelLoadError: If model fails to load.
        """
        if self._model is None:
            try:
                logger.info(f"Loading model {self._model_name} on device {self._device}")
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    cache_folder=str(self._cache_dir),
                )
                logger.info(f"Model loaded successfully on {self._device}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(f"Failed to load model {self._model_name}: {e}") from e
        return self._model

    @overload
    def encode(self, texts: str) -> list[float]:
        """Encode single text to embedding."""
        ...

    @overload
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode list of texts to embeddings."""
        ...

    def encode(self, texts: list[str] | str) -> list[float] | list[list[float]]:
        """Encode texts to embeddings.

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            Single embedding (list[float]) if input is str,
            list of embeddings if input is list[str].

        Raises:
            ValueError: If text is empty or invalid.
            ModelLoadError: If encoding fails.
        """
        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Cannot encode empty text")
            model = self.get_model()
            embedding = model.encode(texts, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, "tolist") else embedding  # type: ignore[no-any-return]
        elif isinstance(texts, list):
            if not texts:
                raise ValueError("Cannot encode empty list")
            if not all(isinstance(t, str) for t in texts):
                raise ValueError("All items must be strings")
            model = self.get_model()
            embeddings = model.encode(texts, convert_to_tensor=False)
            result: list[list[float]] = []
            for emb in embeddings:
                if hasattr(emb, "tolist"):
                    result.append(emb.tolist())
                else:
                    result.append(emb)
            return result
        else:
            raise ValueError("texts must be str or list[str]")

    def get_device(self) -> str:
        """Get device model is loaded on.

        Returns:
            Device string ('cuda', 'cpu', etc).
        """
        return self._device

    def get_model_name(self) -> str:
        """Get name of loaded model.

        Returns:
            Model name string.
        """
        return self._model_name

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding has correct dimension and valid values.

        Args:
            embedding: Embedding vector to validate.

        Returns:
            True if embedding is valid, False otherwise.
        """
        if not isinstance(embedding, list):
            return False
        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            return False
        if not embedding:
            return False
        if not all(isinstance(v, (int, float)) for v in embedding):
            return False
        return True

    @classmethod
    def detect_device(cls) -> str:
        """Detect available device (GPU/CPU).

        Returns:
            Device string ('cuda' if available, else 'cpu').
        """
        if torch.cuda.is_available():
            logger.info("CUDA device detected")
            return "cuda"
        logger.info("Using CPU device")
        return "cpu"

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get default cache directory path.

        Returns:
            Path to cache directory.
        """
        return cls.CACHE_DIR

    def reset_cache(self) -> None:
        """Reset the in-memory model cache.

        Forces reload on next get_model() call. Does NOT delete files
        from disk cache. Used for testing or memory management in
        long-running applications.

        Side effects:
            Sets _model to None, forcing reload on next access.
            Logs cache reset event.

        Example:
            >>> loader = ModelLoader.get_instance()
            >>> model1 = loader.get_model()
            >>> loader.reset_cache()
            >>> model2 = loader.get_model()  # Reloads from cache
        """
        self._model = None
        logger.info(
            f"Model cache reset: {self._model_name}",
        )

    def get_model_dimension(self) -> int:
        """Get embedding dimension of loaded model.

        Returns the dimension of embeddings produced by the model.
        For all-mpnet-base-v2, this is always 768.

        Returns:
            int: Dimension of embeddings.

        Raises:
            ModelLoadError: If model loading fails.

        Example:
            >>> loader = ModelLoader.get_instance()
            >>> dim = loader.get_model_dimension()
            >>> assert dim == 768
        """
        model: SentenceTransformer = self.get_model()
        dimension: int = model.get_sentence_embedding_dimension()
        return dimension
