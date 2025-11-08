"""Type stubs for sentence-transformers model loader with caching strategy.

This stub file defines the complete type interface for ModelLoader,
ensuring type-safe model loading, caching, and error handling.
"""

from pathlib import Path
from typing import Final, overload

from sentence_transformers import SentenceTransformer

# Type aliases for embeddings and dimensions
EmbeddingVector: type[list[float]]
ModelDimension: type[int]
BATCH_TEST_SIZE: Final[int]

# Constants
DEFAULT_MODEL_NAME: Final[str]
EXPECTED_EMBEDDING_DIMENSION: Final[int]

class ModelLoadError(Exception):
    """Raised when model loading fails."""

    message: str

    def __init__(self, message: str) -> None: ...

class ModelValidationError(Exception):
    """Raised when model validation fails."""

    message: str

    def __init__(self, message: str) -> None: ...

class ModelLoader:
    """Type-safe singleton model loader for sentence-transformers.

    Manages loading, caching, and validation of embedding models from
    HuggingFace with automatic in-memory caching.

    Class attributes:
        DEFAULT_MODEL_NAME: Default model identifier.
        EMBEDDING_DIMENSION: Expected embedding dimension (768).
        CACHE_DIR: Default cache directory path.
    """

    DEFAULT_MODEL_NAME: Final[str]
    EMBEDDING_DIMENSION: Final[int]
    CACHE_DIR: Final[Path]
    _instance: ModelLoader | None

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize ModelLoader with configuration.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Directory for caching downloaded models.
            device: Device to load model on ("cpu" or "cuda").

        Raises:
            ModelValidationError: If device is invalid.
        """

    @classmethod
    def get_instance(
        cls,
        model_name: str | None = None,
        cache_dir: Path | str | None = None,
        device: str | None = None,
    ) -> ModelLoader:
        """Get or create singleton ModelLoader instance.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Directory for caching downloaded models.
            device: Device to load model on.

        Returns:
            ModelLoader: Singleton instance.

        Raises:
            ModelValidationError: If device is invalid.
        """

    def get_model(self) -> SentenceTransformer:
        """Get the loaded model, loading if necessary.

        Implements lazy loading with automatic caching.

        Returns:
            SentenceTransformer: Loaded model instance.

        Raises:
            ModelLoadError: If model download/loading fails.
        """

    @overload
    def encode(self, texts: str) -> list[float]: ...
    @overload
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode(self, texts: list[str] | str) -> list[float] | list[list[float]]:
        """Encode texts to embeddings.

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            Single embedding if input is str, list of embeddings otherwise.

        Raises:
            ValueError: If text is empty or invalid.
            ModelLoadError: If encoding fails.
        """

    def get_device(self) -> str:
        """Get device model is loaded on.

        Returns:
            Device string ('cuda', 'cpu', etc).
        """

    def get_model_name(self) -> str:
        """Get name of loaded model.

        Returns:
            Model name string.
        """

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding has correct dimension and valid values.

        Args:
            embedding: Embedding vector to validate.

        Returns:
            True if embedding is valid, False otherwise.
        """

    def reset_cache(self) -> None:
        """Reset the in-memory model cache.

        Forces reload on next get_model() call.
        """

    def get_model_dimension(self) -> int:
        """Get embedding dimension of loaded model.

        Returns:
            int: Dimension of embeddings (768 for all-mpnet-base-v2).

        Raises:
            ModelLoadError: If model loading fails.
        """

    @staticmethod
    def _validate_device(device: str) -> None:
        """Validate device string.

        Args:
            device: Device string to validate.

        Raises:
            ModelValidationError: If device is invalid.
        """

    @classmethod
    def detect_device(cls) -> str:
        """Detect available device (GPU/CPU).

        Returns:
            Device string ('cuda' if available, else 'cpu').
        """

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get default cache directory path.

        Returns:
            Path to cache directory.
        """
