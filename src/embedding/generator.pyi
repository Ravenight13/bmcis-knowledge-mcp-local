"""Type stubs for parallel embedding generation with batch processing.

Provides type definitions for EmbeddingGenerator class that handles
efficient batch processing and parallel embedding generation for
document chunks with progress tracking and error handling.
"""

from typing import Callable, Final
from concurrent.futures import ThreadPoolExecutor
from src.document_parsing.models import ProcessedChunk
from src.embedding.model_loader import ModelLoader
import logging

# Type aliases
EmbeddingVector: type[list[float]]
ProgressCallback: type[Callable[[int, int], None]]
ChunkBatch: type[list[ProcessedChunk]]

class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""
    ...

class EmbeddingValidator:
    """Validates embeddings for correctness and consistency."""

    EXPECTED_DIMENSION: Final[int]

    def __init__(self) -> None:
        """Initialize validator with expected dimension."""
        ...

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate embedding has correct dimension and numeric values.

        Args:
            embedding: Embedding vector to validate.

        Returns:
            True if valid, False otherwise.
        """
        ...

    def validate_batch(self, embeddings: list[list[float]]) -> tuple[int, int]:
        """Validate batch of embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Tuple of (valid_count, invalid_count).
        """
        ...

class EmbeddingGenerator:
    """Generates embeddings for document chunks with parallel processing.

    Provides efficient batch processing, parallel execution, progress tracking,
    and error handling for generating 768-dimensional embeddings for all
    ProcessedChunk objects.

    Attributes:
        DEFAULT_BATCH_SIZE: Recommended batch size for efficiency.
        DEFAULT_NUM_WORKERS: Number of parallel workers.
        MODEL_DIMENSION: Expected embedding dimension (768).
    """

    DEFAULT_BATCH_SIZE: Final[int]
    DEFAULT_NUM_WORKERS: Final[int]
    MODEL_DIMENSION: Final[int]

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str | None = None,
        use_threading: bool = True,
    ) -> None:
        """Initialize EmbeddingGenerator with configuration.

        Args:
            model_loader: ModelLoader instance. Creates default if None.
            batch_size: Number of chunks per batch (32-64 recommended).
            num_workers: Number of parallel workers (2-8 recommended).
            device: Device to use ('cpu', 'cuda'). Auto-detects if None.
            use_threading: Use ThreadPoolExecutor if True.
        """
        ...

    def process_chunks(
        self,
        chunks: list[ProcessedChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        retry_failed: bool = True,
    ) -> list[ProcessedChunk]:
        """Process chunks and generate embeddings.

        Args:
            chunks: List of ProcessedChunk objects to embed.
            progress_callback: Optional callback(processed, total) for progress updates.
            retry_failed: Retry failed chunks if True.

        Returns:
            List of ProcessedChunk objects with embeddings populated.

        Raises:
            EmbeddingGenerationError: If processing fails critically.
        """
        ...

    def process_batch(self, batch: list[ProcessedChunk]) -> list[ProcessedChunk]:
        """Process a single batch of chunks.

        Args:
            batch: Batch of chunks to process.

        Returns:
            Batch of chunks with embeddings populated.

        Raises:
            EmbeddingGenerationError: If batch processing fails.
        """
        ...

    def generate_embeddings_for_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts list is empty.
            EmbeddingGenerationError: If generation fails.
        """
        ...

    def validate_and_enrich_chunk(
        self,
        chunk: ProcessedChunk,
        embedding: list[float],
    ) -> ProcessedChunk:
        """Validate and enrich chunk with embedding.

        Args:
            chunk: ProcessedChunk to enrich.
            embedding: Embedding vector to add.

        Returns:
            Enriched ProcessedChunk with embedding.

        Raises:
            ValueError: If embedding is invalid.
        """
        ...

    def get_progress_summary(self) -> dict[str, int]:
        """Get current progress summary.

        Returns:
            Dictionary with processed, failed, and total counts.
        """
        ...

    def get_statistics(self) -> dict[str, float | int]:
        """Get processing statistics.

        Returns:
            Dictionary with timing, throughput, and error statistics.
        """
        ...
