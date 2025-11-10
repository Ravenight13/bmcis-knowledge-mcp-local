"""Type stubs for embedding generation module.

Provides complete type information for mypy --strict validation.
Enables IDE autocomplete and static type checking across the codebase.

Why this exists:
- Complete type safety for embedding generation pipeline
- IDE support for autocomplete and refactoring
- Early error detection in dependent code
- Documentation of function contracts
"""

from typing import Callable, Final

from src.document_parsing.models import ProcessedChunk
from src.embedding.model_loader import ModelLoader

# Type aliases for clarity
EmbeddingVector = list[float]
ProgressCallback = Callable[[int, int], None]


class EmbeddingGenerationError(Exception):
    """Raised when critical embedding generation failures occur."""

    ...


class EmbeddingValidator:
    """Validates embeddings for correctness and consistency."""

    EXPECTED_DIMENSION: Final[int]

    def __init__(self) -> None: ...
    def validate_embedding(self, embedding: EmbeddingVector) -> bool: ...
    def validate_batch(
        self, embeddings: list[EmbeddingVector]
    ) -> tuple[int, int]: ...


class EmbeddingGenerator:
    """Generates embeddings for document chunks with parallel processing.

    Provides efficient batch processing, parallel execution, progress tracking,
    and error handling for generating 768-dimensional embeddings.
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
            use_threading: Use ThreadPoolExecutor if True, ProcessPoolExecutor if False.
        """

    def process_chunks(
        self,
        chunks: list[ProcessedChunk],
        progress_callback: ProgressCallback | None = None,
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

    def _create_batches(
        self, chunks: list[ProcessedChunk]
    ) -> list[list[ProcessedChunk]]:
        """Create batches from chunks list.

        Args:
            chunks: List of chunks to batch.

        Returns:
            List of batches.
        """

    def _process_batches_parallel(
        self,
        batches: list[list[ProcessedChunk]],
        progress_callback: ProgressCallback | None = None,
    ) -> list[ProcessedChunk]:
        """Process batches in parallel.

        Args:
            batches: List of batches to process.
            progress_callback: Optional progress callback.

        Returns:
            List of processed chunks.
        """

    def process_batch(self, batch: list[ProcessedChunk]) -> list[ProcessedChunk]:
        """Process a single batch of chunks.

        Args:
            batch: Batch of chunks to process.

        Returns:
            Batch of chunks with embeddings populated.

        Raises:
            EmbeddingGenerationError: If batch processing fails.
        """

    def generate_embeddings_for_texts(
        self, texts: list[str]
    ) -> list[EmbeddingVector]:
        """Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts list is empty.
            EmbeddingGenerationError: If generation fails.
        """

    def validate_and_enrich_chunk(
        self,
        chunk: ProcessedChunk,
        embedding: EmbeddingVector,
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

    def get_progress_summary(self) -> dict[str, int]:
        """Get current progress summary.

        Returns:
            Dictionary with processed, failed, and total counts.
        """

    def get_statistics(self) -> dict[str, float | int | str]:
        """Get processing statistics.

        Returns:
            Dictionary with timing, throughput, and error statistics.
        """
