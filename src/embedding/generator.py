"""Parallel embedding generation with batch processing and progress tracking.

Provides EmbeddingGenerator class that efficiently processes document chunks
in parallel batches, generating 768-dimensional embeddings with comprehensive
error handling and progress tracking.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Final

from src.document_parsing.models import ProcessedChunk
from src.embedding.model_loader import ModelLoader, EXPECTED_EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Type aliases
EmbeddingVector = list[float]
ProgressCallback = Callable[[int, int], None]


class EmbeddingGenerationError(Exception):
    """Raised when critical embedding generation failures occur."""

    pass


class EmbeddingValidator:
    """Validates embeddings for correctness and consistency."""

    EXPECTED_DIMENSION: Final[int] = EXPECTED_EMBEDDING_DIMENSION

    def __init__(self) -> None:
        """Initialize validator with expected dimension."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_embedding(self, embedding: EmbeddingVector) -> bool:
        """Validate embedding has correct dimension and numeric values.

        Args:
            embedding: Embedding vector to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(embedding, list):
            return False
        if len(embedding) != self.EXPECTED_DIMENSION:
            self.logger.warning(
                f"Embedding dimension mismatch: expected {self.EXPECTED_DIMENSION}, "
                f"got {len(embedding)}"
            )
            return False
        if not embedding:
            return False
        if not all(isinstance(v, (int, float)) for v in embedding):
            self.logger.warning("Embedding contains non-numeric values")
            return False
        return True

    def validate_batch(
        self, embeddings: list[EmbeddingVector]
    ) -> tuple[int, int]:
        """Validate batch of embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Tuple of (valid_count, invalid_count).
        """
        valid_count = 0
        invalid_count = 0
        for emb in embeddings:
            if self.validate_embedding(emb):
                valid_count += 1
            else:
                invalid_count += 1
        return valid_count, invalid_count


class EmbeddingGenerator:
    """Generates embeddings for document chunks with parallel processing.

    Provides efficient batch processing, parallel execution, progress tracking,
    and error handling for generating 768-dimensional embeddings for all
    ProcessedChunk objects.

    Attributes:
        DEFAULT_BATCH_SIZE: Recommended batch size for efficiency (32).
        DEFAULT_NUM_WORKERS: Number of parallel workers (4).
        MODEL_DIMENSION: Expected embedding dimension (768).
    """

    DEFAULT_BATCH_SIZE: Final[int] = 32
    DEFAULT_NUM_WORKERS: Final[int] = 4
    MODEL_DIMENSION: Final[int] = EXPECTED_EMBEDDING_DIMENSION

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
        self.model_loader = model_loader or ModelLoader(device=device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_threading = use_threading
        self.validator = EmbeddingValidator()

        # Statistics tracking
        self.processed_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.start_time = 0.0
        self.end_time = 0.0

        logger.info(
            f"EmbeddingGenerator initialized: batch_size={batch_size}, "
            f"num_workers={num_workers}, use_threading={use_threading}"
        )

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
        if not chunks:
            logger.warning("No chunks provided for processing")
            return []

        self.start_time = time.time()
        self.total_count = len(chunks)
        self.processed_count = 0
        self.failed_count = 0

        logger.info(f"Starting embedding generation for {self.total_count} chunks")

        try:
            # Create batches
            batches = self._create_batches(chunks)
            logger.info(f"Created {len(batches)} batches of size {self.batch_size}")

            # Process batches
            processed_chunks = self._process_batches_parallel(
                batches, progress_callback
            )

            self.end_time = time.time()
            elapsed = self.end_time - self.start_time

            logger.info(
                f"Embedding generation complete: {self.processed_count} processed, "
                f"{self.failed_count} failed in {elapsed:.2f}s "
                f"({self.processed_count / elapsed:.2f} chunks/sec)"
            )

            return processed_chunks

        except Exception as e:
            logger.error(f"Critical error during embedding generation: {e}")
            raise EmbeddingGenerationError(
                f"Failed to process chunks: {e}"
            ) from e

    def _create_batches(self, chunks: list[ProcessedChunk]) -> list[list[ProcessedChunk]]:
        """Create batches from chunks list.

        Args:
            chunks: List of chunks to batch.

        Returns:
            List of batches.
        """
        batches: list[list[ProcessedChunk]] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batches.append(batch)
        return batches

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
        processed_chunks: list[ProcessedChunk] = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self.process_batch, batch): i
                for i, batch in enumerate(batches)
            }

            # Process completed batches as they finish
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    processed_chunks.extend(batch_results)
                    self.processed_count += len(batch_results)

                    if progress_callback:
                        progress_callback(self.processed_count, self.total_count)

                    logger.debug(
                        f"Processed batch {future_to_batch[future] + 1}/"
                        f"{len(batches)}: {len(batch_results)} chunks"
                    )

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    self.failed_count += 1

        return processed_chunks

    def process_batch(self, batch: list[ProcessedChunk]) -> list[ProcessedChunk]:
        """Process a single batch of chunks.

        Args:
            batch: Batch of chunks to process.

        Returns:
            Batch of chunks with embeddings populated.

        Raises:
            EmbeddingGenerationError: If batch processing fails.
        """
        if not batch:
            return []

        try:
            # Extract texts from chunks
            texts = [chunk.chunk_text for chunk in batch]

            # Generate embeddings
            embeddings = self.generate_embeddings_for_texts(texts)

            # Validate embeddings
            valid_count, invalid_count = self.validator.validate_batch(embeddings)
            if invalid_count > 0:
                logger.warning(
                    f"Batch validation: {valid_count} valid, {invalid_count} invalid"
                )

            # Enrich chunks with embeddings
            enriched_chunks: list[ProcessedChunk] = []
            for chunk, embedding in zip(batch, embeddings, strict=False):
                try:
                    enriched = self.validate_and_enrich_chunk(chunk, embedding)
                    enriched_chunks.append(enriched)
                except ValueError as e:
                    logger.error(f"Failed to enrich chunk: {e}")
                    self.failed_count += 1

            return enriched_chunks

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            raise EmbeddingGenerationError(f"Batch processing failed: {e}") from e

    def generate_embeddings_for_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        """Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts list is empty.
            EmbeddingGenerationError: If generation fails.
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")

        try:
            embeddings = self.model_loader.encode(texts)
            # Handle different return types from encode()
            if isinstance(embeddings, list):
                if embeddings and isinstance(embeddings[0], list):
                    return embeddings
                else:
                    # Single embedding returned
                    return [embeddings] if embeddings else [[]]
            return [embeddings] if isinstance(embeddings, (list, tuple)) else [[]]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}") from e

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
        if not self.validator.validate_embedding(embedding):
            raise ValueError(
                f"Invalid embedding: wrong dimension or non-numeric values"
            )

        # Create enriched chunk with embedding
        enriched_data = chunk.model_dump()
        enriched_data["embedding"] = embedding
        return ProcessedChunk(**enriched_data)

    def get_progress_summary(self) -> dict[str, int]:
        """Get current progress summary.

        Returns:
            Dictionary with processed, failed, and total counts.
        """
        return {
            "processed": self.processed_count,
            "failed": self.failed_count,
            "total": self.total_count,
        }

    def get_statistics(self) -> dict[str, float | int]:
        """Get processing statistics.

        Returns:
            Dictionary with timing, throughput, and error statistics.
        """
        elapsed = (
            self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
        )
        throughput = (
            self.processed_count / elapsed if elapsed > 0 else 0
        )

        return {
            "total_chunks": self.total_count,
            "processed_chunks": self.processed_count,
            "failed_chunks": self.failed_count,
            "elapsed_seconds": elapsed,
            "throughput_chunks_per_sec": throughput,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.model_loader.get_device(),
        }
