"""Batch document processing pipeline for knowledge base ingestion.

Orchestrates end-to-end document processing from markdown files through
parsing, tokenization, chunking, context header generation, and database
insertion. Provides batch processing for efficiency and comprehensive
error handling with progress tracking.

Pipeline stages:
1. Load markdown documents from directory (MarkdownReader)
2. Tokenize document text (Tokenizer)
3. Chunk into 512-token segments (Chunker)
4. Generate context headers (ContextHeaderGenerator)
5. Insert chunks into knowledge_base table (DatabasePool)

Supports concurrent processing, batch database operations, and detailed
logging for production use.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import psycopg2
from pydantic import BaseModel, Field, field_validator

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.document_parsing.models import (
    BatchProcessingStats,
    DocumentMetadata,
    ProcessedChunk,
)

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)


# ============================================================================
# BATCH PROCESSING DATA STRUCTURES
# ============================================================================


class ErrorRecoveryAction(Enum):
    """Enum for error recovery actions.

    Categorizes errors into three response strategies:
    - RETRY: Transient error (network timeout, connection lost) - attempt retry
    - SKIP: Recoverable error (malformed data) - skip chunk and continue
    - FAIL: Permanent error (database constraint) - fail entire batch
    """

    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"


@dataclass
class Batch:
    """Represents a batch of documents for processing.

    Batches partition large document collections into optimized sizes for
    processing. Each batch maintains document order and includes metadata
    for tracking and error recovery.

    Attributes:
        documents: List of documents in this batch.
        batch_index: Zero-indexed position of batch in sequence.
        metadata: Optional metadata about this batch (creation time, source, etc).
    """

    documents: list[ProcessedChunk] = field(
        default_factory=list,
        metadata="Documents in this batch"
    )
    batch_index: int = field(
        default=0,
        metadata="Zero-indexed position in batch sequence"
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata="Optional batch metadata"
    )

    def __post_init__(self) -> None:
        """Validate batch state after initialization."""
        if self.batch_index < 0:
            raise ValueError(f"batch_index must be >= 0, got {self.batch_index}")
        if not isinstance(self.documents, list):
            raise TypeError(f"documents must be list, got {type(self.documents)}")


@dataclass
class BatchProgress:
    """Tracks batch processing progress and metrics.

    Maintains real-time statistics on batch processing including completion
    counts, document progress, and error tracking. Used to monitor processing
    and provide visibility into multi-hour batch operations.

    Attributes:
        batches_completed: Number of batches successfully processed.
        batches_total: Total batches to process.
        documents_processed: Total document chunks processed.
        errors: List of error messages encountered.
        start_time: When processing began.
        current_batch_index: Currently processing batch (for resumption).
    """

    batches_completed: int = field(
        default=0,
        metadata="Batches successfully processed"
    )
    batches_total: int = field(
        default=0,
        metadata="Total batches to process"
    )
    documents_processed: int = field(
        default=0,
        metadata="Total chunks successfully processed"
    )
    errors: list[str] = field(
        default_factory=list,
        metadata="Error messages encountered"
    )
    start_time: datetime | None = field(
        default=None,
        metadata="Processing start timestamp"
    )
    current_batch_index: int = field(
        default=0,
        metadata="Currently processing batch index"
    )

    @property
    def percent_complete(self) -> float:
        """Calculate percentage of batches completed.

        Returns:
            Float from 0.0 to 100.0 representing progress percentage.
        """
        if self.batches_total == 0:
            return 0.0
        return (self.batches_completed / self.batches_total) * 100.0

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing.

        Returns:
            True if errors list is not empty.
        """
        return len(self.errors) > 0


@dataclass
class BatchResult:
    """Result of processing a single batch.

    Contains outcome information from batch processing including success status,
    retry count, error details, and metrics about what was processed.

    Attributes:
        success: Whether batch processed successfully.
        retry_count: Number of retries attempted (0 if successful on first try).
        error: Error message if processing failed (None if successful).
        documents_processed: Number of documents successfully processed.
        batch_index: Index of the batch that was processed.
    """

    success: bool = field(default=False)
    retry_count: int = field(default=0)
    error: str | None = field(default=None)
    documents_processed: int = field(default=0)
    batch_index: int = field(default=0)


# ============================================================================
# PLACEHOLDER IMPLEMENTATIONS FOR TASKS 2.1-2.4
# These will be replaced with actual implementations when tasks are complete
# ============================================================================


class ParseError(Exception):
    """Raised when document parsing fails."""

    pass


class MarkdownReader:
    """Placeholder for Task 2.1: Markdown document reader.

    TODO: Replace with actual implementation from task 2.1.
    """

    @staticmethod
    def read_file(file_path: Path) -> tuple[str, DocumentMetadata]:
        """Read markdown file and extract metadata.

        Args:
            file_path: Path to markdown file.

        Returns:
            Tuple of (content, metadata).

        Raises:
            ParseError: If file cannot be read or parsed.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            metadata = DocumentMetadata(
                title=file_path.stem,
                source_file=str(file_path),
            )
            return content, metadata
        except Exception as e:
            raise ParseError(f"Failed to read {file_path}: {e}") from e


class Tokenizer:
    """Placeholder for Task 2.2: Text tokenizer.

    TODO: Replace with actual implementation from task 2.2.
    """

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        # Placeholder: simple whitespace tokenization
        return text.split()

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(Tokenizer.tokenize(text))


class TextChunk(BaseModel):
    """Represents a text chunk with token information."""

    text: str
    token_count: int
    start_index: int
    end_index: int


class Chunker:
    """Placeholder for Task 2.3: Text chunker.

    TODO: Replace with actual implementation from task 2.3.
    """

    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk.
            overlap: Token overlap between chunks.
        """
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Chunk text into segments.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks with metadata.
        """
        # Placeholder: simple paragraph-based chunking
        paragraphs = text.split("\n\n")
        chunks: list[TextChunk] = []

        for i, para in enumerate(paragraphs):
            if para.strip():
                token_count = Tokenizer.count_tokens(para)
                chunks.append(
                    TextChunk(
                        text=para.strip(),
                        token_count=token_count,
                        start_index=i,
                        end_index=i,
                    )
                )

        return chunks


class ContextHeaderGenerator:
    """Placeholder for Task 2.4: Context header generator.

    TODO: Replace with actual implementation from task 2.4.
    """

    @staticmethod
    def generate_header(
        file_path: Path,
        chunk_index: int,
        content: str,
    ) -> str:
        """Generate context header for chunk.

        Args:
            file_path: Source file path.
            chunk_index: Index of chunk in document.
            content: Chunk content for header extraction.

        Returns:
            Context header string.
        """
        # Placeholder: simple file-based header
        return f"{file_path.name} > Chunk {chunk_index}"


# ============================================================================
# BATCH PROCESSING UTILITY FUNCTIONS
# ============================================================================


def calculate_batch_size(
    total_items: int,
    max_batch_size: int = 32,
) -> int:
    """Calculate optimized batch size based on document count.

    Calculates an appropriate batch size considering memory constraints and
    processing efficiency. Batch sizes are capped at max_batch_size to prevent
    excessive memory usage. For small collections, returns appropriate smaller
    sizes to avoid unnecessary overhead.

    Why batch size matters:
    - Memory efficiency: Each batch is loaded into memory; larger batches use
      more memory but have fewer database round trips.
    - Processing speed: Optimal batch size balances transaction overhead vs
      memory usage, typically 16-32 items.
    - Scalability: Large document collections (>1000s) benefit from smaller
      batches to manage memory and enable progress tracking.

    Heuristic algorithm:
    - For ≤100 items: return total_items (process all at once)
    - For 101-1000: return min(total_items // 10, max_batch_size)
    - For >1000: return max_batch_size (optimize for memory)
    - Always return at least 1

    Args:
        total_items: Total number of items to process.
        max_batch_size: Maximum batch size constraint (default 32).
                       Must be >= 1.

    Returns:
        Calculated batch size, guaranteed to be >= 1 and <= max_batch_size.

    Raises:
        ValueError: If max_batch_size < 1 or total_items < 0.

    Examples:
        >>> calculate_batch_size(50, max_batch_size=32)
        50  # Small collection processed in one batch

        >>> calculate_batch_size(500, max_batch_size=32)
        32  # Medium collection uses max size

        >>> calculate_batch_size(5000, max_batch_size=32)
        32  # Large collection optimized for memory
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if total_items < 0:
        raise ValueError(f"total_items must be >= 0, got {total_items}")

    if total_items == 0:
        return 1

    if total_items <= 100:
        # Small collections: process all at once for efficiency
        return min(total_items, max_batch_size)

    if total_items <= 1000:
        # Medium collections: divide into ~10 batches
        calculated = total_items // 10
        return min(calculated, max_batch_size)

    # Large collections: use maximum to manage memory
    return max_batch_size


def create_batches(
    documents: list[ProcessedChunk],
    batch_size: int | None = None,
) -> list[Batch]:
    """Partition documents into optimized batches.

    Creates Batch objects that maintain document order and include metadata
    for tracking. Calculates optimal batch size automatically if not provided.

    Why batching preserves order and integrity:
    - Documents maintain original sequence for context preservation
    - Batch indices enable resumable processing after failures
    - Metadata tracks source and processing state
    - Small batches enable fine-grained error recovery

    Processing strategy:
    1. Calculate batch size if not provided (via calculate_batch_size)
    2. Partition documents into sequential batches
    3. Add metadata (batch_index, creation_time)
    4. Preserve document order across all batches

    Args:
        documents: List of ProcessedChunk objects to batch.
        batch_size: Batch size (optional). If None, calculated automatically.

    Returns:
        List of Batch objects with balanced document distribution.

    Raises:
        ValueError: If documents empty or batch_size < 1.

    Examples:
        >>> docs = [ProcessedChunk(...) for _ in range(100)]
        >>> batches = create_batches(docs, batch_size=32)
        >>> len(batches)
        4  # 100 items split into 4 batches of 32, 32, 32, 4
    """
    if not documents:
        raise ValueError("documents list cannot be empty")

    if batch_size is None:
        batch_size = calculate_batch_size(len(documents))
    elif batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batches: list[Batch] = []
    creation_time = datetime.now()

    for batch_index in range(0, len(documents), batch_size):
        batch_docs = documents[batch_index : batch_index + batch_size]
        batch = Batch(
            documents=batch_docs,
            batch_index=len(batches),
            metadata={
                "created_at": creation_time.isoformat(),
                "document_count": len(batch_docs),
                "start_index": batch_index,
                "end_index": batch_index + len(batch_docs) - 1,
            },
        )
        batches.append(batch)

    logger.debug(
        "Batches created",
        extra={
            "batch_count": len(batches),
            "total_documents": len(documents),
            "batch_size": batch_size,
        },
    )

    return batches


# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================


class BatchConfig(BaseModel):
    """Configuration for batch document processing.

    Controls processing behavior including input directory, batch size,
    chunk parameters, and database insertion settings.

    Attributes:
        input_dir: Directory containing markdown files to process.
        batch_size: Number of chunks to insert per database transaction.
        chunk_max_tokens: Maximum tokens per chunk.
        chunk_overlap: Token overlap between adjacent chunks.
        recursive: Recursively process subdirectories.
        file_pattern: Glob pattern for matching files (default: *.md).
        skip_existing: Skip files already in database (by hash).
        max_workers: Maximum parallel workers (1 for sequential).
    """

    input_dir: Path = Field(
        description="Input directory with markdown files",
    )
    batch_size: int = Field(
        default=100,
        description="Chunks per database transaction",
        ge=1,
        le=1000,
    )
    chunk_max_tokens: int = Field(
        default=512,
        description="Maximum tokens per chunk",
        ge=100,
        le=2048,
    )
    chunk_overlap: int = Field(
        default=50,
        description="Token overlap between chunks",
        ge=0,
        le=512,
    )
    recursive: bool = Field(
        default=True,
        description="Process subdirectories recursively",
    )
    file_pattern: str = Field(
        default="*.md",
        description="Glob pattern for matching files",
    )
    skip_existing: bool = Field(
        default=True,
        description="Skip chunks already in database",
    )
    max_workers: int = Field(
        default=1,
        description="Maximum parallel workers (1=sequential)",
        ge=1,
        le=16,
    )

    @field_validator("input_dir")
    @classmethod
    def validate_input_dir(cls, v: Path) -> Path:
        """Validate input directory exists and is readable.

        Args:
            v: Input directory path.

        Returns:
            Validated path.

        Raises:
            ValueError: If directory doesn't exist or isn't readable.
        """
        if not v.exists():
            msg = f"Input directory does not exist: {v}"
            raise ValueError(msg)
        if not v.is_dir():
            msg = f"Input path is not a directory: {v}"
            raise ValueError(msg)
        return v


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


class BatchProcessor:
    """Orchestrates batch document processing pipeline.

    Processes markdown files through complete pipeline from file loading
    to database insertion with batch optimization, error handling, and
    progress tracking.

    Integrates all document parsing tasks (2.1-2.4) and Phase 0 infrastructure
    (config, database pool, logging) for production-ready batch processing.

    Attributes:
        config: Batch processing configuration.
        stats: Processing statistics tracker.
    """

    def __init__(self, config: BatchConfig):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration.
        """
        self.config = config
        self.stats = BatchProcessingStats()

        # Progress tracking
        self._progress = BatchProgress()
        self._error_count = 0

        # Initialize components
        self.reader = MarkdownReader()
        self.tokenizer = Tokenizer()
        self.chunker = Chunker(
            max_tokens=config.chunk_max_tokens,
            overlap=config.chunk_overlap,
        )
        self.context_generator = ContextHeaderGenerator()

        logger.info(
            "BatchProcessor initialized",
            extra={
                "input_dir": str(config.input_dir),
                "batch_size": config.batch_size,
                "chunk_max_tokens": config.chunk_max_tokens,
            },
        )

    def calculate_batch_size(
        self,
        total_items: int,
        max_batch_size: int | None = None,
    ) -> int:
        """Calculate optimized batch size for document collection.

        Delegates to module-level calculate_batch_size function with optional
        override for max_batch_size. Determines appropriate batch size based on
        collection size and memory constraints.

        Args:
            total_items: Number of documents to process.
            max_batch_size: Maximum batch size (uses config.batch_size if None).

        Returns:
            Calculated batch size.
        """
        if max_batch_size is None:
            max_batch_size = self.config.batch_size
        return calculate_batch_size(total_items, max_batch_size)

    def create_batches(
        self,
        documents: list[ProcessedChunk],
    ) -> list[Batch]:
        """Create batch groups from document list.

        Partitions documents into batches using configured batch size.
        Maintains document order and tracks batch metadata for progress
        monitoring and error recovery.

        Args:
            documents: List of ProcessedChunk objects.

        Returns:
            List of Batch objects partitioning the documents.

        Raises:
            ValueError: If documents list is empty.
        """
        return create_batches(documents, batch_size=self.config.batch_size)

    def track_progress(
        self,
        batch_index: int,
        total_batches: int,
        documents_count: int = 0,
    ) -> None:
        """Update progress tracking metrics.

        Records batch completion and updates internal progress state. Provides
        percentage complete and enables resumption after failures.

        Why progress tracking aids debugging:
        - Identifies where processing stalled during long operations
        - Enables operator intervention if batch size is inefficient
        - Validates estimated completion time and throughput
        - Supports resumable processing from last completed batch

        Args:
            batch_index: Current batch index (0-based).
            total_batches: Total batches to process.
            documents_count: Documents processed in this batch (optional).
        """
        self._progress.batches_completed = batch_index + 1
        self._progress.batches_total = total_batches
        self._progress.documents_processed += documents_count
        self._progress.current_batch_index = batch_index

        percent = self._progress.percent_complete
        logger.info(
            "Progress update",
            extra={
                "batches_completed": self._progress.batches_completed,
                "batches_total": total_batches,
                "percent_complete": f"{percent:.1f}%",
                "documents_processed": self._progress.documents_processed,
            },
        )

    def get_progress(self) -> BatchProgress:
        """Get current batch processing progress.

        Returns snapshot of progress metrics including completion counts,
        error information, and completion percentage.

        What metrics are important for monitoring:
        - batches_completed / batches_total: Overall progress percentage
        - documents_processed: Throughput metric (chunks/hour)
        - errors: List of problems for post-mortem analysis
        - percent_complete: Easy-to-understand progress indicator

        Returns:
            BatchProgress object with current metrics.
        """
        return BatchProgress(
            batches_completed=self._progress.batches_completed,
            batches_total=self._progress.batches_total,
            documents_processed=self._progress.documents_processed,
            errors=self._progress.errors.copy(),
            start_time=self._progress.start_time,
            current_batch_index=self._progress.current_batch_index,
        )

    def handle_error(
        self,
        error: Exception,
        batch: Batch,
    ) -> ErrorRecoveryAction:
        """Categorize error and determine recovery strategy.

        Analyzes error type to decide whether batch should be retried, skipped,
        or failed. Categorizes common errors and logs details.

        Error categorization strategy:
        - Transient (RETRY): Connection timeouts, server errors, locks
        - Recoverable (SKIP): Validation errors, malformed data
        - Permanent (FAIL): Database constraints, schema mismatch

        Args:
            error: Exception that occurred during processing.
            batch: Batch object being processed when error occurred.

        Returns:
            ErrorRecoveryAction indicating how to proceed.
        """
        error_msg = str(error)
        self._error_count += 1

        # Categorize error by type
        if isinstance(error, (TimeoutError, ConnectionError)):
            # Transient network issues
            action = ErrorRecoveryAction.RETRY
            logger.warning(
                "Transient error encountered, will retry",
                extra={"batch_index": batch.batch_index, "error_type": type(error).__name__},
            )
        elif isinstance(error, (ValueError, TypeError)):
            # Validation or type errors - skip this batch
            action = ErrorRecoveryAction.SKIP
            logger.warning(
                "Recoverable error encountered, will skip batch",
                extra={"batch_index": batch.batch_index, "error": error_msg},
            )
        elif isinstance(error, psycopg2.IntegrityError):
            # Database constraint violation - fail processing
            action = ErrorRecoveryAction.FAIL
            logger.error(
                "Database integrity error, batch processing failed",
                extra={"batch_index": batch.batch_index, "error": error_msg},
            )
        else:
            # Unknown errors default to fail for safety
            action = ErrorRecoveryAction.FAIL
            logger.error(
                "Unknown error, batch processing failed",
                extra={
                    "batch_index": batch.batch_index,
                    "error_type": type(error).__name__,
                    "error": error_msg,
                },
            )

        # Track error in progress
        self._progress.errors.append(f"Batch {batch.batch_index}: {error_msg}")

        return action

    def process_batch_with_retry(
        self,
        batch: Batch,
        processor: Callable[[Batch], int],
        max_retries: int = 3,
    ) -> BatchResult:
        """Process batch with exponential backoff retry logic.

        Processes batch using provided processor function. On failure, retries
        up to max_retries times with exponential backoff (1s, 2s, 4s).
        Gracefully handles transient errors and logs outcomes.

        Retry strategy explanation:
        - Exponential backoff: 1s, 2s, 4s (2^n delay)
        - Max retries: 3 attempts to handle transient failures
        - Backoff prevents overwhelming struggling server
        - Enables recovery from temporary resource contention
        - Logs all retry attempts for post-mortem analysis

        Args:
            batch: Batch to process.
            processor: Callable that processes batch, returns documents count.
            max_retries: Maximum number of retry attempts (default 3).

        Returns:
            BatchResult with success status, retry count, and error info.

        Raises:
            Exception: If max retries exceeded and error is permanent.
        """
        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                documents_processed = processor(batch)
                logger.info(
                    "Batch processed successfully",
                    extra={
                        "batch_index": batch.batch_index,
                        "documents_processed": documents_processed,
                        "retry_count": retry_count,
                    },
                )
                return BatchResult(
                    success=True,
                    retry_count=retry_count,
                    error=None,
                    documents_processed=documents_processed,
                    batch_index=batch.batch_index,
                )

            except Exception as e:
                last_error = e
                action = self.handle_error(e, batch)

                if action == ErrorRecoveryAction.FAIL:
                    # Permanent error - don't retry
                    return BatchResult(
                        success=False,
                        retry_count=retry_count,
                        error=str(e),
                        documents_processed=0,
                        batch_index=batch.batch_index,
                    )

                if action == ErrorRecoveryAction.SKIP:
                    # Skip batch - don't retry
                    logger.info(
                        "Batch skipped due to recoverable error",
                        extra={"batch_index": batch.batch_index},
                    )
                    return BatchResult(
                        success=False,
                        retry_count=0,
                        error=str(e),
                        documents_processed=0,
                        batch_index=batch.batch_index,
                    )

                # Transient error - retry with backoff
                if attempt < max_retries:
                    retry_count += 1
                    backoff_seconds = 2 ** (retry_count - 1)  # 1s, 2s, 4s
                    logger.info(
                        "Transient error, retrying with backoff",
                        extra={
                            "batch_index": batch.batch_index,
                            "retry_count": retry_count,
                            "backoff_seconds": backoff_seconds,
                            "error": str(e),
                        },
                    )
                    time.sleep(backoff_seconds)
                    continue

        # Max retries exhausted
        return BatchResult(
            success=False,
            retry_count=retry_count,
            error=str(last_error) if last_error else "Unknown error",
            documents_processed=0,
            batch_index=batch.batch_index,
        )

    def process_directory(self) -> list[str]:
        """Process all markdown files in configured directory.

        Discovers files matching pattern, processes each through the pipeline,
        and inserts chunks in batches for efficiency. Tracks processing
        statistics and errors.

        Returns:
            List of chunk IDs inserted into database.

        Example:
            >>> config = BatchConfig(input_dir=Path("docs/"))
            >>> processor = BatchProcessor(config)
            >>> chunk_ids = processor.process_directory()
            >>> print(f"Processed {len(chunk_ids)} chunks")
        """
        self.stats.start_time = datetime.now()
        all_chunk_ids: list[str] = []

        try:
            # Discover files
            files = self._discover_files()
            logger.info(
                "Discovered files for processing",
                extra={"file_count": len(files)},
            )

            # Process each file
            for file_path in files:
                try:
                    chunk_ids = self.process_file(file_path)
                    all_chunk_ids.extend(chunk_ids)
                    self.stats.files_processed += 1
                    logger.info(
                        "File processed successfully",
                        extra={
                            "file": str(file_path),
                            "chunks": len(chunk_ids),
                        },
                    )
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {e}"
                    self.stats.add_error(error_msg)
                    logger.error(error_msg, exc_info=True)

        finally:
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                delta = self.stats.end_time - self.stats.start_time
                self.stats.processing_time_seconds = delta.total_seconds()

            logger.info(
                "Batch processing complete",
                extra={
                    "files_processed": self.stats.files_processed,
                    "files_failed": self.stats.files_failed,
                    "chunks_inserted": self.stats.chunks_inserted,
                    "processing_time_seconds": self.stats.processing_time_seconds,
                },
            )

        return all_chunk_ids

    def process_file(self, file_path: Path) -> list[str]:
        """Process single markdown file through pipeline.

        Reads file, tokenizes, chunks, generates context headers, and
        creates ProcessedChunk objects ready for database insertion.

        Args:
            file_path: Path to markdown file.

        Returns:
            List of chunk IDs inserted into database.

        Raises:
            ParseError: If file cannot be read or parsed.
            Exception: For other processing errors.

        Example:
            >>> processor = BatchProcessor(config)
            >>> chunk_ids = processor.process_file(Path("docs/guide.md"))
        """
        logger.debug("Processing file", extra={"file": str(file_path)})

        # Stage 1: Read file and extract metadata
        content, metadata = self.reader.read_file(file_path)

        # Stage 2: Tokenize content
        tokens = self.tokenizer.tokenize(content)
        logger.debug(
            "Tokenized content",
            extra={"file": str(file_path), "token_count": len(tokens)},
        )

        # Stage 3: Chunk into segments
        chunks = self.chunker.chunk_text(content)
        logger.debug(
            "Chunked content",
            extra={"file": str(file_path), "chunk_count": len(chunks)},
        )

        # Stage 4: Generate context headers and create ProcessedChunk objects
        processed_chunks: list[ProcessedChunk] = []
        for i, chunk in enumerate(chunks):
            context_header = self.context_generator.generate_header(
                file_path=file_path,
                chunk_index=i,
                content=chunk.text,
            )

            processed_chunk = ProcessedChunk.create_from_chunk(
                chunk_text=chunk.text,
                context_header=context_header,
                metadata=metadata,
                chunk_index=i,
                total_chunks=len(chunks),
                token_count=chunk.token_count,
            )
            processed_chunks.append(processed_chunk)

        self.stats.chunks_created += len(processed_chunks)

        # Stage 5: Insert into database
        chunk_ids = self._insert_chunks(processed_chunks)
        self.stats.chunks_inserted += len(chunk_ids)

        return chunk_ids

    def _discover_files(self) -> list[Path]:
        """Discover markdown files in input directory.

        Returns:
            List of file paths matching pattern.
        """
        if self.config.recursive:
            pattern = f"**/{self.config.file_pattern}"
        else:
            pattern = self.config.file_pattern

        files = list(self.config.input_dir.glob(pattern))
        return sorted(files)

    def _insert_chunks(self, chunks: list[ProcessedChunk]) -> list[str]:
        """Insert processed chunks into database using batch operations.

        Uses DatabasePool for connection management and performs batch
        inserts for efficiency. Handles duplicate chunks based on hash.

        Args:
            chunks: List of ProcessedChunk objects to insert.

        Returns:
            List of chunk IDs (hashes) successfully inserted.

        Raises:
            psycopg2.Error: If database operation fails.
        """
        if not chunks:
            return []

        inserted_ids: list[str] = []
        batch: list[ProcessedChunk] = []

        with DatabasePool.get_connection() as conn:
            for chunk in chunks:
                batch.append(chunk)

                # Insert when batch is full
                if len(batch) >= self.config.batch_size:
                    ids = self._insert_batch(conn, batch)
                    inserted_ids.extend(ids)
                    batch = []

            # Insert remaining chunks
            if batch:
                ids = self._insert_batch(conn, batch)
                inserted_ids.extend(ids)

        return inserted_ids

    def _insert_batch(
        self,
        conn: Any,
        chunks: list[ProcessedChunk],
    ) -> list[str]:
        """Insert batch of chunks into database.

        Args:
            conn: Database connection.
            chunks: Chunks to insert.

        Returns:
            List of inserted chunk hashes.
        """
        if not chunks:
            return []

        # Build bulk insert query
        insert_sql = """
            INSERT INTO knowledge_base (
                chunk_text,
                chunk_hash,
                embedding,
                source_file,
                source_category,
                document_date,
                chunk_index,
                total_chunks,
                context_header
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (chunk_hash) DO NOTHING
            RETURNING chunk_hash
        """

        inserted_hashes: list[str] = []

        try:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        insert_sql,
                        (
                            chunk.chunk_text,
                            chunk.chunk_hash,
                            None,  # embedding (Phase 2)
                            chunk.source_file,
                            chunk.source_category,
                            chunk.document_date,
                            chunk.chunk_index,
                            chunk.total_chunks,
                            chunk.context_header,
                        ),
                    )

                    # Get inserted hash if not skipped
                    result = cur.fetchone()
                    if result:
                        inserted_hashes.append(result[0])

                conn.commit()

                logger.debug(
                    "Batch inserted",
                    extra={
                        "batch_size": len(chunks),
                        "inserted": len(inserted_hashes),
                        "skipped": len(chunks) - len(inserted_hashes),
                    },
                )

        except psycopg2.Error as e:
            conn.rollback()
            logger.error(
                "Batch insert failed",
                extra={"error": str(e), "batch_size": len(chunks)},
                exc_info=True,
            )
            raise

        return inserted_hashes
