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
from datetime import datetime
from pathlib import Path
from typing import Any

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
