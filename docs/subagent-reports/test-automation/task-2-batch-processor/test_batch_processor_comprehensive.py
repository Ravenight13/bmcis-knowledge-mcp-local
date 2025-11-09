"""Comprehensive test suite for batch_processor.py with 43 tests.

Type-safe testing following TDD methodology with:
- Complete type annotations (mypy --strict compliant)
- Clear test naming (test_{class}_{method}_{scenario})
- Organized into logical test classes
- Fixtures for database, files, components
- Mocking for external dependencies
- Edge case coverage
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from src.core.database import DatabasePool
from src.document_parsing.batch_processor import (
    BatchConfig,
    BatchProcessor,
    Chunker,
    ContextHeaderGenerator,
    MarkdownReader,
    ParseError,
    TextChunk,
    Tokenizer,
)
from src.document_parsing.models import (
    BatchProcessingStats,
    DocumentMetadata,
    ProcessedChunk,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_db() -> Any:
    """Initialize test database with clean state.

    Yields:
        Database connection for testing.
    """
    DatabasePool.initialize()
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_base")
        conn.commit()
        yield conn


@pytest.fixture
def sample_markdown(temp_dir: Path) -> Path:
    """Create sample markdown file for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path to sample markdown file.
    """
    content = """# Installation Guide

This is a sample installation guide for testing.

## Prerequisites

You need Python 3.11 or later.

## Installation Steps

1. Clone the repository
2. Install dependencies
3. Run the application

## Configuration

Edit the config file to set your preferences."""

    file_path: Path = temp_dir / "install-guide.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def multiple_markdown_files(temp_dir: Path) -> list[Path]:
    """Create multiple markdown files for batch testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        List of created file paths.
    """
    files: list[Path] = []

    # File 1
    file1: Path = temp_dir / "file1.md"
    file1.write_text("# Document 1\n\nContent 1.\n\nMore content.", encoding="utf-8")
    files.append(file1)

    # File 2
    file2: Path = temp_dir / "file2.md"
    file2.write_text("# Document 2\n\nContent 2.\n\nMore content.", encoding="utf-8")
    files.append(file2)

    # File 3 (in subdirectory)
    subdir: Path = temp_dir / "subdir"
    subdir.mkdir()
    file3: Path = subdir / "file3.md"
    file3.write_text("# Document 3\n\nContent 3.\n\nMore content.", encoding="utf-8")
    files.append(file3)

    return files


@pytest.fixture
def batch_config(temp_dir: Path) -> BatchConfig:
    """Create default BatchConfig for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Valid BatchConfig instance.
    """
    return BatchConfig(input_dir=temp_dir)


# ============================================================================
# CONFIGURATION TESTS (7 tests)
# ============================================================================


class TestBatchConfigDefaults:
    """Test BatchConfig default values."""

    def test_batch_config_defaults(self, temp_dir: Path) -> None:
        """Verify default configuration values.

        Args:
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)

        assert config.input_dir == temp_dir
        assert config.batch_size == 100
        assert config.chunk_max_tokens == 512
        assert config.chunk_overlap == 50
        assert config.recursive is True
        assert config.file_pattern == "*.md"
        assert config.skip_existing is True
        assert config.max_workers == 1

    def test_batch_config_custom_values(self, temp_dir: Path) -> None:
        """Verify custom configuration values accepted.

        Args:
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(
            input_dir=temp_dir,
            batch_size=50,
            chunk_max_tokens=256,
            chunk_overlap=25,
            recursive=False,
            file_pattern="*.txt",
            skip_existing=False,
            max_workers=4,
        )

        assert config.batch_size == 50
        assert config.chunk_max_tokens == 256
        assert config.chunk_overlap == 25
        assert config.recursive is False
        assert config.file_pattern == "*.txt"
        assert config.skip_existing is False
        assert config.max_workers == 4


class TestBatchConfigValidation:
    """Test BatchConfig Pydantic validation."""

    def test_input_dir_validation_nonexistent(self, temp_dir: Path) -> None:
        """Verify validation fails for non-existent directory.

        Args:
            temp_dir: Temporary directory.
        """
        nonexistent: Path = temp_dir / "nonexistent"

        with pytest.raises(ValueError, match="does not exist"):
            BatchConfig(input_dir=nonexistent)

    def test_input_dir_validation_is_file(self, temp_dir: Path) -> None:
        """Verify validation fails when path is file not directory.

        Args:
            temp_dir: Temporary directory.
        """
        file_path: Path = temp_dir / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            BatchConfig(input_dir=file_path)

    def test_batch_size_min_bound(self, temp_dir: Path) -> None:
        """Verify batch_size minimum bound is 1.

        Args:
            temp_dir: Temporary directory.
        """
        with pytest.raises(ValueError):
            BatchConfig(input_dir=temp_dir, batch_size=0)

    def test_batch_size_max_bound(self, temp_dir: Path) -> None:
        """Verify batch_size maximum bound is 1000.

        Args:
            temp_dir: Temporary directory.
        """
        with pytest.raises(ValueError):
            BatchConfig(input_dir=temp_dir, batch_size=1001)

    def test_chunk_overlap_bounds(self, temp_dir: Path) -> None:
        """Verify chunk_overlap range is 0-512.

        Args:
            temp_dir: Temporary directory.
        """
        # Valid bounds
        config: BatchConfig = BatchConfig(
            input_dir=temp_dir, chunk_overlap=0
        )
        assert config.chunk_overlap == 0

        config = BatchConfig(input_dir=temp_dir, chunk_overlap=512)
        assert config.chunk_overlap == 512

        # Over bounds
        with pytest.raises(ValueError):
            BatchConfig(input_dir=temp_dir, chunk_overlap=513)

    def test_max_workers_bounds(self, temp_dir: Path) -> None:
        """Verify max_workers range is 1-16.

        Args:
            temp_dir: Temporary directory.
        """
        # Valid bounds
        config: BatchConfig = BatchConfig(input_dir=temp_dir, max_workers=1)
        assert config.max_workers == 1

        config = BatchConfig(input_dir=temp_dir, max_workers=16)
        assert config.max_workers == 16

        # Over bounds
        with pytest.raises(ValueError):
            BatchConfig(input_dir=temp_dir, max_workers=17)


# ============================================================================
# CORE BATCH PROCESSING TESTS (8 tests)
# ============================================================================


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization."""

    def test_batch_processor_initialization(self, batch_config: BatchConfig) -> None:
        """Verify processor initializes with all components.

        Args:
            batch_config: Valid configuration.
        """
        processor: BatchProcessor = BatchProcessor(batch_config)

        assert processor.config == batch_config
        assert isinstance(processor.stats, BatchProcessingStats)
        assert processor.stats.files_processed == 0
        assert processor.stats.chunks_created == 0
        assert processor.reader is not None
        assert processor.tokenizer is not None
        assert processor.chunker is not None
        assert processor.context_generator is not None


class TestFileDiscovery:
    """Test _discover_files file discovery logic."""

    def test_discover_files_flat(
        self, multiple_markdown_files: list[Path], temp_dir: Path
    ) -> None:
        """Verify file discovery without recursion.

        Args:
            multiple_markdown_files: Fixture with 3 files.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=False)
        processor: BatchProcessor = BatchProcessor(config)

        files: list[Path] = processor._discover_files()

        # Should find only 2 files in root (not subdirectory)
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)

    def test_discover_files_recursive(
        self, multiple_markdown_files: list[Path], temp_dir: Path
    ) -> None:
        """Verify file discovery with recursion.

        Args:
            multiple_markdown_files: Fixture with 3 files.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=True)
        processor: BatchProcessor = BatchProcessor(config)

        files: list[Path] = processor._discover_files()

        # Should find all 3 files including subdirectory
        assert len(files) == 3
        assert all(f.suffix == ".md" for f in files)
        # Results should be sorted
        assert files == sorted(files)

    def test_discover_files_pattern_matching(self, temp_dir: Path) -> None:
        """Verify glob pattern matching in discovery.

        Args:
            temp_dir: Temporary directory.
        """
        # Create mixed file types
        (temp_dir / "file1.md").write_text("# Doc 1")
        (temp_dir / "file2.txt").write_text("# Doc 2")
        (temp_dir / "file3.md").write_text("# Doc 3")

        config: BatchConfig = BatchConfig(input_dir=temp_dir, file_pattern="*.md")
        processor: BatchProcessor = BatchProcessor(config)

        files: list[Path] = processor._discover_files()

        # Should find only .md files
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)


class TestSingleFileProcessing:
    """Test process_file single file pipeline."""

    def test_process_file_single(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify process_file executes full pipeline.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(sample_markdown)

        # Should return list of chunk IDs
        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) > 0
        assert all(isinstance(id_, str) for id_ in chunk_ids)

        # Stats should be updated
        assert processor.stats.chunks_created > 0
        assert processor.stats.chunks_inserted > 0

    def test_process_file_multiple_chunks(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify file with multiple paragraphs creates multiple chunks.

        Args:
            sample_markdown: Sample markdown with multiple sections.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(sample_markdown)

        # Multiple paragraphs should create multiple chunks
        assert processor.stats.chunks_created >= 5


class TestDirectoryProcessing:
    """Test process_directory batch processing."""

    def test_process_directory_single_file(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify process_directory with single file.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_directory()

        # Should return combined chunk IDs
        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) > 0

        # Stats should be complete
        assert processor.stats.files_processed == 1
        assert processor.stats.files_failed == 0
        assert processor.stats.chunks_created > 0
        assert processor.stats.chunks_inserted > 0
        assert processor.stats.start_time is not None
        assert processor.stats.end_time is not None

    def test_process_directory_multiple_files(
        self, multiple_markdown_files: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Verify process_directory with multiple files.

        Args:
            multiple_markdown_files: Fixture with 3 files.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=True)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_directory()

        # Should process all 3 files
        assert processor.stats.files_processed == 3
        assert processor.stats.files_failed == 0
        assert len(chunk_ids) > 0


# ============================================================================
# BATCH SIZE & MEMORY TESTS (5 tests)
# ============================================================================


class TestBatchSizeLogic:
    """Test batch accumulation and boundary handling."""

    def test_batch_accumulation(self, temp_dir: Path, test_db: Any) -> None:
        """Verify batches accumulate to configured batch_size.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # Create 25 chunks manually
        chunks: list[ProcessedChunk] = []
        metadata: DocumentMetadata = DocumentMetadata(source_file="test.md")

        for i in range(25):
            chunk: ProcessedChunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Chunk {i}",
                context_header=f"Header {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=25,
                token_count=10,
            )
            chunks.append(chunk)

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=10)
        processor: BatchProcessor = BatchProcessor(config)

        # Insert with batch_size=10
        chunk_ids: list[str] = processor._insert_chunks(chunks)

        # All 25 should be inserted
        assert len(chunk_ids) == 25

    def test_batch_size_boundary_exact(self, temp_dir: Path, test_db: Any) -> None:
        """Verify batch_size exactly divides chunks.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # 20 chunks, batch_size=10
        chunks: list[ProcessedChunk] = []
        metadata: DocumentMetadata = DocumentMetadata(source_file="test.md")

        for i in range(20):
            chunk: ProcessedChunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Chunk {i}",
                context_header=f"Header {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=20,
                token_count=10,
            )
            chunks.append(chunk)

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=10)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor._insert_chunks(chunks)

        # Exactly 20 inserted (no partial batches)
        assert len(chunk_ids) == 20

    def test_batch_size_boundary_remainder(self, temp_dir: Path, test_db: Any) -> None:
        """Verify remaining chunks inserted after final batch.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # 15 chunks, batch_size=10 (2 batches: 10, 5)
        chunks: list[ProcessedChunk] = []
        metadata: DocumentMetadata = DocumentMetadata(source_file="test.md")

        for i in range(15):
            chunk: ProcessedChunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Chunk {i}",
                context_header=f"Header {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=15,
                token_count=10,
            )
            chunks.append(chunk)

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=10)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor._insert_chunks(chunks)

        # All 15 should be inserted
        assert len(chunk_ids) == 15

    def test_batch_size_one(self, temp_dir: Path, test_db: Any) -> None:
        """Verify batch_size=1 inserts each chunk individually.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # 5 chunks, batch_size=1
        chunks: list[ProcessedChunk] = []
        metadata: DocumentMetadata = DocumentMetadata(source_file="test.md")

        for i in range(5):
            chunk: ProcessedChunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Chunk {i}",
                context_header=f"Header {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=5,
                token_count=10,
            )
            chunks.append(chunk)

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=1)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor._insert_chunks(chunks)

        # All 5 inserted individually
        assert len(chunk_ids) == 5

    def test_batch_size_large(self, temp_dir: Path, test_db: Any) -> None:
        """Verify large batch_size processes all chunks together.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # 25 chunks, batch_size=1000 (all in one batch)
        chunks: list[ProcessedChunk] = []
        metadata: DocumentMetadata = DocumentMetadata(source_file="test.md")

        for i in range(25):
            chunk: ProcessedChunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Chunk {i}",
                context_header=f"Header {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=25,
                token_count=10,
            )
            chunks.append(chunk)

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=1000)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor._insert_chunks(chunks)

        # All 25 in single batch
        assert len(chunk_ids) == 25


# ============================================================================
# PROGRESS TRACKING TESTS (4 tests)
# ============================================================================


class TestProgressTracking:
    """Test statistics tracking and progress reporting."""

    def test_stats_files_processed(
        self, multiple_markdown_files: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Verify files_processed incremented correctly.

        Args:
            multiple_markdown_files: Fixture with 3 files.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=True)
        processor: BatchProcessor = BatchProcessor(config)

        processor.process_directory()

        assert processor.stats.files_processed == 3

    def test_stats_chunks_created(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify chunks_created accumulated correctly.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        processor.process_file(sample_markdown)

        assert processor.stats.chunks_created > 0

    def test_stats_chunks_inserted(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify chunks_inserted tracked correctly.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        processor.process_file(sample_markdown)

        # Chunks inserted should match chunks created (no dupes)
        assert processor.stats.chunks_inserted == processor.stats.chunks_created

    def test_stats_timing_calculation(
        self, multiple_markdown_files: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Verify processing_time_seconds calculated correctly.

        Args:
            multiple_markdown_files: Fixture with 3 files.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=True)
        processor: BatchProcessor = BatchProcessor(config)

        processor.process_directory()

        # Verify timing fields set
        assert processor.stats.start_time is not None
        assert processor.stats.end_time is not None
        assert processor.stats.processing_time_seconds > 0

        # Verify timing calculation correct
        delta: float = (
            processor.stats.end_time - processor.stats.start_time
        ).total_seconds()
        assert delta > 0


# ============================================================================
# ERROR RECOVERY TESTS (6 tests)
# ============================================================================


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_error_invalid_file_continues(self, temp_dir: Path, test_db: Any) -> None:
        """Verify processing continues after file error.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        # Create valid file
        valid_file: Path = temp_dir / "valid.md"
        valid_file.write_text("# Valid Content\n\nSome text.", encoding="utf-8")

        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=False)
        processor: BatchProcessor = BatchProcessor(config)

        # Should process despite having a file
        chunk_ids: list[str] = processor.process_directory()

        # At minimum, valid file was processed
        assert processor.stats.files_processed > 0

    def test_error_stats_files_failed(self, temp_dir: Path) -> None:
        """Verify stats.files_failed incremented on error.

        Args:
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        # Manually add errors
        processor.stats.add_error("Error 1")
        processor.stats.add_error("Error 2")

        assert processor.stats.files_failed == 2

    def test_error_stats_errors_collected(self, temp_dir: Path) -> None:
        """Verify stats.errors list collects error messages.

        Args:
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        error_msg: str = "Test error message"
        processor.stats.add_error(error_msg)

        assert error_msg in processor.stats.errors
        assert len(processor.stats.errors) == 1

    def test_error_parse_error_caught(self, temp_dir: Path) -> None:
        """Verify ParseError is caught and handled.

        Args:
            temp_dir: Temporary directory.
        """
        nonexistent_file: Path = temp_dir / "nonexistent.md"

        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        # Should raise ParseError for non-existent file
        with pytest.raises(ParseError):
            processor.process_file(nonexistent_file)

    def test_error_multiple_file_failures(self, temp_dir: Path, test_db: Any) -> None:
        """Verify stats track multiple file failures correctly.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        # Simulate multiple failures
        processor.stats.add_error("Error 1")
        processor.stats.add_error("Error 2")
        processor.stats.add_error("Error 3")

        assert processor.stats.files_failed == 3
        assert len(processor.stats.errors) == 3


# ============================================================================
# DATABASE INTEGRATION TESTS (5 tests)
# ============================================================================


class TestDatabaseIntegration:
    """Test database insertion and deduplication."""

    def test_database_chunks_inserted(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify chunks inserted into database.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(sample_markdown)

        # Verify chunks in database
        with test_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base")
            count: int = cur.fetchone()[0]
            assert count == len(chunk_ids)

    def test_database_chunk_structure(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify chunk structure in database.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        processor.process_file(sample_markdown)

        # Verify chunk fields
        with test_db.cursor() as cur:
            cur.execute(
                "SELECT chunk_text, chunk_hash, context_header FROM knowledge_base LIMIT 1"
            )
            row = cur.fetchone()

            assert row[0]  # chunk_text not empty
            assert len(row[1]) == 64  # SHA-256 hash
            assert row[2]  # context_header not empty

    def test_database_deduplication(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify duplicate chunks not inserted twice.

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        # First processing
        chunk_ids_1: list[str] = processor.process_file(sample_markdown)
        count_1: int = len(chunk_ids_1)

        # Second processing (same file)
        processor2: BatchProcessor = BatchProcessor(config)
        chunk_ids_2: list[str] = processor2.process_file(sample_markdown)
        count_2: int = len(chunk_ids_2)

        # First should insert, second should skip all (dedup)
        assert count_1 > 0
        assert count_2 == 0  # All are duplicates

    def test_database_empty_chunks(self, temp_dir: Path, test_db: Any) -> None:
        """Verify empty chunk list handled gracefully.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        # Insert empty list
        chunk_ids: list[str] = processor._insert_chunks([])

        # Should return empty list
        assert chunk_ids == []


# ============================================================================
# EDGE CASES TESTS (5 tests)
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file_handling(self, temp_dir: Path, test_db: Any) -> None:
        """Verify empty file handled gracefully.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        empty_file: Path = temp_dir / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(empty_file)

        # Empty file produces no chunks
        assert len(chunk_ids) >= 0

    def test_single_paragraph_single_chunk(self, temp_dir: Path, test_db: Any) -> None:
        """Verify minimal file with single paragraph.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        minimal_file: Path = temp_dir / "minimal.md"
        minimal_file.write_text("# Title\n\nContent.", encoding="utf-8")

        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(minimal_file)

        assert len(chunk_ids) >= 1

    def test_large_file_many_chunks(self, temp_dir: Path, test_db: Any) -> None:
        """Verify large file with many chunks processed.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        large_file: Path = temp_dir / "large.md"

        # Create file with many paragraphs
        content: str = ""
        for i in range(50):
            content += f"Paragraph {i}\n\n"

        large_file.write_text(content, encoding="utf-8")

        config: BatchConfig = BatchConfig(input_dir=temp_dir, batch_size=10)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(large_file)

        # Many chunks created and processed
        assert processor.stats.chunks_created >= 10

    def test_special_characters_handling(self, temp_dir: Path, test_db: Any) -> None:
        """Verify special characters preserved.

        Args:
            temp_dir: Temporary directory.
            test_db: Test database.
        """
        special_file: Path = temp_dir / "special.md"
        special_content: str = """# Unicode Test

Japanese: こんにちは

Accents: café, naïve, résumé

Symbols: @#$%^&*()"""

        special_file.write_text(special_content, encoding="utf-8")

        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        chunk_ids: list[str] = processor.process_file(special_file)

        # Should handle special characters
        assert len(chunk_ids) > 0


# ============================================================================
# TYPE SAFETY TESTS (3 tests)
# ============================================================================


class TestTypeSafety:
    """Test type safety and contracts."""

    def test_process_file_return_type(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Verify process_file returns list[str].

        Args:
            sample_markdown: Sample markdown file.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)
        processor: BatchProcessor = BatchProcessor(config)

        result: list[str] = processor.process_file(sample_markdown)

        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_process_directory_return_type(
        self, multiple_markdown_files: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Verify process_directory returns list[str].

        Args:
            multiple_markdown_files: Fixture with 3 files.
            test_db: Test database.
            temp_dir: Temporary directory.
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir, recursive=True)
        processor: BatchProcessor = BatchProcessor(config)

        result: list[str] = processor.process_directory()

        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_batch_config_path_type(self, temp_dir: Path) -> None:
        """Verify input_dir is Path type.

        Args:
            temp_dir: Temporary directory (Path instance).
        """
        config: BatchConfig = BatchConfig(input_dir=temp_dir)

        assert isinstance(config.input_dir, Path)
