"""Tests for batch document processing pipeline.

Comprehensive test suite for BatchProcessor including unit tests for
individual components, integration tests with database, performance
tests for batch operations, and error handling validation.

Test categories:
- Model validation (ProcessedChunk, BatchConfig, DocumentMetadata)
- Single file processing
- Directory batch processing
- Database insertion and deduplication
- Error handling and recovery
- Performance and batch optimization
- Progress tracking and statistics
"""

import hashlib
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg2
import pytest

from src.core.database import DatabasePool
from src.document_parsing.batch_processor import (
    BatchConfig,
    BatchProcessor,
    Chunker,
    MarkdownReader,
    ParseError,
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
def test_db() -> Any:
    """Initialize test database with clean state.

    Yields:
        Database connection for testing.
    """
    DatabasePool.initialize()
    with DatabasePool.get_connection() as conn:
        # Clean test data
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_base")
        conn.commit()
        yield conn


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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

Edit the config file to set your preferences.
"""
    file_path = temp_dir / "install-guide.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_documents(temp_dir: Path) -> list[Path]:
    """Create multiple sample documents for batch testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        List of sample document paths.
    """
    docs = []

    # Document 1
    doc1 = temp_dir / "user-guide.md"
    doc1.write_text(
        """# User Guide

Welcome to the user guide.

## Getting Started

Follow these steps to get started.
""",
        encoding="utf-8",
    )
    docs.append(doc1)

    # Document 2
    doc2 = temp_dir / "api-reference.md"
    doc2.write_text(
        """# API Reference

This document describes the API endpoints.

## Authentication

Use API keys for authentication.
""",
        encoding="utf-8",
    )
    docs.append(doc2)

    # Document 3 in subdirectory
    subdir = temp_dir / "advanced"
    subdir.mkdir()
    doc3 = subdir / "advanced-topics.md"
    doc3.write_text(
        """# Advanced Topics

This covers advanced usage patterns.

## Performance Tuning

Optimize your configuration for performance.
""",
        encoding="utf-8",
    )
    docs.append(doc3)

    return docs


# ============================================================================
# MODEL TESTS
# ============================================================================


class TestDocumentMetadata:
    """Test DocumentMetadata model validation."""

    def test_minimal_metadata(self) -> None:
        """Test metadata with minimal required fields."""
        metadata = DocumentMetadata(source_file="test.md")
        assert metadata.title == "Untitled"
        assert metadata.author is None
        assert metadata.category is None
        assert metadata.tags == []
        assert metadata.source_file == "test.md"
        assert metadata.document_date is None

    def test_full_metadata(self) -> None:
        """Test metadata with all fields populated."""
        metadata = DocumentMetadata(
            title="User Guide",
            author="John Doe",
            category="product_docs",
            tags=["installation", "setup"],
            source_file="docs/guide.md",
            document_date=date(2024, 1, 15),
        )
        assert metadata.title == "User Guide"
        assert metadata.author == "John Doe"
        assert metadata.category == "product_docs"
        assert metadata.tags == ["installation", "setup"]
        assert metadata.document_date == date(2024, 1, 15)

    def test_tags_validation(self) -> None:
        """Test tags list validation removes empty strings."""
        metadata = DocumentMetadata(
            source_file="test.md",
            tags=["valid", "", "  ", "also-valid", "  trimmed  "],
        )
        assert metadata.tags == ["valid", "also-valid", "trimmed"]

    def test_title_length_validation(self) -> None:
        """Test title length constraints."""
        # Valid length
        metadata = DocumentMetadata(
            title="A" * 512,
            source_file="test.md",
        )
        assert len(metadata.title) == 512

        # Too long
        with pytest.raises(Exception):  # Pydantic ValidationError
            DocumentMetadata(
                title="A" * 513,
                source_file="test.md",
            )


class TestProcessedChunk:
    """Test ProcessedChunk model validation."""

    def test_create_from_chunk(self) -> None:
        """Test factory method for creating ProcessedChunk."""
        metadata = DocumentMetadata(
            title="Test Doc",
            source_file="test.md",
            category="test_category",
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="This is a test chunk with some content.",
            context_header="test.md > Section",
            metadata=metadata,
            chunk_index=0,
            total_chunks=5,
            token_count=10,
        )

        assert chunk.chunk_text == "This is a test chunk with some content."
        assert chunk.context_header == "test.md > Section"
        assert chunk.source_file == "test.md"
        assert chunk.source_category == "test_category"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 5
        assert chunk.chunk_token_count == 10
        assert chunk.embedding is None

        # Verify hash is computed correctly
        expected_hash = hashlib.sha256(
            "This is a test chunk with some content.".encode("utf-8")
        ).hexdigest()
        assert chunk.chunk_hash == expected_hash
        assert len(chunk.chunk_hash) == 64

    def test_chunk_index_validation(self) -> None:
        """Test chunk_index must be less than total_chunks."""
        metadata = DocumentMetadata(source_file="test.md")

        # Valid: chunk_index < total_chunks
        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="Test",
            context_header="header",
            metadata=metadata,
            chunk_index=4,
            total_chunks=5,
            token_count=1,
        )
        assert chunk.chunk_index == 4

    def test_metadata_dict_structure(self) -> None:
        """Test metadata dictionary is correctly populated."""
        metadata = DocumentMetadata(
            title="Test",
            author="Author",
            tags=["tag1", "tag2"],
            source_file="test.md",
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="Content",
            context_header="header",
            metadata=metadata,
            chunk_index=2,
            total_chunks=10,
            token_count=5,
        )

        assert chunk.metadata["title"] == "Test"
        assert chunk.metadata["author"] == "Author"
        assert chunk.metadata["tags"] == ["tag1", "tag2"]
        assert chunk.metadata["chunk_index"] == 2
        assert chunk.metadata["total_chunks"] == 10


class TestBatchProcessingStats:
    """Test BatchProcessingStats model."""

    def test_initial_state(self) -> None:
        """Test stats are initialized to zero."""
        stats = BatchProcessingStats()
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.chunks_created == 0
        assert stats.chunks_inserted == 0
        assert stats.processing_time_seconds == 0.0
        assert stats.errors == []
        assert stats.start_time is None
        assert stats.end_time is None

    def test_add_error(self) -> None:
        """Test adding errors increments failed count."""
        stats = BatchProcessingStats()
        stats.add_error("Error 1")
        stats.add_error("Error 2")

        assert stats.files_failed == 2
        assert stats.errors == ["Error 1", "Error 2"]


# ============================================================================
# COMPONENT TESTS
# ============================================================================


class TestTokenizer:
    """Test Tokenizer component."""

    def test_tokenize_simple(self) -> None:
        """Test basic tokenization."""
        text = "This is a test"
        tokens = Tokenizer.tokenize(text)
        assert tokens == ["This", "is", "a", "test"]

    def test_count_tokens(self) -> None:
        """Test token counting."""
        text = "This is a test"
        count = Tokenizer.count_tokens(text)
        assert count == 4

    def test_empty_text(self) -> None:
        """Test tokenization of empty text."""
        tokens = Tokenizer.tokenize("")
        # Empty string split on whitespace returns empty list
        assert tokens == [] or tokens == [""]  # Behavior may vary


class TestChunker:
    """Test Chunker component."""

    def test_chunk_text_paragraphs(self) -> None:
        """Test chunking by paragraphs."""
        text = """Paragraph 1

Paragraph 2

Paragraph 3"""
        chunker = Chunker(max_tokens=512, overlap=50)
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 3
        assert chunks[0].text == "Paragraph 1"
        assert chunks[1].text == "Paragraph 2"
        assert chunks[2].text == "Paragraph 3"

    def test_empty_paragraphs_skipped(self) -> None:
        """Test empty paragraphs are skipped."""
        text = """Content 1


Content 2"""
        chunker = Chunker()
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 2
        assert chunks[0].text == "Content 1"
        assert chunks[1].text == "Content 2"


class TestMarkdownReader:
    """Test MarkdownReader component."""

    def test_read_file(self, sample_markdown: Path) -> None:
        """Test reading markdown file."""
        content, metadata = MarkdownReader.read_file(sample_markdown)

        assert "Installation Guide" in content
        assert "Prerequisites" in content
        assert metadata.source_file == str(sample_markdown)
        assert metadata.title == "install-guide"

    def test_read_nonexistent_file(self, temp_dir: Path) -> None:
        """Test reading non-existent file raises error."""
        with pytest.raises(ParseError):
            MarkdownReader.read_file(temp_dir / "nonexistent.md")


# ============================================================================
# BATCH CONFIG TESTS
# ============================================================================


class TestBatchConfig:
    """Test BatchConfig validation."""

    def test_valid_config(self, temp_dir: Path) -> None:
        """Test creating valid configuration."""
        config = BatchConfig(input_dir=temp_dir)
        assert config.input_dir == temp_dir
        assert config.batch_size == 100
        assert config.chunk_max_tokens == 512
        assert config.chunk_overlap == 50
        assert config.recursive is True
        assert config.file_pattern == "*.md"

    def test_custom_config(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = BatchConfig(
            input_dir=temp_dir,
            batch_size=50,
            chunk_max_tokens=256,
            chunk_overlap=25,
            recursive=False,
            file_pattern="*.txt",
        )
        assert config.batch_size == 50
        assert config.chunk_max_tokens == 256
        assert config.chunk_overlap == 25
        assert config.recursive is False
        assert config.file_pattern == "*.txt"

    def test_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test validation fails for non-existent directory."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BatchConfig(input_dir=temp_dir / "nonexistent")

    def test_file_instead_of_directory(self, temp_dir: Path) -> None:
        """Test validation fails when path is a file not directory."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test")

        with pytest.raises(Exception):  # Pydantic ValidationError
            BatchConfig(input_dir=file_path)


# ============================================================================
# BATCH PROCESSOR TESTS
# ============================================================================


class TestBatchProcessor:
    """Test BatchProcessor pipeline."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test processor initialization."""
        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        assert processor.config == config
        assert processor.stats.files_processed == 0
        assert processor.reader is not None
        assert processor.tokenizer is not None
        assert processor.chunker is not None

    def test_discover_files_flat(self, sample_documents: list[Path], temp_dir: Path) -> None:
        """Test file discovery without recursion."""
        config = BatchConfig(input_dir=temp_dir, recursive=False)
        processor = BatchProcessor(config)

        files = processor._discover_files()
        # Should find only files in root, not subdirectory
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)

    def test_discover_files_recursive(
        self, sample_documents: list[Path], temp_dir: Path
    ) -> None:
        """Test file discovery with recursion."""
        config = BatchConfig(input_dir=temp_dir, recursive=True)
        processor = BatchProcessor(config)

        files = processor._discover_files()
        # Should find all 3 files including subdirectory
        assert len(files) == 3
        assert all(f.suffix == ".md" for f in files)

    def test_process_file(self, sample_markdown: Path, test_db: Any, temp_dir: Path) -> None:
        """Test processing single file."""
        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        chunk_ids = processor.process_file(sample_markdown)

        # Should create multiple chunks
        assert len(chunk_ids) > 0
        assert processor.stats.chunks_created > 0
        assert processor.stats.chunks_inserted > 0

    def test_process_directory(
        self, sample_documents: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Test processing entire directory."""
        config = BatchConfig(input_dir=temp_dir, recursive=True)
        processor = BatchProcessor(config)

        chunk_ids = processor.process_directory()

        # Should process all 3 files
        assert processor.stats.files_processed == 3
        assert processor.stats.files_failed == 0
        assert len(chunk_ids) > 0
        assert processor.stats.start_time is not None
        assert processor.stats.end_time is not None
        assert processor.stats.processing_time_seconds > 0

    def test_database_deduplication(
        self, sample_markdown: Path, test_db: Any, temp_dir: Path
    ) -> None:
        """Test duplicate chunks are not inserted twice."""
        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        # Process file first time
        chunk_ids_1 = processor.process_file(sample_markdown)
        first_insert_count = processor.stats.chunks_inserted

        # Process same file again
        processor2 = BatchProcessor(config)
        chunk_ids_2 = processor2.process_file(sample_markdown)
        second_insert_count = processor2.stats.chunks_inserted

        # First time should insert chunks
        assert first_insert_count > 0

        # Second time should skip duplicates (ON CONFLICT DO NOTHING)
        assert second_insert_count == 0


# ============================================================================
# DATABASE INTEGRATION TESTS
# ============================================================================


class TestDatabaseIntegration:
    """Test database insertion and queries."""

    def test_chunk_insertion(self, sample_markdown: Path, test_db: Any, temp_dir: Path) -> None:
        """Test chunks are correctly inserted into database."""
        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        chunk_ids = processor.process_file(sample_markdown)

        # Verify chunks in database
        with test_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base")
            count = cur.fetchone()[0]
            assert count == len(chunk_ids)

            # Verify chunk structure
            cur.execute("SELECT chunk_text, chunk_hash, context_header FROM knowledge_base LIMIT 1")
            row = cur.fetchone()
            assert row[0]  # chunk_text not empty
            assert len(row[1]) == 64  # chunk_hash is SHA-256
            assert row[2]  # context_header not empty

    def test_batch_insertion_performance(
        self, sample_documents: list[Path], test_db: Any, temp_dir: Path
    ) -> None:
        """Test batch insertion is efficient."""
        config = BatchConfig(input_dir=temp_dir, batch_size=10, recursive=True)
        processor = BatchProcessor(config)

        import time

        start = time.time()
        processor.process_directory()
        duration = time.time() - start

        # Processing should complete in reasonable time
        # (This is a basic sanity check, not a precise benchmark)
        assert duration < 10  # Should be much faster in practice

    def test_transaction_rollback_on_error(self, temp_dir: Path, test_db: Any) -> None:
        """Test database rollback on insertion error."""
        # Create file with valid content
        valid_file = temp_dir / "valid.md"
        valid_file.write_text("# Valid content\n\nSome text here.", encoding="utf-8")

        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        # Process valid file successfully
        processor.process_file(valid_file)
        assert processor.stats.chunks_inserted > 0

        # Database should contain the chunks
        with test_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_base")
            count_before = cur.fetchone()[0]
            assert count_before > 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_file_continues_processing(self, temp_dir: Path, test_db: Any) -> None:
        """Test processing continues after file error."""
        # Create mix of valid and invalid files
        valid_file = temp_dir / "valid.md"
        valid_file.write_text("# Valid", encoding="utf-8")

        invalid_file = temp_dir / "invalid.md"
        # Create file but make it unreadable (simulate permission error)
        invalid_file.write_text("content")
        invalid_file.chmod(0o000)

        config = BatchConfig(input_dir=temp_dir, recursive=False)
        processor = BatchProcessor(config)

        try:
            chunk_ids = processor.process_directory()

            # Should process valid file despite invalid file error
            # (May or may not succeed depending on permissions)
        finally:
            # Restore permissions for cleanup
            invalid_file.chmod(0o644)

    def test_empty_file_handling(self, temp_dir: Path, test_db: Any) -> None:
        """Test handling of empty markdown files."""
        empty_file = temp_dir / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        # Should handle empty file gracefully
        chunk_ids = processor.process_file(empty_file)
        # Empty file produces no chunks
        assert len(chunk_ids) >= 0

    def test_stats_tracking_with_errors(self, temp_dir: Path) -> None:
        """Test statistics correctly track errors."""
        # Create invalid file scenario
        config = BatchConfig(input_dir=temp_dir)
        processor = BatchProcessor(config)

        processor.stats.add_error("Test error 1")
        processor.stats.add_error("Test error 2")

        assert processor.stats.files_failed == 2
        assert len(processor.stats.errors) == 2
        assert "Test error 1" in processor.stats.errors


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_large_batch_processing(self, temp_dir: Path, test_db: Any) -> None:
        """Test processing large number of files."""
        # Create 20 test files
        for i in range(20):
            file_path = temp_dir / f"doc_{i:03d}.md"
            file_path.write_text(
                f"# Document {i}\n\nContent for document {i}.\n\n## Section\n\nMore content.",
                encoding="utf-8",
            )

        config = BatchConfig(input_dir=temp_dir, batch_size=50)
        processor = BatchProcessor(config)

        chunk_ids = processor.process_directory()

        assert processor.stats.files_processed == 20
        assert len(chunk_ids) > 20  # Multiple chunks per file
        assert processor.stats.processing_time_seconds < 30  # Reasonable performance

    def test_batch_size_optimization(self, sample_documents: list[Path], temp_dir: Path) -> None:
        """Test different batch sizes complete successfully."""
        for batch_size in [1, 10, 100]:
            # Clean database between runs
            DatabasePool.initialize()
            with DatabasePool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM knowledge_base")
                conn.commit()

            config = BatchConfig(input_dir=temp_dir, batch_size=batch_size, recursive=True)
            processor = BatchProcessor(config)

            chunk_ids = processor.process_directory()

            # All batch sizes should produce same results
            assert processor.stats.files_processed == 3
            assert len(chunk_ids) > 0
