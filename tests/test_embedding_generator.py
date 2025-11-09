"""Unit tests for parallel embedding generation with batch processing.

Tests cover:
- Batch processing and creation
- Parallel embedding generation
- Progress tracking
- Error handling and validation
- Statistics collection
- Integration with ModelLoader
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.document_parsing.models import ProcessedChunk, DocumentMetadata
from src.embedding.generator import (
    EmbeddingGenerationError,
    EmbeddingGenerator,
    EmbeddingValidator,
)
from src.embedding.model_loader import ModelLoader, EXPECTED_EMBEDDING_DIMENSION


class TestEmbeddingValidator:
    """Tests for EmbeddingValidator class."""

    def test_validator_initialization(self) -> None:
        """Test EmbeddingValidator can be initialized."""
        validator = EmbeddingValidator()
        assert validator.EXPECTED_DIMENSION == EXPECTED_EMBEDDING_DIMENSION

    def test_validate_embedding_valid(self) -> None:
        """Test validation of valid embedding."""
        validator = EmbeddingValidator()
        embedding = [0.1] * EXPECTED_EMBEDDING_DIMENSION
        assert validator.validate_embedding(embedding) is True

    def test_validate_embedding_wrong_dimension(self) -> None:
        """Test validation fails for wrong dimension."""
        validator = EmbeddingValidator()
        embedding = [0.1] * 512
        assert validator.validate_embedding(embedding) is False

    def test_validate_embedding_non_numeric(self) -> None:
        """Test validation fails for non-numeric values."""
        validator = EmbeddingValidator()
        embedding = [0.1] * (EXPECTED_EMBEDDING_DIMENSION - 1) + ["invalid"]
        assert validator.validate_embedding(embedding) is False

    def test_validate_embedding_empty(self) -> None:
        """Test validation fails for empty embedding."""
        validator = EmbeddingValidator()
        assert validator.validate_embedding([]) is False

    def test_validate_embedding_not_list(self) -> None:
        """Test validation fails for non-list embedding."""
        validator = EmbeddingValidator()
        assert validator.validate_embedding("not a list") is False

    def test_validate_batch_all_valid(self) -> None:
        """Test batch validation with all valid embeddings."""
        validator = EmbeddingValidator()
        embeddings = [[0.1] * EXPECTED_EMBEDDING_DIMENSION for _ in range(5)]
        valid, invalid = validator.validate_batch(embeddings)
        assert valid == 5
        assert invalid == 0

    def test_validate_batch_mixed(self) -> None:
        """Test batch validation with mixed valid/invalid embeddings."""
        validator = EmbeddingValidator()
        embeddings = [
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
            [0.1] * 512,  # Wrong dimension
            [0.1] * EXPECTED_EMBEDDING_DIMENSION,
        ]
        valid, invalid = validator.validate_batch(embeddings)
        assert valid == 2
        assert invalid == 1


class TestEmbeddingGeneratorInitialization:
    """Tests for EmbeddingGenerator initialization."""

    def test_initialization_with_defaults(self) -> None:
        """Test EmbeddingGenerator initialization with default parameters."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            assert gen.batch_size == 32
            assert gen.num_workers == 4
            assert gen.use_threading is True

    def test_initialization_with_custom_params(self) -> None:
        """Test EmbeddingGenerator initialization with custom parameters."""
        mock_loader = Mock(spec=ModelLoader)
        gen = EmbeddingGenerator(
            model_loader=mock_loader,
            batch_size=64,
            num_workers=8,
            use_threading=False,
        )
        assert gen.batch_size == 64
        assert gen.num_workers == 8
        assert gen.use_threading is False
        assert gen.model_loader is mock_loader

    def test_initialization_creates_default_loader(self) -> None:
        """Test EmbeddingGenerator creates ModelLoader if not provided."""
        with patch("src.embedding.generator.ModelLoader") as mock_loader_class:
            gen = EmbeddingGenerator()
            mock_loader_class.assert_called_once()

    def test_initialization_sets_statistics(self) -> None:
        """Test EmbeddingGenerator initializes statistics to zero."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            assert gen.processed_count == 0
            assert gen.failed_count == 0
            assert gen.total_count == 0


class TestBatchCreation:
    """Tests for batch creation from chunks list."""

    def test_create_single_batch(self) -> None:
        """Test creating single batch from small list."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator(batch_size=10)
            chunks = [
                self._create_test_chunk(i) for i in range(5)
            ]
            batches = gen._create_batches(chunks)
            assert len(batches) == 1
            assert len(batches[0]) == 5

    def test_create_multiple_batches(self) -> None:
        """Test creating multiple batches."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator(batch_size=10)
            chunks = [
                self._create_test_chunk(i) for i in range(25)
            ]
            batches = gen._create_batches(chunks)
            assert len(batches) == 3
            assert len(batches[0]) == 10
            assert len(batches[1]) == 10
            assert len(batches[2]) == 5

    def test_create_batches_empty(self) -> None:
        """Test creating batches from empty list."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            batches = gen._create_batches([])
            assert len(batches) == 0

    def test_batch_size_boundary(self) -> None:
        """Test batch creation with exact batch size multiple."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator(batch_size=10)
            chunks = [
                self._create_test_chunk(i) for i in range(30)
            ]
            batches = gen._create_batches(chunks)
            assert len(batches) == 3
            for batch in batches:
                assert len(batch) == 10

    @staticmethod
    def _create_test_chunk(index: int) -> ProcessedChunk:
        """Create a test ProcessedChunk."""
        return ProcessedChunk(
            chunk_text=f"Test chunk {index}",
            chunk_hash=f"hash_{index}",
            context_header=f"doc > section {index}",
            source_file="test.md",
            source_category="test",
            document_date=None,
            chunk_index=index,
            total_chunks=10,
            chunk_token_count=100,
            metadata={},
            embedding=None,
        )


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    def test_generate_embeddings_valid_texts(self) -> None:
        """Test generating embeddings for valid text list."""
        mock_loader = Mock(spec=ModelLoader)
        mock_embeddings = [[0.1] * EXPECTED_EMBEDDING_DIMENSION for _ in range(3)]
        mock_loader.encode.return_value = mock_embeddings

        gen = EmbeddingGenerator(model_loader=mock_loader)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = gen.generate_embeddings_for_texts(texts)

        assert len(result) == 3
        assert all(len(emb) == EXPECTED_EMBEDDING_DIMENSION for emb in result)

    def test_generate_embeddings_empty_list(self) -> None:
        """Test generating embeddings for empty list raises error."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            with pytest.raises(ValueError, match="empty text list"):
                gen.generate_embeddings_for_texts([])

    def test_generate_embeddings_encode_failure(self) -> None:
        """Test handling of encode failure."""
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.encode.side_effect = RuntimeError("Encode failed")

        gen = EmbeddingGenerator(model_loader=mock_loader)
        with pytest.raises(EmbeddingGenerationError):
            gen.generate_embeddings_for_texts(["text"])


class TestChunkEnrichment:
    """Tests for chunk enrichment with embeddings."""

    def test_enrich_chunk_valid_embedding(self) -> None:
        """Test enriching chunk with valid embedding."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            chunk = self._create_test_chunk()
            embedding = [0.1] * EXPECTED_EMBEDDING_DIMENSION

            enriched = gen.validate_and_enrich_chunk(chunk, embedding)
            assert enriched.embedding == embedding
            assert enriched.chunk_text == chunk.chunk_text

    def test_enrich_chunk_invalid_embedding(self) -> None:
        """Test enriching chunk with invalid embedding raises error."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            chunk = self._create_test_chunk()
            embedding = [0.1] * 512  # Wrong dimension

            with pytest.raises(ValueError, match="Invalid embedding"):
                gen.validate_and_enrich_chunk(chunk, embedding)

    def test_enrich_chunk_preserves_metadata(self) -> None:
        """Test enriching chunk preserves all original metadata."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            chunk = self._create_test_chunk()
            embedding = [0.1] * EXPECTED_EMBEDDING_DIMENSION

            enriched = gen.validate_and_enrich_chunk(chunk, embedding)
            assert enriched.source_file == chunk.source_file
            assert enriched.chunk_index == chunk.chunk_index
            assert enriched.metadata == chunk.metadata

    @staticmethod
    def _create_test_chunk() -> ProcessedChunk:
        """Create a test ProcessedChunk."""
        return ProcessedChunk(
            chunk_text="Test chunk text",
            chunk_hash="test_hash",
            context_header="doc > section",
            source_file="test.md",
            source_category="test",
            document_date=None,
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=100,
            metadata={"key": "value"},
            embedding=None,
        )


class TestProgressTracking:
    """Tests for progress tracking."""

    def test_progress_summary_initial(self) -> None:
        """Test initial progress summary."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            summary = gen.get_progress_summary()

            assert summary["processed"] == 0
            assert summary["failed"] == 0
            assert summary["total"] == 0

    def test_progress_summary_updated(self) -> None:
        """Test progress summary after updates."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            gen.processed_count = 100
            gen.failed_count = 5
            gen.total_count = 150

            summary = gen.get_progress_summary()
            assert summary["processed"] == 100
            assert summary["failed"] == 5
            assert summary["total"] == 150

    def test_statistics_collection(self) -> None:
        """Test statistics collection."""
        with patch("src.embedding.generator.ModelLoader") as mock_loader_class:
            mock_loader = Mock(spec=ModelLoader)
            mock_loader.get_device.return_value = "cuda"
            mock_loader_class.return_value = mock_loader

            gen = EmbeddingGenerator(batch_size=32, num_workers=4)
            gen.total_count = 100
            gen.processed_count = 100
            gen.failed_count = 0

            stats = gen.get_statistics()
            assert stats["total_chunks"] == 100
            assert stats["processed_chunks"] == 100
            assert stats["failed_chunks"] == 0
            assert stats["batch_size"] == 32
            assert stats["num_workers"] == 4
            assert stats["device"] == "cuda"

    def test_statistics_throughput(self) -> None:
        """Test throughput calculation in statistics."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            gen.total_count = 1000
            gen.processed_count = 1000
            gen.start_time = 0
            gen.end_time = 10  # 10 second processing

            stats = gen.get_statistics()
            assert stats["throughput_chunks_per_sec"] > 0
            assert stats["elapsed_seconds"] == 10


class TestProcessChunks:
    """Tests for full chunk processing pipeline."""

    def test_process_empty_chunks_list(self) -> None:
        """Test processing empty chunks list."""
        with patch("src.embedding.generator.ModelLoader"):
            gen = EmbeddingGenerator()
            result = gen.process_chunks([])
            assert result == []

    def test_process_chunks_with_progress_callback(self) -> None:
        """Test processing chunks with progress callback."""
        mock_loader = Mock(spec=ModelLoader)
        mock_embeddings = [[0.1] * EXPECTED_EMBEDDING_DIMENSION for _ in range(5)]
        mock_loader.encode.return_value = mock_embeddings

        callback = Mock()
        gen = EmbeddingGenerator(model_loader=mock_loader, batch_size=5)
        chunks = [self._create_test_chunk(i) for i in range(5)]

        result = gen.process_chunks(chunks, progress_callback=callback)
        assert len(result) == 5
        # Callback should be called at least once
        assert callback.called

    def test_process_chunks_statistics_updated(self) -> None:
        """Test that statistics are updated during processing."""
        mock_loader = Mock(spec=ModelLoader)
        mock_embeddings = [[0.1] * EXPECTED_EMBEDDING_DIMENSION for _ in range(5)]
        mock_loader.encode.return_value = mock_embeddings
        mock_loader.get_device.return_value = "cpu"

        gen = EmbeddingGenerator(model_loader=mock_loader, batch_size=5)
        chunks = [self._create_test_chunk(i) for i in range(5)]

        result = gen.process_chunks(chunks)
        assert gen.total_count == 5
        assert gen.processed_count == 5
        assert gen.start_time > 0
        assert gen.end_time > 0

    def test_process_batch_error_handling(self) -> None:
        """Test error handling in batch processing."""
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.encode.side_effect = RuntimeError("Encode error")

        gen = EmbeddingGenerator(model_loader=mock_loader)
        chunks = [self._create_test_chunk(i) for i in range(3)]

        with pytest.raises(EmbeddingGenerationError):
            gen.process_batch(chunks)

    @staticmethod
    def _create_test_chunk(index: int) -> ProcessedChunk:
        """Create a test ProcessedChunk."""
        return ProcessedChunk(
            chunk_text=f"Test chunk {index}",
            chunk_hash=f"hash_{index}",
            context_header=f"doc > section {index}",
            source_file="test.md",
            source_category="test",
            document_date=None,
            chunk_index=index,
            total_chunks=10,
            chunk_token_count=100,
            metadata={},
            embedding=None,
        )


class TestEmbeddingGenerationError:
    """Tests for EmbeddingGenerationError."""

    def test_error_initialization(self) -> None:
        """Test error can be initialized with message."""
        error = EmbeddingGenerationError("Test error")
        assert str(error) == "Test error"

    def test_error_inheritance(self) -> None:
        """Test EmbeddingGenerationError inherits from Exception."""
        error = EmbeddingGenerationError("test")
        assert isinstance(error, Exception)

    def test_error_with_cause(self) -> None:
        """Test error can wrap another exception."""
        original = RuntimeError("Original error")
        error = EmbeddingGenerationError("Wrapper")
        error.__cause__ = original
        assert error.__cause__ is original
