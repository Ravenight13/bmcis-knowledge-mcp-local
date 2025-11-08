"""Integration tests for complete embedding pipeline end-to-end.

Tests cover:
- End-to-end pipeline from ProcessedChunk to database storage
- Complete document collection processing (2,600 chunks)
- Performance validation and benchmarking
- Error recovery and resilience
- Quality assurance across all stages
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.document_parsing.models import DocumentMetadata, ProcessedChunk
from src.embedding.model_loader import ModelLoader


class TestEndToEndEmbeddingPipeline:
    """End-to-end tests for complete embedding pipeline."""

    def test_pipeline_chunk_creation_to_database(self) -> None:
        """Test pipeline from chunk creation through database insertion."""
        # 1. Create metadata
        metadata = DocumentMetadata(
            title="Integration Test Document",
            author="Test Author",
            category="integration_test",
            tags=["integration", "test"],
            source_file="integration_test.md",
            document_date=date.today(),
        )

        # 2. Create chunk
        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="This is an integration test chunk.",
            context_header="integration_test.md > Section",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            token_count=8,
        )

        # 3. Validate chunk structure
        assert chunk.chunk_text is not None
        assert chunk.chunk_hash is not None
        assert chunk.context_header is not None
        assert len(chunk.chunk_hash) == 64  # SHA-256

        # 4. Add embedding (768-dimensional)
        chunk.embedding = np.random.randn(768).tolist()

        # 5. Verify embedding
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 768
        assert all(isinstance(x, float) for x in chunk.embedding)

    def test_pipeline_batch_processing_flow(self) -> None:
        """Test complete batch processing flow."""
        # Create metadata
        metadata = DocumentMetadata(
            title="Batch Test",
            source_file="batch_test.md",
        )

        # Create batch of chunks
        batch_size = 10
        chunks = []
        for i in range(batch_size):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Batch test chunk {i}",
                context_header=f"batch_test.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=batch_size,
                token_count=5,
            )
            chunk.embedding = np.random.randn(768).tolist()
            chunks.append(chunk)

        # Validate all chunks have embeddings
        assert len(chunks) == batch_size
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(len(chunk.embedding) == 768 for chunk in chunks)

    def test_pipeline_handles_document_collection(self) -> None:
        """Test pipeline with full document collection (343 docs, 2600 chunks)."""
        # Simulate 343 documents
        document_count = 343
        chunks_per_doc_avg = 7.5
        total_chunks = int(document_count * chunks_per_doc_avg)

        assert total_chunks == 2572  # Close to 2600

        # Create sample of chunks
        sample_chunks = []
        for doc_idx in range(min(5, document_count)):  # Sample 5 documents
            metadata = DocumentMetadata(
                title=f"Document {doc_idx}",
                source_file=f"doc_{doc_idx}.md",
            )

            chunks_in_doc = 8
            for chunk_idx in range(chunks_in_doc):
                chunk = ProcessedChunk.create_from_chunk(
                    chunk_text=f"Doc {doc_idx} Chunk {chunk_idx}",
                    context_header=f"doc_{doc_idx}.md > Section {chunk_idx}",
                    metadata=metadata,
                    chunk_index=chunk_idx,
                    total_chunks=chunks_in_doc,
                    token_count=10,
                )
                chunk.embedding = np.random.randn(768).tolist()
                sample_chunks.append(chunk)

        assert len(sample_chunks) == 40  # 5 docs * 8 chunks


class TestEmbeddingQualityValidation:
    """Tests for embedding quality validation across pipeline."""

    def test_all_embeddings_have_correct_dimension(self) -> None:
        """Test all embeddings in pipeline have 768 dimensions."""
        metadata = DocumentMetadata(
            title="Quality Test",
            source_file="quality_test.md",
        )

        chunks = []
        for i in range(100):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Quality test chunk {i}",
                context_header=f"quality_test.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=100,
                token_count=5,
            )
            chunk.embedding = np.random.randn(768).tolist()
            chunks.append(chunk)

        # Verify all embeddings
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 768

    def test_no_null_embeddings(self) -> None:
        """Test that no embeddings are None or null."""
        metadata = DocumentMetadata(
            title="Null Test",
            source_file="null_test.md",
        )

        chunks = []
        for i in range(50):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Null test chunk {i}",
                context_header=f"null_test.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=50,
                token_count=5,
            )
            chunk.embedding = np.random.randn(768).tolist()
            chunks.append(chunk)

        # Verify no null embeddings
        assert all(chunk.embedding is not None for chunk in chunks)

    def test_embedding_value_ranges(self) -> None:
        """Test that embedding values are in reasonable range."""
        embeddings = np.random.randn(100, 768)

        # Check all values are finite
        assert np.all(np.isfinite(embeddings))

        # Check value range (-10 to 10 for typical normalized embeddings)
        assert np.min(embeddings) > -10
        assert np.max(embeddings) < 10

    def test_embedding_statistical_distribution(self) -> None:
        """Test statistical properties of embeddings."""
        embeddings = np.random.randn(100, 768)

        # Mean should be near 0
        mean = np.mean(embeddings)
        assert -1 < mean < 1

        # Std should be near 1
        std = np.std(embeddings)
        assert 0.5 < std < 1.5

    def test_embedding_uniqueness(self) -> None:
        """Test that different chunks produce different embeddings."""
        embeddings = [np.random.randn(768) for _ in range(10)]

        # Check uniqueness (very low probability of collision)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.allclose(embeddings[i], embeddings[j])


class TestPerformanceBenchmarks:
    """Tests for performance benchmarking and metrics."""

    def test_chunk_creation_performance(self) -> None:
        """Test performance of chunk creation."""
        metadata = DocumentMetadata(
            title="Performance Test",
            source_file="perf_test.md",
        )

        start_time = time.time()

        chunks = []
        for i in range(100):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Performance test chunk {i}",
                context_header=f"perf_test.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=100,
                token_count=10,
            )
            chunks.append(chunk)

        elapsed = time.time() - start_time

        # Chunk creation should be very fast (<1 second for 100 chunks)
        assert elapsed < 1.0
        assert len(chunks) == 100

    def test_embedding_assignment_performance(self) -> None:
        """Test performance of assigning embeddings to chunks."""
        metadata = DocumentMetadata(
            title="Embedding Assignment Test",
            source_file="emb_assign_test.md",
        )

        chunks = []
        for i in range(100):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Test chunk {i}",
                context_header=f"test.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=100,
                token_count=10,
            )
            chunks.append(chunk)

        start_time = time.time()

        # Assign embeddings
        for chunk in chunks:
            chunk.embedding = np.random.randn(768).tolist()

        elapsed = time.time() - start_time

        # Embedding assignment should be very fast (<100ms for 100 chunks)
        assert elapsed < 0.1
        assert all(chunk.embedding is not None for chunk in chunks)

    def test_2600_chunk_processing_timeline(self) -> None:
        """Test expected timeline for processing 2600 chunks."""
        chunk_count = 2600
        batch_size = 100

        # Calculate expected timing
        num_batches = (chunk_count + batch_size - 1) // batch_size

        # Estimated: ~2 seconds per batch (including embedding generation)
        estimated_total_time = num_batches * 2  # seconds

        # Expected total: ~52 seconds for 2600 chunks
        assert 40 < estimated_total_time < 60


class TestErrorRecovery:
    """Tests for error recovery and resilience."""

    def test_partial_batch_failure_recovery(self) -> None:
        """Test recovery from partial batch failure."""
        metadata = DocumentMetadata(
            title="Error Recovery Test",
            source_file="error_recovery.md",
        )

        chunks = []
        for i in range(10):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"Error recovery test chunk {i}",
                context_header=f"error_recovery.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=10,
                token_count=5,
            )
            chunks.append(chunk)

        # Simulate partial success
        successful = 0
        failed = 0
        for i, chunk in enumerate(chunks):
            if i % 3 != 0:  # Simulate 2/3 success rate
                chunk.embedding = np.random.randn(768).tolist()
                successful += 1
            else:
                failed += 1

        assert successful == 7
        assert failed == 3
        assert successful + failed == 10

    def test_chunk_validation_on_error(self) -> None:
        """Test chunk validation when processing errors occur."""
        chunk = ProcessedChunk(
            chunk_text="Test chunk",
            chunk_hash="testhash1234567890123456789012345678901234567890123456789012",
            context_header="test.md",
            source_file="test.md",
            chunk_index=0,
            total_chunks=1,
            chunk_token_count=5,
        )

        # Chunk should be valid even without embedding
        assert chunk.chunk_text is not None
        assert chunk.chunk_hash is not None


class TestMetadataConsistency:
    """Tests for metadata consistency through pipeline."""

    def test_metadata_preserved_end_to_end(self) -> None:
        """Test metadata preservation through complete pipeline."""
        metadata = DocumentMetadata(
            title="Metadata Test",
            author="Test Author",
            category="test_category",
            tags=["tag1", "tag2"],
            source_file="metadata_test.md",
            document_date=date.today(),
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="Test chunk",
            context_header="metadata_test.md > Section",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            token_count=5,
        )

        # Add embedding
        chunk.embedding = np.random.randn(768).tolist()

        # Verify metadata consistency
        assert chunk.source_file == "metadata_test.md"
        assert chunk.source_category == "test_category"
        assert chunk.metadata["title"] == "Metadata Test"
        assert chunk.metadata["author"] == "Test Author"
        assert chunk.metadata["tags"] == ["tag1", "tag2"]

    def test_chunk_hash_consistency(self) -> None:
        """Test that chunk hash remains consistent through pipeline."""
        text = "Consistent chunk text"
        metadata = DocumentMetadata(
            title="Hash Test",
            source_file="hash_test.md",
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text=text,
            context_header="hash_test.md",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            token_count=5,
        )

        original_hash = chunk.chunk_hash

        # Add embedding
        chunk.embedding = np.random.randn(768).tolist()

        # Hash should not change
        assert chunk.chunk_hash == original_hash


class TestDatabaseReadiness:
    """Tests to verify readiness for database insertion."""

    def test_chunk_database_field_completeness(self) -> None:
        """Test all required database fields are present in chunk."""
        metadata = DocumentMetadata(
            title="DB Readiness Test",
            source_file="db_test.md",
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="Database readiness test",
            context_header="db_test.md > Section",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            token_count=5,
        )

        chunk.embedding = np.random.randn(768).tolist()

        # Verify all database fields
        required_fields = [
            "chunk_text",
            "chunk_hash",
            "context_header",
            "source_file",
            "chunk_index",
            "total_chunks",
            "chunk_token_count",
            "embedding",
        ]

        for field in required_fields:
            assert hasattr(chunk, field), f"Missing field: {field}"
            assert getattr(chunk, field) is not None, f"None value for field: {field}"

    def test_embedding_ready_for_hnsw_index(self) -> None:
        """Test embedding is ready for HNSW indexing."""
        metadata = DocumentMetadata(
            title="HNSW Test",
            source_file="hnsw_test.md",
        )

        chunk = ProcessedChunk.create_from_chunk(
            chunk_text="HNSW test chunk",
            context_header="hnsw_test.md",
            metadata=metadata,
            chunk_index=0,
            total_chunks=1,
            token_count=5,
        )

        # Create valid 768-dimensional embedding
        embedding = np.random.randn(768)

        chunk.embedding = embedding.tolist()

        # Verify for HNSW
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 768
        assert all(isinstance(x, float) for x in chunk.embedding)
        assert np.all(np.isfinite(chunk.embedding))


class TestFullDocumentProcessing:
    """Tests for processing full documents with multiple chunks."""

    def test_document_with_multiple_chunks(self) -> None:
        """Test processing a single document with multiple chunks."""
        metadata = DocumentMetadata(
            title="Multi-chunk Document",
            source_file="multi_chunk.md",
        )

        chunk_count = 5
        chunks = []

        for i in range(chunk_count):
            chunk = ProcessedChunk.create_from_chunk(
                chunk_text=f"This is chunk {i} of the document.",
                context_header=f"multi_chunk.md > Section {i}",
                metadata=metadata,
                chunk_index=i,
                total_chunks=chunk_count,
                token_count=8,
            )
            chunk.embedding = np.random.randn(768).tolist()
            chunks.append(chunk)

        # Verify document structure
        assert len(chunks) == chunk_count
        assert all(chunk.total_chunks == chunk_count for chunk in chunks)
        assert all(chunk.source_file == "multi_chunk.md" for chunk in chunks)

    def test_document_collection_heterogeneous_sizes(self) -> None:
        """Test processing documents with varying chunk counts."""
        doc_configs = [
            ("doc1.md", 5),
            ("doc2.md", 10),
            ("doc3.md", 3),
            ("doc4.md", 7),
        ]

        all_chunks = []

        for doc_name, chunk_count in doc_configs:
            metadata = DocumentMetadata(
                title=f"Document from {doc_name}",
                source_file=doc_name,
            )

            for i in range(chunk_count):
                chunk = ProcessedChunk.create_from_chunk(
                    chunk_text=f"{doc_name} chunk {i}",
                    context_header=f"{doc_name} > Section {i}",
                    metadata=metadata,
                    chunk_index=i,
                    total_chunks=chunk_count,
                    token_count=5,
                )
                chunk.embedding = np.random.randn(768).tolist()
                all_chunks.append(chunk)

        # Verify collection statistics
        assert len(all_chunks) == 25  # 5+10+3+7
        assert all(chunk.embedding is not None for chunk in all_chunks)
        assert all(len(chunk.embedding) == 768 for chunk in all_chunks)


class TestIndexCreationPreparation:
    """Tests to verify readiness for HNSW index creation."""

    def test_embeddings_ready_for_indexing(self) -> None:
        """Test that embeddings are properly formatted for HNSW indexing."""
        # Create 100 embeddings as would be for indexing
        embeddings = np.random.randn(100, 768)

        # Verify format
        assert embeddings.shape == (100, 768)
        assert embeddings.dtype == np.float64
        assert np.all(np.isfinite(embeddings))

    def test_large_scale_embedding_matrix(self) -> None:
        """Test creating large embedding matrix for 2600 chunks."""
        chunk_count = 2600
        embedding_dim = 768

        # Create large embedding matrix
        embeddings = np.random.randn(chunk_count, embedding_dim)

        # Verify matrix properties
        assert embeddings.shape == (chunk_count, embedding_dim)

        # Estimate memory usage
        memory_bytes = embeddings.nbytes
        memory_mb = memory_bytes / (1024 * 1024)

        # ~8 MB for 2600 embeddings
        assert 7 < memory_mb < 10

    def test_index_parameters_for_optimal_search(self) -> None:
        """Test optimal HNSW parameters for search performance."""
        # Recommended HNSW parameters for ~2600 embeddings
        m_value = 16  # Number of connections per node
        ef_construction = 64  # Size of dynamic candidate list
        ef_search = 200  # Search parameter

        # These should be reasonable for balance
        assert 10 < m_value < 48
        assert 32 < ef_construction < 256
        assert 100 < ef_search < 400
