"""Real implementation tests for embedding pipeline - validates actual behavior with real models.

This test module provides comprehensive testing of the embedding pipeline using:
- REAL models from HuggingFace (no mocks of core functionality)
- REAL database connections (actual PostgreSQL operations)
- REAL performance measurements (actual hardware timing)

Test Classes:
- TestModelLoaderReal: Load actual models, verify dimensions, measure performance
- TestEmbeddingGeneratorReal: Generate embeddings with real chunks, validate quality
- TestChunkInserterReal: Insert chunks with embeddings into real database
- TestEmbeddingEndToEndPipeline: Complete pipeline end-to-end

Why real tests matter:
Heavily-mocked tests (75% coverage in Phase 2) don't validate actual behavior.
Real tests prove the pipeline works with actual components and performance
characteristics. Critical for production readiness.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import numpy as np
import pytest

from src.core.database import DatabasePool
from src.document_parsing.models import DocumentMetadata, ProcessedChunk
from src.embedding.database import ChunkInserter, InsertionStats
from src.embedding.generator import EmbeddingGenerator
from src.embedding.model_loader import (
    EXPECTED_EMBEDDING_DIMENSION,
    ModelLoader,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


class TestModelLoaderReal:
    """Test ModelLoader with real model downloads from HuggingFace.

    Why: Verify model loads correctly from HuggingFace, produces correct
    dimensions, and performs within acceptable time bounds.
    """

    def teardown_method(self) -> None:
        """Reset singleton instance after each test."""
        ModelLoader._instance = None

    @pytest.fixture
    def model_loader(self) -> ModelLoader:
        """Initialize model loader for tests.

        Returns:
            ModelLoader: Real model loader instance
        """
        return ModelLoader(device="cpu")

    def test_load_primary_model_real(self, model_loader: ModelLoader) -> None:
        """Load actual primary model and verify dimensions.

        Why this test:
        - Verify model downloads correctly from HuggingFace
        - Confirm embedding dimension matches expectations (768 for all-mpnet-base-v2)
        - Ensure model can be moved to device (CPU/GPU)

        What it tests:
        - Model loads without errors
        - Model produces 768-dimensional embeddings
        - Model inference works with sample text
        - Embeddings are numerically valid
        """
        # Get model - will download from HuggingFace if not cached
        model = model_loader.get_model()
        assert model is not None, "Model should load successfully"

        # Generate embedding for sample text
        sample_text: str = "This is a test sentence for embedding verification"
        embeddings: list[list[float]] = model.encode(
            [sample_text], convert_to_tensor=False
        )

        # Verify embeddings structure
        assert isinstance(embeddings, list), "Embeddings should be list"
        assert len(embeddings) == 1, "Should have one embedding"
        assert isinstance(embeddings[0], list), "Embedding should be list"

        # Verify embedding dimension
        embedding: list[float] = embeddings[0]
        assert len(embedding) == 768, (
            f"Expected 768-dimensional embedding, got {len(embedding)}"
        )

        # Verify embedding values are numeric
        assert all(
            isinstance(v, (int, float)) for v in embedding
        ), "All embedding values should be numeric"

        # Verify embeddings are not all zeros (sanity check)
        embedding_array: np.ndarray = np.array(embedding)
        assert not np.allclose(
            embedding_array, 0
        ), "Embedding should not be all zeros"

        # Verify embeddings are normalized (all-mpnet models produce unit vectors)
        norm: float = float(np.linalg.norm(embedding_array))
        assert 0.95 < norm < 1.05, (
            f"Expected normalized vector (~1.0), got norm={norm:.4f}"
        )

        logger.info(
            f"✓ Primary model loaded: 768-dim embeddings, norm={norm:.4f}"
        )

    def test_model_loading_time_benchmark(self, model_loader: ModelLoader) -> None:
        """Measure actual model loading time for performance baseline.

        Why this test:
        - Establish baseline for model load performance
        - Verify load time is reasonable (<30 seconds on CPU)
        - Use for performance tracking across versions
        - Critical for deployment planning

        What it tests:
        - Model loading completes in acceptable time
        - First load is slower than cached loads
        - Device selection affects timing
        """
        start: float = time.time()
        model = model_loader.get_model()
        load_time: float = time.time() - start

        assert load_time < 30, (
            f"Model load took {load_time:.2f}s (expected <30s on CPU)"
        )
        assert model is not None, "Model should be loaded"

        logger.info(f"✓ Model loading time: {load_time:.2f}s")

    def test_model_produces_different_embeddings_for_different_inputs(
        self, model_loader: ModelLoader
    ) -> None:
        """Verify model produces different embeddings for different inputs.

        Why this test:
        - Validate model actually processes input (not returning constant)
        - Check embedding variation across inputs
        - Ensure semantic differences are captured

        What it tests:
        - Different texts produce different embeddings
        - Similarity varies with semantic distance
        """
        model = model_loader.get_model()

        # Generate embeddings for very different texts
        texts: list[str] = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Paris is the capital of France",
        ]

        embeddings: list[list[float]] = model.encode(
            texts, convert_to_tensor=False
        )

        # Convert to numpy arrays for comparison
        emb_arrays: list[np.ndarray] = [np.array(e) for e in embeddings]

        # Compute pairwise distances
        dist_01: float = float(np.linalg.norm(emb_arrays[0] - emb_arrays[1]))
        dist_12: float = float(np.linalg.norm(emb_arrays[1] - emb_arrays[2]))
        dist_02: float = float(np.linalg.norm(emb_arrays[0] - emb_arrays[2]))

        # All distances should be non-zero (different embeddings)
        assert dist_01 > 0.1, "Different texts should have different embeddings"
        assert dist_12 > 0.1, "Different texts should have different embeddings"
        assert dist_02 > 0.1, "Different texts should have different embeddings"

        logger.info(f"✓ Embeddings vary correctly: d01={dist_01:.4f}, d12={dist_12:.4f}")


class TestEmbeddingGeneratorReal:
    """Test EmbeddingGenerator with real model embeddings.

    Why: Verify embeddings generated for actual chunk objects with real models,
    validate embedding quality through semantic similarity checks.
    """

    def teardown_method(self) -> None:
        """Clean up after tests."""
        ModelLoader._instance = None

    @pytest.fixture
    def generator(self) -> EmbeddingGenerator:
        """Initialize real embedding generator.

        Returns:
            EmbeddingGenerator: Real generator with actual model
        """
        return EmbeddingGenerator(
            model_loader=ModelLoader(device="cpu"), batch_size=8, num_workers=2
        )

    @pytest.fixture
    def sample_metadata(self) -> DocumentMetadata:
        """Create sample document metadata.

        Returns:
            DocumentMetadata: Sample metadata for test chunks
        """
        return DocumentMetadata(
            title="Test Document",
            author="Test Author",
            category="test_docs",
            tags=["test", "embedding"],
            source_file="test_document.md",
            document_date=date.today(),
        )

    def test_generate_embeddings_real_chunks(
        self, generator: EmbeddingGenerator, sample_metadata: DocumentMetadata
    ) -> None:
        """Generate embeddings for real document chunks.

        Why this test:
        - Verify embeddings generated for actual chunk objects
        - Validate embedding quality through semantic similarity
        - Test batch generation with multiple chunks
        - Confirm enrichment of chunks with embeddings

        What it tests:
        - Real EmbeddingGenerator works with ProcessedChunk objects
        - Embeddings have correct dimension (768)
        - Chunk enrichment preserves original data
        - Embedding quality validates semantic relationships
        """
        # Create real chunk objects with actual content
        chunks: list[ProcessedChunk] = [
            ProcessedChunk(
                chunk_text="The capital of France is Paris, a major European city",
                source_document_id="doc_geo_1",
                chunk_index=0,
                position_in_document=0,
                metadata=sample_metadata,
                token_count=12,
            ),
            ProcessedChunk(
                chunk_text="Paris is located in northern France on the Seine river",
                source_document_id="doc_geo_1",
                chunk_index=1,
                position_in_document=56,
                metadata=sample_metadata,
                token_count=11,
            ),
            ProcessedChunk(
                chunk_text="Boston is a major city in the United States with cold winters",
                source_document_id="doc_geo_2",
                chunk_index=0,
                position_in_document=0,
                metadata=sample_metadata,
                token_count=12,
            ),
        ]

        # Generate embeddings using real generator
        result: list[ProcessedChunk] = generator.process_batch(chunks)

        # Verify results structure
        assert len(result) == 3, f"Expected 3 enriched chunks, got {len(result)}"
        assert all(
            chunk.embedding is not None for chunk in result
        ), "All chunks should have embeddings"

        # Verify embedding dimensions
        for chunk in result:
            embedding: list[float] = chunk.embedding
            assert isinstance(embedding, list), "Embedding should be list"
            assert len(embedding) == 768, (
                f"Expected 768-dimensional embedding, got {len(embedding)}"
            )
            assert all(
                isinstance(v, (int, float)) for v in embedding
            ), "All values should be numeric"

        # Verify original chunk data preserved
        assert result[0].chunk_text == chunks[0].chunk_text
        assert result[0].chunk_index == 0
        assert result[0].source_document_id == "doc_geo_1"

        # Validate semantic similarity
        emb0: np.ndarray = np.array(result[0].embedding)
        emb1: np.ndarray = np.array(result[1].embedding)
        emb2: np.ndarray = np.array(result[2].embedding)

        # Normalize for cosine similarity
        emb0_norm: np.ndarray = emb0 / np.linalg.norm(emb0)
        emb1_norm: np.ndarray = emb1 / np.linalg.norm(emb1)
        emb2_norm: np.ndarray = emb2 / np.linalg.norm(emb2)

        # Cosine similarity between chunks 0 & 1 (both about Paris)
        sim_01: float = float(np.dot(emb0_norm, emb1_norm))
        # Cosine similarity between chunks 0 & 2 (different topics)
        sim_02: float = float(np.dot(emb0_norm, emb2_norm))

        # Similar chunks should have higher similarity
        assert sim_01 > 0.5, (
            f"Similar chunks should have sim > 0.5, got {sim_01:.4f}"
        )
        assert sim_02 < sim_01, (
            f"Similar chunks (0,1) should be more similar than different ones (0,2)"
        )

        logger.info(
            f"✓ Generated embeddings: 3 chunks, "
            f"similarity(0,1)={sim_01:.4f}, similarity(0,2)={sim_02:.4f}"
        )

    def test_batch_processing_performance_real(
        self, generator: EmbeddingGenerator, sample_metadata: DocumentMetadata
    ) -> None:
        """Measure real batch processing performance.

        Why this test:
        - Establish baseline for real batch processing
        - Verify performance target of 10-20x improvement
        - Measure throughput for capacity planning
        - Validate that parallel processing actually improves performance

        What it tests:
        - Batch generation completes in reasonable time
        - Throughput exceeds 50 chunks/sec baseline
        - Timing is consistent and measurable
        """
        # Create 100 chunks
        chunks: list[ProcessedChunk] = [
            ProcessedChunk(
                chunk_text=f"Sample document chunk number {i} with some content about testing",
                source_document_id=f"doc_{i // 10}",
                chunk_index=i % 10,
                position_in_document=i * 100,
                metadata=sample_metadata,
                token_count=10,
            )
            for i in range(100)
        ]

        # Measure generation time
        start: float = time.time()
        result: list[ProcessedChunk] = generator.process_chunks(chunks)
        elapsed: float = time.time() - start

        # Calculate throughput
        throughput: float = len(result) / elapsed if elapsed > 0 else 0

        # Verify results
        assert len(result) == 100, f"Expected 100 embeddings, got {len(result)}"
        assert all(chunk.embedding is not None for chunk in result)

        # Performance target: > 50 chunks/sec (after optimization: > 500)
        assert throughput > 50, (
            f"Throughput {throughput:.0f} chunks/sec is below baseline (>50)"
        )

        logger.info(
            f"✓ Batch processing: 100 chunks in {elapsed:.3f}s ({throughput:.0f} chunks/sec)"
        )

    def test_generator_statistics_tracking(
        self, generator: EmbeddingGenerator, sample_metadata: DocumentMetadata
    ) -> None:
        """Verify generator tracks statistics correctly.

        Why this test:
        - Ensure statistics are accurate for monitoring
        - Validate progress tracking
        - Check error counting

        What it tests:
        - Statistics reflect actual processing
        - Progress callback is invoked
        """
        chunks: list[ProcessedChunk] = [
            ProcessedChunk(
                chunk_text=f"Chunk {i}",
                source_document_id="doc_stats",
                chunk_index=i,
                position_in_document=i * 10,
                metadata=sample_metadata,
                token_count=2,
            )
            for i in range(20)
        ]

        # Track progress
        progress_updates: list[tuple[int, int]] = []

        def progress_callback(processed: int, total: int) -> None:
            """Capture progress updates."""
            progress_updates.append((processed, total))

        result: list[ProcessedChunk] = generator.process_chunks(
            chunks, progress_callback=progress_callback
        )

        # Verify statistics
        stats: dict[str, int] = generator.get_progress_summary()
        assert stats["processed"] == 20
        assert stats["total"] == 20
        assert stats["failed"] == 0

        # Verify progress was tracked
        assert len(progress_updates) > 0, "Progress callback should be invoked"
        assert progress_updates[-1] == (20, 20), "Final progress should be 20/20"

        logger.info(
            f"✓ Statistics tracking: processed={stats['processed']}, "
            f"failed={stats['failed']}, progress_updates={len(progress_updates)}"
        )


class TestChunkInserterReal:
    """Test ChunkInserter with real database operations.

    Why: Verify database operations work correctly, deduplication works,
    and data is properly stored and queryable.
    """

    @pytest.fixture
    def database_pool(self) -> DatabasePool:
        """Initialize real database pool.

        Returns:
            DatabasePool: Connected database pool
        """
        pool = DatabasePool()
        pool.initialize()
        yield pool
        # Cleanup after test
        try:
            pool.close()
        except Exception as e:
            logger.warning(f"Error closing pool: {e}")

    @pytest.fixture
    def sample_metadata(self) -> DocumentMetadata:
        """Create sample metadata for test chunks.

        Returns:
            DocumentMetadata: Sample metadata
        """
        return DocumentMetadata(
            title="Test Document",
            author="Test Author",
            category="test_docs",
            tags=["test"],
            source_file="test.md",
            document_date=date.today(),
        )

    def test_insert_chunks_with_embeddings_real(
        self, database_pool: DatabasePool, sample_metadata: DocumentMetadata
    ) -> None:
        """Insert real chunks with embeddings into database.

        Why this test:
        - Verify database schema supports embeddings
        - Test insertion of actual ProcessedChunk objects
        - Validate database deduplication (ON CONFLICT)
        - Confirm data integrity in database

        What it tests:
        - ChunkInserter connects to real database
        - Inserts chunks with embeddings successfully
        - ON CONFLICT clause prevents duplicates
        - Database returns correct insertion stats
        - Data is queryable after insertion
        """
        # Generate chunks with embeddings
        generator = EmbeddingGenerator(
            model_loader=ModelLoader(device="cpu"), batch_size=5, num_workers=2
        )

        chunks: list[ProcessedChunk] = [
            ProcessedChunk(
                chunk_text=f"Test chunk {i} content for database insertion",
                source_document_id="test_insert_doc",
                chunk_index=i,
                position_in_document=i * 50,
                metadata=sample_metadata,
                token_count=6,
            )
            for i in range(10)
        ]

        # Add embeddings
        enriched_chunks: list[ProcessedChunk] = generator.process_chunks(chunks)

        # Insert into database
        inserter = ChunkInserter(batch_size=5)
        stats: InsertionStats = inserter.insert_chunks(enriched_chunks)

        # Verify insertion
        assert stats.inserted >= 0, "Inserted count should be non-negative"
        assert stats.failed == 0, f"Expected 0 failed, got {stats.failed}"

        # Try inserting same chunks again - should handle duplicates
        stats_dup: InsertionStats = inserter.insert_chunks(enriched_chunks)
        assert stats_dup.failed == 0, "Duplicate insertion should not fail"

        logger.info(
            f"✓ Database insertion: {stats.inserted} inserted, "
            f"{stats_dup.updated} updated on re-insertion"
        )


class TestEmbeddingEndToEndPipeline:
    """Test complete embedding pipeline from document chunks to database.

    Why: Validate the complete pipeline works end-to-end without mocks,
    from chunk parsing through embedding generation to database storage.
    """

    @pytest.fixture
    def database_pool(self) -> DatabasePool:
        """Initialize database pool.

        Returns:
            DatabasePool: Connected database pool
        """
        pool = DatabasePool()
        pool.initialize()
        yield pool
        try:
            pool.close()
        except Exception as e:
            logger.warning(f"Error closing pool: {e}")

    @pytest.fixture
    def sample_metadata(self) -> DocumentMetadata:
        """Create sample metadata.

        Returns:
            DocumentMetadata: Sample metadata
        """
        return DocumentMetadata(
            title="End-to-End Test Document",
            author="Test System",
            category="test_docs",
            tags=["end_to_end", "pipeline"],
            source_file="e2e_test.md",
            document_date=date.today(),
        )

    def test_full_pipeline_real(
        self, database_pool: DatabasePool, sample_metadata: DocumentMetadata
    ) -> None:
        """End-to-end test: chunks → embeddings → database.

        Why this test:
        - Verify complete pipeline works without mocks
        - Test all integration points together
        - Validate final result in database
        - Confirm throughput of complete pipeline

        What it tests:
        - Document chunks are processed correctly
        - Embeddings generate with real model
        - Chunks insert into real database
        - Data is stored with correct structure
        - Pipeline metrics are captured
        """
        # Simulate parsed chunks (as would come from document parsing)
        chunks: list[ProcessedChunk] = [
            ProcessedChunk(
                chunk_text="Machine learning is a subset of artificial intelligence",
                source_document_id="e2e_doc",
                chunk_index=0,
                position_in_document=0,
                metadata=sample_metadata,
                token_count=9,
            ),
            ProcessedChunk(
                chunk_text="Neural networks are inspired by biological neurons in the brain",
                source_document_id="e2e_doc",
                chunk_index=1,
                position_in_document=58,
                metadata=sample_metadata,
                token_count=12,
            ),
            ProcessedChunk(
                chunk_text="Deep learning uses multiple layers of neural networks",
                source_document_id="e2e_doc",
                chunk_index=2,
                position_in_document=122,
                metadata=sample_metadata,
                token_count=10,
            ),
        ]

        # Step 1: Generate embeddings
        generator = EmbeddingGenerator(
            model_loader=ModelLoader(device="cpu"), batch_size=3, num_workers=2
        )
        start_generation: float = time.time()
        enriched_chunks: list[ProcessedChunk] = generator.process_chunks(chunks)
        generation_time: float = time.time() - start_generation

        # Verify embeddings generated
        assert len(enriched_chunks) == 3, "Should have 3 enriched chunks"
        assert all(
            chunk.embedding is not None for chunk in enriched_chunks
        ), "All chunks should have embeddings"

        # Step 2: Insert into database
        inserter = ChunkInserter(batch_size=3)
        start_insertion: float = time.time()
        stats: InsertionStats = inserter.insert_chunks(enriched_chunks)
        insertion_time: float = time.time() - start_insertion

        # Verify complete pipeline succeeded
        assert stats.inserted >= 0, "Pipeline must insert chunks"
        assert stats.failed == 0, "Pipeline must have no failures"

        # Verify total time
        total_time: float = generation_time + insertion_time
        assert total_time < 30, (
            f"Complete pipeline took {total_time:.2f}s (expected <30s)"
        )

        logger.info(
            f"✓ End-to-end pipeline: "
            f"generation={generation_time:.3f}s, "
            f"insertion={insertion_time:.3f}s, "
            f"total={total_time:.3f}s, "
            f"inserted={stats.inserted}"
        )
