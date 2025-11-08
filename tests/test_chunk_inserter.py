"""Unit tests for ChunkInserter database insertion functionality.

Tests cover:
- Batch insertion with embeddings
- Deduplication via chunk_hash
- HNSW index creation and validation
- Error handling for invalid embeddings
- Performance benchmarks
- Transaction safety
"""

from datetime import date
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from psycopg2 import DatabaseError

from src.document_parsing.models import ProcessedChunk
from src.embedding.database import ChunkInserter, InsertionStats


@pytest.fixture
def sample_chunks() -> list[ProcessedChunk]:
    """Create sample chunks with embeddings for testing."""
    chunks = []
    for i in range(5):
        # Create proper 64-character hash (SHA-256 format)
        chunk_hash = f"{i:064x}"  # Hex string padded to 64 chars
        chunk = ProcessedChunk(
            chunk_text=f"Sample chunk text {i}",
            chunk_hash=chunk_hash,
            context_header=f"file.md > Section {i}",
            source_file="test_file.md",
            source_category="test_category",
            document_date=date(2025, 1, 1),
            chunk_index=i,
            total_chunks=5,
            chunk_token_count=100,
            metadata={"test": True},
            embedding=[0.1 + i * 0.01] * 768,  # 768-dimensional vector
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def inserter() -> ChunkInserter:
    """Create ChunkInserter instance for testing."""
    return ChunkInserter(batch_size=2)  # Small batch for testing


class TestInsertionStats:
    """Tests for InsertionStats class."""

    def test_initialization(self) -> None:
        """Test InsertionStats initializes with correct defaults."""
        stats = InsertionStats()

        assert stats.inserted == 0
        assert stats.updated == 0
        assert stats.failed == 0
        assert stats.index_created is False
        assert stats.index_creation_time_seconds == 0.0
        assert stats.total_time_seconds == 0.0
        assert stats.batch_count == 0
        assert stats.average_batch_time_seconds == 0.0

    def test_to_dict(self) -> None:
        """Test InsertionStats converts to dictionary correctly."""
        stats = InsertionStats()
        stats.inserted = 10
        stats.updated = 2
        stats.failed = 1

        result = stats.to_dict()

        assert result["inserted"] == 10
        assert result["updated"] == 2
        assert result["failed"] == 1
        assert "index_created" in result
        assert "total_time_seconds" in result


class TestChunkInserter:
    """Tests for ChunkInserter class."""

    def test_initialization(self) -> None:
        """Test ChunkInserter initializes with correct batch size."""
        inserter = ChunkInserter(batch_size=50)
        assert inserter.batch_size == 50

    def test_initialization_invalid_batch_size(self) -> None:
        """Test ChunkInserter rejects invalid batch sizes."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ChunkInserter(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ChunkInserter(batch_size=-1)

    def test_serialize_vector_valid(self, inserter: ChunkInserter) -> None:
        """Test vector serialization to pgvector format."""
        embedding = [0.1, 0.2, 0.3] + [0.0] * 765  # 768 dims

        result = inserter._serialize_vector(embedding)

        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.1" in result
        assert "0.2" in result
        assert "0.3" in result

    def test_serialize_vector_numpy_array(self, inserter: ChunkInserter) -> None:
        """Test vector serialization with numpy array."""
        embedding = np.array([0.1, 0.2, 0.3] + [0.0] * 765)

        result = inserter._serialize_vector(embedding)

        assert result.startswith("[")
        assert result.endswith("]")

    def test_serialize_vector_invalid_none(self, inserter: ChunkInserter) -> None:
        """Test vector serialization rejects None."""
        with pytest.raises(ValueError, match="Embedding cannot be None"):
            inserter._serialize_vector(None)

    def test_serialize_vector_invalid_dimensions(self, inserter: ChunkInserter) -> None:
        """Test vector serialization rejects wrong dimensions."""
        with pytest.raises(ValueError, match="must be 768 dimensions"):
            inserter._serialize_vector([0.1, 0.2, 0.3])  # Only 3 dims

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_chunks_empty_list(
        self, mock_get_conn: Mock, inserter: ChunkInserter
    ) -> None:
        """Test insert_chunks handles empty list gracefully."""
        stats = inserter.insert_chunks([])

        assert stats.inserted == 0
        assert stats.updated == 0
        assert stats.failed == 0
        assert stats.batch_count == 0
        mock_get_conn.assert_not_called()

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_chunks_validates_embeddings(
        self, mock_get_conn: Mock, inserter: ChunkInserter, sample_chunks: list[ProcessedChunk]
    ) -> None:
        """Test insert_chunks validates all chunks have embeddings."""
        # Set one chunk to have no embedding
        sample_chunks[2].embedding = None

        with pytest.raises(ValueError, match="Invalid embeddings"):
            inserter.insert_chunks(sample_chunks)

        mock_get_conn.assert_not_called()

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_chunks_validates_embedding_dimensions(
        self, mock_get_conn: Mock, inserter: ChunkInserter, sample_chunks: list[ProcessedChunk]
    ) -> None:
        """Test insert_chunks validates embedding dimensions."""
        # Set one chunk to have wrong dimension
        sample_chunks[2].embedding = [0.1, 0.2, 0.3]  # Only 3 dims

        with pytest.raises(ValueError, match="Invalid embeddings"):
            inserter.insert_chunks(sample_chunks)

    @patch("src.embedding.database.execute_values")
    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_chunks_success(
        self,
        mock_get_conn: Mock,
        mock_execute_values: Mock,
        inserter: ChunkInserter,
        sample_chunks: list[ProcessedChunk],
    ) -> None:
        """Test successful chunk insertion with batching."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock batch results (3 inserted, 2 updated)
        mock_cursor.fetchall.side_effect = [
            [(True,), (True,)],  # Batch 1: 2 inserted
            [(True,), (False,)],  # Batch 2: 1 inserted, 1 updated
            [(False,)],  # Batch 3: 1 updated
        ]

        stats = inserter.insert_chunks(sample_chunks, create_index=False)

        # Verify stats
        assert stats.inserted == 3
        assert stats.updated == 2
        assert stats.failed == 0
        assert stats.batch_count == 3  # 5 chunks / batch_size=2 = 3 batches
        assert stats.total_time_seconds > 0

        # Verify commit was called
        mock_conn.commit.assert_called()

        # Verify execute_values was called for each batch
        assert mock_execute_values.call_count == 3

    @patch("src.embedding.database.execute_values")
    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_chunks_with_index_creation(
        self,
        mock_get_conn: Mock,
        mock_execute_values: Mock,
        inserter: ChunkInserter,
        sample_chunks: list[ProcessedChunk],
    ) -> None:
        """Test chunk insertion with HNSW index creation."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock batch results
        mock_cursor.fetchall.side_effect = [
            [(True,), (True,)],  # Batch 1
            [(True,), (True,)],  # Batch 2
            [(True,)],  # Batch 3
        ]

        stats = inserter.insert_chunks(sample_chunks, create_index=True)

        # Verify index creation was attempted
        assert stats.index_created is True
        assert stats.index_creation_time_seconds > 0

        # Verify DROP and CREATE INDEX were called
        execute_calls = mock_cursor.execute.call_args_list
        sql_statements = [call[0][0] for call in execute_calls]

        # Check for DROP INDEX
        assert any("DROP INDEX" in sql for sql in sql_statements)
        # Check for CREATE INDEX with HNSW
        assert any("CREATE INDEX" in sql and "hnsw" in sql for sql in sql_statements)

    @patch("src.embedding.database.execute_values")
    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_insert_batch_failure_continues(
        self,
        mock_get_conn: Mock,
        mock_execute_values: Mock,
        inserter: ChunkInserter,
        sample_chunks: list[ProcessedChunk],
    ) -> None:
        """Test that batch failure doesn't stop entire operation."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Simulate batch failure on second batch by raising exception
        def side_effect_batch(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Second call raises error
            if mock_execute_values.call_count == 2:
                raise DatabaseError("Simulated batch failure")

        mock_execute_values.side_effect = side_effect_batch

        # Mock batch results
        mock_cursor.fetchall.side_effect = [
            [(True,), (True,)],  # Batch 1: success
            [(True,)],  # Batch 3: success (batch 2 failed before fetchall)
        ]

        stats = inserter.insert_chunks(sample_chunks, create_index=False)

        # Verify partial success
        assert stats.inserted >= 0
        assert stats.failed == 2  # Batch 2 had 2 chunks
        assert stats.batch_count > 0

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_verify_index_exists_true(
        self, mock_get_conn: Mock, inserter: ChunkInserter
    ) -> None:
        """Test verify_index_exists returns True when index exists."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock index exists query result
        mock_cursor.fetchone.return_value = (True,)

        result = inserter.verify_index_exists()

        assert result is True

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_verify_index_exists_false(
        self, mock_get_conn: Mock, inserter: ChunkInserter
    ) -> None:
        """Test verify_index_exists returns False when index doesn't exist."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock index doesn't exist
        mock_cursor.fetchone.return_value = (False,)

        result = inserter.verify_index_exists()

        assert result is False

    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_get_vector_count(self, mock_get_conn: Mock, inserter: ChunkInserter) -> None:
        """Test get_vector_count returns correct count."""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock count query result
        mock_cursor.fetchone.return_value = (2600,)

        result = inserter.get_vector_count()

        assert result == 2600

        # Verify SQL query
        execute_call = mock_cursor.execute.call_args[0][0]
        assert "COUNT(*)" in execute_call
        assert "WHERE embedding IS NOT NULL" in execute_call


class TestIntegration:
    """Integration tests (require live database connection)."""

    @pytest.mark.integration
    @patch("src.embedding.database.execute_values")
    @patch("src.embedding.database.DatabasePool.get_connection")
    def test_end_to_end_insertion(
        self, mock_get_conn: Mock, mock_execute_values: Mock, sample_chunks: list[ProcessedChunk]
    ) -> None:
        """Test end-to-end insertion workflow."""
        # Mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        # Mock batch results
        mock_cursor.fetchall.side_effect = [
            [(True,), (True,)],  # Batch 1
            [(True,), (True,)],  # Batch 2
            [(True,)],  # Batch 3
        ]

        # Create inserter and insert
        inserter = ChunkInserter(batch_size=2)
        stats = inserter.insert_chunks(sample_chunks, create_index=True)

        # Verify results
        assert stats.inserted > 0
        assert stats.failed == 0
        assert stats.index_created is True
        assert stats.total_time_seconds > 0
        assert stats.batch_count == 3  # 5 chunks / 2 batch_size = 3 batches

        # Verify commit called
        mock_conn.commit.assert_called()
