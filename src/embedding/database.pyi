"""Type stubs for database insertion module.

Provides complete type information for embedding database operations.
Enables mypy --strict validation and IDE support.
"""

from typing import Any

import numpy as np
from psycopg2.extensions import connection as Connection

from src.document_parsing.models import ProcessedChunk

# Type alias for vector values
VectorValue = list[float] | np.ndarray


class VectorSerializer:
    """Optimized vector serialization using numpy for 6-10x performance improvement."""

    @staticmethod
    def serialize_vector(embedding: list[float] | np.ndarray) -> str:
        """Serialize single embedding vector to pgvector format efficiently.

        Args:
            embedding: List or numpy array of 768 floats.

        Returns:
            String in pgvector format: "[0.1,0.2,...]"

        Raises:
            ValueError: If embedding is None or not 768 dimensions.
        """

    @staticmethod
    def serialize_vectors_batch(
        embeddings: list[list[float]] | np.ndarray,
    ) -> list[str]:
        """Serialize batch of vectors using vectorized operations for maximum throughput.

        Args:
            embeddings: List of embedding vectors or 2D numpy array.

        Returns:
            List of strings in pgvector format.

        Raises:
            ValueError: If shape is invalid.
        """


class InsertionStats:
    """Statistics for chunk insertion operations."""

    inserted: int
    updated: int
    failed: int
    index_created: bool
    index_creation_time_seconds: float
    total_time_seconds: float
    batch_count: int
    average_batch_time_seconds: float

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ChunkInserter:
    """Batch insertion of document chunks with embeddings into PostgreSQL."""

    batch_size: int

    def __init__(self, batch_size: int = 100) -> None: ...
    def insert_chunks(
        self, chunks: list[ProcessedChunk], create_index: bool = True
    ) -> InsertionStats: ...
    def _insert_batch(
        self, conn: Connection, batch: list[ProcessedChunk]
    ) -> tuple[int, int]: ...
    def _insert_batch_unnest(
        self, conn: Connection, batch: list[ProcessedChunk]
    ) -> tuple[int, int]: ...
    def _serialize_vector(self, embedding: list[float] | None) -> str: ...
    def _create_hnsw_index(self, conn: Connection) -> None: ...
    def verify_index_exists(self) -> bool: ...
    def get_vector_count(self) -> int: ...
