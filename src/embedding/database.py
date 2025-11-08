"""Database insertion for document chunks with HNSW index management.

Provides batch insertion of document chunks with embeddings into PostgreSQL,
deduplication via chunk_hash, and HNSW index creation for fast similarity search.
Uses connection pooling from Phase 0 infrastructure for reliable database access.

The module handles:
- Batch insertion with configurable batch sizes (default 100 chunks)
- Deduplication via ON CONFLICT on chunk_hash unique constraint
- HNSW index creation with optimized parameters (m=16, ef_construction=64)
- Transaction safety with automatic rollback on errors
- Comprehensive error handling and logging
- Performance monitoring and statistics reporting
"""

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.document_parsing.models import ProcessedChunk

# Module logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Type alias for vector values
VectorValue = list[float] | np.ndarray


class InsertionStats:
    """Statistics for chunk insertion operations.

    Tracks insertion metrics including counts, timing, and index creation
    status for monitoring and debugging batch operations.

    Attributes:
        inserted: Number of chunks inserted (new records).
        updated: Number of chunks updated (duplicates via ON CONFLICT).
        failed: Number of chunks that failed insertion.
        index_created: Whether HNSW index was created successfully.
        index_creation_time_seconds: Time taken to create HNSW index.
        total_time_seconds: Total time for entire operation.
        batch_count: Number of batches processed.
        average_batch_time_seconds: Average time per batch.
    """

    def __init__(self) -> None:
        """Initialize insertion statistics."""
        self.inserted: int = 0
        self.updated: int = 0
        self.failed: int = 0
        self.index_created: bool = False
        self.index_creation_time_seconds: float = 0.0
        self.total_time_seconds: float = 0.0
        self.batch_count: int = 0
        self.average_batch_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging and reporting.

        Returns:
            Dictionary with all statistics fields.
        """
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "failed": self.failed,
            "index_created": self.index_created,
            "index_creation_time_seconds": self.index_creation_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "batch_count": self.batch_count,
            "average_batch_time_seconds": self.average_batch_time_seconds,
        }


class ChunkInserter:
    """Batch insertion of document chunks with embeddings into PostgreSQL.

    Provides methods for efficient batch insertion of ProcessedChunk objects
    with embeddings into the knowledge_base table, with automatic deduplication,
    HNSW index creation, and comprehensive error handling.

    The inserter uses connection pooling from DatabasePool for reliable
    database access and implements batch processing for optimal performance.

    Example:
        >>> inserter = ChunkInserter(batch_size=100)
        >>> chunks = [ProcessedChunk(...), ...]  # with embeddings populated
        >>> stats = inserter.insert_chunks(chunks)
        >>> print(f"Inserted: {stats.inserted}, Updated: {stats.updated}")
        >>> if stats.index_created:
        ...     print(f"Index created in {stats.index_creation_time_seconds}s")
    """

    def __init__(self, batch_size: int = 100) -> None:
        """Initialize ChunkInserter with batch configuration.

        Args:
            batch_size: Number of chunks to insert per batch (default: 100).
                       Recommended range: 50-200 for optimal performance.

        Raises:
            ValueError: If batch_size < 1.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.batch_size = batch_size
        logger.info(f"ChunkInserter initialized with batch_size={batch_size}")

    def insert_chunks(
        self, chunks: list[ProcessedChunk], create_index: bool = True
    ) -> InsertionStats:
        """Insert chunks with embeddings into knowledge_base table.

        Performs batch insertion with deduplication, transaction safety,
        and optional HNSW index creation. Chunks without embeddings are skipped
        with a warning logged.

        The method:
        1. Validates all chunks have embeddings and correct dimensions (768)
        2. Batches chunks for efficient insertion (batch_size chunks per batch)
        3. Uses ON CONFLICT UPDATE for deduplication via chunk_hash
        4. Optionally creates/recreates HNSW index after data load
        5. Returns comprehensive statistics including timing and counts

        Args:
            chunks: List of ProcessedChunk objects with embeddings populated.
            create_index: Whether to create HNSW index after insertion (default: True).
                         Set to False if index already exists and inserting incrementally.

        Returns:
            InsertionStats: Statistics including inserted/updated/failed counts,
                           timing information, and index creation status.

        Raises:
            ValueError: If any chunk has invalid embedding (None, wrong dimension).
            psycopg2.DatabaseError: If database operation fails.
            RuntimeError: If connection pool unavailable.

        Example:
            >>> chunks = [
            ...     ProcessedChunk(
            ...         chunk_text="content",
            ...         chunk_hash="abc123...",
            ...         embedding=[0.1, 0.2, ...],  # 768 floats
            ...         # ... other fields
            ...     ),
            ... ]
            >>> stats = inserter.insert_chunks(chunks)
            >>> print(f"{stats.inserted} chunks inserted")
        """
        stats = InsertionStats()
        start_time = time.time()

        if not chunks:
            logger.warning("No chunks provided for insertion")
            stats.total_time_seconds = time.time() - start_time
            return stats

        # Validate all chunks have embeddings
        invalid_chunks = [
            i for i, chunk in enumerate(chunks) if chunk.embedding is None or len(chunk.embedding) != 768
        ]

        if invalid_chunks:
            error_msg = (
                f"Invalid embeddings in {len(invalid_chunks)} chunks at indices: "
                f"{invalid_chunks[:10]}{'...' if len(invalid_chunks) > 10 else ''}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Starting insertion of {len(chunks)} chunks in batches of {self.batch_size}")

        try:
            with DatabasePool.get_connection() as conn:
                # Process chunks in batches
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i : i + self.batch_size]
                    batch_start = time.time()

                    try:
                        inserted, updated = self._insert_batch(conn, batch)
                        stats.inserted += inserted
                        stats.updated += updated
                        stats.batch_count += 1

                        batch_time = time.time() - batch_start
                        logger.info(
                            f"Batch {stats.batch_count}: inserted={inserted}, "
                            f"updated={updated}, time={batch_time:.2f}s"
                        )

                    except Exception as e:
                        stats.failed += len(batch)
                        logger.error(f"Batch {stats.batch_count + 1} failed: {e}", exc_info=True)
                        # Continue with next batch instead of failing entire operation
                        continue

                # Commit all batches
                conn.commit()
                logger.info(
                    f"Committed {stats.inserted + stats.updated} chunks "
                    f"({stats.inserted} new, {stats.updated} updated)"
                )

                # Create HNSW index if requested
                if create_index:
                    logger.info("Creating HNSW index for similarity search...")
                    index_start = time.time()

                    try:
                        self._create_hnsw_index(conn)
                        stats.index_created = True
                        stats.index_creation_time_seconds = time.time() - index_start

                        logger.info(
                            f"HNSW index created successfully in "
                            f"{stats.index_creation_time_seconds:.2f}s"
                        )

                    except Exception as e:
                        logger.error(f"HNSW index creation failed: {e}", exc_info=True)
                        stats.index_created = False

        except Exception as e:
            logger.error(f"Chunk insertion failed: {e}", exc_info=True)
            raise

        # Calculate final statistics
        stats.total_time_seconds = time.time() - start_time
        if stats.batch_count > 0:
            stats.average_batch_time_seconds = (
                stats.total_time_seconds - stats.index_creation_time_seconds
            ) / stats.batch_count

        logger.info(
            f"Insertion complete: {stats.inserted} inserted, {stats.updated} updated, "
            f"{stats.failed} failed in {stats.total_time_seconds:.2f}s "
            f"(avg {stats.average_batch_time_seconds:.2f}s/batch)"
        )

        return stats

    def _insert_batch(self, conn: Connection, batch: list[ProcessedChunk]) -> tuple[int, int]:
        """Insert a single batch of chunks using execute_values.

        Uses PostgreSQL's ON CONFLICT UPDATE for deduplication via chunk_hash.
        Returns counts of inserted vs updated records by examining the result.

        Args:
            conn: Database connection from pool.
            batch: List of ProcessedChunk objects to insert.

        Returns:
            Tuple of (inserted_count, updated_count).

        Raises:
            psycopg2.DatabaseError: If batch insertion fails.
        """
        # Prepare batch data as tuples matching INSERT statement
        batch_data = [
            (
                chunk.chunk_text,
                chunk.chunk_hash,
                self._serialize_vector(chunk.embedding),  # Convert to pgvector format
                chunk.source_file,
                chunk.source_category,
                chunk.document_date,
                chunk.chunk_index,
                chunk.total_chunks,
                chunk.context_header,
                chunk.chunk_token_count,
                psycopg2.extras.Json(chunk.metadata),  # Convert dict to JSONB
            )
            for chunk in batch
        ]

        # SQL with ON CONFLICT UPDATE for deduplication
        # Note: knowledge_base table has metadata JSONB column (not in original schema)
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
                context_header,
                chunk_token_count,
                metadata
            ) VALUES %s
            ON CONFLICT (chunk_hash) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                source_file = EXCLUDED.source_file,
                source_category = EXCLUDED.source_category,
                document_date = EXCLUDED.document_date,
                chunk_index = EXCLUDED.chunk_index,
                total_chunks = EXCLUDED.total_chunks,
                context_header = EXCLUDED.context_header,
                chunk_token_count = EXCLUDED.chunk_token_count,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING (xmax = 0) AS inserted
        """

        with conn.cursor() as cur:
            # Use execute_values for efficient batch insert
            execute_values(
                cur,
                insert_sql,
                batch_data,
                template=None,
                page_size=len(batch_data),
            )

            # Fetch results to determine inserted vs updated
            results = cur.fetchall()

            # xmax = 0 means INSERT (new row), xmax > 0 means UPDATE (existing row)
            inserted = sum(1 for (is_insert,) in results if is_insert)
            updated = len(results) - inserted

            return inserted, updated

    def _serialize_vector(self, embedding: list[float] | None) -> str:
        """Serialize embedding vector to pgvector format.

        Converts Python list of floats to PostgreSQL vector string format
        for insertion into vector(768) column.

        Args:
            embedding: List of 768 floats representing the embedding vector.

        Returns:
            String in pgvector format: "[0.1,0.2,0.3,...]"

        Raises:
            ValueError: If embedding is None or not 768 dimensions.
        """
        if embedding is None:
            raise ValueError("Embedding cannot be None")

        if len(embedding) != 768:
            raise ValueError(f"Embedding must be 768 dimensions, got {len(embedding)}")

        # pgvector expects format: [0.1,0.2,0.3,...]
        # Convert to numpy for efficient string serialization if needed
        if isinstance(embedding, np.ndarray):
            vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
        else:
            vector_str = "[" + ",".join(str(x) for x in embedding) + "]"

        return vector_str

    def _create_hnsw_index(self, conn: Connection) -> None:
        """Create or recreate HNSW index for similarity search.

        Drops existing index if present and creates new HNSW index with
        optimized parameters for 768-dimensional vectors:
        - m=16: connections per node (good balance of speed/accuracy)
        - ef_construction=64: size of dynamic candidate list during construction

        Args:
            conn: Database connection from pool.

        Raises:
            psycopg2.DatabaseError: If index creation fails.
        """
        with conn.cursor() as cur:
            # Drop existing index if present (idempotent)
            logger.debug("Dropping existing HNSW index if present...")
            cur.execute("DROP INDEX IF EXISTS idx_knowledge_embedding")

            # Create HNSW index with optimized parameters
            logger.debug("Creating HNSW index with m=16, ef_construction=64...")
            cur.execute("""
                CREATE INDEX idx_knowledge_embedding ON knowledge_base
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            conn.commit()

    def verify_index_exists(self) -> bool:
        """Verify that HNSW index exists and is usable.

        Checks PostgreSQL system catalogs to confirm the idx_knowledge_embedding
        index exists on the knowledge_base table.

        Returns:
            True if index exists, False otherwise.

        Raises:
            RuntimeError: If database connection fails.
        """
        try:
            with DatabasePool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename = 'knowledge_base'
                            AND indexname = 'idx_knowledge_embedding'
                        )
                    """)

                    exists = cur.fetchone()[0]
                    logger.info(f"HNSW index exists: {exists}")
                    return exists

        except Exception as e:
            logger.error(f"Failed to verify index existence: {e}", exc_info=True)
            raise RuntimeError(f"Index verification failed: {e}") from e

    def get_vector_count(self) -> int:
        """Get count of vectors in knowledge_base table.

        Returns:
            Number of rows with non-NULL embeddings.

        Raises:
            RuntimeError: If database query fails.
        """
        try:
            with DatabasePool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL")
                    count = cur.fetchone()[0]
                    logger.info(f"Vector count in knowledge_base: {count}")
                    return count

        except Exception as e:
            logger.error(f"Failed to get vector count: {e}", exc_info=True)
            raise RuntimeError(f"Vector count query failed: {e}") from e
