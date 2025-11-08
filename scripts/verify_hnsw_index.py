#!/usr/bin/env python3
"""Verification script for HNSW index creation and similarity search.

This script:
1. Connects to the database
2. Verifies the HNSW index exists
3. Tests similarity search performance
4. Reports index statistics and query times

Usage:
    python scripts/verify_hnsw_index.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger

logger = StructuredLogger.get_logger(__name__)


def verify_index_exists() -> bool:
    """Verify HNSW index exists on knowledge_base table."""
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = 'knowledge_base'
                AND indexname = 'idx_knowledge_embedding'
            """)

            result = cur.fetchone()
            if result:
                indexname, indexdef = result
                logger.info(f"HNSW index found: {indexname}")
                logger.info(f"Index definition: {indexdef}")
                return True
            else:
                logger.warning("HNSW index not found")
                return False


def get_index_stats() -> dict:
    """Get statistics about the HNSW index."""
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cur:
            # Get index size
            cur.execute("""
                SELECT
                    pg_size_pretty(pg_relation_size('idx_knowledge_embedding')) AS index_size,
                    pg_relation_size('idx_knowledge_embedding') AS index_size_bytes
            """)
            index_size, index_size_bytes = cur.fetchone()

            # Get table stats
            cur.execute("""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(embedding) AS rows_with_embeddings,
                    pg_size_pretty(pg_total_relation_size('knowledge_base')) AS table_size
                FROM knowledge_base
            """)
            total_rows, rows_with_embeddings, table_size = cur.fetchone()

            # Get index usage stats
            cur.execute("""
                SELECT
                    idx_scan AS times_used,
                    idx_tup_read AS tuples_read,
                    idx_tup_fetch AS tuples_fetched
                FROM pg_stat_user_indexes
                WHERE indexname = 'idx_knowledge_embedding'
            """)
            usage_stats = cur.fetchone()

            stats = {
                "index_size": index_size,
                "index_size_bytes": index_size_bytes,
                "total_rows": total_rows,
                "rows_with_embeddings": rows_with_embeddings,
                "table_size": table_size,
            }

            if usage_stats:
                stats["times_used"] = usage_stats[0]
                stats["tuples_read"] = usage_stats[1]
                stats["tuples_fetched"] = usage_stats[2]

            return stats


def test_similarity_search() -> tuple[float, int]:
    """Test similarity search performance with random query vector.

    Returns:
        Tuple of (query_time_ms, result_count)
    """
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cur:
            # Get a sample embedding from the database to use as query
            cur.execute("""
                SELECT embedding
                FROM knowledge_base
                WHERE embedding IS NOT NULL
                LIMIT 1
            """)

            result = cur.fetchone()
            if not result:
                logger.warning("No embeddings found in database")
                return 0.0, 0

            # Use sample embedding as query vector
            query_vector = result[0]

            # Test similarity search with timing
            start_time = time.time()

            cur.execute(
                """
                SELECT
                    id,
                    chunk_text,
                    source_file,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_base
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 10
            """,
                (query_vector, query_vector),
            )

            results = cur.fetchall()
            query_time_ms = (time.time() - start_time) * 1000

            return query_time_ms, len(results)


def main() -> None:
    """Run HNSW index verification."""
    logger.info("=" * 80)
    logger.info("HNSW Index Verification Script")
    logger.info("=" * 80)

    try:
        # Initialize database connection pool
        DatabasePool.initialize()

        # Verify index exists
        logger.info("\n[1/3] Verifying HNSW index exists...")
        index_exists = verify_index_exists()

        if not index_exists:
            logger.error("HNSW index does not exist. Please run insert_chunks() first.")
            sys.exit(1)

        # Get index statistics
        logger.info("\n[2/3] Getting index statistics...")
        stats = get_index_stats()

        logger.info("\nIndex Statistics:")
        logger.info(f"  - Index size: {stats['index_size']} ({stats['index_size_bytes']} bytes)")
        logger.info(f"  - Table size: {stats['table_size']}")
        logger.info(f"  - Total rows: {stats['total_rows']}")
        logger.info(f"  - Rows with embeddings: {stats['rows_with_embeddings']}")

        if "times_used" in stats:
            logger.info(f"  - Index scans: {stats['times_used']}")
            logger.info(f"  - Tuples read: {stats['tuples_read']}")
            logger.info(f"  - Tuples fetched: {stats['tuples_fetched']}")

        # Test similarity search
        logger.info("\n[3/3] Testing similarity search performance...")
        query_time_ms, result_count = test_similarity_search()

        logger.info(f"\nSimilarity Search Results:")
        logger.info(f"  - Query time: {query_time_ms:.2f} ms")
        logger.info(f"  - Results returned: {result_count}")

        if query_time_ms < 500:
            logger.info(f"  - Performance: EXCELLENT (target: <500ms)")
        elif query_time_ms < 1000:
            logger.info(f"  - Performance: GOOD (target: <500ms)")
        else:
            logger.warning(f"  - Performance: NEEDS OPTIMIZATION (target: <500ms)")

        logger.info("\n" + "=" * 80)
        logger.info("Verification complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        DatabasePool.close_all()


if __name__ == "__main__":
    main()
