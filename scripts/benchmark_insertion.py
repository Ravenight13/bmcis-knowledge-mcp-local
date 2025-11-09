#!/usr/bin/env python3
"""Performance benchmark for ChunkInserter database insertion.

Measures insertion performance with various batch sizes and chunk counts
to optimize database configuration for production workloads.

Usage:
    python scripts/benchmark_insertion.py [--chunks N] [--batch-sizes B1,B2,B3]
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.document_parsing.models import ProcessedChunk
from src.embedding.database import ChunkInserter

logger = StructuredLogger.get_logger(__name__)


def generate_test_chunks(count: int) -> list[ProcessedChunk]:
    """Generate test chunks with realistic embeddings.

    Args:
        count: Number of chunks to generate.

    Returns:
        List of ProcessedChunk objects with 768-dimensional embeddings.
    """
    chunks = []
    for i in range(count):
        # Generate random 768-dimensional embedding
        embedding = np.random.randn(768).tolist()

        chunk = ProcessedChunk(
            chunk_text=f"Test chunk content {i} " * 50,  # ~50 words
            chunk_hash=f"{i:064x}",  # Unique 64-char hash
            context_header=f"benchmark.md > Section {i // 100} > Subsection {i % 100}",
            source_file=f"benchmark/test_{i // 100}.md",
            source_category="benchmark",
            document_date=date(2025, 1, 1),
            chunk_index=i % 100,
            total_chunks=100,
            chunk_token_count=256,
            metadata={"benchmark": True, "batch_id": i // 100},
            embedding=embedding,
        )
        chunks.append(chunk)

    return chunks


def run_benchmark(
    chunk_count: int, batch_size: int, create_index: bool = False
) -> dict:
    """Run insertion benchmark with specified parameters.

    Args:
        chunk_count: Number of chunks to insert.
        batch_size: Batch size for insertion.
        create_index: Whether to create HNSW index after insertion.

    Returns:
        Dictionary with benchmark results.
    """
    logger.info(f"\nBenchmark: {chunk_count} chunks, batch_size={batch_size}")

    # Generate test chunks
    logger.info("Generating test chunks...")
    gen_start = time.time()
    chunks = generate_test_chunks(chunk_count)
    gen_time = time.time() - gen_start
    logger.info(f"Generated {len(chunks)} chunks in {gen_time:.2f}s")

    # Run insertion
    inserter = ChunkInserter(batch_size=batch_size)
    stats = inserter.insert_chunks(chunks, create_index=create_index)

    # Calculate metrics
    total_time = stats.total_time_seconds
    insertion_time = total_time - stats.index_creation_time_seconds
    chunks_per_second = chunk_count / insertion_time if insertion_time > 0 else 0
    ms_per_chunk = (insertion_time * 1000) / chunk_count if chunk_count > 0 else 0

    results = {
        "chunk_count": chunk_count,
        "batch_size": batch_size,
        "inserted": stats.inserted,
        "updated": stats.updated,
        "failed": stats.failed,
        "total_time_seconds": total_time,
        "insertion_time_seconds": insertion_time,
        "index_creation_time_seconds": stats.index_creation_time_seconds,
        "chunks_per_second": chunks_per_second,
        "ms_per_chunk": ms_per_chunk,
        "batch_count": stats.batch_count,
        "average_batch_time_seconds": stats.average_batch_time_seconds,
    }

    logger.info(f"\nResults:")
    logger.info(f"  - Inserted: {stats.inserted}, Updated: {stats.updated}, Failed: {stats.failed}")
    logger.info(f"  - Total time: {total_time:.2f}s")
    logger.info(f"  - Insertion time: {insertion_time:.2f}s")
    logger.info(f"  - Index creation: {stats.index_creation_time_seconds:.2f}s")
    logger.info(f"  - Throughput: {chunks_per_second:.0f} chunks/sec")
    logger.info(f"  - Latency: {ms_per_chunk:.2f} ms/chunk")
    logger.info(f"  - Batches: {stats.batch_count} ({stats.average_batch_time_seconds:.3f}s avg)")

    return results


def cleanup_benchmark_data() -> None:
    """Remove benchmark data from database."""
    logger.info("\nCleaning up benchmark data...")
    try:
        with DatabasePool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge_base WHERE source_category = 'benchmark'")
                deleted = cur.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted} benchmark rows")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)


def main() -> None:
    """Run performance benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark ChunkInserter performance")
    parser.add_argument(
        "--chunks",
        type=int,
        default=100,
        help="Number of chunks to insert (default: 100)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="50,100,200",
        help="Comma-separated batch sizes to test (default: 50,100,200)",
    )
    parser.add_argument(
        "--with-index",
        action="store_true",
        help="Create HNSW index after insertion",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of benchmark data",
    )

    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    logger.info("=" * 80)
    logger.info("ChunkInserter Performance Benchmark")
    logger.info("=" * 80)
    logger.info(f"Chunk count: {args.chunks}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Index creation: {'Yes' if args.with_index else 'No'}")

    try:
        # Initialize database
        DatabasePool.initialize()

        all_results = []

        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            try:
                results = run_benchmark(
                    chunk_count=args.chunks,
                    batch_size=batch_size,
                    create_index=args.with_index and batch_size == batch_sizes[-1],
                )
                all_results.append(results)

                # Clean up between runs (except last)
                if batch_size != batch_sizes[-1] and not args.no_cleanup:
                    cleanup_benchmark_data()

            except Exception as e:
                logger.error(f"Benchmark failed for batch_size={batch_size}: {e}", exc_info=True)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Summary")
        logger.info("=" * 80)
        logger.info(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<15} {'Latency':<12}")
        logger.info("-" * 80)

        for r in all_results:
            logger.info(
                f"{r['batch_size']:<12} "
                f"{r['insertion_time_seconds']:<10.2f} "
                f"{r['chunks_per_second']:<15.0f} "
                f"{r['ms_per_chunk']:<12.2f}"
            )

        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("Recommendations")
        logger.info("=" * 80)

        best = min(all_results, key=lambda r: r["insertion_time_seconds"])
        logger.info(f"Best batch size: {best['batch_size']}")
        logger.info(f"Best throughput: {best['chunks_per_second']:.0f} chunks/sec")

        if args.with_index:
            logger.info(
                f"\nHNSW index creation time: "
                f"{best['index_creation_time_seconds']:.2f}s "
                f"({best['index_creation_time_seconds'] / best['chunk_count'] * 1000:.2f} ms/chunk)"
            )

        # Cleanup
        if not args.no_cleanup:
            cleanup_benchmark_data()

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        DatabasePool.close_all()


if __name__ == "__main__":
    main()
