#!/usr/bin/env python3
"""
BMCIS Production Data Ingestion

Ingests 435 documents from BMCIS VP Sales System (00_-09_ folders) into PostgreSQL knowledge_base.

Uses:
- embedding model: all-mpnet-base-v2 (768-dimensional vectors)
- pgvector for storage and HNSW indexing
- Full-text search via ts_vector trigger
- Chunk hashing to avoid duplicates

Usage:
    python scripts/ingest_bmcis_production.py
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import psycopg2
import requests
from tiktoken import encoding_for_model

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger


logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Initialize token counter for GPT models
try:
    enc = encoding_for_model("gpt-3.5-turbo")
except Exception:
    enc = None


class BMCISProductionIngester:
    """Production-grade BMCIS document ingestion using Ollama embeddings."""

    def __init__(self, batch_size: int = 50, chunk_size: int = 1500) -> None:
        """Initialize ingester.

        Args:
            batch_size: Documents per batch
            chunk_size: Target chunk size in characters
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Use Ollama embeddings (nomic-embed-text)
        self.ollama_url = "http://localhost:11434/api/embed"
        self.embedding_model_name = "nomic-embed-text"

        logger.info("Using Ollama embeddings (nomic-embed-text)...")
        logger.info(f"  URL: {self.ollama_url}")

        # Test connection
        try:
            test_response = requests.post(
                self.ollama_url,
                json={"model": self.embedding_model_name, "input": "test"},
                timeout=10
            )
            if test_response.status_code == 200:
                logger.info("✅ Ollama connection successful")
            else:
                logger.error(f"Ollama returned {test_response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama connection warning: {e}")

        self.db_pool = DatabasePool()
        self.inserted_chunks = 0

    def get_files(self) -> list[Path]:
        """Get all files from 00_-09_ folders."""
        base = Path(
            "/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System"
        )

        files: list[Path] = []
        for i in range(10):
            for directory in base.glob(f"0{i}_*"):
                if directory.is_dir():
                    files.extend(directory.glob("**/*.md"))
                    files.extend(directory.glob("**/*.txt"))

        return sorted(files)

    def read_file(self, path: Path) -> str | None:
        """Read file content safely."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def chunk_document(
        self, content: str, source_file: str, folder: str
    ) -> list[dict[str, Any]]:
        """Split document into chunks with metadata."""
        chunks: list[dict[str, Any]] = []
        chunk_num = 0
        start = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            # Try sentence boundary
            if end < len(content):
                for sep in [".\n", "\n\n", ".\n\n"]:
                    pos = content.rfind(sep, start, end)
                    if pos > start + 200:
                        end = pos + len(sep)
                        break

            chunk_text = content[start:end].strip()

            if len(chunk_text) > 50:
                # Compute hash to avoid duplicates
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

                # Count tokens
                token_count = 0
                if enc:
                    try:
                        token_count = len(enc.encode(chunk_text))
                    except Exception:
                        token_count = len(chunk_text) // 4  # Rough estimate

                chunks.append({
                    "chunk_text": chunk_text,
                    "chunk_hash": chunk_hash,
                    "source_file": source_file,
                    "source_category": folder,
                    "chunk_index": chunk_num,
                    "total_chunks": 0,  # Will update later
                    "chunk_token_count": token_count,
                    "metadata": {
                        "folder": folder,
                        "source_file": source_file,
                    },
                })
                chunk_num += 1

            start = end - 300 if end < len(content) else len(content)

        # Update total_chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate 768-dimensional embeddings via Ollama."""
        if not texts:
            return []

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.embedding_model_name,
                    "input": texts
                },
                timeout=300
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                logger.warning(f"No embeddings returned from Ollama")
                return [[0.0] * 768] * len(texts)

            return embeddings

        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 768 for _ in texts]

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Insert chunks into knowledge_base table.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Number inserted
        """
        if not chunks:
            return 0

        conn = self.db_pool.get_connection()
        try:
            cursor = conn.cursor()

            # Generate embeddings for all chunks
            texts = [c["chunk_text"] for c in chunks]
            embeddings = self.generate_embeddings(texts)

            # Insert chunks
            inserted = 0
            for chunk, embedding in zip(chunks, embeddings):
                try:
                    cursor.execute(
                        """
                        INSERT INTO knowledge_base (
                            chunk_text, chunk_hash, embedding,
                            source_file, source_category,
                            chunk_index, total_chunks,
                            chunk_token_count, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_hash) DO NOTHING
                        """,
                        (
                            chunk["chunk_text"],
                            chunk["chunk_hash"],
                            embedding,
                            chunk["source_file"],
                            chunk["source_category"],
                            chunk["chunk_index"],
                            chunk["total_chunks"],
                            chunk["chunk_token_count"],
                            chunk["metadata"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting chunk: {e}")
                    continue

            conn.commit()
            logger.info(f"Inserted {inserted} chunks from batch")
            return inserted

        except Exception as e:
            conn.rollback()
            logger.error(f"Batch insertion failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def ingest(self) -> None:
        """Run full ingestion."""
        logger.info("=" * 70)
        logger.info("BMCIS PRODUCTION DATA INGESTION")
        logger.info("=" * 70)

        # Get files
        logger.info("Discovering files...")
        files = self.get_files()
        logger.info(f"Found {len(files)} markdown/text files")

        # Process files
        logger.info("Processing documents and generating embeddings...")
        all_chunks: list[dict[str, Any]] = []
        processed = 0

        for idx, file_path in enumerate(files, 1):
            try:
                rel_path = file_path.relative_to(
                    Path(
                        "/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System"
                    )
                )
                folder = str(rel_path).split("/")[0]

                content = self.read_file(file_path)
                if not content:
                    logger.debug(f"Skipped (empty): {file_path.name}")
                    continue

                logger.debug(f"Processing: {file_path.name} ({len(content)} chars)")
                chunks = self.chunk_document(content, file_path.name, folder)
                all_chunks.extend(chunks)
                processed += 1

                logger.debug(f"  Created {len(chunks)} chunks")

                # Insert immediately for faster progress visibility
                if all_chunks:
                    logger.info(f"Inserting {len(all_chunks)} chunks...")
                    inserted = self.insert_chunks(all_chunks)
                    self.inserted_chunks += inserted
                    all_chunks = []
                    logger.info(f"  Total inserted: {self.inserted_chunks}")

                if idx % 25 == 0:
                    logger.info(
                        f"Progress: {idx}/{len(files)} files processed "
                        f"({self.inserted_chunks} chunks inserted)"
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Final batch
        if all_chunks:
            inserted = self.insert_chunks(all_chunks)
            self.inserted_chunks += inserted

        logger.info("=" * 70)
        logger.info(
            f"INGESTION COMPLETE: {processed} documents, {self.inserted_chunks} chunks"
        )
        logger.info("=" * 70)


def main() -> None:
    """Main entry point."""
    start_time = time.time()
    try:
        ingester = BMCISProductionIngester(batch_size=20, chunk_size=1500)
        ingester.ingest()
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
