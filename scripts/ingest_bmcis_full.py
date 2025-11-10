#!/usr/bin/env python3
"""
BMCIS Data Ingestion - Full Production Script

Ingests 436 documents from BMCIS VP Sales System (00_-09_ folders) into PostgreSQL with pgvector.

Features:
- Recursive directory traversal (00_-09_ only)
- Intelligent document chunking with overlap
- Batch embedding generation (via sentence-transformers)
- Transactional database insertion
- Progress tracking and error handling
- Index building for vector and BM25 search

Usage:
    python scripts/ingest_bmcis_full.py
    python scripts/ingest_bmcis_full.py --dry-run  # Show plan without inserting
    python scripts/ingest_bmcis_full.py --skip-embeddings  # Insert without computing vectors

Environment:
    DATABASE_URL: PostgreSQL connection (default: postgresql://localhost/bmcis_knowledge_dev)
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger


logger: logging.Logger = StructuredLogger.get_logger(__name__)


class BMCISDataIngester:
    """Handles BMCIS document ingestion into PostgreSQL knowledge base."""

    def __init__(
        self,
        dry_run: bool = False,
        skip_embeddings: bool = False,
        batch_size: int = 50,
    ) -> None:
        """Initialize ingester.

        Args:
            dry_run: If True, show plan but don't insert
            skip_embeddings: If True, insert without computing vectors
            batch_size: Number of documents per batch
        """
        self.dry_run = dry_run
        self.skip_embeddings = skip_embeddings
        self.batch_size = batch_size

        self.db_pool = DatabasePool()
        self.embedding_model = None

        if not skip_embeddings:
            logger.info("Loading embedding model (sentence-transformers)...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_ingestion_files(self) -> list[Path]:
        """Get all files from 00_-09_ folders.

        Returns:
            List of Path objects sorted by folder
        """
        base_path = Path(
            "/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System"
        )

        if not base_path.exists():
            raise FileNotFoundError(f"BMCIS directory not found: {base_path}")

        files: list[Path] = []

        # Iterate through 00_-09_ directories
        for i in range(10):
            # Use glob to handle wildcard folder names
            for directory in base_path.glob(f"0{i}_*"):
                if directory.is_dir():
                    # Get all markdown and text files
                    files.extend(directory.glob("**/*.md"))
                    files.extend(directory.glob("**/*.txt"))

        # Sort by path for consistent ordering
        return sorted(files)

    def read_document(self, file_path: Path) -> str | None:
        """Read document content.

        Args:
            file_path: Path to document

        Returns:
            Document content or None if unreadable
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

    def chunk_document(
        self,
        content: str,
        doc_id: str,
        source_file: str = "",
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[dict[str, Any]]:
        """Split document into overlapping chunks.

        Args:
            content: Document content
            doc_id: Document identifier
            source_file: Original source filename
            chunk_size: Target chunk size (characters)
            overlap: Overlap between chunks (characters)

        Returns:
            List of chunk dictionaries
        """
        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_num = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at sentence boundaries
            if end < len(content):
                for separator in [".\n", ".\r\n", "\n\n", ".\n\n"]:
                    pos = content.rfind(separator, start, end)
                    if pos > start + 100:  # Minimum chunk size
                        end = pos + len(separator)
                        break

            chunk_text = content[start:end].strip()

            if len(chunk_text) > 50:  # Skip very small chunks
                chunks.append({
                    "document_id": doc_id,
                    "content": chunk_text,
                    "chunk_number": chunk_num,
                    "source_file": source_file,
                })
                chunk_num += 1

            start = end - overlap if end < len(content) else len(content)

        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")

        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def insert_documents(self, documents: list[dict[str, Any]]) -> int:
        """Insert documents and chunks into database.

        Args:
            documents: List of document dictionaries with chunks

        Returns:
            Number of chunks inserted
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(documents)} documents")
            return 0

        total_chunks = 0
        conn = self.db_pool.get_connection()

        try:
            cursor = conn.cursor()

            for doc in documents:
                doc_id = doc["id"]
                source_file = doc["source_file"]
                folder = doc["folder"]

                # Insert document
                cursor.execute(
                    """
                    INSERT INTO knowledge_base (title, source, metadata, created_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (source) DO NOTHING
                    RETURNING id
                    """,
                    (doc_id, source_file, {"folder": folder}),
                )

                result = cursor.fetchone()
                if not result:
                    # Document already exists
                    logger.debug(f"Document already exists: {doc_id}")
                    continue

                kb_id = result[0]

                # Insert chunks
                for chunk in doc["chunks"]:
                    # Generate embedding if needed
                    embedding = None
                    if not self.skip_embeddings and self.embedding_model:
                        embeddings = self.generate_embeddings([chunk["content"]])
                        embedding = embeddings[0]

                    cursor.execute(
                        """
                        INSERT INTO knowledge_base (
                            knowledge_base_id, content, chunk_number,
                            source_file, embedding, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        """,
                        (kb_id, chunk["content"], chunk["chunk_number"],
                         chunk["source_file"], embedding),
                    )

                    total_chunks += 1

            conn.commit()
            logger.info(f"Inserted {total_chunks} chunks successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Insertion failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

        return total_chunks

    def ingest(self) -> None:
        """Run full ingestion workflow."""
        logger.info("=" * 60)
        logger.info("BMCIS Data Ingestion Starting")
        logger.info("=" * 60)

        # Discover files
        logger.info("Discovering files from 00_-09_ folders...")
        files = self.get_ingestion_files()
        logger.info(f"Found {len(files)} markdown/text files")

        if not files:
            logger.error("No files found to ingest!")
            return

        # Process files
        logger.info("Processing documents...")
        all_documents: list[dict[str, Any]] = []
        total_chunks = 0

        for idx, file_path in enumerate(files, 1):
            try:
                # Get relative path and folder
                try:
                    rel_path = file_path.relative_to(
                        Path(
                            "/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System"
                        )
                    )
                    folder = str(rel_path).split("/")[0]
                except ValueError:
                    folder = "unknown"

                # Read content
                content = self.read_document(file_path)
                if not content:
                    continue

                # Chunk document
                chunks = self.chunk_document(
                    content,
                    file_path.stem,
                    source_file=file_path.name,
                    chunk_size=1000,
                    overlap=200,
                )

                if chunks:
                    all_documents.append({
                        "id": file_path.stem,
                        "source_file": file_path.name,
                        "folder": folder,
                        "chunks": chunks,
                    })
                    total_chunks += len(chunks)

                if idx % 50 == 0:
                    logger.info(
                        f"  {idx}/{len(files)} files processed "
                        f"({total_chunks} chunks so far)"
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(
            f"Processing complete: {len(all_documents)} documents, {total_chunks} chunks"
        )

        # Insert into database
        if not self.dry_run:
            logger.info("Inserting documents into database...")
            inserted = self.insert_documents(all_documents)
            logger.info(f"Successfully inserted {inserted} chunks")

            # Build indexes
            logger.info("Building indexes...")
            self._build_indexes()

        logger.info("=" * 60)
        logger.info("Ingestion Complete!")
        logger.info("=" * 60)

    def _build_indexes(self) -> None:
        """Build vector and BM25 indexes."""
        if self.dry_run:
            logger.info("[DRY RUN] Would build indexes")
            return

        conn = self.db_pool.get_connection()
        try:
            cursor = conn.cursor()

            # Build HNSW index for vector search
            logger.info("Building HNSW index...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
                ON knowledge_base USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)

            # Build GIN index for text search
            logger.info("Building GIN index...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_gin
                ON knowledge_base USING gin (to_tsvector('english', content))
            """)

            conn.commit()
            logger.info("Indexes built successfully")

        except Exception as e:
            logger.error(f"Index building failed: {e}")
        finally:
            cursor.close()
            conn.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest BMCIS VP Sales System documents into knowledge base"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without inserting data",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Insert documents without computing embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )

    args = parser.parse_args()

    # Create ingester
    ingester = BMCISDataIngester(
        dry_run=args.dry_run,
        skip_embeddings=args.skip_embeddings,
        batch_size=args.batch_size,
    )

    # Run ingestion
    start_time = time.time()
    try:
        ingester.ingest()
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
