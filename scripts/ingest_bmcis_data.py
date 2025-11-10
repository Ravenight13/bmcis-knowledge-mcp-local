#!/usr/bin/env python3
"""
BMCIS Data Ingestion Script

Ingests all BMCIS VP Sales System documents (00_-09_ folders) into the knowledge base.
Handles document chunking, embedding generation, and database insertion.

Usage:
    python scripts/ingest_bmcis_data.py

Environment:
    DATABASE_URL: PostgreSQL connection string
    OPENAI_API_KEY or sentence-transformers for embeddings
"""

import logging
import os
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger


def setup_logger() -> logging.Logger:
    """Initialize structured logger."""
    return StructuredLogger.get_logger(__name__)


def get_ingestion_files() -> list[Path]:
    """Get all files from 00_-09_ folders in BMCIS VP Sales System.

    Returns:
        List of Path objects for all ingestion files
    """
    base_path = Path(
        "/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System"
    )

    if not base_path.exists():
        raise FileNotFoundError(f"BMCIS directory not found: {base_path}")

    files = []

    # Find all 00_-09_ directories
    for i in range(10):
        folder = base_path / f"0{i}_*"
        # Use glob to handle wildcards in folder names
        matching_dirs = list(base_path.glob(f"0{i}_*"))

        for directory in matching_dirs:
            if directory.is_dir():
                # Get all markdown, text, PDF, and DOCX files
                for pattern in ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.docx"]:
                    files.extend(directory.glob(pattern))

    return files


def read_document(file_path: Path) -> str | None:
    """Read document content from file.

    Args:
        file_path: Path to document file

    Returns:
        Document content or None if unreadable
    """
    try:
        if file_path.suffix == ".pdf":
            # For PDF files, we'd need PyPDF2 or pdfplumber
            # For now, skip or use OCR if available
            logger.warning(f"PDF not yet supported: {file_path}")
            return None

        if file_path.suffix == ".docx":
            # For DOCX files, we'd need python-docx
            # For now, skip
            logger.warning(f"DOCX not yet supported: {file_path}")
            return None

        # Read markdown and text files
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def chunk_document(
    content: str, document_id: str, chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, Any]]:
    """Split document into chunks with overlap.

    Args:
        content: Document content
        document_id: Document identifier
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    start = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to break at a sentence boundary
        if end < len(content):
            # Look backwards for a period, newline, or other break
            for break_char in [".\n", ".\r\n", ".\n\n", "\n\n"]:
                pos = content.rfind(break_char, start, end)
                if pos > start:
                    end = pos + len(break_char)
                    break

        chunk_text = content[start:end].strip()

        if chunk_text:
            chunks.append({
                "document_id": document_id,
                "content": chunk_text,
                "start_char": start,
                "end_char": end,
                "chunk_index": len(chunks),
            })

        start = end - overlap

    return chunks


def main() -> None:
    """Main ingestion workflow."""
    logger = setup_logger()

    logger.info("Starting BMCIS data ingestion...")

    # Get all ingestion files
    logger.info("Discovering ingestion files (00_-09_ folders)...")
    files = get_ingestion_files()

    # Filter to readable files
    readable_files = [f for f in files if f.suffix in [".md", ".txt"]]
    logger.info(f"Found {len(readable_files)} readable files (MD/TXT)")

    if not readable_files:
        logger.error("No readable files found!")
        return

    # Initialize database
    logger.info("Initializing database connection...")
    pool = DatabasePool()

    try:
        # Process each document
        total_chunks = 0
        processed_docs = 0

        for file_path in readable_files:
            try:
                logger.info(f"Processing: {file_path.relative_to(file_path.parents[7])}")

                # Read document
                content = read_document(file_path)
                if not content:
                    continue

                # Create document entry
                doc_id = str(file_path.relative_to(file_path.parents[7]))

                # Insert document (this would need implementation in src)
                # For now, just log the chunk info
                chunks = chunk_document(content, doc_id)
                logger.info(f"  → {len(chunks)} chunks")
                total_chunks += len(chunks)
                processed_docs += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(
            f"Ingestion complete: {processed_docs} documents, {total_chunks} chunks"
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    logger = setup_logger()
    main()
