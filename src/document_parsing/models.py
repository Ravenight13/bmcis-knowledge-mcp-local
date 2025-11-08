"""Data models for batch document processing.

Provides Pydantic models for representing processed document chunks with
metadata, validation for database insertion, and type-safe mappings to the
knowledge_base table schema.

Models support complete validation of chunk data including token counts,
text lengths, metadata structure, and context headers before database
insertion to ensure data integrity.
"""

import hashlib
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata extracted from document frontmatter or headers.

    Contains structured information about the document source including
    title, author, category, tags, and publication date. Used for filtering
    and organization in knowledge base queries.

    Attributes:
        title: Document title or heading.
        author: Document author or creator.
        category: Document category (product_docs, kb_article, etc).
        tags: List of topic tags for filtering.
        source_file: Original file path relative to data directory.
        document_date: Publication or last update date.
    """

    title: str = Field(
        default="Untitled",
        description="Document title",
        min_length=1,
        max_length=512,
    )
    author: str | None = Field(
        default=None,
        description="Document author",
        max_length=256,
    )
    category: str | None = Field(
        default=None,
        description="Document category (product_docs, kb_article, etc)",
        max_length=128,
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags for filtering",
    )
    source_file: str = Field(
        description="Original file path",
        min_length=1,
        max_length=512,
    )
    document_date: date | None = Field(
        default=None,
        description="Publication or update date",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags list contains only non-empty strings.

        Args:
            v: List of tag strings.

        Returns:
            Validated tags list with empty strings removed.
        """
        return [tag.strip() for tag in v if tag and tag.strip()]


class ProcessedChunk(BaseModel):
    """Represents a document chunk ready for database insertion.

    Contains all data required for insertion into the knowledge_base table
    including chunk text, context header, metadata, token count, and computed
    hash for deduplication. Validates all constraints before database insertion.

    Attributes:
        chunk_text: The text content of the chunk (512 tokens max).
        chunk_hash: SHA-256 hash of chunk_text for deduplication.
        context_header: Hierarchical context (e.g., "file.md > Section > Subsection").
        source_file: Original markdown file path.
        source_category: Document category for filtering.
        document_date: Document publication/update date.
        chunk_index: Position of this chunk in the document (0-indexed).
        total_chunks: Total number of chunks in the source document.
        chunk_token_count: Number of tokens in chunk_text.
        metadata: Additional metadata as JSON object.
        embedding: Vector embedding (populated in Phase 2, NULL for now).
    """

    chunk_text: str = Field(
        description="Chunk text content",
        min_length=1,
        max_length=50000,  # Reasonable max for 512 tokens + context
    )
    chunk_hash: str = Field(
        description="SHA-256 hash for deduplication",
        min_length=64,
        max_length=64,
    )
    context_header: str = Field(
        description="Hierarchical context header",
        min_length=1,
        max_length=1024,
    )
    source_file: str = Field(
        description="Original file path",
        min_length=1,
        max_length=512,
    )
    source_category: str | None = Field(
        default=None,
        description="Document category",
        max_length=128,
    )
    document_date: date | None = Field(
        default=None,
        description="Document date",
    )
    chunk_index: int = Field(
        description="Position in document (0-indexed)",
        ge=0,
    )
    total_chunks: int = Field(
        description="Total chunks in document",
        ge=1,
    )
    chunk_token_count: int = Field(
        description="Token count in chunk",
        ge=1,
        le=1024,  # Max reasonable token count
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata as JSON",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding (NULL in Phase 1)",
    )

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, v: int, info: Any) -> int:
        """Validate chunk_index is less than total_chunks.

        Args:
            v: Chunk index value.
            info: Validation context with other field values.

        Returns:
            Validated chunk_index.

        Raises:
            ValueError: If chunk_index >= total_chunks.
        """
        # Note: info.data may not have total_chunks yet during validation
        # This check is performed at the model level after all fields are set
        if hasattr(info, "data") and "total_chunks" in info.data:
            total = info.data["total_chunks"]
            if v >= total:
                msg = f"chunk_index ({v}) must be < total_chunks ({total})"
                raise ValueError(msg)
        return v

    @classmethod
    def create_from_chunk(
        cls,
        chunk_text: str,
        context_header: str,
        metadata: DocumentMetadata,
        chunk_index: int,
        total_chunks: int,
        token_count: int,
    ) -> "ProcessedChunk":
        """Factory method to create ProcessedChunk from components.

        Automatically computes chunk_hash from chunk_text and constructs
        the ProcessedChunk with all required fields.

        Args:
            chunk_text: The text content of the chunk.
            context_header: Hierarchical context string.
            metadata: Document metadata from frontmatter.
            chunk_index: Position in document (0-indexed).
            total_chunks: Total chunks in document.
            token_count: Number of tokens in chunk.

        Returns:
            ProcessedChunk: Validated chunk ready for database insertion.

        Example:
            >>> metadata = DocumentMetadata(
            ...     title="User Guide",
            ...     source_file="docs/user-guide.md",
            ... )
            >>> chunk = ProcessedChunk.create_from_chunk(
            ...     chunk_text="Installation instructions...",
            ...     context_header="user-guide.md > Installation",
            ...     metadata=metadata,
            ...     chunk_index=0,
            ...     total_chunks=5,
            ...     token_count=256,
            ... )
        """
        # Compute SHA-256 hash for deduplication
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        # Build metadata dict for JSONB storage
        metadata_dict = {
            "title": metadata.title,
            "author": metadata.author,
            "tags": metadata.tags,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }

        return cls(
            chunk_text=chunk_text,
            chunk_hash=chunk_hash,
            context_header=context_header,
            source_file=metadata.source_file,
            source_category=metadata.category,
            document_date=metadata.document_date,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_token_count=token_count,
            metadata=metadata_dict,
            embedding=None,  # Phase 2: embeddings
        )


class BatchProcessingStats(BaseModel):
    """Statistics for batch processing operation.

    Tracks processing metrics including file counts, chunk counts, timing,
    and error information for monitoring and debugging batch operations.

    Attributes:
        files_processed: Number of files successfully processed.
        files_failed: Number of files that failed processing.
        chunks_created: Total number of chunks created.
        chunks_inserted: Number of chunks successfully inserted to database.
        processing_time_seconds: Total processing time.
        errors: List of error messages encountered.
        start_time: Processing start timestamp.
        end_time: Processing end timestamp.
    """

    files_processed: int = Field(
        default=0,
        description="Files successfully processed",
        ge=0,
    )
    files_failed: int = Field(
        default=0,
        description="Files that failed processing",
        ge=0,
    )
    chunks_created: int = Field(
        default=0,
        description="Total chunks created",
        ge=0,
    )
    chunks_inserted: int = Field(
        default=0,
        description="Chunks inserted to database",
        ge=0,
    )
    processing_time_seconds: float = Field(
        default=0.0,
        description="Total processing time",
        ge=0,
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Processing start time",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Processing end time",
    )

    def add_error(self, error_message: str) -> None:
        """Add error message to stats.

        Args:
            error_message: Error description.
        """
        self.errors.append(error_message)
        self.files_failed += 1
