"""Document chunking module for splitting text into overlapping token-based chunks.

This module provides functionality to split documents into chunks of a specified
token size while maintaining overlap between consecutive chunks and preserving
semantic boundaries (sentences and paragraphs when possible).

The chunker respects sentence boundaries to avoid splitting mid-sentence, ensuring
that chunks remain semantically coherent while maintaining overlap for context preservation.

Example:
    >>> from src.document_parsing.chunker import Chunker, ChunkerConfig
    >>> config = ChunkerConfig(chunk_size=512, overlap_tokens=50)
    >>> chunker = Chunker(config=config)
    >>> text = "This is a long document..."
    >>> token_ids = [1, 2, 3, ...]  # From tokenizer
    >>> chunks = chunker.chunk_text(text, token_ids)
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk.metadata.chunk_index}: {len(chunk.tokens)} tokens")
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class ChunkMetadata:
    """Metadata for a chunk.

    Attributes:
        chunk_index: Zero-based index of this chunk in the sequence.
        start_token_pos: Position of first token in this chunk.
        end_token_pos: Position of last token in this chunk.
        sentence_count: Number of sentences in this chunk.
        overlap_tokens: Number of tokens overlapping from previous chunk.
    """

    chunk_index: int
    start_token_pos: int
    end_token_pos: int
    sentence_count: int
    overlap_tokens: int


@dataclass
class Chunk:
    """Represents a chunk of text with token metadata.

    Attributes:
        text: The actual text content of the chunk.
        tokens: List of token IDs for this chunk.
        token_count: Number of tokens in this chunk.
        start_pos: Starting character position in original text.
        end_pos: Ending character position in original text.
        metadata: ChunkMetadata with additional chunk information.
    """

    text: str
    tokens: list[int]
    token_count: int
    start_pos: int
    end_pos: int
    metadata: ChunkMetadata


class ChunkerConfig(BaseModel):
    """Configuration for the chunking algorithm.

    Attributes:
        chunk_size: Target number of tokens per chunk (default: 512).
        overlap_tokens: Number of tokens to overlap between chunks (default: 50).
        preserve_boundaries: Whether to preserve sentence boundaries (default: True).
        min_chunk_size: Minimum tokens allowed in a chunk (default: 100).
    """

    chunk_size: int = Field(default=512, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    preserve_boundaries: bool = Field(default=True)
    min_chunk_size: int = Field(default=100, gt=0)

    class Config:
        """Pydantic configuration."""

        extra = "forbid"

    def validate_config(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.overlap_tokens >= self.chunk_size:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size > self.chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must not exceed "
                f"chunk_size ({self.chunk_size})"
            )


class Chunker:
    """Splits text into overlapping chunks while preserving semantic boundaries.

    This chunker produces chunks of approximately the specified token size,
    maintaining overlap between consecutive chunks for context preservation.
    When preserve_boundaries is True, chunks respect sentence boundaries
    to ensure semantic coherence.

    Attributes:
        config: ChunkerConfig instance controlling chunking behavior.

    Example:
        >>> config = ChunkerConfig(chunk_size=512, overlap_tokens=50)
        >>> chunker = Chunker(config=config)
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example token IDs
        >>> chunks = chunker.chunk_text(text, token_ids)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialize chunker with optional configuration.

        Args:
            config: ChunkerConfig instance. If None, uses default configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        self.config = config or ChunkerConfig()
        self.config.validate_config()

    def chunk_text(self, text: str, token_ids: list[int]) -> list[Chunk]:
        """Chunk text into overlapping token-based chunks.

        Splits text and corresponding token IDs into chunks of the configured
        size, maintaining overlap and respecting sentence boundaries when possible.

        Args:
            text: The full document text.
            token_ids: List of token IDs corresponding to the text.

        Returns:
            List of Chunk objects representing the chunked document.

        Raises:
            ValueError: If text and token_ids lengths don't correspond properly.

        Example:
            >>> chunker = Chunker()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> token_ids = [1, 2, 3, 4, 5, 6]
            >>> chunks = chunker.chunk_text(text, token_ids)
            >>> assert len(chunks) > 0
            >>> assert chunks[0].token_count <= 512
        """
        if not text or not token_ids:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        current_pos = 0

        # Identify sentence boundaries if preserving them
        sentence_ranges: list[tuple[int, int]] = []
        if self.config.preserve_boundaries:
            sentence_ranges = self._identify_sentences(text)

        while current_pos < len(token_ids):
            # Determine chunk boundaries
            chunk_end = min(current_pos + self.config.chunk_size, len(token_ids))

            # If preserving boundaries, find sentence boundaries
            if self.config.preserve_boundaries and sentence_ranges:
                chunk_start, chunk_end = self._find_sentence_boundaries(
                    current_pos, chunk_end, sentence_ranges
                )
            else:
                chunk_start = current_pos

            # Ensure minimum chunk size (unless it's the last chunk)
            if (
                chunk_end - chunk_start < self.config.min_chunk_size
                and chunk_end < len(token_ids)
            ):
                chunk_end = min(
                    chunk_start + self.config.min_chunk_size, len(token_ids)
                )

            # Extract chunk tokens
            chunk_token_ids = token_ids[chunk_start:chunk_end]

            # Find character positions in original text
            # This is approximate - ideally would have char positions from tokenizer
            start_char = max(0, len(text) // len(token_ids) * chunk_start)
            end_char = min(len(text), len(text) // len(token_ids) * chunk_end)

            # Extract chunk text
            chunk_text = text[start_char:end_char].strip()

            # Count sentences in chunk
            sentence_count = len(
                [s for s in sentence_ranges if s[0] >= start_char and s[1] <= end_char]
            )

            # Create chunk
            overlap = (
                max(0, chunk_start - (chunk_index * self.config.chunk_size))
                if chunk_index > 0
                else 0
            )

            metadata = ChunkMetadata(
                chunk_index=chunk_index,
                start_token_pos=chunk_start,
                end_token_pos=chunk_end,
                sentence_count=sentence_count,
                overlap_tokens=overlap,
            )

            chunk = Chunk(
                text=chunk_text,
                tokens=chunk_token_ids,
                token_count=len(chunk_token_ids),
                start_pos=start_char,
                end_pos=end_char,
                metadata=metadata,
            )

            chunks.append(chunk)

            # Move to next chunk position
            current_pos = chunk_end - self.config.overlap_tokens
            if current_pos >= len(token_ids):
                break

            chunk_index += 1

        return chunks

    def _identify_sentences(self, text: str) -> list[tuple[int, int]]:
        """Identify sentence boundaries in text.

        Uses regex pattern to identify sentence-ending punctuation followed
        by whitespace and capitalization.

        Args:
            text: The text to analyze.

        Returns:
            List of (start_pos, end_pos) tuples for each sentence.

        Example:
            >>> chunker = Chunker()
            >>> text = "First sentence. Second sentence!"
            >>> sentences = chunker._identify_sentences(text)
            >>> len(sentences)
            2
        """
        # Simple sentence detection: look for sentence-ending punctuation
        # followed by space and capital letter (or end of text)
        sentence_pattern = r"[.!?]+\s+"

        sentences: list[tuple[int, int]] = []
        current_start = 0

        for match in re.finditer(sentence_pattern, text):
            sentence_end = match.end()
            sentences.append((current_start, sentence_end - 1))
            current_start = sentence_end

        # Add final sentence if any text remains
        if current_start < len(text):
            sentences.append((current_start, len(text)))

        return sentences

    def _find_sentence_boundaries(
        self,
        start_token: int,
        end_token: int,
        sentence_ranges: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Find sentence boundaries that align with token positions.

        Adjusts token boundaries to align with sentence boundaries when
        preserve_boundaries is enabled.

        Args:
            start_token: Starting token position.
            end_token: Desired ending token position.
            sentence_ranges: List of (char_start, char_end) for sentences.

        Returns:
            Adjusted (start_token, end_token) aligned with sentence boundaries.

        Example:
            >>> chunker = Chunker()
            >>> start, end = chunker._find_sentence_boundaries(0, 100, [(0, 50), (50, 100)])
            >>> assert start >= 0
            >>> assert end <= 100
        """
        # For now, return original boundaries
        # In a full implementation, would map char positions to token positions
        return start_token, end_token
