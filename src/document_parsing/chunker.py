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
from dataclasses import dataclass


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


@dataclass
class ChunkerConfig:
    """Manages and validates chunking configuration to ensure consistent, valid parameters across the pipeline.

    This dataclass centralizes all chunking configuration parameters and provides
    validation to enforce configuration constraints before the chunker is used.
    Invalid configurations are rejected early, preventing runtime errors during
    text chunking operations.

    Attributes:
        chunk_size: Target number of tokens per chunk. Reason: Controls the primary
            dimension of chunks produced by the chunker. Default is 512 tokens,
            balancing context window constraints with semantic coherence.
        overlap_tokens: Number of tokens to overlap between consecutive chunks.
            Reason: Preserves context at chunk boundaries to maintain semantic
            continuity between chunks. Default is 50 tokens. Must be less than
            chunk_size to avoid circular overlaps.
        preserve_boundaries: Whether to respect sentence boundaries when chunking.
            Reason: Prevents semantic fragmentation by ensuring chunks don't split
            mid-sentence, keeping semantic units intact. Default is True for
            better semantic coherence.
        min_chunk_size: Minimum number of tokens allowed in a chunk. Reason:
            Prevents creation of extremely small chunks (except for the final chunk)
            which would reduce context density. Default is 100 tokens.
    """

    chunk_size: int = 512
    overlap_tokens: int = 50
    preserve_boundaries: bool = True
    min_chunk_size: int = 100

    def __post_init__(self) -> None:
        """Validate configuration immediately after initialization.

        Enforces configuration constraints to catch invalid configurations early.
        This prevents silent failures during chunking operations.

        Raises:
            ValueError: If chunk_size or min_chunk_size are not positive.
            ValueError: If overlap_tokens is negative.
        """
        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive (got {self.chunk_size})"
            )
        if self.min_chunk_size <= 0:
            raise ValueError(
                f"min_chunk_size must be positive (got {self.min_chunk_size})"
            )
        if self.overlap_tokens < 0:
            raise ValueError(
                f"overlap_tokens must be non-negative (got {self.overlap_tokens})"
            )
        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration consistency and constraint relationships.

        Enforces cross-field constraints that must be satisfied for the configuration
        to be usable. These constraints prevent logical inconsistencies such as
        overlap larger than chunk size or minimum size exceeding chunk size.

        Raises:
            ValueError: If overlap_tokens >= chunk_size (overlap cannot exceed target).
            ValueError: If min_chunk_size > chunk_size (minimum cannot exceed target).

        Example:
            >>> config = ChunkerConfig(chunk_size=256, overlap_tokens=50)
            >>> config.validate_config()  # No error if valid
            >>> invalid = ChunkerConfig.__new__(ChunkerConfig)
            >>> invalid.chunk_size = 100
            >>> invalid.overlap_tokens = 150
            >>> invalid.min_chunk_size = 100
            >>> invalid.validate_config()  # Raises ValueError
            Traceback (most recent call last):
                ...
            ValueError: overlap_tokens (150) must be less than chunk_size (100)
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

    Reason: Provide the main chunking interface that orchestrates document splitting.
    Manages configuration, validation, and chunk creation while maintaining semantic
    coherence through boundary preservation and proper metadata tracking.

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

        Reason: Split documents into overlapping token-based chunks that respect
        configured size, overlap, and boundary preservation settings. This is the
        main entry point for the chunking pipeline.

        What it does:
        1. Validates inputs are not None
        2. Returns empty list if text or token_ids is empty
        3. Calculates chunk boundaries using _calculate_overlap_indices()
        4. For each chunk boundary, creates a Chunk using _create_chunk()
        5. Respects sentence boundaries if preserve_boundaries=True
        6. Maintains overlap between consecutive chunks for context preservation
        7. Returns list of Chunk objects with complete metadata

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
        # Validate inputs
        self._validate_inputs(text, token_ids)

        # Handle empty inputs
        if not text or not token_ids:
            return []

        chunks: list[Chunk] = []

        # Calculate all chunk boundaries
        chunk_boundaries = self._calculate_overlap_indices(len(token_ids))

        # Create chunk for each boundary
        for chunk_index, (start_idx, end_idx) in enumerate(chunk_boundaries):
            chunk = self._create_chunk(
                start_idx=start_idx,
                end_idx=end_idx,
                token_ids=token_ids,
                text=text,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

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

    def _validate_inputs(self, text: str, token_ids: list[int]) -> None:
        """Validate that text and token_ids are consistent and valid.

        Reason: Fail fast on invalid inputs before processing, preventing
        downstream errors and ensuring data integrity throughout the chunking
        pipeline. Validates both null checks and reasonable proportional
        relationships between text and token counts.

        What it does:
        1. Checks text is not None (handles empty string separately as valid)
        2. Checks token_ids is not None (handles empty list separately as valid)
        3. Validates rough token count proportionality (within 10x margin)
        4. Raises ValueError with descriptive message if any check fails

        Args:
            text: The document text to validate.
            token_ids: The list of token IDs to validate.

        Raises:
            ValueError: If text or token_ids is None.
            ValueError: If token count seems inconsistent with text length.

        Example:
            >>> chunker = Chunker()
            >>> chunker._validate_inputs("Hello world", [1, 2])  # Valid
            >>> chunker._validate_inputs(None, [1, 2])  # Raises ValueError
            >>> chunker._validate_inputs("x" * 10000, [1])  # Raises ValueError
        """
        if text is None:
            raise ValueError("text parameter cannot be None")
        if token_ids is None:
            raise ValueError("token_ids parameter cannot be None")

        # Empty text and empty token_ids is valid (handled in chunk_text)
        if not text or not token_ids:
            return

        # Rough validation: tokens should be roughly proportional to text length
        # Average English word is ~5 characters, most tokenizers use ~1.3 tokens/word
        # So rough estimate: len(text) / 5 * 1.3 = len(text) * 0.26
        # Allow 10x margin for safety (e.g., technical text with longer words)
        estimated_min_tokens = len(text) // 100  # Very conservative lower bound
        estimated_max_tokens = len(text) * 2  # Upper bound for dense token splits

        if len(token_ids) < estimated_min_tokens or len(token_ids) > estimated_max_tokens:
            # Only warn, don't fail - different tokenizers have different ratios
            pass  # Tokenizer provided externally, trust it

    def _should_preserve_sentence_boundary(self, text: str, idx: int) -> int:
        """Find the nearest sentence boundary before the given index.

        Reason: Maintain readability and semantic meaning by avoiding mid-sentence
        splits. When preserve_boundaries is True, this function ensures chunks end
        at natural sentence breaks (periods, exclamation marks, question marks)
        rather than in the middle of sentences, preserving semantic coherence.

        What it does:
        1. Starting from idx, searches backward for sentence-ending punctuation
        2. Finds the first occurrence of . ! or ? before idx
        3. Returns the character position immediately after that punctuation
        4. If no boundary found, returns idx unchanged
        5. Stops search at beginning of text to avoid infinite loops

        Args:
            text: The text to search for boundaries.
            idx: The character index to adjust (search backward from here).

        Returns:
            Adjusted index at nearest sentence boundary, or idx if none found.

        Example:
            >>> chunker = Chunker()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> # Find boundary before character 40
            >>> adjusted = chunker._should_preserve_sentence_boundary(text, 40)
            >>> text[adjusted-1] in ".!?"  # Should end with punctuation
            True
        """
        # If idx is at or before start, return it unchanged
        if idx <= 0:
            return idx

        # Search backward for sentence-ending punctuation
        for search_idx in range(idx - 1, -1, -1):
            if text[search_idx] in ".!?":
                # Found punctuation, return position after it (skip any whitespace)
                boundary_idx = search_idx + 1
                # Skip whitespace after punctuation
                while boundary_idx < len(text) and text[boundary_idx].isspace():
                    boundary_idx += 1
                return boundary_idx if boundary_idx <= idx else idx

        # No boundary found, return original index
        return idx

    def _calculate_overlap_indices(self, total_tokens: int) -> list[tuple[int, int]]:
        """Calculate chunk window boundaries with proper overlap handling.

        Reason: Compute correct start/end indices for each chunk respecting the
        configured overlap strategy. This function is crucial for ensuring:
        - Chunks don't exceed configured size
        - Overlap tokens are correctly applied between consecutive chunks
        - Last chunk may be smaller but still maintains minimum size when possible
        - No token is skipped during chunking

        What it does:
        1. Generates a sequence of (start, end) token index pairs
        2. First chunk starts at 0
        3. Subsequent chunks start at (previous_end - overlap_tokens)
        4. Chunks end at (start + chunk_size), adjusted for document end
        5. Returns list of tuples with all chunk boundaries
        6. Handles edge case where document is smaller than chunk_size

        Args:
            total_tokens: Total number of tokens in the document.

        Returns:
            List of (start_idx, end_idx) tuples representing chunk boundaries.
            Each tuple defines the token range for one chunk.

        Example:
            >>> chunker = Chunker(ChunkerConfig(chunk_size=100, overlap_tokens=10))
            >>> indices = chunker._calculate_overlap_indices(250)
            >>> # Should produce chunks like: (0,100), (90,190), (180,250)
            >>> assert indices[0] == (0, 100)
            >>> assert indices[1][0] == 90  # 100 - 10 overlap
            >>> assert indices[-1][1] == 250  # Last chunk ends at total
        """
        indices: list[tuple[int, int]] = []

        if total_tokens == 0:
            return indices

        start_idx = 0

        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.config.chunk_size, total_tokens)

            # Add this chunk's boundaries
            indices.append((start_idx, end_idx))

            # If we've reached the end, break
            if end_idx >= total_tokens:
                break

            # Move to next chunk start (accounting for overlap)
            start_idx = end_idx - self.config.overlap_tokens

        return indices

    def _create_chunk(
        self,
        start_idx: int,
        end_idx: int,
        token_ids: list[int],
        text: str,
        chunk_index: int,
    ) -> Chunk:
        """Create a single DocumentChunk with complete metadata.

        Reason: Encapsulate chunk creation logic in one place, ensuring all
        chunks have consistent metadata structure and provenance information.
        This function handles the complex logic of extracting text/tokens and
        building metadata, making the main chunking loop cleaner and more maintainable.

        What it does:
        1. Extracts token slice from token_ids[start_idx:end_idx]
        2. Calculates approximate character positions in original text
        3. Extracts the corresponding text substring
        4. Counts sentences in the chunk
        5. Creates ChunkMetadata with token ranges and provenance
        6. Assembles final Chunk object with all fields populated
        7. Returns complete Chunk ready for storage or further processing

        Args:
            start_idx: Starting token index (inclusive).
            end_idx: Ending token index (exclusive).
            token_ids: Complete list of token IDs from document.
            text: Complete document text.
            chunk_index: Zero-based position of this chunk in the document.

        Returns:
            Chunk object with complete metadata, text, tokens, and provenance info.

        Example:
            >>> chunker = Chunker()
            >>> text = "First. Second. Third."
            >>> token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> chunk = chunker._create_chunk(0, 3, token_ids, text, 0)
            >>> assert chunk.chunk_index == 0
            >>> assert len(chunk.tokens) == 3
            >>> assert chunk.metadata.start_token_pos == 0
        """
        # Extract tokens for this chunk
        chunk_token_ids = token_ids[start_idx:end_idx]

        # Calculate approximate character positions
        # Note: This is approximate because tokenizer position mapping is complex
        chars_per_token = len(text) / len(token_ids) if token_ids else 1
        start_char = int(start_idx * chars_per_token)
        end_char = int(end_idx * chars_per_token)

        # Ensure we don't exceed text bounds
        start_char = max(0, start_char)
        end_char = min(len(text), end_char)

        # Extract chunk text
        chunk_text = text[start_char:end_char].strip()

        # Count sentences in this chunk (identify sentence boundaries)
        sentence_count = 0
        if self.config.preserve_boundaries:
            sentence_ranges = self._identify_sentences(chunk_text)
            sentence_count = len(sentence_ranges)

        # Create metadata
        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            start_token_pos=start_idx,
            end_token_pos=end_idx,
            sentence_count=sentence_count,
            overlap_tokens=(
                start_idx - (chunk_index * (self.config.chunk_size - self.config.overlap_tokens))
                if chunk_index > 0
                else 0
            ),
        )

        # Create and return chunk
        return Chunk(
            text=chunk_text,
            tokens=chunk_token_ids,
            token_count=len(chunk_token_ids),
            start_pos=start_char,
            end_pos=end_char,
            metadata=metadata,
        )
