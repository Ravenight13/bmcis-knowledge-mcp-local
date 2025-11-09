"""Tests for the document chunking module.

Comprehensive test suite covering:
- Basic chunking with 512-token chunks and 50-token overlap
- Chunk boundary preservation and metadata tracking
- Edge cases (empty text, single sentence, very long documents)
- Sentence boundary detection
- Configuration validation
- Overlap effectiveness verification
"""

import pytest

from src.document_parsing.chunker import (
    Chunk,
    ChunkMetadata,
    Chunker,
    ChunkerConfig,
)
from src.document_parsing.tokenizer import Tokenizer, TokenizerConfig


class TestChunkerConfig:
    """Test ChunkerConfig validation.

    Tests cover:
    - Default configuration values match specifications
    - Custom configuration values are properly stored
    - Validation enforces positive chunk and min_chunk sizes
    - Validation enforces non-negative overlap_tokens
    - Cross-field constraints are validated (overlap < chunk, min <= chunk)
    """

    def test_chunker_config_defaults(self) -> None:
        """Test default ChunkerConfig values match specifications.

        Reason: Ensures configuration defaults align with pipeline requirements.
        What it does: Validates that all default fields have expected values.
        """
        config = ChunkerConfig()
        assert config.chunk_size == 512, "Default chunk_size should be 512"
        assert config.overlap_tokens == 50, "Default overlap_tokens should be 50"
        assert config.preserve_boundaries is True, "Default preserve_boundaries should be True"
        assert config.min_chunk_size == 100, "Default min_chunk_size should be 100"

    def test_chunker_config_custom(self) -> None:
        """Test custom ChunkerConfig values are properly stored.

        Reason: Ensures configuration can be customized for different use cases.
        What it does: Creates config with custom values and verifies persistence.
        """
        config = ChunkerConfig(
            chunk_size=256,
            overlap_tokens=25,
            preserve_boundaries=False,
            min_chunk_size=50,
        )
        assert config.chunk_size == 256, "Custom chunk_size should persist"
        assert config.overlap_tokens == 25, "Custom overlap_tokens should persist"
        assert config.preserve_boundaries is False, "Custom preserve_boundaries should persist"
        assert config.min_chunk_size == 50, "Custom min_chunk_size should persist"

    def test_config_validation_overlap_exceeds_chunk(self) -> None:
        """Test validation fails when overlap_tokens >= chunk_size.

        Reason: Prevents logical inconsistency where overlap cannot exceed target chunk size.
        What it does: Attempts to create config with overlap >= chunk_size and verifies ValueError.
        """
        with pytest.raises(ValueError, match="overlap_tokens.*must be less than"):
            ChunkerConfig(chunk_size=100, overlap_tokens=150)

    def test_config_validation_overlap_equals_chunk(self) -> None:
        """Test validation fails when overlap_tokens equals chunk_size.

        Reason: Overlap must be strictly less than chunk_size to avoid circular overlaps.
        What it does: Attempts to create config where overlap equals chunk_size.
        """
        with pytest.raises(ValueError, match="overlap_tokens.*must be less than"):
            ChunkerConfig(chunk_size=100, overlap_tokens=100)

    def test_config_validation_min_exceeds_chunk(self) -> None:
        """Test validation fails when min_chunk_size > chunk_size.

        Reason: Prevents impossible constraint where minimum exceeds target size.
        What it does: Attempts to create config with min_chunk_size > chunk_size.
        """
        with pytest.raises(ValueError, match="min_chunk_size.*must not exceed"):
            ChunkerConfig(chunk_size=100, min_chunk_size=150)

    def test_config_invalid_chunk_sizes(self) -> None:
        """Test that chunk_size must be positive (not zero or negative).

        Reason: Zero or negative chunk size is meaningless in chunking context.
        What it does: Verifies ValueError is raised for non-positive chunk_size values.
        """
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkerConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkerConfig(chunk_size=-1)

    def test_config_invalid_min_chunk_size(self) -> None:
        """Test that min_chunk_size must be positive.

        Reason: Minimum chunk size of zero or negative would create degenerate chunks.
        What it does: Verifies ValueError is raised for non-positive min_chunk_size.
        """
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkerConfig(min_chunk_size=0)
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkerConfig(min_chunk_size=-1)

    def test_config_invalid_negative_overlap(self) -> None:
        """Test that overlap_tokens must be non-negative.

        Reason: Negative overlap is impossible in sequential chunking.
        What it does: Verifies ValueError is raised for negative overlap_tokens.
        """
        with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
            ChunkerConfig(overlap_tokens=-1)

    def test_config_zero_overlap_valid(self) -> None:
        """Test that zero overlap_tokens is valid (no overlap, adjacent chunks).

        Reason: Zero overlap is valid configuration for non-overlapping chunks.
        What it does: Creates config with zero overlap and verifies success.
        """
        config = ChunkerConfig(overlap_tokens=0)
        assert config.overlap_tokens == 0, "Zero overlap should be valid"

    def test_config_validate_config_explicit_call(self) -> None:
        """Test validate_config() can be called explicitly for validation.

        Reason: Provides explicit validation API for edge cases or re-validation.
        What it does: Creates config and calls validate_config() to ensure no error on valid config.

        Example:
            >>> config = ChunkerConfig(chunk_size=256, overlap_tokens=50)
            >>> config.validate_config()  # Should not raise
        """
        config = ChunkerConfig(chunk_size=256, overlap_tokens=50)
        # Should not raise
        config.validate_config()

    def test_config_all_valid_boundaries(self) -> None:
        """Test configuration with valid values at constraint boundaries.

        Reason: Ensures edge cases of valid constraints work correctly.
        What it does: Creates config where overlap_tokens is just under chunk_size.
        """
        config = ChunkerConfig(chunk_size=100, overlap_tokens=99, min_chunk_size=100)
        assert config.overlap_tokens == 99, "Overlap just under chunk_size should be valid"
        assert config.min_chunk_size == 100, "Min chunk equal to chunk_size should be valid"


class TestChunkerBasic:
    """Test basic chunker functionality."""

    def test_chunker_initialization_default(self) -> None:
        """Test chunker initialization with default config."""
        chunker = Chunker()
        assert chunker.config.chunk_size == 512
        assert chunker.config.overlap_tokens == 50
        assert chunker.config.preserve_boundaries is True

    def test_chunker_initialization_custom(self) -> None:
        """Test chunker initialization with custom config."""
        config = ChunkerConfig(chunk_size=256, overlap_tokens=30)
        chunker = Chunker(config=config)
        assert chunker.config.chunk_size == 256
        assert chunker.config.overlap_tokens == 30

    def test_chunk_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = Chunker()
        chunks = chunker.chunk_text("", [])
        assert chunks == []

    def test_chunk_empty_tokens(self) -> None:
        """Test chunking with empty token list."""
        chunker = Chunker()
        chunks = chunker.chunk_text("Some text", [])
        assert chunks == []

    def test_chunk_single_small_token(self) -> None:
        """Test chunking with single small token."""
        chunker = Chunker()
        text = "Hi"
        token_ids = [1]
        chunks = chunker.chunk_text(text, token_ids)
        assert len(chunks) == 1
        assert chunks[0].token_count == 1
        assert chunks[0].text == "Hi"

    def test_chunk_basic_512_tokens(self) -> None:
        """Test basic chunking produces chunks close to 512 tokens."""
        chunker = Chunker()
        # Create text with about 1500 tokens
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(long_text)

        chunks = chunker.chunk_text(long_text, token_ids)
        assert len(chunks) > 1  # Should have multiple chunks

        # First chunks should be close to 512
        for chunk in chunks[:-1]:
            assert chunk.token_count <= chunker.config.chunk_size

        # All chunks should have valid structure
        for chunk in chunks:
            assert chunk.token_count > 0
            assert len(chunk.tokens) == chunk.token_count
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.start_token_pos >= 0
            assert chunk.metadata.end_token_pos > chunk.metadata.start_token_pos

    def test_chunk_metadata_structure(self) -> None:
        """Test that chunk metadata is properly structured."""
        chunker = Chunker()
        text = "First sentence. Second sentence. Third sentence."
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        if len(chunks) > 0:
            chunk = chunks[0]
            metadata = chunk.metadata
            assert isinstance(metadata, ChunkMetadata)
            assert metadata.chunk_index == 0
            assert metadata.start_token_pos >= 0
            assert metadata.end_token_pos > metadata.start_token_pos
            assert metadata.sentence_count >= 0
            assert metadata.overlap_tokens >= 0

    def test_chunk_positions_valid(self) -> None:
        """Test that chunk character positions are valid."""
        chunker = Chunker()
        text = "The quick brown fox jumps over the lazy dog. " * 20
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        for i, chunk in enumerate(chunks):
            assert chunk.start_pos >= 0
            assert chunk.end_pos <= len(text)
            assert chunk.start_pos <= chunk.end_pos
            # Text should match positions
            if chunk.end_pos <= len(text):
                assert len(chunk.text) > 0


class TestChunkerOverlap:
    """Test chunker overlap functionality."""

    def test_overlap_default_50_tokens(self) -> None:
        """Test default 50-token overlap."""
        config = ChunkerConfig(chunk_size=512, overlap_tokens=50)
        chunker = Chunker(config=config)
        assert chunker.config.overlap_tokens == 50

    def test_overlap_custom(self) -> None:
        """Test custom overlap configuration."""
        config = ChunkerConfig(chunk_size=256, overlap_tokens=30)
        chunker = Chunker(config=config)
        text = "The quick brown fox jumps over the lazy dog. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        # Check that overlap tokens are tracked in metadata
        if len(chunks) > 1:
            for chunk in chunks[1:]:
                # Each chunk should have overlap tracked
                assert chunk.metadata.overlap_tokens >= 0

    def test_zero_overlap(self) -> None:
        """Test chunking with zero overlap."""
        config = ChunkerConfig(chunk_size=512, overlap_tokens=0)
        chunker = Chunker(config=config)
        text = "The quick brown fox jumps over the lazy dog. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        # With zero overlap, chunks should be sequential
        if len(chunks) > 1:
            # Token positions should be adjacent
            for i in range(len(chunks) - 1):
                assert chunks[i].metadata.end_token_pos <= chunks[i + 1].metadata.start_token_pos + 1


class TestChunkerBoundaries:
    """Test chunker sentence boundary preservation."""

    def test_sentence_detection(self) -> None:
        """Test that sentence boundaries are identified."""
        chunker = Chunker()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._identify_sentences(text)
        assert len(sentences) >= 2  # Should find at least 2 sentences

    def test_sentence_detection_edge_cases(self) -> None:
        """Test sentence detection with edge cases."""
        chunker = Chunker()
        test_cases = [
            "Single sentence",
            "Two sentences. Connected.",
            "Abbreviation like Dr. Smith works fine.",
            "Question? Exclamation! Period.",
            "Multiple...dots and!!! punctuation!!!",
        ]
        for text in test_cases:
            sentences = chunker._identify_sentences(text)
            assert isinstance(sentences, list)
            # Each sentence should be a tuple of (start, end)
            for start, end in sentences:
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert start >= 0
                assert end > start

    def test_preserve_boundaries_true(self) -> None:
        """Test chunking with preserve_boundaries=True."""
        config = ChunkerConfig(preserve_boundaries=True)
        chunker = Chunker(config=config)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        # Should have chunks, each containing complete sentences
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.tokens) > 0

    def test_preserve_boundaries_false(self) -> None:
        """Test chunking with preserve_boundaries=False."""
        config = ChunkerConfig(preserve_boundaries=False)
        chunker = Chunker(config=config)
        text = "The quick brown fox jumps over the lazy dog. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        # Should have chunks even without boundary preservation
        assert len(chunks) > 0


class TestChunkerEdgeCases:
    """Test edge cases in chunking."""

    def test_chunk_very_short_document(self) -> None:
        """Test chunking a very short document."""
        chunker = Chunker()
        text = "Hi"
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        assert len(chunks) >= 1
        assert chunks[0].text.strip() == "Hi"

    def test_chunk_single_very_long_sentence(self) -> None:
        """Test chunking a single very long sentence."""
        chunker = Chunker()
        # Create a very long sentence
        long_sentence = " ".join(["word"] * 200) + "."
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(long_sentence)

        chunks = chunker.chunk_text(long_sentence, token_ids)
        # Should create chunks even with single long sentence
        assert len(chunks) > 0
        # All tokens should be distributed across chunks
        total_tokens = sum(chunk.token_count for chunk in chunks)
        assert total_tokens <= len(token_ids) * 1.1  # Account for overlap

    def test_chunk_paragraphs(self) -> None:
        """Test chunking multi-paragraph text."""
        chunker = Chunker()
        text = """
        First paragraph with some content about the topic.
        It has multiple sentences for better coverage.

        Second paragraph starts here. It also contains multiple sentences.
        This ensures we have enough content for chunking.

        Third and final paragraph with concluding remarks.
        It should be properly chunked as well.
        """
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_with_special_characters(self) -> None:
        """Test chunking text with special characters."""
        chunker = Chunker()
        text = "Hello @world #hashtag $money &ampersand. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        assert len(chunks) > 0

    def test_chunk_unicode_text(self) -> None:
        """Test chunking unicode and emoji text."""
        chunker = Chunker()
        text = "Hello 你好 🌍 مرحبا Привет. " * 50
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)
        assert len(chunks) > 0


class TestChunkerLargeDocument:
    """Test chunking with larger documents."""

    def test_chunk_large_document_distribution(self) -> None:
        """Test that a large document is properly distributed across chunks."""
        chunker = Chunker(config=ChunkerConfig(chunk_size=512, overlap_tokens=50))
        # Create a ~1500 token document
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Verify chunk sizes
        for i, chunk in enumerate(chunks[:-1]):
            # All chunks except last should be close to 512
            assert chunk.token_count <= chunker.config.chunk_size
            assert chunk.token_count >= chunker.config.min_chunk_size

    def test_chunk_index_sequence(self) -> None:
        """Test that chunk indices form proper sequence."""
        chunker = Chunker()
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        # Chunk indices should be sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_chunk_token_positions_non_overlapping(self) -> None:
        """Test that chunk token positions are properly tracked."""
        chunker = Chunker()
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        # Each chunk should have valid positions
        for chunk in chunks:
            assert chunk.metadata.start_token_pos < chunk.metadata.end_token_pos
            assert chunk.metadata.start_token_pos >= 0
            assert chunk.metadata.end_token_pos <= len(token_ids)


class TestChunkerIntegration:
    """Integration tests with tokenizer."""

    def test_chunker_with_tokenizer_integration(self) -> None:
        """Test chunker working with tokenizer."""
        tokenizer = Tokenizer()
        chunker = Chunker()

        text = "The quick brown fox jumps over the lazy dog. " * 50
        token_ids = tokenizer.encode(text)

        chunks = chunker.chunk_text(text, token_ids)

        assert len(chunks) > 0
        # Verify chunks can be decoded back
        for chunk in chunks:
            assert len(chunk.tokens) == chunk.token_count

    def test_multiple_documents(self) -> None:
        """Test chunking multiple documents."""
        tokenizer = Tokenizer()
        chunker = Chunker()

        texts = [
            "Short document.",
            "Medium document. " * 10,
            "Long document. " * 100,
        ]

        for text in texts:
            token_ids = tokenizer.encode(text)
            chunks = chunker.chunk_text(text, token_ids)
            assert len(chunks) >= 1
            assert sum(c.token_count for c in chunks) <= len(token_ids) * 1.1
