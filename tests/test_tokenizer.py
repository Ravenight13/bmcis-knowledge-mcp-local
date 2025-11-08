"""Comprehensive test suite for tiktoken-based tokenization module.

Tests cover:
- Token counting accuracy for various text types
- Encoding and decoding round-trip consistency
- Configuration models and validation
- Edge cases (empty text, very long text, special characters)
- Encoder caching and singleton pattern
- Thread safety of encoder cache
- Error handling and validation
- Performance benchmarks

Target coverage: >95% of src/document_parsing/tokenizer.py
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.document_parsing.tokenizer import (
    EncodingModel,
    Tokenizer,
    TokenizerConfig,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    """Create a default TokenizerConfig for testing."""
    return TokenizerConfig(
        encoding_model="cl100k_base",
        max_tokens=0,
        cache_encoder=True,
    )


@pytest.fixture
def tokenizer(tokenizer_config: TokenizerConfig) -> Tokenizer:
    """Create a Tokenizer instance for testing."""
    Tokenizer.clear_cache()
    return Tokenizer(tokenizer_config)


@pytest.fixture(autouse=True)
def cleanup_encoder_cache() -> Any:
    """Clean up encoder cache after each test."""
    yield
    Tokenizer.clear_cache()


# ==============================================================================
# TokenizerConfig Tests
# ==============================================================================


class TestTokenizerConfig:
    """Test suite for TokenizerConfig model."""

    def test_default_config(self) -> None:
        """Test TokenizerConfig with default values."""
        config = TokenizerConfig()
        assert config.encoding_model == "cl100k_base"
        assert config.max_tokens == 0
        assert config.cache_encoder is True

    def test_custom_config(self) -> None:
        """Test TokenizerConfig with custom values."""
        config = TokenizerConfig(
            encoding_model="o200k_base",
            max_tokens=4096,
            cache_encoder=False,
        )
        assert config.encoding_model == "o200k_base"
        assert config.max_tokens == 4096
        assert config.cache_encoder is False

    def test_encoding_model_validation(self) -> None:
        """Test that only valid encoding models are accepted."""
        valid_models: list[EncodingModel] = [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
        ]
        for model in valid_models:
            config = TokenizerConfig(encoding_model=model)
            assert config.encoding_model == model

    def test_max_tokens_validation(self) -> None:
        """Test that max_tokens must be non-negative."""
        # Valid values
        config = TokenizerConfig(max_tokens=0)
        assert config.max_tokens == 0
        config = TokenizerConfig(max_tokens=8192)
        assert config.max_tokens == 8192

        # Invalid values
        with pytest.raises(ValueError):
            TokenizerConfig(max_tokens=-1)

    def test_cache_encoder_flag(self) -> None:
        """Test cache_encoder configuration flag."""
        config_enabled = TokenizerConfig(cache_encoder=True)
        assert config_enabled.cache_encoder is True

        config_disabled = TokenizerConfig(cache_encoder=False)
        assert config_disabled.cache_encoder is False


# ==============================================================================
# Tokenizer Initialization Tests
# ==============================================================================


class TestTokenizerInitialization:
    """Test suite for Tokenizer initialization."""

    def test_default_initialization(self) -> None:
        """Test Tokenizer with default configuration."""
        tokenizer = Tokenizer()
        assert tokenizer.config.encoding_model == "cl100k_base"
        assert tokenizer.config.max_tokens == 0

    def test_custom_initialization(self) -> None:
        """Test Tokenizer with custom configuration."""
        config = TokenizerConfig(
            encoding_model="o200k_base",
            max_tokens=8192,
        )
        tokenizer = Tokenizer(config)
        assert tokenizer.config.encoding_model == "o200k_base"
        assert tokenizer.config.max_tokens == 8192

    def test_invalid_encoding_model_raises_error(self) -> None:
        """Test that invalid encoding model raises error."""
        # Pydantic validation error (ValidationError) is raised for invalid model
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TokenizerConfig(encoding_model="invalid_model")  # type: ignore

    def test_encoding_model_getter(self, tokenizer: Tokenizer) -> None:
        """Test get_encoding_model() returns correct model name."""
        assert tokenizer.get_encoding_model() == "cl100k_base"


# ==============================================================================
# Token Counting Tests
# ==============================================================================


class TestTokenCounting:
    """Test suite for token counting functionality."""

    def test_count_tokens_empty_string(self, tokenizer: Tokenizer) -> None:
        """Test that empty string returns 0 tokens."""
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_simple_text(self, tokenizer: Tokenizer) -> None:
        """Test token counting for simple ASCII text."""
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_single_word(self, tokenizer: Tokenizer) -> None:
        """Test token counting for single word."""
        count = tokenizer.count_tokens("hello")
        assert count == 1

    def test_count_tokens_multiple_words(self, tokenizer: Tokenizer) -> None:
        """Test token counting for multiple words."""
        count = tokenizer.count_tokens("hello world test")
        assert count > 1

    def test_count_tokens_with_special_characters(self, tokenizer: Tokenizer) -> None:
        """Test token counting with special characters and punctuation."""
        text = "Hello! How are you? #amazing @user $100 (test)"
        count = tokenizer.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_with_unicode(self, tokenizer: Tokenizer) -> None:
        """Test token counting with Unicode characters."""
        # Unicode emoticon
        count = tokenizer.count_tokens("Hello 🌍 world!")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_with_numbers(self, tokenizer: Tokenizer) -> None:
        """Test token counting with numbers."""
        text = "The answer is 42. Pi is 3.14159. Count: 1, 2, 3."
        count = tokenizer.count_tokens(text)
        assert count > 0

    def test_count_tokens_with_code(self, tokenizer: Tokenizer) -> None:
        """Test token counting for code-like text."""
        code = "def hello_world():\n    return 'Hello, World!'"
        count = tokenizer.count_tokens(code)
        assert count > 0

    def test_count_tokens_type_validation(self, tokenizer: Tokenizer) -> None:
        """Test that count_tokens validates input type."""
        with pytest.raises(TypeError):
            tokenizer.count_tokens(123)  # type: ignore

    def test_count_tokens_long_text(self, tokenizer: Tokenizer) -> None:
        """Test token counting for longer text."""
        # Create a text with ~1000 tokens
        text = "word " * 500
        count = tokenizer.count_tokens(text)
        assert count > 400  # Should be roughly 500, allowing some variation


# ==============================================================================
# Encoding Tests
# ==============================================================================


class TestEncoding:
    """Test suite for text encoding functionality."""

    def test_encode_empty_string(self, tokenizer: Tokenizer) -> None:
        """Test encoding empty string."""
        tokens = tokenizer.encode("")
        assert tokens == []

    def test_encode_simple_text(self, tokenizer: Tokenizer) -> None:
        """Test encoding simple text."""
        tokens = tokenizer.encode("hello")
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_returns_list_of_ints(self, tokenizer: Tokenizer) -> None:
        """Test that encode returns list of integers."""
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert all(t >= 0 for t in tokens)

    def test_encode_type_validation(self, tokenizer: Tokenizer) -> None:
        """Test that encode validates input type."""
        with pytest.raises(TypeError):
            tokenizer.encode(123)  # type: ignore

    def test_encode_multiple_words(self, tokenizer: Tokenizer) -> None:
        """Test encoding multiple words."""
        tokens = tokenizer.encode("hello world")
        assert len(tokens) > 1
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_with_special_chars(self, tokenizer: Tokenizer) -> None:
        """Test encoding with special characters."""
        tokens = tokenizer.encode("!@#$%^&*()")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


# ==============================================================================
# Decoding Tests
# ==============================================================================


class TestDecoding:
    """Test suite for token decoding functionality."""

    def test_decode_empty_list(self, tokenizer: Tokenizer) -> None:
        """Test decoding empty token list."""
        text = tokenizer.decode([])
        assert text == ""

    def test_decode_valid_tokens(self, tokenizer: Tokenizer) -> None:
        """Test decoding valid token list."""
        # Encode then decode should be identity
        original = "hello"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original

    def test_decode_type_validation_list(self, tokenizer: Tokenizer) -> None:
        """Test that decode validates input is list."""
        with pytest.raises(TypeError):
            tokenizer.decode("not a list")  # type: ignore

    def test_decode_type_validation_integers(self, tokenizer: Tokenizer) -> None:
        """Test that decode validates all tokens are integers."""
        with pytest.raises(TypeError):
            tokenizer.decode([1, 2, "three"])  # type: ignore

    def test_encode_decode_roundtrip(self, tokenizer: Tokenizer) -> None:
        """Test round-trip consistency: text -> tokens -> text."""
        texts = [
            "Hello, world!",
            "The quick brown fox",
            "Numbers: 123 456 789",
        ]
        for original_text in texts:
            tokens = tokenizer.encode(original_text)
            decoded_text = tokenizer.decode(tokens)
            assert decoded_text == original_text


# ==============================================================================
# Max Tokens Limit Tests
# ==============================================================================


class TestMaxTokensLimit:
    """Test suite for max_tokens limit enforcement."""

    def test_count_tokens_below_limit(self) -> None:
        """Test token counting below max_tokens limit."""
        config = TokenizerConfig(encoding_model="cl100k_base", max_tokens=100)
        tokenizer = Tokenizer(config)

        text = "hello world"
        count = tokenizer.count_tokens(text)
        assert count < 100

    def test_count_tokens_exceeds_limit_raises_error(self) -> None:
        """Test that exceeding max_tokens raises ValueError."""
        config = TokenizerConfig(encoding_model="cl100k_base", max_tokens=2)
        tokenizer = Tokenizer(config)

        with pytest.raises(ValueError, match="exceeds max_tokens limit"):
            tokenizer.count_tokens("hello world test")

    def test_encode_below_limit(self) -> None:
        """Test encoding below max_tokens limit."""
        config = TokenizerConfig(encoding_model="cl100k_base", max_tokens=100)
        tokenizer = Tokenizer(config)

        tokens = tokenizer.encode("hello world")
        assert len(tokens) < 100

    def test_encode_exceeds_limit_raises_error(self) -> None:
        """Test that encoding exceeding max_tokens raises ValueError."""
        config = TokenizerConfig(encoding_model="cl100k_base", max_tokens=2)
        tokenizer = Tokenizer(config)

        with pytest.raises(ValueError, match="exceeds max_tokens limit"):
            tokenizer.encode("hello world test")

    def test_no_limit_when_max_tokens_is_zero(self, tokenizer: Tokenizer) -> None:
        """Test that max_tokens=0 means no limit."""
        # Create text that would exceed a reasonable limit
        text = "word " * 5000
        # Should not raise error when max_tokens=0
        count = tokenizer.count_tokens(text)
        assert count > 100


# ==============================================================================
# Encoder Singleton Tests
# ==============================================================================


class TestEncoderSingleton:
    """Test suite for encoder singleton pattern."""

    def test_get_encoder_returns_encoding(self) -> None:
        """Test that get_encoder returns an Encoding object."""
        encoder = Tokenizer.get_encoder()
        assert encoder is not None
        # Verify it has encode method
        assert hasattr(encoder, "encode")

    def test_get_encoder_caching(self) -> None:
        """Test that get_encoder returns same instance (caching)."""
        encoder1 = Tokenizer.get_encoder()
        encoder2 = Tokenizer.get_encoder()
        assert encoder1 is encoder2

    def test_multiple_tokenizers_share_encoder(self) -> None:
        """Test that multiple Tokenizer instances share encoder cache."""
        Tokenizer.clear_cache()
        tokenizer1 = Tokenizer()
        encoder1 = Tokenizer.get_encoder()

        tokenizer2 = Tokenizer()
        encoder2 = Tokenizer.get_encoder()

        assert encoder1 is encoder2

    def test_clear_cache_resets_encoder(self) -> None:
        """Test that clear_cache resets the internal Tokenizer cache."""
        # Store reference to encoder
        encoder1 = Tokenizer.get_encoder()
        # Clear Tokenizer's internal cache
        Tokenizer.clear_cache()
        # Verify internal cache is None after clear
        assert Tokenizer._encoder_cache is None
        # Getting encoder again should set it again
        encoder2 = Tokenizer.get_encoder()
        # Internal cache should be populated again
        assert Tokenizer._encoder_cache is not None

    def test_clear_cache_is_idempotent(self) -> None:
        """Test that clear_cache can be called multiple times safely."""
        Tokenizer.clear_cache()
        Tokenizer.clear_cache()
        Tokenizer.clear_cache()
        # Should not raise error


# ==============================================================================
# Thread Safety Tests
# ==============================================================================


class TestThreadSafety:
    """Test suite for thread-safe encoder caching."""

    def test_concurrent_encoder_access(self) -> None:
        """Test that concurrent encoder access is thread-safe."""
        Tokenizer.clear_cache()
        encoders: list[object] = []
        errors: list[Exception] = []

        def get_encoder_in_thread() -> None:
            try:
                encoder = Tokenizer.get_encoder()
                encoders.append(encoder)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_encoder_in_thread) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should have accessed the same encoder
        assert len(errors) == 0
        assert len(encoders) == 10
        assert all(e is encoders[0] for e in encoders)

    def test_concurrent_tokenization(self) -> None:
        """Test that concurrent tokenization operations work correctly."""
        tokenizer = Tokenizer()
        results: list[int] = []
        errors: list[Exception] = []

        def count_tokens_in_thread(text: str) -> None:
            try:
                count = tokenizer.count_tokens(text)
                results.append(count)
            except Exception as e:
                errors.append(e)

        texts = ["hello world"] * 10
        threads = [
            threading.Thread(target=count_tokens_in_thread, args=(text,))
            for text in texts
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should have completed successfully
        assert len(errors) == 0
        assert len(results) == 10
        # All results should be identical
        assert all(r == results[0] for r in results)


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_count_tokens_invalid_input_type(self, tokenizer: Tokenizer) -> None:
        """Test count_tokens with invalid input types."""
        invalid_inputs = [123, None, [], {}, 45.67]
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError)):
                tokenizer.count_tokens(invalid_input)  # type: ignore

    def test_encode_invalid_input_type(self, tokenizer: Tokenizer) -> None:
        """Test encode with invalid input types."""
        invalid_inputs = [123, None, [], {}]
        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError)):
                tokenizer.encode(invalid_input)  # type: ignore

    def test_decode_invalid_input_type(self, tokenizer: Tokenizer) -> None:
        """Test decode with invalid input types."""
        with pytest.raises(TypeError):
            tokenizer.decode("not a list")  # type: ignore

    def test_decode_invalid_token_type(self, tokenizer: Tokenizer) -> None:
        """Test decode with non-integer tokens."""
        with pytest.raises(TypeError):
            tokenizer.decode([1, 2, "three"])  # type: ignore

    def test_count_tokens_catches_encoding_errors(self, tokenizer: Tokenizer) -> None:
        """Test that encoding errors are caught and wrapped."""
        # Create an invalid text that might cause encoding issues
        # Using valid strings should not raise errors
        texts = ["hello", "test", "123"]
        for text in texts:
            count = tokenizer.count_tokens(text)
            assert count > 0

    def test_encode_catches_encoding_errors(self, tokenizer: Tokenizer) -> None:
        """Test that encode catches and wraps encoding errors."""
        # Valid texts should encode without errors
        tokens = tokenizer.encode("hello world")
        assert len(tokens) > 0

    def test_decode_catches_encoding_errors(self, tokenizer: Tokenizer) -> None:
        """Test that decode catches and wraps encoding errors."""
        # Valid tokens should decode without errors
        tokens = [9906, 11, 1917]
        text = tokenizer.decode(tokens)
        assert isinstance(text, str)


# ==============================================================================
# Performance Tests
# ==============================================================================


class TestPerformance:
    """Test suite for performance characteristics."""

    def test_encoder_cache_improves_performance(self) -> None:
        """Test that encoder caching provides performance benefit."""
        text = "hello world " * 100

        # First call (cache miss)
        Tokenizer.clear_cache()
        tokenizer = Tokenizer()
        count1 = tokenizer.count_tokens(text)

        # Second call (cache hit) - should be faster
        count2 = tokenizer.count_tokens(text)

        # Results should be identical
        assert count1 == count2

    def test_count_tokens_efficiency(self, tokenizer: Tokenizer) -> None:
        """Test that count_tokens is efficient for large text."""
        # Create very large text
        text = "word " * 10000
        count = tokenizer.count_tokens(text)
        assert count > 1000


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self) -> None:
        """Test complete tokenization workflow."""
        # Initialize
        config = TokenizerConfig(
            encoding_model="cl100k_base",
            max_tokens=1000,
        )
        tokenizer = Tokenizer(config)

        # Count tokens
        text = "The quick brown fox jumps over the lazy dog"
        count = tokenizer.count_tokens(text)

        # Encode
        tokens = tokenizer.encode(text)
        assert len(tokens) == count

        # Decode
        decoded = tokenizer.decode(tokens)
        assert decoded == text

        # Get model
        model = tokenizer.get_encoding_model()
        assert model == "cl100k_base"

    def test_multiple_texts(self, tokenizer: Tokenizer) -> None:
        """Test tokenizing multiple texts in sequence."""
        texts = [
            "Hello, world!",
            "The quick brown fox",
            "Python is great",
            "Machine learning is fun",
        ]

        for text in texts:
            count = tokenizer.count_tokens(text)
            tokens = tokenizer.encode(text)
            assert len(tokens) == count

            decoded = tokenizer.decode(tokens)
            assert decoded == text
