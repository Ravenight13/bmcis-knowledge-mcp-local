"""Type stubs for tiktoken-based tokenization module.

Provides type definitions for token counting and encoding functionality
using OpenAI's tiktoken library with support for multiple encoding models.
"""

from typing import ClassVar, Literal

from pydantic_settings import BaseSettings
from tiktoken.core import Encoding

# Type aliases for encoding models
EncodingModel = Literal["cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base"]

class TokenizerConfig(BaseSettings):
    """Configuration for tokenizer behavior.

    Attributes:
        encoding_model: The tiktoken encoding model to use (default: cl100k_base).
        max_tokens: Maximum tokens allowed in a single text (0 = unlimited).
        cache_encoder: Whether to cache encoder instance in memory.
    """

    encoding_model: EncodingModel
    max_tokens: int
    cache_encoder: bool

class Tokenizer:
    """Token counting and encoding for documents using OpenAI's tiktoken.

    Implements singleton pattern for encoder instance with thread-safe caching.
    Supports multiple encoding models commonly used by OpenAI's language models.

    Attributes:
        config: TokenizerConfig instance for this tokenizer.
        _encoder_cache: Cached tiktoken encoder instance (class variable).
    """

    _encoder_cache: ClassVar[Encoding | None]

    config: TokenizerConfig

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Initialize tokenizer with optional configuration.

        Args:
            config: TokenizerConfig instance. If None, uses defaults.

        Raises:
            ValueError: If encoding_model is invalid.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured encoding model.

        Accurately counts tokens that will be used by the target model,
        handling special tokens and encoding edge cases.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.

        Raises:
            ValueError: If text exceeds max_tokens and limit is enforced.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using configured encoding model.

        Args:
            text: Text to encode into tokens.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If text exceeds max_tokens and limit is enforced.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back into text.

        Args:
            tokens: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        ...

    def get_encoding_model(self) -> str:
        """Get the encoding model name currently in use.

        Returns:
            Name of the encoding model (e.g., 'cl100k_base').
        """
        ...

    @classmethod
    def get_encoder(cls) -> Encoding:
        """Get or create the cached tiktoken encoder instance.

        Implements lazy-loading singleton pattern for encoder efficiency.

        Returns:
            tiktoken encoder instance.
        """
        ...

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached encoder instance.

        Used primarily for testing and cleanup. After calling this,
        the next token operation will load a fresh encoder.
        """
        ...
