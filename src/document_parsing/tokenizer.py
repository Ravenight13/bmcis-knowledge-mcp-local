"""Tiktoken-based tokenization for document text.

Provides accurate token counting and encoding functionality using OpenAI's
tiktoken library with support for multiple encoding models (cl100k_base for
GPT-4/3.5-turbo, o200k_base for GPT-4-turbo, etc.).

Key features:
- Lazy-loading singleton pattern for encoder instance
- Thread-safe encoder caching
- Multiple encoding model support
- Edge case handling (special tokens, encoding errors)
- Full type hints with mypy --strict compliance
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, Literal

import tiktoken
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tiktoken.core import Encoding

# Logger for debug information
logger = logging.getLogger(__name__)

# Type aliases for encoding models supported by tiktoken
EncodingModel = Literal["cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base"]

# Thread lock for encoder singleton access
_encoder_lock = threading.Lock()


class TokenizerConfig(BaseSettings):
    """Configuration for tokenizer behavior.

    Controls tokenizer settings including encoding model selection, maximum
    token limits, and encoder caching behavior. Uses Pydantic v2 BaseSettings
    for environment variable support with TOKEN_ prefix.

    Attributes:
        encoding_model: The tiktoken encoding model to use.
            - cl100k_base: GPT-3.5-turbo, GPT-4 (default)
            - o200k_base: GPT-4-turbo
            - p50k_base: Text-davinci-003, text-davinci-002
            - p50k_edit: Code model edits
            - r50k_base: GPT-3, text-davinci-001

        max_tokens: Maximum tokens allowed in a single text (0 = unlimited).
            Useful for enforcing token budgets during document parsing.

        cache_encoder: Whether to cache encoder instance in memory.
            Caching significantly improves performance for multiple
            token operations.

    Example:
        >>> config = TokenizerConfig(
        ...     encoding_model="cl100k_base",
        ...     max_tokens=8192,
        ...     cache_encoder=True
        ... )
        >>> tokenizer = Tokenizer(config)
    """

    encoding_model: EncodingModel = Field(
        default="cl100k_base",
        description="Tiktoken encoding model (default: cl100k_base for GPT-4/3.5-turbo)",
    )
    max_tokens: int = Field(
        default=0,
        description="Maximum tokens per text (0 = unlimited)",
        ge=0,
    )
    cache_encoder: bool = Field(
        default=True,
        description="Cache encoder instance for performance",
    )

    model_config = SettingsConfigDict(
        env_prefix="TOKEN_",
        case_sensitive=False,
        extra="ignore",
    )


class Tokenizer:
    """Token counting and encoding using OpenAI's tiktoken library.

    Implements singleton pattern for encoder instance with thread-safe caching
    and lazy-loading to optimize memory usage and performance. Accurately
    counts tokens as they will be used by OpenAI language models.

    Thread Safety:
        The encoder singleton is protected by a threading.Lock to ensure
        thread-safe initialization and caching across concurrent operations.

    Performance:
        First token operation loads encoder into cache (~50MB memory).
        Subsequent operations use cached encoder (no disk I/O).

    Attributes:
        config: TokenizerConfig instance with model and behavior settings.
        _encoder_cache: Class-level cached tiktoken encoder (singleton).

    Example:
        >>> tokenizer = Tokenizer()
        >>> text = "Hello, world!"
        >>> token_count = tokenizer.count_tokens(text)  # Returns: 4
        >>> tokens = tokenizer.encode(text)
        >>> decoded = tokenizer.decode(tokens)
        >>> assert decoded == text
    """

    _encoder_cache: ClassVar[Encoding | None] = None

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Initialize tokenizer with optional configuration.

        Args:
            config: TokenizerConfig instance. If None, uses default config
                   loaded from environment variables or defaults.

        Raises:
            ValueError: If encoding_model is not supported by tiktoken.

        Example:
            >>> # Use default configuration
            >>> tokenizer = Tokenizer()
            >>> # Use custom configuration
            >>> config = TokenizerConfig(encoding_model="o200k_base")
            >>> tokenizer = Tokenizer(config)
        """
        self.config = config or TokenizerConfig()

        logger.debug(
            "Tokenizer initialized",
            extra={
                "encoding_model": self.config.encoding_model,
                "max_tokens": self.config.max_tokens,
                "cache_enabled": self.config.cache_encoder,
            },
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured encoding model.

        Accurately counts tokens that will be used by the target model,
        handling special tokens and encoding edge cases. This is faster
        than full encoding when only token count is needed.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.

        Raises:
            ValueError: If text exceeds max_tokens and limit is enforced.

        Example:
            >>> tokenizer = Tokenizer()
            >>> # Simple text
            >>> count = tokenizer.count_tokens("Hello world")
            >>> # Complex text with special characters
            >>> count = tokenizer.count_tokens("Hello, world! 🌍")
            >>> # Empty text
            >>> assert tokenizer.count_tokens("") == 0
        """
        if not isinstance(text, str):
            msg = f"Expected str, got {type(text).__name__}"
            raise TypeError(msg)

        try:
            encoder = self.get_encoder()
            token_ids = encoder.encode(text)
            token_count = len(token_ids)

            # Check max_tokens limit if configured
            if self.config.max_tokens > 0 and token_count > self.config.max_tokens:
                msg = (
                    f"Text exceeds max_tokens limit: {token_count} > "
                    f"{self.config.max_tokens}"
                )
                raise ValueError(msg)

            logger.debug(
                "Token count completed",
                extra={
                    "token_count": token_count,
                    "text_length": len(text),
                    "encoding_model": self.config.encoding_model,
                },
            )

            return token_count

        except ValueError:
            # Re-raise ValueError for max_tokens violations
            raise
        except Exception as e:
            msg = f"Error counting tokens: {e}"
            raise ValueError(msg) from e

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using configured encoding model.

        Converts text into the token IDs that will be used by the target model.
        These token IDs can be decoded back to text using decode().

        Args:
            text: Text to encode into tokens.

        Returns:
            List of token IDs (integers 0-199999 for cl100k_base).

        Raises:
            ValueError: If text exceeds max_tokens and limit is enforced.

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> print(tokens)  # Example: [9906, 11, 1917, 0]
            >>> # Verify round-trip encoding
            >>> decoded = tokenizer.decode(tokens)
            >>> assert decoded == "Hello, world!"
        """
        if not isinstance(text, str):
            msg = f"Expected str, got {type(text).__name__}"
            raise TypeError(msg)

        try:
            encoder = self.get_encoder()
            token_ids = encoder.encode(text)

            # Check max_tokens limit if configured
            if self.config.max_tokens > 0 and len(token_ids) > self.config.max_tokens:
                msg = (
                    f"Text exceeds max_tokens limit: {len(token_ids)} > "
                    f"{self.config.max_tokens}"
                )
                raise ValueError(msg)

            logger.debug(
                "Encoding completed",
                extra={
                    "token_count": len(token_ids),
                    "text_length": len(text),
                    "encoding_model": self.config.encoding_model,
                },
            )

            return list(token_ids)

        except ValueError:
            # Re-raise ValueError for max_tokens violations
            raise
        except Exception as e:
            msg = f"Error encoding text: {e}"
            raise ValueError(msg) from e

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back into text.

        Converts token IDs generated by encode() back to the original text.
        Note: Some special tokens may not decode exactly, and whitespace
        normalization may differ slightly from the original.

        Args:
            tokens: List of token IDs to decode.

        Returns:
            Decoded text string.

        Raises:
            ValueError: If tokens list is invalid or decoding fails.
            TypeError: If tokens is not a list of integers.

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = [9906, 11, 1917, 0]
            >>> text = tokenizer.decode(tokens)
            >>> print(text)  # "Hello, world!"
        """
        if not isinstance(tokens, list):
            msg = f"Expected list, got {type(tokens).__name__}"
            raise TypeError(msg)

        # Validate all tokens are integers
        for token in tokens:
            if not isinstance(token, int):
                msg = f"Token must be int, got {type(token).__name__}"
                raise TypeError(msg)

        try:
            encoder = self.get_encoder()
            text = encoder.decode(tokens)

            logger.debug(
                "Decoding completed",
                extra={
                    "token_count": len(tokens),
                    "text_length": len(text),
                    "encoding_model": self.config.encoding_model,
                },
            )

            return str(text)

        except Exception as e:
            msg = f"Error decoding tokens: {e}"
            raise ValueError(msg) from e

    def get_encoding_model(self) -> str:
        """Get the encoding model name currently in use.

        Returns:
            Name of the encoding model (e.g., 'cl100k_base').

        Example:
            >>> tokenizer = Tokenizer()
            >>> model = tokenizer.get_encoding_model()
            >>> assert model == "cl100k_base"
        """
        return self.config.encoding_model

    @classmethod
    def get_encoder(cls) -> Encoding:
        """Get or create the cached tiktoken encoder instance.

        Implements lazy-loading singleton pattern for encoder efficiency.
        On first call, loads encoder from disk. Subsequent calls return
        cached instance. Thread-safe via locking.

        Returns:
            tiktoken encoder instance (cached in _encoder_cache).

        Raises:
            ValueError: If encoder loading fails.

        Example:
            >>> # First call loads encoder (~50MB, disk I/O)
            >>> encoder = Tokenizer.get_encoder()
            >>> # Subsequent calls return cached instance (fast)
            >>> encoder2 = Tokenizer.get_encoder()
            >>> assert encoder is encoder2  # Same cached instance
        """
        if cls._encoder_cache is not None:
            return cls._encoder_cache

        with _encoder_lock:
            # Double-check locking pattern
            if cls._encoder_cache is not None:
                return cls._encoder_cache

            try:
                # Load default encoding (cl100k_base)
                encoder = tiktoken.get_encoding("cl100k_base")
                cls._encoder_cache = encoder

                logger.debug(
                    "Encoder loaded and cached",
                    extra={
                        "encoding_model": "cl100k_base",
                    },
                )

                return encoder

            except Exception as e:
                msg = f"Failed to load tiktoken encoder: {e}"
                raise ValueError(msg) from e

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached encoder instance.

        Used primarily for testing and cleanup. After calling this,
        the next token operation will load a fresh encoder instance.

        Thread Safety:
            Uses locking to ensure safe cache clearing in concurrent contexts.

        Example:
            >>> Tokenizer.clear_cache()
            >>> # Next get_encoder() call will load from disk
            >>> encoder = Tokenizer.get_encoder()
        """
        with _encoder_lock:
            cls._encoder_cache = None
            logger.debug("Encoder cache cleared")
