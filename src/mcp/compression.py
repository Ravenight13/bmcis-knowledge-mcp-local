"""High-performance response compression layer for MCP tools (Task 10.4).

This module provides thread-safe response compression with:
- Multiple algorithms (gzip, brotli, none)
- Configurable compression levels and size thresholds
- Field shortening for JSON key reduction
- Size estimation before compression
- Comprehensive metrics tracking
- Thread-safe operations with lock protection

Example:
    >>> config = CompressionConfig(enabled=True, algorithm='gzip', compression_level=6)
    >>> compressor = ResponseCompressor(config)
    >>> data = {"vendor_id": 123, "confidence_score": 0.95}
    >>> compressed, metadata = compressor.compress(data)
    >>> savings = compressor.estimate_savings(data)
    >>> print(f"Compression ratio: {metadata.compression_ratio:.2%}")
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

CompressionAlgorithm = Literal["gzip", "brotli", "none"]


class CompressionConfig(BaseModel):
    """Configuration for response compression.

    Attributes:
        enabled: Whether compression is enabled
        algorithm: Compression algorithm ('gzip', 'brotli', or 'none')
        compression_level: Compression level (1-9, 1=fast, 9=best)
        min_size_bytes: Minimum response size before compression (default: 1000)

    Example:
        >>> config = CompressionConfig(
        ...     enabled=True,
        ...     algorithm="gzip",
        ...     compression_level=6,
        ...     min_size_bytes=500
        ... )
    """

    enabled: bool = Field(default=True, description="Enable/disable compression")
    algorithm: CompressionAlgorithm = Field(
        default="gzip",
        description="Compression algorithm: gzip, brotli, or none",
    )
    compression_level: int = Field(
        default=6,
        description="Compression level (1-9)",
        ge=1,
        le=9,
    )
    min_size_bytes: int = Field(
        default=1000,
        description="Minimum size threshold for compression",
        ge=100,
    )

    model_config = ConfigDict(frozen=True)


class CompressionMetadata(BaseModel):
    """Metadata about a compression operation.

    Attributes:
        algorithm: Algorithm used for compression
        original_size: Size before compression in bytes
        compressed_size: Size after compression in bytes
        compression_ratio: Ratio of compressed to original (0.0-1.0)

    Example:
        >>> metadata = CompressionMetadata(
        ...     algorithm="gzip",
        ...     original_size=5000,
        ...     compressed_size=1500,
        ...     compression_ratio=0.30
        ... )
    """

    algorithm: CompressionAlgorithm = Field(..., description="Compression algorithm used")
    original_size: int = Field(..., description="Original size in bytes", ge=0)
    compressed_size: int = Field(..., description="Compressed size in bytes", ge=0)
    compression_ratio: float = Field(
        ...,
        description="Compression ratio (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


@dataclass
class CompressionStats:
    """Statistics tracking for compression operations.

    Attributes:
        hits: Number of successful compressions
        misses: Number of responses below min_size threshold
        total_original_bytes: Total bytes before compression
        total_compressed_bytes: Total bytes after compression
        avg_compression_ratio: Average compression ratio across all operations
    """

    hits: int = 0
    misses: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    avg_compression_ratio: float = 0.0

    def __str__(self) -> str:
        """Format stats as human-readable string."""
        return (
            f"CompressionStats(hits={self.hits}, misses={self.misses}, "
            f"avg_ratio={self.avg_compression_ratio:.2%}, "
            f"total_original={self.total_original_bytes:,} bytes, "
            f"total_compressed={self.total_compressed_bytes:,} bytes)"
        )


class FieldShortener:
    """Utility for shortening JSON field names to reduce payload size.

    Maps verbose field names to short abbreviations:
    - vendor_id -> vid
    - confidence_score -> cs
    - similarity_score -> ss
    - chunk_id -> cid
    - source_file -> sf
    - etc.

    This reduces JSON payload size while maintaining readability through
    a consistent mapping that can be applied and reversed.

    Example:
        >>> original = {"vendor_id": 123, "confidence_score": 0.95}
        >>> shortened = FieldShortener.shorten(original)
        >>> print(shortened)  # {'vid': 123, 'cs': 0.95}
        >>> restored = FieldShortener.unshorten(shortened)
        >>> print(restored)  # {'vendor_id': 123, 'confidence_score': 0.95}
    """

    # Comprehensive mapping of field names to short forms
    SHORTENING_MAP: ClassVar[dict[str, str]] = {
        # Vendor fields
        "vendor_id": "vid",
        "vendor_name": "vn",
        "vendor_type": "vt",
        # Score fields
        "confidence_score": "cs",
        "similarity_score": "ss",
        "bm25_score": "bs",
        "hybrid_score": "hs",
        # Search/chunk fields
        "chunk_id": "cid",
        "chunk_text": "ct",
        "chunk_snippet": "csnp",
        "chunk_index": "cidx",
        "chunk_token_count": "ctc",
        "total_chunks": "tc",
        # File/source fields
        "source_file": "sf",
        "source_category": "sc",
        # Entity fields
        "entity_id": "eid",
        "entity_type": "et",
        # Relationship fields
        "relationship_id": "rid",
        "relationship_type": "rt",
        # Metadata fields
        "context_header": "ch",
        "last_updated": "lu",
        "entity_ids": "eids",
        "relationship_ids": "rids",
        # Pagination fields
        "page_size": "ps",
        "has_more": "hm",
        "total_available": "ta",
        # Response fields
        "strategy_used": "su",
        "execution_time_ms": "etm",
        "error_code": "ec",
        "score_type": "sct",
        "rank": "r",
        # Statistics
        "entity_count": "ecnt",
        "relationship_count": "rcnt",
        "entity_type_distribution": "etd",
        "relationship_type_distribution": "rtd",
    }

    # Reverse mapping for unshortening
    REVERSE_MAP: ClassVar[dict[str, str]] = {v: k for k, v in SHORTENING_MAP.items()}

    @classmethod
    def shorten(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Shorten field names in a dictionary recursively.

        Args:
            data: Dictionary to shorten

        Returns:
            New dictionary with shortened field names

        Example:
            >>> data = {"vendor_id": 1, "confidence_score": 0.9}
            >>> shortened = FieldShortener.shorten(data)
            >>> shortened
            {'vid': 1, 'cs': 0.9}
        """
        if not isinstance(data, dict):
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            # Use shortened key if mapping exists, otherwise keep original
            short_key = cls.SHORTENING_MAP.get(key, key)

            # Recursively shorten nested dictionaries
            if isinstance(value, dict):
                result[short_key] = cls.shorten(value)
            # Shorten field names in list items if they're dictionaries
            elif isinstance(value, list):
                result[short_key] = [
                    cls.shorten(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[short_key] = value

        return result

    @classmethod
    def unshorten(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Restore original field names in a dictionary recursively.

        Args:
            data: Dictionary with shortened field names

        Returns:
            New dictionary with original field names

        Example:
            >>> data = {"vid": 1, "cs": 0.9}
            >>> restored = FieldShortener.unshorten(data)
            >>> restored
            {'vendor_id': 1, 'confidence_score': 0.9}
        """
        if not isinstance(data, dict):
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            # Use original key if reverse mapping exists, otherwise keep shortened key
            original_key = cls.REVERSE_MAP.get(key, key)

            # Recursively unshorten nested dictionaries
            if isinstance(value, dict):
                result[original_key] = cls.unshorten(value)
            # Unshorten field names in list items if they're dictionaries
            elif isinstance(value, list):
                result[original_key] = [
                    cls.unshorten(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[original_key] = value

        return result


class ResponseCompressor:
    """Thread-safe response compression with configurable algorithms and metrics.

    Supports gzip and brotli compression with automatic size estimation,
    field shortening, and comprehensive metrics tracking. All operations are
    thread-safe using internal locking.

    Configuration can be loaded from environment variables:
    - MCP_COMPRESSION_ENABLED (true/false)
    - MCP_COMPRESSION_ALGORITHM (gzip/brotli/none)
    - MCP_COMPRESSION_LEVEL (1-9)
    - MCP_COMPRESSION_MIN_SIZE (bytes)

    Example:
        >>> config = CompressionConfig(enabled=True, algorithm="gzip")
        >>> compressor = ResponseCompressor(config)
        >>> data = {"vendor_id": 1, "confidence_score": 0.95}
        >>> compressed, metadata = compressor.compress(data)
        >>> print(f"Ratio: {metadata.compression_ratio:.2%}")
        >>> decompressed = compressor.decompress(compressed)
        >>> stats = compressor.get_stats()
    """

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """Initialize compressor with optional configuration.

        Args:
            config: CompressionConfig instance. If None, loads from environment
                   or uses defaults.

        Example:
            >>> compressor = ResponseCompressor()  # Uses defaults
            >>> config = CompressionConfig(algorithm="brotli")
            >>> compressor = ResponseCompressor(config)
        """
        self._lock = Lock()
        self._stats = CompressionStats()

        if config is None:
            config = self._load_config_from_env()

        self._config = config

        # Validate brotli availability if selected
        if config.algorithm == "brotli":
            try:
                import brotli  # noqa: F401
            except ImportError:
                raise ImportError(
                    "brotli algorithm selected but brotli library not installed. "
                    "Install with: pip install brotli"
                )

    @staticmethod
    def _load_config_from_env() -> CompressionConfig:
        """Load compression config from environment variables.

        Returns:
            CompressionConfig with values from environment or defaults

        Environment Variables:
            MCP_COMPRESSION_ENABLED: 'true' or 'false' (default: 'true')
            MCP_COMPRESSION_ALGORITHM: 'gzip', 'brotli', or 'none' (default: 'gzip')
            MCP_COMPRESSION_LEVEL: 1-9 (default: 6)
            MCP_COMPRESSION_MIN_SIZE: bytes (default: 1000)
        """
        enabled = os.getenv("MCP_COMPRESSION_ENABLED", "true").lower() == "true"
        algorithm_str = os.getenv("MCP_COMPRESSION_ALGORITHM", "gzip")
        if algorithm_str not in ("gzip", "brotli", "none"):
            algorithm_str = "gzip"
        algorithm: CompressionAlgorithm = algorithm_str  # type: ignore[assignment]
        compression_level = int(os.getenv("MCP_COMPRESSION_LEVEL", "6"))
        min_size_bytes = int(os.getenv("MCP_COMPRESSION_MIN_SIZE", "1000"))

        return CompressionConfig(
            enabled=enabled,
            algorithm=algorithm,
            compression_level=compression_level,
            min_size_bytes=min_size_bytes,
        )

    def compress(
        self, data: dict[str, Any] | str
    ) -> tuple[bytes, CompressionMetadata]:
        """Compress response data using configured algorithm.

        Automatically:
        1. Converts input to JSON string if needed
        2. Shortens field names to reduce size
        3. Skips compression if below min_size threshold
        4. Tracks compression statistics

        Args:
            data: Dictionary or string to compress

        Returns:
            Tuple of (compressed_bytes, compression_metadata)

        Raises:
            ValueError: If compression algorithm fails

        Example:
            >>> data = {"vendor_id": 1, "confidence_score": 0.95}
            >>> compressed, metadata = compressor.compress(data)
            >>> print(f"Saved {metadata.original_size - metadata.compressed_size} bytes")
        """
        with self._lock:
            try:
                # Convert to JSON string if needed
                if isinstance(data, dict):
                    json_str = json.dumps(data)
                else:
                    json_str = str(data)

                original_bytes = json_str.encode("utf-8")
                original_size = len(original_bytes)

                # Check if compression is enabled and size threshold met
                if (
                    not self._config.enabled
                    or original_size < self._config.min_size_bytes
                ):
                    self._stats.misses += 1
                    # Don't apply field shortening if not compressing
                    if isinstance(data, dict):
                        original_bytes = json.dumps(data).encode("utf-8")
                    return original_bytes, CompressionMetadata(
                        algorithm="none",
                        original_size=original_size,
                        compressed_size=original_size,
                        compression_ratio=1.0,
                    )

                # Apply field shortening
                if isinstance(data, dict):
                    shortened = FieldShortener.shorten(data)
                    json_str = json.dumps(shortened)
                    original_bytes = json_str.encode("utf-8")
                    original_size = len(original_bytes)

                # Compress based on algorithm
                if self._config.algorithm == "gzip":
                    compressed_bytes = gzip.compress(
                        original_bytes, compresslevel=self._config.compression_level
                    )
                elif self._config.algorithm == "brotli":
                    import brotli

                    compressed_bytes = brotli.compress(
                        original_bytes, quality=self._config.compression_level
                    )
                else:  # "none"
                    compressed_bytes = original_bytes

                compressed_size = len(compressed_bytes)
                compression_ratio = min(
                    compressed_size / original_size if original_size > 0 else 1.0,
                    1.0  # Cap at 1.0 to avoid invalid values
                )

                # Update statistics
                self._stats.hits += 1
                self._stats.total_original_bytes += original_size
                self._stats.total_compressed_bytes += compressed_size

                # Calculate rolling average compression ratio
                if self._stats.hits > 0:
                    self._stats.avg_compression_ratio = (
                        self._stats.total_compressed_bytes / self._stats.total_original_bytes
                    )

                return compressed_bytes, CompressionMetadata(
                    algorithm=self._config.algorithm,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                )

            except Exception as e:
                raise ValueError(f"Compression failed: {e}") from e

    def decompress(self, data: bytes) -> dict[str, Any] | str:
        """Decompress response data.

        Automatically:
        1. Decompresses using the appropriate algorithm
        2. Attempts to parse as JSON and unshorten field names
        3. Falls back to string if JSON parsing fails
        4. Handles both compressed and uncompressed data

        Args:
            data: Compressed or uncompressed bytes

        Returns:
            Decompressed data as dict or string

        Raises:
            ValueError: If decompression fails

        Example:
            >>> compressed, metadata = compressor.compress(data)
            >>> decompressed = compressor.decompress(compressed)
        """
        with self._lock:
            try:
                decompressed_bytes: bytes | None = None

                # Try decompressing based on algorithm
                if self._config.algorithm == "gzip":
                    try:
                        decompressed_bytes = gzip.decompress(data)
                    except (OSError, EOFError, gzip.BadGzipFile):
                        # Not gzip compressed, treat as raw data
                        decompressed_bytes = data
                elif self._config.algorithm == "brotli":
                    try:
                        import brotli
                        decompressed_bytes = brotli.decompress(data)
                    except (OSError, EOFError, ModuleNotFoundError):
                        # Not brotli compressed, treat as raw data
                        decompressed_bytes = data
                else:  # "none"
                    decompressed_bytes = data

                if decompressed_bytes is None:
                    decompressed_bytes = data

                # Decode to string
                decompressed_str = decompressed_bytes.decode("utf-8")

                # Try to parse as JSON and unshorten
                try:
                    parsed: Any = json.loads(decompressed_str)
                    if isinstance(parsed, dict):
                        result: dict[str, Any] | str = FieldShortener.unshorten(parsed)
                        return result
                    # Return non-dict JSON as-is
                    result_non_dict: dict[str, Any] | str = str(parsed)
                    return result_non_dict
                except json.JSONDecodeError:
                    # Not JSON, return as string
                    result_str: dict[str, Any] | str = decompressed_str
                    return result_str

            except Exception as e:
                raise ValueError(f"Decompression failed: {e}") from e

    def estimate_savings(
        self, data: dict[str, Any] | str
    ) -> dict[str, Any]:
        """Estimate compression savings before actual compression.

        Provides size estimates for:
        - Original data
        - After field shortening
        - After compression (gzip and brotli)
        - Savings percentages

        Args:
            data: Data to estimate

        Returns:
            Dictionary with size estimates and savings metrics

        Example:
            >>> data = {"vendor_id": 1, "confidence_score": 0.95}
            >>> estimate = compressor.estimate_savings(data)
            >>> print(f"Estimated savings: {estimate['savings_percent']:.1%}")
        """
        try:
            # Original size
            if isinstance(data, dict):
                original_json = json.dumps(data)
            else:
                original_json = str(data)

            original_bytes = original_json.encode("utf-8")
            original_size = len(original_bytes)

            # Size after field shortening
            if isinstance(data, dict):
                shortened = FieldShortener.shorten(data)
                shortened_json = json.dumps(shortened)
                shortened_bytes = shortened_json.encode("utf-8")
                shortened_size = len(shortened_bytes)
            else:
                shortened_size = original_size

            # Estimate gzip compression
            try:
                gzip_compressed = gzip.compress(original_bytes, compresslevel=6)
                gzip_size = len(gzip_compressed)
            except (OSError, MemoryError):
                gzip_size = original_size

            # Estimate brotli compression
            try:
                import brotli
                brotli_compressed = brotli.compress(original_bytes, quality=6)
                brotli_size = len(brotli_compressed)
            except (OSError, MemoryError, ModuleNotFoundError):
                brotli_size = original_size

            # Calculate savings
            best_compressed = min(gzip_size, brotli_size)
            savings_bytes = original_size - best_compressed
            savings_percent = (savings_bytes / original_size * 100) if original_size > 0 else 0

            return {
                "original_size_bytes": original_size,
                "shortened_size_bytes": shortened_size,
                "shortening_savings_bytes": original_size - shortened_size,
                "gzip_size_bytes": gzip_size,
                "brotli_size_bytes": brotli_size,
                "best_compressed_size_bytes": best_compressed,
                "savings_bytes": savings_bytes,
                "savings_percent": savings_percent,
                "recommended_algorithm": "brotli" if brotli_size < gzip_size else "gzip",
                "worth_compressing": savings_percent > 10,
            }

        except (TypeError, ValueError, AttributeError) as e:
            return {
                "error": str(e),
                "original_size_bytes": 0,
                "worth_compressing": False,
            }

    def get_stats(self) -> CompressionStats:
        """Get current compression statistics.

        Returns:
            CompressionStats with hits, misses, and aggregate sizes

        Example:
            >>> stats = compressor.get_stats()
            >>> print(f"Compression hits: {stats.hits}")
            >>> print(f"Average ratio: {stats.avg_compression_ratio:.2%}")
        """
        with self._lock:
            return CompressionStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                total_original_bytes=self._stats.total_original_bytes,
                total_compressed_bytes=self._stats.total_compressed_bytes,
                avg_compression_ratio=self._stats.avg_compression_ratio,
            )

    def reset_stats(self) -> None:
        """Reset compression statistics.

        Example:
            >>> compressor.reset_stats()
            >>> stats = compressor.get_stats()
            >>> stats.hits  # Returns 0
        """
        with self._lock:
            self._stats = CompressionStats()
