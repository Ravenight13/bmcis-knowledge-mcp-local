"""Type stubs for response compression layer (Task 10.4).

Complete type definitions for compression functionality with full type safety.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

CompressionAlgorithm = Literal["gzip", "brotli", "none"]

class CompressionConfig(BaseModel):
    """Configuration for response compression."""

    enabled: bool
    algorithm: CompressionAlgorithm
    compression_level: int
    min_size_bytes: int

    class Config:
        frozen: bool

class CompressionMetadata(BaseModel):
    """Metadata about compression operation."""

    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float

@dataclass
class CompressionStats:
    """Statistics about compression operations."""

    hits: int
    misses: int
    total_original_bytes: int
    total_compressed_bytes: int
    avg_compression_ratio: float

class FieldShortener:
    """Utility for shortening JSON field names."""

    SHORTENING_MAP: dict[str, str]

    @classmethod
    def shorten(cls, data: dict[str, Any]) -> dict[str, Any]: ...

    @classmethod
    def unshorten(cls, data: dict[str, Any]) -> dict[str, Any]: ...

class ResponseCompressor:
    """Thread-safe response compression with metrics tracking."""

    def __init__(self, config: CompressionConfig | None = None) -> None: ...

    def compress(self, data: dict[str, Any] | str) -> tuple[bytes, CompressionMetadata]: ...

    def decompress(self, data: bytes) -> dict[str, Any] | str: ...

    def estimate_savings(self, data: dict[str, Any] | str) -> dict[str, Any]: ...

    def get_stats(self) -> CompressionStats: ...

    def reset_stats(self) -> None: ...
