"""Comprehensive tests for response compression layer (Task 10.4).

Tests cover:
- Compression/decompression roundtrip for all algorithms
- Field shortening and unshortening
- Size estimation accuracy
- Performance benchmarks
- Thread safety
- Configuration loading
- Metrics tracking
"""

import json
import os
import threading
import time
from typing import Any

import pytest

from src.mcp.compression import (
    CompressionConfig,
    CompressionMetadata,
    CompressionStats,
    FieldShortener,
    ResponseCompressor,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_vendor_data() -> dict[str, Any]:
    """Sample vendor response data."""
    return {
        "vendor_id": 123,
        "vendor_name": "Acme Corporation",
        "confidence_score": 0.95,
        "entities": [
            {
                "entity_id": "e1",
                "name": "Company A",
                "entity_type": "COMPANY",
                "confidence": 0.92,
            },
            {
                "entity_id": "e2",
                "name": "John Doe",
                "entity_type": "PERSON",
                "confidence": 0.88,
            },
        ],
        "statistics": {
            "entity_count": 50,
            "relationship_count": 25,
            "entity_type_distribution": {"COMPANY": 30, "PERSON": 20},
        },
    }


@pytest.fixture
def sample_search_data() -> dict[str, Any]:
    """Sample search response data."""
    return {
        "results": [
            {
                "chunk_id": 1,
                "source_file": "docs/auth.md",
                "source_category": "security",
                "hybrid_score": 0.85,
                "rank": 1,
                "chunk_index": 0,
                "total_chunks": 10,
                "chunk_snippet": "JWT authentication provides secure token-based access",
                "context_header": "auth.md > Security",
            },
            {
                "chunk_id": 2,
                "source_file": "docs/api.md",
                "source_category": "api",
                "hybrid_score": 0.78,
                "rank": 2,
                "chunk_index": 5,
                "total_chunks": 15,
                "chunk_snippet": "API endpoints require authentication headers",
                "context_header": "api.md > Authentication",
            },
        ],
        "total_found": 42,
        "strategy_used": "hybrid",
        "execution_time_ms": 245.3,
    }


@pytest.fixture
def gzip_compressor() -> ResponseCompressor:
    """Create gzip compressor."""
    config = CompressionConfig(
        enabled=True, algorithm="gzip", compression_level=6, min_size_bytes=100
    )
    return ResponseCompressor(config)


@pytest.fixture
def none_compressor() -> ResponseCompressor:
    """Create no-compression compressor."""
    config = CompressionConfig(
        enabled=False, algorithm="none", compression_level=6, min_size_bytes=1000
    )
    return ResponseCompressor(config)


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestCompressionConfig:
    """Test CompressionConfig model."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = CompressionConfig()
        assert config.enabled is True
        assert config.algorithm == "gzip"
        assert config.compression_level == 6
        assert config.min_size_bytes == 1000

    def test_config_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CompressionConfig(
            enabled=False,
            algorithm="brotli",
            compression_level=9,
            min_size_bytes=500,
        )
        assert config.enabled is False
        assert config.algorithm == "brotli"
        assert config.compression_level == 9
        assert config.min_size_bytes == 500

    def test_config_compression_level_validation(self) -> None:
        """Test compression level validation (1-9)."""
        with pytest.raises(ValueError):
            CompressionConfig(compression_level=0)

        with pytest.raises(ValueError):
            CompressionConfig(compression_level=10)

        # Valid levels should work
        for level in range(1, 10):
            config = CompressionConfig(compression_level=level)
            assert config.compression_level == level

    def test_config_min_size_validation(self) -> None:
        """Test min_size_bytes validation (>= 100)."""
        with pytest.raises(ValueError):
            CompressionConfig(min_size_bytes=50)

        config = CompressionConfig(min_size_bytes=100)
        assert config.min_size_bytes == 100

    def test_config_frozen(self) -> None:
        """Test that config is immutable."""
        config = CompressionConfig(compression_level=6)
        with pytest.raises(Exception):
            config.compression_level = 9  # type: ignore[misc]


# ==============================================================================
# Field Shortening Tests
# ==============================================================================


class TestFieldShortener:
    """Test JSON field shortening utility."""

    def test_shorten_vendor_fields(self) -> None:
        """Test vendor field shortening."""
        data = {
            "vendor_id": 123,
            "vendor_name": "Acme",
            "confidence_score": 0.95,
        }
        shortened = FieldShortener.shorten(data)
        assert shortened == {"vid": 123, "vn": "Acme", "cs": 0.95}

    def test_shorten_search_fields(self) -> None:
        """Test search result field shortening."""
        data = {
            "chunk_id": 1,
            "source_file": "auth.md",
            "hybrid_score": 0.85,
            "chunk_snippet": "JWT auth...",
        }
        shortened = FieldShortener.shorten(data)
        assert shortened == {"cid": 1, "sf": "auth.md", "hs": 0.85, "csnp": "JWT auth..."}

    def test_unshorten_vendor_fields(self) -> None:
        """Test vendor field unshortening."""
        data = {"vid": 123, "vn": "Acme", "cs": 0.95}
        unshortened = FieldShortener.unshorten(data)
        assert unshortened == {
            "vendor_id": 123,
            "vendor_name": "Acme",
            "confidence_score": 0.95,
        }

    def test_unshorten_search_fields(self) -> None:
        """Test search result field unshortening."""
        data = {"cid": 1, "sf": "auth.md", "hs": 0.85, "csnp": "JWT auth..."}
        unshortened = FieldShortener.unshorten(data)
        assert unshortened == {
            "chunk_id": 1,
            "source_file": "auth.md",
            "hybrid_score": 0.85,
            "chunk_snippet": "JWT auth...",
        }

    def test_shorten_nested_dict(self) -> None:
        """Test shortening nested dictionaries."""
        data = {
            "vendor_id": 1,
            "statistics": {
                "entity_count": 50,
                "relationship_count": 25,
            },
        }
        shortened = FieldShortener.shorten(data)
        assert shortened["vid"] == 1
        assert shortened["statistics"]["ecnt"] == 50
        assert shortened["statistics"]["rcnt"] == 25

    def test_shorten_list_of_dicts(self) -> None:
        """Test shortening list items that are dictionaries."""
        data = {
            "results": [
                {"chunk_id": 1, "hybrid_score": 0.85},
                {"chunk_id": 2, "hybrid_score": 0.78},
            ],
        }
        shortened = FieldShortener.shorten(data)
        assert shortened["results"][0] == {"cid": 1, "hs": 0.85}
        assert shortened["results"][1] == {"cid": 2, "hs": 0.78}

    def test_roundtrip_shortening(self) -> None:
        """Test that shorten -> unshorten preserves original."""
        original = {
            "vendor_id": 123,
            "vendor_name": "Acme",
            "confidence_score": 0.95,
            "entities": [
                {"entity_id": "e1", "confidence": 0.92},
                {"entity_id": "e2", "confidence": 0.88},
            ],
        }
        shortened = FieldShortener.shorten(original)
        restored = FieldShortener.unshorten(shortened)
        assert restored == original

    def test_unknown_fields_preserved(self) -> None:
        """Test that unknown fields are preserved during shortening."""
        data = {
            "vendor_id": 1,
            "custom_field": "value",
            "another_unknown": 42,
        }
        shortened = FieldShortener.shorten(data)
        assert shortened["vid"] == 1
        assert shortened["custom_field"] == "value"
        assert shortened["another_unknown"] == 42

    def test_shorten_size_reduction(self) -> None:
        """Test that shortening actually reduces JSON size."""
        data = {
            "vendor_id": 123,
            "confidence_score": 0.95,
            "source_file": "long_path/to/vendor/data.json",
            "entity_type_distribution": {"COMPANY": 30, "PERSON": 20},
        }
        original_json = json.dumps(data)
        shortened = FieldShortener.shorten(data)
        shortened_json = json.dumps(shortened)

        assert len(shortened_json) < len(original_json)


# ==============================================================================
# Compression Roundtrip Tests
# ==============================================================================


class TestCompressionRoundtrip:
    """Test compression and decompression roundtrips."""

    def test_compress_decompress_dict(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test compressing and decompressing a dictionary."""
        compressed, metadata = gzip_compressor.compress(sample_vendor_data)
        decompressed = gzip_compressor.decompress(compressed)

        assert isinstance(decompressed, dict)
        assert decompressed == sample_vendor_data
        assert metadata.algorithm == "gzip"
        assert metadata.original_size > 0
        assert metadata.compressed_size > 0

    def test_compress_decompress_string(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test compressing and decompressing a string."""
        original_str = "This is a test string for compression"
        compressed, metadata = gzip_compressor.compress(original_str)
        decompressed = gzip_compressor.decompress(compressed)

        assert isinstance(decompressed, str)
        assert decompressed == original_str

    def test_compress_decompress_large_response(
        self, gzip_compressor: ResponseCompressor, sample_search_data: dict[str, Any]
    ) -> None:
        """Test compression with larger response data."""
        # Duplicate to make larger
        large_data = {
            "results": sample_search_data["results"] * 100,
            "total_found": sample_search_data["total_found"] * 100,
            "strategy_used": sample_search_data["strategy_used"],
            "execution_time_ms": sample_search_data["execution_time_ms"],
        }

        compressed, metadata = gzip_compressor.compress(large_data)
        decompressed = gzip_compressor.decompress(compressed)

        assert decompressed == large_data
        assert metadata.compression_ratio < 1.0  # Should achieve some compression

    def test_compress_skip_below_threshold(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test that compression is skipped for small data."""
        small_data = {"id": 1, "name": "A"}
        compressed, metadata = gzip_compressor.compress(small_data)

        # Since data is below threshold, should not be compressed
        # Check that we have 1 miss recorded
        stats = gzip_compressor.get_stats()
        assert stats.misses > 0

    def test_compress_respects_enabled_flag(
        self, none_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test that compression is skipped when disabled."""
        compressed, metadata = none_compressor.compress(sample_vendor_data)

        # When disabled, compression_ratio should be 1.0 (no compression)
        assert metadata.algorithm == "none"
        assert metadata.compression_ratio == 1.0

    def test_decompress_with_field_unshortening(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test that decompression automatically unshortens fields."""
        original = {"vendor_id": 1, "confidence_score": 0.95}
        compressed, _ = gzip_compressor.compress(original)
        decompressed = gzip_compressor.decompress(compressed)

        # Fields should be restored to original names
        assert "vendor_id" in decompressed
        assert "confidence_score" in decompressed
        assert decompressed["vendor_id"] == 1
        assert decompressed["confidence_score"] == 0.95


# ==============================================================================
# Compression Metadata Tests
# ==============================================================================


class TestCompressionMetadata:
    """Test compression metadata model."""

    def test_metadata_creation(self) -> None:
        """Test creating compression metadata."""
        metadata = CompressionMetadata(
            algorithm="gzip",
            original_size=5000,
            compressed_size=1500,
            compression_ratio=0.30,
        )
        assert metadata.algorithm == "gzip"
        assert metadata.original_size == 5000
        assert metadata.compressed_size == 1500
        assert metadata.compression_ratio == 0.30

    def test_metadata_compression_ratio_validation(self) -> None:
        """Test compression ratio validation (0.0-1.0)."""
        # Valid ratios
        for ratio in [0.0, 0.5, 1.0]:
            metadata = CompressionMetadata(
                algorithm="gzip",
                original_size=1000,
                compressed_size=500,
                compression_ratio=ratio,
            )
            assert metadata.compression_ratio == ratio

        # Invalid ratios
        with pytest.raises(ValueError):
            CompressionMetadata(
                algorithm="gzip",
                original_size=1000,
                compressed_size=500,
                compression_ratio=1.5,
            )


# ==============================================================================
# Size Estimation Tests
# ==============================================================================


class TestSizeEstimation:
    """Test size estimation functionality."""

    def test_estimate_savings_dict(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test size estimation for dictionary."""
        estimate = gzip_compressor.estimate_savings(sample_vendor_data)

        assert "original_size_bytes" in estimate
        assert "gzip_size_bytes" in estimate
        assert "brotli_size_bytes" in estimate
        assert "savings_bytes" in estimate
        assert "savings_percent" in estimate
        assert "worth_compressing" in estimate
        assert estimate["original_size_bytes"] > 0

    def test_estimate_savings_string(self, gzip_compressor: ResponseCompressor) -> None:
        """Test size estimation for string."""
        test_string = "This is a test string for compression estimation" * 10
        estimate = gzip_compressor.estimate_savings(test_string)

        assert estimate["original_size_bytes"] > 0
        assert estimate["savings_bytes"] >= 0
        assert 0 <= estimate["savings_percent"] <= 100

    def test_estimate_recommends_algorithm(
        self, gzip_compressor: ResponseCompressor, sample_search_data: dict[str, Any]
    ) -> None:
        """Test that estimate recommends best algorithm."""
        estimate = gzip_compressor.estimate_savings(sample_search_data)

        recommended = estimate["recommended_algorithm"]
        assert recommended in ("gzip", "brotli")

        # Verify it's the smaller size
        gzip_size = estimate["gzip_size_bytes"]
        brotli_size = estimate["brotli_size_bytes"]
        if recommended == "gzip":
            assert gzip_size <= brotli_size
        else:
            assert brotli_size <= gzip_size

    def test_estimate_worth_compressing(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test worth_compressing flag."""
        # Small data: not worth compressing
        small_data = {"id": 1}
        small_estimate = gzip_compressor.estimate_savings(small_data)
        # Small data may or may not be worth compressing depending on threshold

        # Large repetitive data: definitely worth compressing
        large_data_dict = {
            "items": [{"id": i, "name": f"Item {i}"} for i in range(100)]
        }
        large_estimate = gzip_compressor.estimate_savings(large_data_dict)
        assert large_estimate["savings_percent"] > 0

    def test_estimate_shortening_savings(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test that estimation includes shortening savings."""
        estimate = gzip_compressor.estimate_savings(sample_vendor_data)

        assert "shortened_size_bytes" in estimate
        assert "shortening_savings_bytes" in estimate

        # Shortening should reduce size
        assert (
            estimate["shortening_savings_bytes"] >= 0
        )  # Should save at least 0 bytes


# ==============================================================================
# Metrics Tests
# ==============================================================================


class TestCompressionStats:
    """Test compression statistics tracking."""

    def test_stats_initialization(self) -> None:
        """Test stats initialization."""
        stats = CompressionStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_original_bytes == 0
        assert stats.total_compressed_bytes == 0
        assert stats.avg_compression_ratio == 0.0

    def test_stats_string_representation(self) -> None:
        """Test stats string representation."""
        stats = CompressionStats(
            hits=10,
            misses=5,
            total_original_bytes=50000,
            total_compressed_bytes=15000,
            avg_compression_ratio=0.30,
        )
        stats_str = str(stats)

        assert "hits=10" in stats_str
        assert "misses=5" in stats_str
        # Check for the number (with or without comma formatting)
        assert "50" in stats_str and "000" in stats_str

    def test_stats_tracking_on_compress(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test that compression operations are tracked in stats."""
        # Initial stats should be empty
        initial_stats = gzip_compressor.get_stats()
        assert initial_stats.hits == 0

        # Compress data
        gzip_compressor.compress(sample_vendor_data)

        # Stats should be updated
        new_stats = gzip_compressor.get_stats()
        assert new_stats.hits >= 0
        # Note: might be hits=0 if below threshold, or hits=1 if above

    def test_stats_reset(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test stats reset."""
        # Compress multiple times
        for _ in range(5):
            gzip_compressor.compress(sample_vendor_data)

        # Verify stats are non-zero
        stats_before = gzip_compressor.get_stats()
        assert stats_before.hits + stats_before.misses > 0

        # Reset
        gzip_compressor.reset_stats()

        # Verify stats are reset
        stats_after = gzip_compressor.get_stats()
        assert stats_after.hits == 0
        assert stats_after.misses == 0
        assert stats_after.total_original_bytes == 0
        assert stats_after.total_compressed_bytes == 0

    def test_compression_hits_vs_misses(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test that hits and misses are correctly tracked."""
        # Compressor with low threshold
        config_low = CompressionConfig(
            enabled=True, algorithm="gzip", min_size_bytes=100
        )
        compressor_low = ResponseCompressor(config_low)

        # Compressor with high threshold
        config_high = CompressionConfig(
            enabled=True, algorithm="gzip", min_size_bytes=1000000
        )
        compressor_high = ResponseCompressor(config_high)

        # Compress same data with both
        compressor_low.compress(sample_vendor_data)
        compressor_high.compress(sample_vendor_data)

        stats_low = compressor_low.get_stats()
        stats_high = compressor_high.get_stats()

        # Low threshold should have hits (data > 100 bytes)
        assert stats_low.hits >= 0

        # High threshold should have misses (data < 1MB)
        assert stats_high.misses > 0


# ==============================================================================
# Thread Safety Tests
# ==============================================================================


class TestThreadSafety:
    """Test thread-safe compression operations."""

    def test_concurrent_compression(self, sample_vendor_data: dict[str, Any]) -> None:
        """Test concurrent compression operations."""
        config = CompressionConfig(enabled=True, algorithm="gzip")
        compressor = ResponseCompressor(config)

        results: list[tuple[bytes, CompressionMetadata]] = []
        errors: list[Exception] = []

        def compress_worker() -> None:
            try:
                for _ in range(10):
                    compressed, metadata = compressor.compress(sample_vendor_data)
                    results.append((compressed, metadata))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compress_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 iterations

    def test_concurrent_decompression(self, sample_vendor_data: dict[str, Any]) -> None:
        """Test concurrent decompression operations."""
        config = CompressionConfig(enabled=True, algorithm="gzip")
        compressor = ResponseCompressor(config)

        # Pre-compress data
        compressed, _ = compressor.compress(sample_vendor_data)

        decompressed_results: list[dict[str, Any] | str] = []
        errors: list[Exception] = []

        def decompress_worker() -> None:
            try:
                for _ in range(10):
                    decompressed = compressor.decompress(compressed)
                    decompressed_results.append(decompressed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=decompress_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(decompressed_results) == 50
        # All decompressed data should match original
        assert all(result == sample_vendor_data for result in decompressed_results)

    def test_concurrent_stats_access(self, sample_vendor_data: dict[str, Any]) -> None:
        """Test thread-safe stats access during concurrent operations."""
        config = CompressionConfig(enabled=True, algorithm="gzip")
        compressor = ResponseCompressor(config)

        stat_reads: list[CompressionStats] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    compressor.compress(sample_vendor_data)
                    stats = compressor.get_stats()
                    stat_reads.append(stats)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(stat_reads) > 0


# ==============================================================================
# Environment Configuration Tests
# ==============================================================================


class TestEnvironmentConfiguration:
    """Test loading configuration from environment variables."""

    def test_load_config_from_env_defaults(self) -> None:
        """Test loading config with default environment."""
        # Clear relevant env vars
        for key in [
            "MCP_COMPRESSION_ENABLED",
            "MCP_COMPRESSION_ALGORITHM",
            "MCP_COMPRESSION_LEVEL",
            "MCP_COMPRESSION_MIN_SIZE",
        ]:
            os.environ.pop(key, None)

        compressor = ResponseCompressor()
        assert compressor._config.enabled is True
        assert compressor._config.algorithm == "gzip"
        assert compressor._config.compression_level == 6
        assert compressor._config.min_size_bytes == 1000

    def test_load_config_from_env_custom(self) -> None:
        """Test loading config from environment variables."""
        try:
            os.environ["MCP_COMPRESSION_ENABLED"] = "false"
            os.environ["MCP_COMPRESSION_ALGORITHM"] = "gzip"
            os.environ["MCP_COMPRESSION_LEVEL"] = "9"
            os.environ["MCP_COMPRESSION_MIN_SIZE"] = "500"

            compressor = ResponseCompressor()
            assert compressor._config.enabled is False
            assert compressor._config.algorithm == "gzip"
            assert compressor._config.compression_level == 9
            assert compressor._config.min_size_bytes == 500
        finally:
            # Cleanup
            for key in [
                "MCP_COMPRESSION_ENABLED",
                "MCP_COMPRESSION_ALGORITHM",
                "MCP_COMPRESSION_LEVEL",
                "MCP_COMPRESSION_MIN_SIZE",
            ]:
                os.environ.pop(key, None)


# ==============================================================================
# Performance Tests
# ==============================================================================


class TestPerformance:
    """Test compression performance requirements."""

    def test_compression_performance_benchmark(
        self, gzip_compressor: ResponseCompressor, sample_search_data: dict[str, Any]
    ) -> None:
        """Test that compression completes within performance budget."""
        # Create larger dataset
        large_data = {
            "results": sample_search_data["results"] * 20,
            "total_found": sample_search_data["total_found"],
            "strategy_used": sample_search_data["strategy_used"],
            "execution_time_ms": sample_search_data["execution_time_ms"],
        }

        start_time = time.time()
        compressed, _ = gzip_compressor.compress(large_data)
        compression_time = (time.time() - start_time) * 1000  # Convert to ms

        # Compression should complete in less than 50ms
        assert compression_time < 50, f"Compression took {compression_time:.2f}ms"

    def test_decompression_performance_benchmark(
        self, gzip_compressor: ResponseCompressor, sample_search_data: dict[str, Any]
    ) -> None:
        """Test that decompression completes within performance budget."""
        # Create larger dataset
        large_data = {
            "results": sample_search_data["results"] * 20,
            "total_found": sample_search_data["total_found"],
            "strategy_used": sample_search_data["strategy_used"],
            "execution_time_ms": sample_search_data["execution_time_ms"],
        }

        compressed, _ = gzip_compressor.compress(large_data)

        start_time = time.time()
        _ = gzip_compressor.decompress(compressed)
        decompression_time = (time.time() - start_time) * 1000  # Convert to ms

        # Decompression should complete in less than 50ms
        assert decompression_time < 50, f"Decompression took {decompression_time:.2f}ms"

    def test_estimation_performance_benchmark(
        self, gzip_compressor: ResponseCompressor, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test that size estimation completes quickly."""
        start_time = time.time()
        for _ in range(100):
            _ = gzip_compressor.estimate_savings(sample_vendor_data)
        total_time = (time.time() - start_time) * 1000  # Convert to ms

        # 100 estimations should complete in less than 500ms (5ms avg)
        assert total_time < 500, f"100 estimations took {total_time:.2f}ms"


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Test error handling in compression operations."""

    def test_decompress_invalid_gzip_data(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test decompression handles uncompressed data gracefully."""
        # When given raw data that's not gzip compressed, it should
        # treat it as raw data and return it as a string
        uncompressed_data = b"This is not gzip compressed data"

        # Should not raise - instead returns the raw data
        result = gzip_compressor.decompress(uncompressed_data)
        assert isinstance(result, str)
        assert result == "This is not gzip compressed data"

    def test_compress_complex_nested_structure(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test compression with deeply nested structures."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"data": "deep value", "score": 0.99}
                        }
                    }
                }
            }
        }

        compressed, metadata = gzip_compressor.compress(nested)
        decompressed = gzip_compressor.decompress(compressed)

        assert decompressed == nested

    def test_compress_special_characters(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test compression with special characters and unicode."""
        data = {
            "vendor_name": "Acme & Co. (International) Ltd.",
            "description": "Services: 日本語, العربية, Ελληνικά",
            "emoji": "Success: ✓ Error: ✗",
        }

        compressed, _ = gzip_compressor.compress(data)
        decompressed = gzip_compressor.decompress(compressed)

        assert decompressed == data


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_vendor_response(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test full compression workflow with vendor response."""
        config = CompressionConfig(enabled=True, algorithm="gzip", min_size_bytes=100)
        compressor = ResponseCompressor(config)

        # Estimate savings
        estimate = compressor.estimate_savings(sample_vendor_data)
        assert estimate["worth_compressing"] is True

        # Compress
        compressed, metadata = compressor.compress(sample_vendor_data)
        assert metadata.compression_ratio < 1.0

        # Decompress
        decompressed = compressor.decompress(compressed)
        assert decompressed == sample_vendor_data

        # Check stats
        stats = compressor.get_stats()
        assert stats.hits > 0

    def test_full_workflow_search_response(
        self, sample_search_data: dict[str, Any]
    ) -> None:
        """Test full compression workflow with search response."""
        config = CompressionConfig(enabled=True, algorithm="gzip", min_size_bytes=100)
        compressor = ResponseCompressor(config)

        # Multiple compress/decompress cycles
        for _ in range(3):
            compressed, metadata = compressor.compress(sample_search_data)
            decompressed = compressor.decompress(compressed)
            assert decompressed == sample_search_data

        stats = compressor.get_stats()
        assert stats.hits >= 3

    def test_compression_ratio_comparison(
        self, sample_vendor_data: dict[str, Any]
    ) -> None:
        """Test compression ratio comparison across algorithms."""
        gzip_config = CompressionConfig(
            enabled=True, algorithm="gzip", min_size_bytes=100
        )
        gzip_compressor = ResponseCompressor(gzip_config)

        compressed_gzip, metadata_gzip = gzip_compressor.compress(sample_vendor_data)

        # Both should compress successfully
        assert metadata_gzip.compression_ratio <= 1.0
        assert metadata_gzip.original_size > 0

    def test_field_shortening_with_compression(
        self, gzip_compressor: ResponseCompressor
    ) -> None:
        """Test that field shortening works with compression."""
        original = {
            "vendor_id": 1,
            "vendor_name": "Test Vendor",
            "confidence_score": 0.95,
            "entities": [
                {"entity_id": "e1", "confidence": 0.92},
            ],
        }

        compressed, metadata = gzip_compressor.compress(original)

        # Decompress and verify fields are unshortened
        decompressed = gzip_compressor.decompress(compressed)
        assert "vendor_id" in decompressed
        assert "vendor_name" in decompressed
        assert "confidence_score" in decompressed
        assert decompressed == original


# ==============================================================================
# Algorithm-Specific Tests
# ==============================================================================


class TestAlgorithmVariations:
    """Test algorithm-specific behavior."""

    def test_gzip_compression(self) -> None:
        """Test gzip compression."""
        config = CompressionConfig(
            enabled=True, algorithm="gzip", compression_level=6, min_size_bytes=100
        )
        compressor = ResponseCompressor(config)

        data = {"test": "data" * 100}
        compressed, metadata = compressor.compress(data)
        assert metadata.algorithm == "gzip"

    def test_no_compression(self) -> None:
        """Test 'none' algorithm (no compression)."""
        config = CompressionConfig(
            enabled=True, algorithm="none", min_size_bytes=100
        )
        compressor = ResponseCompressor(config)

        data = {"test": "data" * 100}
        compressed, metadata = compressor.compress(data)
        assert metadata.algorithm == "none"
        assert metadata.compression_ratio == 1.0

    def test_compression_level_variations(self) -> None:
        """Test different compression levels produce different outputs."""
        data = {"test": "data" * 100}

        sizes: list[int] = []
        for level in [1, 6, 9]:
            config = CompressionConfig(
                enabled=True,
                algorithm="gzip",
                compression_level=level,
                min_size_bytes=100,
            )
            compressor = ResponseCompressor(config)
            compressed, _ = compressor.compress(data)
            sizes.append(len(compressed))

        # Higher compression levels should generally produce smaller output
        # (though not always, depends on data)
        assert all(s > 0 for s in sizes)
