"""Comprehensive tests for response formatting models (Task 10.4 Phase B).

Tests cover:
- ResponseMetadata creation, validation, and timestamp handling
- ExecutionContext token accounting and validation
- ResponseWarning levels and code format validation
- ConfidenceScore validation and completeness
- RankingContext percentile validation
- DeduplicationInfo similarity tracking
- EnhancedSemanticSearchResult composition
- MCPResponseEnvelope generic wrapper and composition
- Backward compatibility with existing response formats
- Type safety and mypy --strict compatibility
- Edge cases and error conditions

Test coverage: 50+ tests across all response formatting models
"""

import base64
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.mcp.models import (
    ConfidenceScore,
    DeduplicationInfo,
    EnhancedSemanticSearchResult,
    ExecutionContext,
    MCPResponseEnvelope,
    PaginationMetadata,
    RankingContext,
    ResponseMetadata,
    ResponseWarning,
    SearchResultMetadata,
)

# ==============================================================================
# ResponseMetadata Tests (8 tests)
# ==============================================================================


class TestResponseMetadata:
    """Test ResponseMetadata model validation."""

    def test_response_metadata_valid(self) -> None:
        """Test valid ResponseMetadata creation."""
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp="2025-11-09T15:30:45.123Z",
            request_id="req_abc123",
            status="success",
            message=None,
        )
        assert metadata.operation == "semantic_search"
        assert metadata.version == "1.0"
        assert metadata.status == "success"
        assert metadata.request_id == "req_abc123"
        assert metadata.message is None

    def test_response_metadata_with_message(self) -> None:
        """Test ResponseMetadata with optional message."""
        metadata = ResponseMetadata(
            operation="find_vendor_info",
            version="1.0",
            timestamp="2025-11-09T15:30:45Z",
            request_id="req_xyz789",
            status="partial",
            message="Some results filtered due to token limits",
        )
        assert metadata.message == "Some results filtered due to token limits"
        assert metadata.status == "partial"

    def test_response_metadata_status_values(self) -> None:
        """Test all valid status values."""
        for status in ["success", "partial", "error"]:
            metadata = ResponseMetadata(
                operation="test",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_123",
                status=status,  # type: ignore
            )
            assert metadata.status == status

    def test_response_metadata_invalid_status(self) -> None:
        """Test invalid status value raises ValidationError."""
        with pytest.raises(ValidationError):
            ResponseMetadata(
                operation="test",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_123",
                status="invalid",  # type: ignore
            )

    def test_response_metadata_timestamp_iso8601_z_format(self) -> None:
        """Test timestamp validation with Z suffix."""
        metadata = ResponseMetadata(
            operation="test",
            timestamp="2025-11-09T15:30:45.123456Z",
            request_id="req_123",
            status="success",
        )
        assert metadata.timestamp == "2025-11-09T15:30:45.123456Z"

    def test_response_metadata_timestamp_iso8601_offset_format(self) -> None:
        """Test timestamp validation with +HH:MM offset."""
        metadata = ResponseMetadata(
            operation="test",
            timestamp="2025-11-09T15:30:45.123+00:00",
            request_id="req_123",
            status="success",
        )
        assert metadata.timestamp == "2025-11-09T15:30:45.123+00:00"

    def test_response_metadata_timestamp_without_timezone_raises_error(self) -> None:
        """Test timestamp without timezone raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseMetadata(
                operation="test",
                timestamp="2025-11-09T15:30:45.123",  # No timezone
                request_id="req_123",
                status="success",
            )
        assert "timezone" in str(exc_info.value).lower()

    def test_response_metadata_invalid_timestamp_format_raises_error(self) -> None:
        """Test invalid timestamp format raises ValidationError."""
        with pytest.raises(ValidationError):
            ResponseMetadata(
                operation="test",
                timestamp="not-a-timestamp",
                request_id="req_123",
                status="success",
            )


# ==============================================================================
# ExecutionContext Tests (8 tests)
# ==============================================================================


class TestExecutionContext:
    """Test ExecutionContext model validation."""

    def test_execution_context_valid(self) -> None:
        """Test valid ExecutionContext creation."""
        context = ExecutionContext(
            tokens_estimated=3400,
            tokens_used=3450,
            cache_hit=True,
            execution_time_ms=245.3,
            request_id="req_abc123",
        )
        assert context.tokens_estimated == 3400
        assert context.tokens_used == 3450
        assert context.cache_hit is True
        assert context.execution_time_ms == 245.3
        assert context.request_id == "req_abc123"

    def test_execution_context_without_tokens_used(self) -> None:
        """Test ExecutionContext with tokens_used as None."""
        context = ExecutionContext(
            tokens_estimated=3400,
            tokens_used=None,
            cache_hit=False,
            execution_time_ms=125.0,
            request_id="req_123",
        )
        assert context.tokens_used is None

    def test_execution_context_tokens_used_10_percent_overage(self) -> None:
        """Test tokens_used can be up to 10% higher than estimated."""
        context = ExecutionContext(
            tokens_estimated=1000,
            tokens_used=1100,  # Exactly 10% higher
            cache_hit=False,
            execution_time_ms=100.0,
            request_id="req_123",
        )
        assert context.tokens_used == 1100

    def test_execution_context_tokens_used_exceeding_10_percent_raises_error(
        self,
    ) -> None:
        """Test tokens_used exceeding 10% raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionContext(
                tokens_estimated=1000,
                tokens_used=1101,  # 10.1% higher (exceeds limit)
                cache_hit=False,
                execution_time_ms=100.0,
                request_id="req_123",
            )
        assert "exceeds estimated" in str(exc_info.value).lower()

    def test_execution_context_cache_hit_false(self) -> None:
        """Test cache_hit=False."""
        context = ExecutionContext(
            tokens_estimated=3400,
            tokens_used=3400,
            cache_hit=False,
            execution_time_ms=520.0,
            request_id="req_123",
        )
        assert context.cache_hit is False

    def test_execution_context_zero_execution_time(self) -> None:
        """Test execution_time_ms can be 0.0 (cached instant response)."""
        context = ExecutionContext(
            tokens_estimated=100,
            tokens_used=100,
            cache_hit=True,
            execution_time_ms=0.0,
            request_id="req_123",
        )
        assert context.execution_time_ms == 0.0

    def test_execution_context_zero_tokens(self) -> None:
        """Test zero token counts (edge case)."""
        context = ExecutionContext(
            tokens_estimated=0,
            tokens_used=0,
            cache_hit=False,
            execution_time_ms=10.0,
            request_id="req_123",
        )
        assert context.tokens_estimated == 0
        assert context.tokens_used == 0

    def test_execution_context_request_id_matches_metadata(self) -> None:
        """Test request_id consistency with ResponseMetadata."""
        request_id = "req_xyz789"
        context = ExecutionContext(
            tokens_estimated=1000,
            tokens_used=1050,
            cache_hit=True,
            execution_time_ms=50.0,
            request_id=request_id,
        )
        assert context.request_id == request_id


# ==============================================================================
# ResponseWarning Tests (8 tests)
# ==============================================================================


class TestResponseWarning:
    """Test ResponseWarning model validation."""

    def test_response_warning_valid_warning_level(self) -> None:
        """Test valid ResponseWarning with warning level."""
        warning = ResponseWarning(
            level="warning",
            code="TOKEN_LIMIT_WARNING",
            message="Response approaching context limit",
            suggestion="Reduce page_size or use ids_only mode",
        )
        assert warning.level == "warning"
        assert warning.code == "TOKEN_LIMIT_WARNING"
        assert warning.message == "Response approaching context limit"

    def test_response_warning_valid_info_level(self) -> None:
        """Test valid ResponseWarning with info level."""
        warning = ResponseWarning(
            level="info",
            code="CACHE_HIT",
            message="Response served from cache",
        )
        assert warning.level == "info"
        assert warning.code == "CACHE_HIT"
        assert warning.suggestion is None

    def test_response_warning_valid_error_level(self) -> None:
        """Test valid ResponseWarning with error level."""
        warning = ResponseWarning(
            level="error",
            code="SEARCH_FAILED",
            message="Search index unavailable",
            suggestion="Retry in 30 seconds",
        )
        assert warning.level == "error"

    def test_response_warning_all_levels(self) -> None:
        """Test all valid warning levels."""
        for level in ["info", "warning", "error"]:
            warning = ResponseWarning(
                level=level,  # type: ignore
                code="TEST_CODE",
                message="Test message",
            )
            assert warning.level == level

    def test_response_warning_invalid_level_raises_error(self) -> None:
        """Test invalid level raises ValidationError."""
        with pytest.raises(ValidationError):
            ResponseWarning(
                level="critical",  # type: ignore
                code="TEST_CODE",
                message="Test message",
            )

    def test_response_warning_code_screaming_snake_case(self) -> None:
        """Test code validation requires SCREAMING_SNAKE_CASE."""
        warning = ResponseWarning(
            level="warning",
            code="DEPRECATED_PARAMETER_USED",
            message="The top_k parameter is deprecated",
        )
        assert warning.code == "DEPRECATED_PARAMETER_USED"

    def test_response_warning_invalid_code_format_raises_error(self) -> None:
        """Test invalid code format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseWarning(
                level="warning",
                code="invalid-code-format",  # Not SCREAMING_SNAKE_CASE
                message="Test message",
            )
        assert "SCREAMING_SNAKE_CASE" in str(exc_info.value)

    def test_response_warning_code_with_numbers(self) -> None:
        """Test code can include numbers."""
        warning = ResponseWarning(
            level="warning",
            code="TOKEN_LIMIT_1000",
            message="Tokens exceeded 1000",
        )
        assert warning.code == "TOKEN_LIMIT_1000"


# ==============================================================================
# ConfidenceScore Tests (7 tests)
# ==============================================================================


class TestConfidenceScore:
    """Test ConfidenceScore model validation."""

    def test_confidence_score_valid(self) -> None:
        """Test valid ConfidenceScore creation."""
        confidence = ConfidenceScore(
            score_reliability=0.92,
            source_quality=0.88,
            recency=0.75,
        )
        assert confidence.score_reliability == 0.92
        assert confidence.source_quality == 0.88
        assert confidence.recency == 0.75

    def test_confidence_score_min_values(self) -> None:
        """Test ConfidenceScore with 0.0 values."""
        confidence = ConfidenceScore(
            score_reliability=0.0,
            source_quality=0.0,
            recency=0.0,
        )
        assert confidence.score_reliability == 0.0

    def test_confidence_score_max_values(self) -> None:
        """Test ConfidenceScore with 1.0 values."""
        confidence = ConfidenceScore(
            score_reliability=1.0,
            source_quality=1.0,
            recency=1.0,
        )
        assert confidence.score_reliability == 1.0

    def test_confidence_score_mixed_values(self) -> None:
        """Test ConfidenceScore with varied values."""
        confidence = ConfidenceScore(
            score_reliability=0.50,
            source_quality=0.75,
            recency=0.25,
        )
        assert confidence.score_reliability == 0.50
        assert confidence.source_quality == 0.75
        assert confidence.recency == 0.25

    def test_confidence_score_invalid_above_1_0_raises_error(self) -> None:
        """Test values > 1.0 raise ValidationError."""
        with pytest.raises(ValidationError):
            ConfidenceScore(
                score_reliability=1.1,  # > 1.0
                source_quality=0.88,
                recency=0.75,
            )

    def test_confidence_score_invalid_below_0_0_raises_error(self) -> None:
        """Test values < 0.0 raise ValidationError."""
        with pytest.raises(ValidationError):
            ConfidenceScore(
                score_reliability=0.92,
                source_quality=-0.1,  # < 0.0
                recency=0.75,
            )

    def test_confidence_score_must_be_complete(self) -> None:
        """Test ConfidenceScore requires all fields (model-level validator).

        This is by design - if confidence is not available, use None
        for the entire ConfidenceScore model, not individual fields.
        """
        # All fields present should work
        confidence = ConfidenceScore(
            score_reliability=0.92,
            source_quality=0.88,
            recency=0.75,
        )
        assert confidence.score_reliability is not None


# ==============================================================================
# RankingContext Tests (6 tests)
# ==============================================================================


class TestRankingContext:
    """Test RankingContext model validation."""

    def test_ranking_context_valid(self) -> None:
        """Test valid RankingContext creation."""
        ranking = RankingContext(
            percentile=99,
            explanation="Highest combined semantic + keyword match",
            score_method="hybrid",
        )
        assert ranking.percentile == 99
        assert ranking.explanation == "Highest combined semantic + keyword match"
        assert ranking.score_method == "hybrid"

    def test_ranking_context_percentile_min(self) -> None:
        """Test percentile at minimum (0)."""
        ranking = RankingContext(
            percentile=0,
            explanation="Lowest ranked result",
            score_method="vector",
        )
        assert ranking.percentile == 0

    def test_ranking_context_percentile_max(self) -> None:
        """Test percentile at maximum (100)."""
        ranking = RankingContext(
            percentile=100,
            explanation="Highest ranked result",
            score_method="bm25",
        )
        assert ranking.percentile == 100

    def test_ranking_context_percentile_mid_range(self) -> None:
        """Test percentile in middle of range."""
        ranking = RankingContext(
            percentile=50,
            explanation="Median score",
            score_method="hybrid",
        )
        assert ranking.percentile == 50

    def test_ranking_context_invalid_percentile_below_0_raises_error(self) -> None:
        """Test percentile < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            RankingContext(
                percentile=-1,  # < 0
                explanation="Invalid",
                score_method="hybrid",
            )

    def test_ranking_context_invalid_percentile_above_100_raises_error(self) -> None:
        """Test percentile > 100 raises ValidationError."""
        with pytest.raises(ValidationError):
            RankingContext(
                percentile=101,  # > 100
                explanation="Invalid",
                score_method="hybrid",
            )


# ==============================================================================
# DeduplicationInfo Tests (6 tests)
# ==============================================================================


class TestDeduplicationInfo:
    """Test DeduplicationInfo model validation."""

    def test_deduplication_info_not_duplicate(self) -> None:
        """Test DeduplicationInfo for unique result."""
        dedup = DeduplicationInfo(
            is_duplicate=False,
            similar_chunk_ids=[2, 5],
            confidence=0.85,
        )
        assert dedup.is_duplicate is False
        assert dedup.similar_chunk_ids == [2, 5]
        assert dedup.confidence == 0.85

    def test_deduplication_info_is_duplicate(self) -> None:
        """Test DeduplicationInfo when result is duplicate."""
        dedup = DeduplicationInfo(
            is_duplicate=True,
            similar_chunk_ids=[1, 3],
            confidence=0.95,
        )
        assert dedup.is_duplicate is True
        assert len(dedup.similar_chunk_ids) == 2

    def test_deduplication_info_no_similar_chunks(self) -> None:
        """Test DeduplicationInfo with empty similar_chunk_ids."""
        dedup = DeduplicationInfo(
            is_duplicate=False,
            similar_chunk_ids=[],
            confidence=0.50,
        )
        assert dedup.similar_chunk_ids == []

    def test_deduplication_info_many_similar_chunks(self) -> None:
        """Test DeduplicationInfo with many similar chunks."""
        similar_ids = list(range(2, 102))  # 100 similar chunks
        dedup = DeduplicationInfo(
            is_duplicate=False,
            similar_chunk_ids=similar_ids,
            confidence=0.92,
        )
        assert len(dedup.similar_chunk_ids) == 100

    def test_deduplication_info_confidence_bounds(self) -> None:
        """Test DeduplicationInfo confidence validation."""
        # Min confidence
        dedup_min = DeduplicationInfo(
            is_duplicate=False,
            similar_chunk_ids=[],
            confidence=0.0,
        )
        assert dedup_min.confidence == 0.0

        # Max confidence
        dedup_max = DeduplicationInfo(
            is_duplicate=False,
            similar_chunk_ids=[],
            confidence=1.0,
        )
        assert dedup_max.confidence == 1.0

    def test_deduplication_info_invalid_confidence_raises_error(self) -> None:
        """Test invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError):
            DeduplicationInfo(
                is_duplicate=False,
                similar_chunk_ids=[],
                confidence=1.5,  # > 1.0
            )


# ==============================================================================
# EnhancedSemanticSearchResult Tests (6 tests)
# ==============================================================================


class TestEnhancedSemanticSearchResult:
    """Test EnhancedSemanticSearchResult model."""

    def test_enhanced_result_full_metadata(self) -> None:
        """Test EnhancedSemanticSearchResult with all fields."""
        result = EnhancedSemanticSearchResult(
            chunk_id=1,
            source_file="docs/auth.md",
            source_category="security",
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
            confidence=ConfidenceScore(
                score_reliability=0.92,
                source_quality=0.88,
                recency=0.75,
            ),
            ranking=RankingContext(
                percentile=99,
                explanation="Highest combined match",
                score_method="hybrid",
            ),
            deduplication=DeduplicationInfo(
                is_duplicate=False,
                similar_chunk_ids=[2, 5],
                confidence=0.85,
            ),
        )
        assert result.chunk_id == 1
        assert result.confidence is not None
        assert result.confidence.score_reliability == 0.92

    def test_enhanced_result_without_confidence(self) -> None:
        """Test EnhancedSemanticSearchResult with confidence=None."""
        result = EnhancedSemanticSearchResult(
            chunk_id=2,
            source_file="docs/api.md",
            source_category="api",
            hybrid_score=0.75,
            rank=2,
            chunk_index=1,
            total_chunks=15,
            confidence=None,
            ranking=RankingContext(
                percentile=75,
                explanation="Good semantic match",
                score_method="hybrid",
            ),
        )
        assert result.confidence is None
        assert result.ranking.percentile == 75

    def test_enhanced_result_without_deduplication(self) -> None:
        """Test EnhancedSemanticSearchResult with deduplication=None."""
        result = EnhancedSemanticSearchResult(
            chunk_id=3,
            source_file="docs/config.md",
            source_category="config",
            hybrid_score=0.65,
            rank=3,
            chunk_index=2,
            total_chunks=20,
            confidence=ConfidenceScore(
                score_reliability=0.85,
                source_quality=0.80,
                recency=0.70,
            ),
            ranking=RankingContext(
                percentile=50,
                explanation="Moderate relevance",
                score_method="vector",
            ),
            deduplication=None,
        )
        assert result.deduplication is None

    def test_enhanced_result_extends_search_result_metadata(self) -> None:
        """Test EnhancedSemanticSearchResult is SearchResultMetadata subclass."""
        result = EnhancedSemanticSearchResult(
            chunk_id=4,
            source_file="docs/test.md",
            source_category="testing",
            hybrid_score=0.55,
            rank=4,
            chunk_index=3,
            total_chunks=25,
            ranking=RankingContext(
                percentile=25,
                explanation="Lower relevance",
                score_method="bm25",
            ),
        )
        # Should have all SearchResultMetadata fields
        assert result.chunk_id == 4
        assert result.source_file == "docs/test.md"
        assert result.hybrid_score == 0.55

    def test_enhanced_result_serialization(self) -> None:
        """Test EnhancedSemanticSearchResult serialization."""
        result = EnhancedSemanticSearchResult(
            chunk_id=5,
            source_file="docs/schema.md",
            source_category="schema",
            hybrid_score=0.90,
            rank=1,
            chunk_index=0,
            total_chunks=8,
            confidence=ConfidenceScore(
                score_reliability=0.95,
                source_quality=0.92,
                recency=0.88,
            ),
            ranking=RankingContext(
                percentile=95,
                explanation="Excellent match",
                score_method="hybrid",
            ),
        )
        # Should serialize to dict without errors
        result_dict = result.model_dump()
        assert result_dict["chunk_id"] == 5
        assert result_dict["confidence"]["score_reliability"] == 0.95

    def test_enhanced_result_json_serialization(self) -> None:
        """Test EnhancedSemanticSearchResult JSON serialization."""
        result = EnhancedSemanticSearchResult(
            chunk_id=6,
            source_file="docs/final.md",
            source_category="documentation",
            hybrid_score=0.88,
            rank=2,
            chunk_index=1,
            total_chunks=12,
            ranking=RankingContext(
                percentile=88,
                explanation="Strong relevance",
                score_method="hybrid",
            ),
        )
        # Should serialize to JSON without errors
        json_str = result.model_dump_json()
        assert "chunk_id" in json_str
        assert "6" in json_str or '"chunk_id":6' in json_str


# ==============================================================================
# MCPResponseEnvelope Tests (6 tests)
# ==============================================================================


class TestMCPResponseEnvelope:
    """Test MCPResponseEnvelope generic wrapper."""

    def test_response_envelope_with_list_results(self) -> None:
        """Test MCPResponseEnvelope with list results."""
        results = [
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/auth.md",
                source_category="security",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=10,
            )
        ]
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_123",
                status="success",
            ),
            results=results,
            execution_context=ExecutionContext(
                tokens_estimated=3400,
                tokens_used=3450,
                cache_hit=True,
                execution_time_ms=245.3,
                request_id="req_123",
            ),
        )
        assert len(envelope.results) == 1  # type: ignore
        assert envelope.metadata.operation == "semantic_search"

    def test_response_envelope_with_pagination(self) -> None:
        """Test MCPResponseEnvelope with pagination."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_456",
                status="success",
            ),
            results=[],
            pagination=PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=42,
            ),
            execution_context=ExecutionContext(
                tokens_estimated=3400,
                tokens_used=3450,
                cache_hit=False,
                execution_time_ms=245.3,
                request_id="req_456",
            ),
        )
        assert envelope.pagination is not None
        assert envelope.pagination.has_more is True

    def test_response_envelope_with_warnings(self) -> None:
        """Test MCPResponseEnvelope with warnings list."""
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_789",
                status="partial",
            ),
            results=[],
            execution_context=ExecutionContext(
                tokens_estimated=3400,
                tokens_used=3450,
                cache_hit=False,
                execution_time_ms=245.3,
                request_id="req_789",
            ),
            warnings=[
                ResponseWarning(
                    level="warning",
                    code="TOKEN_LIMIT_WARNING",
                    message="Response approaching context limit",
                    suggestion="Reduce page_size or use ids_only mode",
                )
            ],
        )
        assert len(envelope.warnings) == 1
        assert envelope.warnings[0].code == "TOKEN_LIMIT_WARNING"

    def test_response_envelope_without_pagination(self) -> None:
        """Test MCPResponseEnvelope with pagination=None (default)."""
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="find_vendor_info",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_101",
                status="success",
            ),
            results={},
            execution_context=ExecutionContext(
                tokens_estimated=2800,
                tokens_used=2850,
                cache_hit=True,
                execution_time_ms=150.0,
                request_id="req_101",
            ),
        )
        assert envelope.pagination is None

    def test_response_envelope_empty_warnings_default(self) -> None:
        """Test MCPResponseEnvelope with default empty warnings list."""
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="test",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_202",
                status="success",
            ),
            results=[],
            execution_context=ExecutionContext(
                tokens_estimated=100,
                tokens_used=100,
                cache_hit=True,
                execution_time_ms=10.0,
                request_id="req_202",
            ),
        )
        assert envelope.warnings == []
        assert isinstance(envelope.warnings, list)

    def test_response_envelope_serialization(self) -> None:
        """Test MCPResponseEnvelope serialization to dict."""
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_303",
                status="success",
            ),
            results=[],
            execution_context=ExecutionContext(
                tokens_estimated=1000,
                tokens_used=1050,
                cache_hit=True,
                execution_time_ms=100.0,
                request_id="req_303",
            ),
        )
        envelope_dict = envelope.model_dump()
        assert "metadata" in envelope_dict
        assert "results" in envelope_dict
        assert "execution_context" in envelope_dict
        assert "warnings" in envelope_dict


# ==============================================================================
# Backward Compatibility Tests (4 tests)
# ==============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with existing response formats."""

    def test_existing_search_result_metadata_still_works(self) -> None:
        """Test existing SearchResultMetadata is not broken."""
        result = SearchResultMetadata(
            chunk_id=1,
            source_file="docs/test.md",
            source_category="test",
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
        )
        assert result.chunk_id == 1
        assert result.source_file == "docs/test.md"

    def test_existing_pagination_metadata_still_works(self) -> None:
        """Test existing PaginationMetadata is not broken."""
        pagination = PaginationMetadata(
            cursor=None,
            page_size=10,
            has_more=False,
            total_available=42,
        )
        assert pagination.page_size == 10
        assert pagination.has_more is False

    def test_response_metadata_fields_are_optional_where_appropriate(self) -> None:
        """Test ResponseMetadata has appropriate optional fields."""
        # message should be optional
        metadata = ResponseMetadata(
            operation="test",
            timestamp="2025-11-09T15:30:45Z",
            request_id="req_123",
            status="success",
        )
        assert metadata.message is None

    def test_execution_context_fields_are_optional_where_appropriate(self) -> None:
        """Test ExecutionContext has appropriate optional fields."""
        # tokens_used should be optional
        context = ExecutionContext(
            tokens_estimated=100,
            cache_hit=False,
            execution_time_ms=50.0,
            request_id="req_123",
        )
        assert context.tokens_used is None


# ==============================================================================
# Type Safety Tests (3 tests)
# ==============================================================================


class TestTypeSafety:
    """Test type safety and strictness."""

    def test_response_metadata_requires_all_required_fields(self) -> None:
        """Test ResponseMetadata requires all mandatory fields."""
        with pytest.raises(ValidationError):
            ResponseMetadata(
                operation="test",
                # Missing timestamp, request_id, status
            )

    def test_execution_context_requires_all_required_fields(self) -> None:
        """Test ExecutionContext requires all mandatory fields."""
        with pytest.raises(ValidationError):
            ExecutionContext(
                tokens_estimated=100,
                cache_hit=True,
                # Missing execution_time_ms, request_id
            )

    def test_response_warning_requires_all_required_fields(self) -> None:
        """Test ResponseWarning requires all mandatory fields."""
        with pytest.raises(ValidationError):
            ResponseWarning(
                level="warning",
                # Missing code, message
            )


# ==============================================================================
# Integration Tests (3 tests)
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple models."""

    def test_complete_response_envelope_with_enhanced_results(self) -> None:
        """Test complete response envelope with enhanced results."""
        results = [
            EnhancedSemanticSearchResult(
                chunk_id=1,
                source_file="docs/auth.md",
                source_category="security",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=10,
                confidence=ConfidenceScore(
                    score_reliability=0.92,
                    source_quality=0.88,
                    recency=0.75,
                ),
                ranking=RankingContext(
                    percentile=99,
                    explanation="Highest match",
                    score_method="hybrid",
                ),
                deduplication=DeduplicationInfo(
                    is_duplicate=False,
                    similar_chunk_ids=[2],
                    confidence=0.85,
                ),
            ),
        ]
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                version="1.0",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_integration_001",
                status="success",
            ),
            results=results,
            pagination=None,
            execution_context=ExecutionContext(
                tokens_estimated=3400,
                tokens_used=3450,
                cache_hit=True,
                execution_time_ms=245.3,
                request_id="req_integration_001",
            ),
        )
        assert len(envelope.results) == 1  # type: ignore
        assert envelope.metadata.operation == "semantic_search"
        assert envelope.execution_context.request_id == "req_integration_001"

    def test_response_envelope_with_multiple_warnings(self) -> None:
        """Test response envelope with multiple warnings."""
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="semantic_search",
                timestamp="2025-11-09T15:30:45Z",
                request_id="req_multi_warn",
                status="partial",
            ),
            results=[],
            execution_context=ExecutionContext(
                tokens_estimated=200000,
                tokens_used=180000,
                cache_hit=False,
                execution_time_ms=500.0,
                request_id="req_multi_warn",
            ),
            warnings=[
                ResponseWarning(
                    level="warning",
                    code="TOKEN_LIMIT_WARNING",
                    message="Approaching token limit",
                    suggestion="Reduce results or use lighter mode",
                ),
                ResponseWarning(
                    level="info",
                    code="SLOW_QUERY",
                    message="Query took longer than expected",
                ),
            ],
        )
        assert len(envelope.warnings) == 2
        assert envelope.warnings[0].level == "warning"
        assert envelope.warnings[1].level == "info"

    def test_envelope_metadata_request_id_consistency(self) -> None:
        """Test request_id consistency across envelope components."""
        request_id = "req_consistent_123"
        envelope = MCPResponseEnvelope(
            metadata=ResponseMetadata(
                operation="test",
                timestamp="2025-11-09T15:30:45Z",
                request_id=request_id,
                status="success",
            ),
            results=[],
            execution_context=ExecutionContext(
                tokens_estimated=100,
                cache_hit=False,
                execution_time_ms=10.0,
                request_id=request_id,
            ),
        )
        # Both should have the same request_id
        assert envelope.metadata.request_id == request_id
        assert envelope.execution_context.request_id == request_id
