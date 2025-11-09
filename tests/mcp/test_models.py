"""Test Pydantic models for semantic_search tool.

Tests comprehensive validation, serialization, and edge cases for all
request/response models supporting progressive disclosure.

Test Coverage:
- Request schema validation (query, top_k, response_mode)
- Response schema validation (all 4 levels)
- Field constraints (min/max, ranges, types)
- Invalid inputs raise ValidationError
- Serialization/deserialization

Performance:
- Fast unit tests (<100ms total)
- No database dependencies
"""

import pytest
from pydantic import ValidationError

from src.mcp.models import (
    SearchResultFull,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


class TestSemanticSearchRequest:
    """Test request schema validation."""

    def test_valid_request_defaults(self) -> None:
        """Test valid request with default parameters."""
        req = SemanticSearchRequest(query="test query")
        assert req.query == "test query"
        assert req.top_k == 10
        assert req.response_mode == "metadata"

    def test_valid_request_all_params(self) -> None:
        """Test valid request with all parameters specified."""
        req = SemanticSearchRequest(
            query="JWT authentication", top_k=5, response_mode="full"
        )
        assert req.query == "JWT authentication"
        assert req.top_k == 5
        assert req.response_mode == "full"

    def test_valid_request_all_response_modes(self) -> None:
        """Test all valid response_mode values."""
        for mode in ["ids_only", "metadata", "preview", "full"]:
            req = SemanticSearchRequest(query="test", response_mode=mode)  # type: ignore[arg-type]
            assert req.response_mode == mode

    def test_invalid_query_empty(self) -> None:
        """Test empty query raises ValidationError."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            SemanticSearchRequest(query="")

    def test_invalid_query_whitespace_only(self) -> None:
        """Test whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            SemanticSearchRequest(query="   ")

    def test_invalid_query_too_long(self) -> None:
        """Test query exceeding max length raises ValidationError."""
        with pytest.raises(ValidationError, match="at most 500 characters"):
            SemanticSearchRequest(query="a" * 501)

    def test_valid_query_max_length(self) -> None:
        """Test query at exactly max length is valid."""
        req = SemanticSearchRequest(query="a" * 500)
        assert len(req.query) == 500

    def test_invalid_top_k_zero(self) -> None:
        """Test top_k = 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SemanticSearchRequest(query="test", top_k=0)

    def test_invalid_top_k_negative(self) -> None:
        """Test top_k < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SemanticSearchRequest(query="test", top_k=-1)

    def test_invalid_top_k_too_large(self) -> None:
        """Test top_k > 50 raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 50"):
            SemanticSearchRequest(query="test", top_k=51)

    def test_valid_top_k_boundary_values(self) -> None:
        """Test top_k at min and max boundaries."""
        req1 = SemanticSearchRequest(query="test", top_k=1)
        assert req1.top_k == 1
        req2 = SemanticSearchRequest(query="test", top_k=50)
        assert req2.top_k == 50

    def test_invalid_response_mode(self) -> None:
        """Test invalid response_mode raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be"):
            SemanticSearchRequest(query="test", response_mode="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "emoji_query",
        [
            "🔐 authentication",
            "🚀 rocket science",
            "🔑 JWT tokens",
            "🛡️ security best practices",
            "🌐 API gateway",
            "⚡ performance optimization",
            "🐛 debugging techniques",
            "📝 documentation",
            "🔗 blockchain",
            "🤖 machine learning",
        ],
    )
    def test_valid_emoji_queries(self, emoji_query: str) -> None:
        """Test queries with emoji characters are valid."""
        req = SemanticSearchRequest(query=emoji_query)
        assert req.query == emoji_query
        assert len(req.query) > 0

    @pytest.mark.parametrize(
        "multibyte_query",
        [
            "認証 authentication",  # Japanese
            "أمان security",  # Arabic
            "安全 security",  # Chinese Simplified
            "보안 security",  # Korean
            "Безопасность security",  # Russian (Cyrillic)
            "ασφάλεια security",  # Greek
            "אבטחה security",  # Hebrew
            "ตรวจสอบ verification",  # Thai
            "हम authentication",  # Hindi
            "მხარდამჭერი support",  # Georgian
        ],
    )
    def test_valid_multibyte_character_queries(self, multibyte_query: str) -> None:
        """Test queries with multi-byte characters from various languages."""
        req = SemanticSearchRequest(query=multibyte_query)
        assert req.query == multibyte_query
        assert len(req.query) > 0

    @pytest.mark.parametrize(
        "rtl_query",
        [
            "أمان البيانات",  # Arabic: Data security
            "בדיקת אבטחה",  # Hebrew: Security test
            "امنیت سیستم",  # Persian: System security
            "حماية المعلومات",  # Arabic: Information protection
        ],
    )
    def test_valid_rtl_language_queries(self, rtl_query: str) -> None:
        """Test queries with Right-to-Left (RTL) text."""
        req = SemanticSearchRequest(query=rtl_query)
        assert req.query == rtl_query
        # Ensure serialization works
        assert req.model_dump()["query"] == rtl_query

    @pytest.mark.parametrize(
        "mixed_script_query",
        [
            "hello مرحبا 世界",  # English, Arabic, Chinese
            "API الرابط 接口",  # English, Arabic, Chinese
            "JWT токен 토큰",  # English, Russian, Korean
            "🚀 emoji αβγ 中文",  # Emoji, Greek, Chinese
        ],
    )
    def test_valid_mixed_script_queries(self, mixed_script_query: str) -> None:
        """Test queries with mixed scripts (English, Arabic, Chinese, emoji)."""
        req = SemanticSearchRequest(query=mixed_script_query)
        assert req.query == mixed_script_query

    @pytest.mark.parametrize(
        "composed_decomposed",
        [
            ("café", "cafe\u0301"),  # Composed vs decomposed accents
            ("naïve", "nai\u0308ve"),  # Composed vs decomposed diaeresis
        ],
    )
    def test_query_unicode_normalization_variants(
        self, composed_decomposed: tuple[str, str]
    ) -> None:
        """Test that both composed and decomposed Unicode are accepted."""
        composed, decomposed = composed_decomposed

        req_composed = SemanticSearchRequest(query=composed)
        req_decomposed = SemanticSearchRequest(query=decomposed)

        # Both should be valid (normalization handled at search layer)
        assert req_composed.query == composed
        assert req_decomposed.query == decomposed

    def test_query_with_zero_width_characters(self) -> None:
        """Test query with zero-width characters is valid."""
        # Zero-width space: U+200B
        query = "test\u200bquery"
        req = SemanticSearchRequest(query=query)
        assert req.query == query

    def test_query_with_bidi_characters(self) -> None:
        """Test query with bidirectional control characters."""
        # Contains LTR and RTL marks
        query = "hello\u202bمرحبا\u202c"  # LTR hello + RTL Arabic + explicit terminator
        req = SemanticSearchRequest(query=query)
        assert req.query == query

    def test_emoji_with_skin_tone_modifier(self) -> None:
        """Test emoji with skin tone modifiers."""
        query = "👨‍💻‍🔧 developer"  # Emoji with ZWJ and modifier
        req = SemanticSearchRequest(query=query)
        assert req.query == query

    def test_combining_diacritical_marks(self) -> None:
        """Test query with combining diacritical marks."""
        query = "e\u0301\u0302\u0303 test"  # Multiple combining marks
        req = SemanticSearchRequest(query=query)
        assert req.query == query


class TestSearchResultIDs:
    """Test IDs-only result model (Level 0)."""

    def test_valid_result_ids(self) -> None:
        """Test valid SearchResultIDs creation."""
        result = SearchResultIDs(chunk_id=1, hybrid_score=0.85, rank=1)
        assert result.chunk_id == 1
        assert result.hybrid_score == 0.85
        assert result.rank == 1

    def test_invalid_score_too_high(self) -> None:
        """Test hybrid_score > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SearchResultIDs(chunk_id=1, hybrid_score=1.1, rank=1)

    def test_invalid_score_negative(self) -> None:
        """Test hybrid_score < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SearchResultIDs(chunk_id=1, hybrid_score=-0.1, rank=1)

    def test_invalid_rank_zero(self) -> None:
        """Test rank = 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SearchResultIDs(chunk_id=1, hybrid_score=0.85, rank=0)


class TestSearchResultMetadata:
    """Test metadata result model (Level 1)."""

    def test_valid_result_metadata(self) -> None:
        """Test valid SearchResultMetadata creation."""
        result = SearchResultMetadata(
            chunk_id=1,
            source_file="docs/guide.md",
            source_category="guide",
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
        )
        assert result.chunk_id == 1
        assert result.source_file == "docs/guide.md"
        assert result.source_category == "guide"
        assert result.chunk_index == 0
        assert result.total_chunks == 10

    def test_optional_source_category_null(self) -> None:
        """Test source_category can be None."""
        result = SearchResultMetadata(
            chunk_id=1,
            source_file="docs/guide.md",
            source_category=None,
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
        )
        assert result.source_category is None

    def test_invalid_chunk_index_negative(self) -> None:
        """Test chunk_index < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/guide.md",
                source_category="guide",
                hybrid_score=0.85,
                rank=1,
                chunk_index=-1,
                total_chunks=10,
            )

    def test_invalid_total_chunks_zero(self) -> None:
        """Test total_chunks = 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SearchResultMetadata(
                chunk_id=1,
                source_file="docs/guide.md",
                source_category="guide",
                hybrid_score=0.85,
                rank=1,
                chunk_index=0,
                total_chunks=0,
            )


class TestSearchResultPreview:
    """Test preview result model (Level 2)."""

    def test_valid_result_preview(self) -> None:
        """Test valid SearchResultPreview creation."""
        result = SearchResultPreview(
            chunk_id=1,
            source_file="docs/guide.md",
            source_category="guide",
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
            chunk_snippet="This is a preview...",
            context_header="guide.md > Section 1",
        )
        assert result.chunk_snippet == "This is a preview..."
        assert result.context_header == "guide.md > Section 1"

    def test_snippet_can_be_long(self) -> None:
        """Test chunk_snippet can be up to 203 chars (200 + '...')."""
        long_snippet = "a" * 200 + "..."
        result = SearchResultPreview(
            chunk_id=1,
            source_file="docs/guide.md",
            source_category="guide",
            hybrid_score=0.85,
            rank=1,
            chunk_index=0,
            total_chunks=10,
            chunk_snippet=long_snippet,
            context_header="guide.md > Section 1",
        )
        assert len(result.chunk_snippet) == 203


class TestSearchResultFull:
    """Test full result model (Level 3)."""

    def test_valid_result_full(self) -> None:
        """Test valid SearchResultFull creation."""
        result = SearchResultFull(
            chunk_id=1,
            chunk_text="Full chunk content here...",
            similarity_score=0.80,
            bm25_score=0.70,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="docs/guide.md",
            source_category="guide",
            context_header="guide.md > Section 1",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=512,
        )
        assert result.chunk_text == "Full chunk content here..."
        assert result.similarity_score == 0.80
        assert result.bm25_score == 0.70
        assert result.chunk_token_count == 512

    def test_long_chunk_text(self) -> None:
        """Test chunk_text can be very long (1000+ chars)."""
        long_text = "a" * 5000
        result = SearchResultFull(
            chunk_id=1,
            chunk_text=long_text,
            similarity_score=0.80,
            bm25_score=0.70,
            hybrid_score=0.85,
            rank=1,
            score_type="hybrid",
            source_file="docs/guide.md",
            source_category="guide",
            context_header="guide.md > Section 1",
            chunk_index=0,
            total_chunks=10,
            chunk_token_count=1200,
        )
        assert len(result.chunk_text) == 5000

    def test_invalid_similarity_score(self) -> None:
        """Test similarity_score > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SearchResultFull(
                chunk_id=1,
                chunk_text="Full chunk content here...",
                similarity_score=1.5,
                bm25_score=0.70,
                hybrid_score=0.85,
                rank=1,
                score_type="hybrid",
                source_file="docs/guide.md",
                source_category="guide",
                context_header="guide.md > Section 1",
                chunk_index=0,
                total_chunks=10,
                chunk_token_count=512,
            )


class TestSemanticSearchResponse:
    """Test response schema."""

    def test_response_with_ids_results(self) -> None:
        """Test response with IDs-only results."""
        response = SemanticSearchResponse(
            results=[SearchResultIDs(chunk_id=1, hybrid_score=0.85, rank=1)],
            total_found=1,
            strategy_used="hybrid",
            execution_time_ms=100.5,
        )
        assert len(response.results) == 1
        assert response.total_found == 1
        assert response.strategy_used == "hybrid"
        assert response.execution_time_ms == 100.5

    def test_response_with_metadata_results(self) -> None:
        """Test response with metadata-level results."""
        response = SemanticSearchResponse(
            results=[
                SearchResultMetadata(
                    chunk_id=1,
                    source_file="docs/guide.md",
                    source_category="guide",
                    hybrid_score=0.85,
                    rank=1,
                    chunk_index=0,
                    total_chunks=10,
                )
            ],
            total_found=1,
            strategy_used="hybrid",
            execution_time_ms=250.5,
        )
        assert len(response.results) == 1
        assert response.total_found == 1

    def test_response_with_preview_results(self) -> None:
        """Test response with preview-level results."""
        response = SemanticSearchResponse(
            results=[
                SearchResultPreview(
                    chunk_id=1,
                    source_file="docs/guide.md",
                    source_category="guide",
                    hybrid_score=0.85,
                    rank=1,
                    chunk_index=0,
                    total_chunks=10,
                    chunk_snippet="This is a preview...",
                    context_header="guide.md > Section 1",
                )
            ],
            total_found=1,
            strategy_used="hybrid",
            execution_time_ms=300.5,
        )
        assert len(response.results) == 1

    def test_response_with_full_results(self) -> None:
        """Test response with full-level results."""
        response = SemanticSearchResponse(
            results=[
                SearchResultFull(
                    chunk_id=1,
                    chunk_text="Full content here...",
                    similarity_score=0.80,
                    bm25_score=0.70,
                    hybrid_score=0.85,
                    rank=1,
                    score_type="hybrid",
                    source_file="docs/guide.md",
                    source_category="guide",
                    context_header="guide.md > Section 1",
                    chunk_index=0,
                    total_chunks=10,
                    chunk_token_count=512,
                )
            ],
            total_found=1,
            strategy_used="hybrid",
            execution_time_ms=450.5,
        )
        assert len(response.results) == 1

    def test_response_empty_results(self) -> None:
        """Test response with no results."""
        response = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=50.0,
        )
        assert len(response.results) == 0
        assert response.total_found == 0

    def test_invalid_negative_execution_time(self) -> None:
        """Test negative execution_time_ms raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SemanticSearchResponse(
                results=[],
                total_found=0,
                strategy_used="hybrid",
                execution_time_ms=-10.0,
            )

    def test_invalid_negative_total_found(self) -> None:
        """Test negative total_found raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SemanticSearchResponse(
                results=[],
                total_found=-1,
                strategy_used="hybrid",
                execution_time_ms=100.0,
            )
