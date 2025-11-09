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


# ==============================================================================
# PHASE A: VendorInfo Models Tests (Task 10.2)
# ==============================================================================
# Comprehensive validation tests for VendorInfo request/response models
# Test Categories: 60+ test cases covering all models and edge cases
# ==============================================================================


class TestFindVendorInfoRequest:
    """Test FindVendorInfoRequest schema validation (15+ cases)."""

    # ========================================================================
    # Valid Request Cases
    # ========================================================================

    def test_valid_request_minimal(self) -> None:
        """Test valid request with only vendor_name."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(vendor_name="Acme Corp")
        assert req.vendor_name == "Acme Corp"
        assert req.response_mode == "metadata"
        assert req.include_relationships is False

    def test_valid_request_with_all_parameters(self) -> None:
        """Test valid request with all parameters specified."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(
            vendor_name="TechCorp Inc",
            response_mode="full",
            include_relationships=True,
        )
        assert req.vendor_name == "TechCorp Inc"
        assert req.response_mode == "full"
        assert req.include_relationships is True

    @pytest.mark.parametrize(
        "response_mode",
        ["ids_only", "metadata", "preview", "full"],
    )
    def test_valid_request_each_response_mode(
        self, response_mode: str
    ) -> None:
        """Test all valid response_mode values."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(
            vendor_name="Test Vendor", response_mode=response_mode  # type: ignore[arg-type]
        )
        assert req.response_mode == response_mode

    @pytest.mark.parametrize(
        "include_relationships", [True, False]
    )
    def test_valid_request_include_relationships(
        self, include_relationships: bool
    ) -> None:
        """Test both values of include_relationships."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(
            vendor_name="Test Vendor",
            include_relationships=include_relationships,
        )
        assert req.include_relationships == include_relationships

    @pytest.mark.parametrize(
        "unicode_name",
        [
            "你好公司",  # Chinese
            "مرحبا شركة",  # Arabic
            "Привет Компания",  # Russian
            "こんにちは会社",  # Japanese
            "🚀 RocketCorp",  # Emoji
            "Café Société",  # Accents
        ],
    )
    def test_valid_request_unicode_vendor_names(
        self, unicode_name: str
    ) -> None:
        """Test vendor names with unicode and emoji characters."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(vendor_name=unicode_name)
        assert req.vendor_name == unicode_name

    def test_vendor_name_whitespace_stripped(self) -> None:
        """Test that vendor_name is stripped of leading/trailing whitespace."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(vendor_name="  Acme Corp  ")
        assert req.vendor_name == "Acme Corp"

    def test_vendor_name_single_character(self) -> None:
        """Test vendor_name with single character is valid."""
        from src.mcp.models import FindVendorInfoRequest

        req = FindVendorInfoRequest(vendor_name="A")
        assert req.vendor_name == "A"

    def test_vendor_name_exactly_max_length(self) -> None:
        """Test vendor_name at exactly max length (200 chars)."""
        from src.mcp.models import FindVendorInfoRequest

        long_name = "A" * 200
        req = FindVendorInfoRequest(vendor_name=long_name)
        assert len(req.vendor_name) == 200
        assert req.vendor_name == long_name

    # ========================================================================
    # Invalid Request Cases
    # ========================================================================

    def test_invalid_request_empty_vendor_name(self) -> None:
        """Test empty vendor_name raises ValidationError."""
        from src.mcp.models import FindVendorInfoRequest

        with pytest.raises(ValidationError, match="empty or whitespace"):
            FindVendorInfoRequest(vendor_name="")

    def test_invalid_request_whitespace_only_vendor_name(self) -> None:
        """Test whitespace-only vendor_name raises ValidationError."""
        from src.mcp.models import FindVendorInfoRequest

        with pytest.raises(ValidationError, match="empty or whitespace"):
            FindVendorInfoRequest(vendor_name="   ")

    def test_invalid_request_vendor_name_too_long(self) -> None:
        """Test vendor_name exceeding 200 chars raises ValidationError."""
        from src.mcp.models import FindVendorInfoRequest

        with pytest.raises(ValidationError, match="at most 200 characters"):
            FindVendorInfoRequest(vendor_name="A" * 201)

    def test_invalid_request_invalid_response_mode(self) -> None:
        """Test invalid response_mode raises ValidationError."""
        from src.mcp.models import FindVendorInfoRequest

        with pytest.raises(ValidationError):
            FindVendorInfoRequest(
                vendor_name="Test",
                response_mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_invalid_request_wrong_type_response_mode(self) -> None:
        """Test non-string response_mode raises ValidationError."""
        from src.mcp.models import FindVendorInfoRequest

        with pytest.raises(ValidationError):
            FindVendorInfoRequest(
                vendor_name="Test",
                response_mode=123,  # type: ignore[arg-type]
            )


class TestVendorEntity:
    """Test VendorEntity model validation (8+ cases)."""

    # ========================================================================
    # Valid Entity Cases
    # ========================================================================

    def test_valid_entity_minimal(self) -> None:
        """Test valid VendorEntity with required fields only."""
        from src.mcp.models import VendorEntity

        entity = VendorEntity(
            entity_id="vendor_123",
            name="Acme Corporation",
            entity_type="COMPANY",
            confidence=0.95,
        )
        assert entity.entity_id == "vendor_123"
        assert entity.name == "Acme Corporation"
        assert entity.entity_type == "COMPANY"
        assert entity.confidence == 0.95

    def test_valid_entity_with_snippet(self) -> None:
        """Test VendorEntity with snippet field."""
        from src.mcp.models import VendorEntity

        snippet = "Acme Corporation is a leading provider of..." * 2
        entity = VendorEntity(
            entity_id="vendor_456",
            name="TechCorp",
            entity_type="ORGANIZATION",
            confidence=0.85,
            snippet=snippet,
        )
        assert len(entity.snippet) <= 200
        assert "Acme" in entity.snippet or entity.snippet == snippet[:200]

    @pytest.mark.parametrize(
        "confidence",
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    def test_valid_entity_confidence_boundaries(
        self, confidence: float
    ) -> None:
        """Test confidence at all boundary values."""
        from src.mcp.models import VendorEntity

        entity = VendorEntity(
            entity_id="vendor_789",
            name="TestCorp",
            entity_type="BUSINESS",
            confidence=confidence,
        )
        assert entity.confidence == confidence

    # ========================================================================
    # Invalid Entity Cases
    # ========================================================================

    def test_invalid_entity_confidence_negative(self) -> None:
        """Test confidence < 0.0 raises ValidationError."""
        from src.mcp.models import VendorEntity

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            VendorEntity(
                entity_id="vendor_999",
                name="BadCorp",
                entity_type="COMPANY",
                confidence=-0.1,
            )

    def test_invalid_entity_confidence_over_one(self) -> None:
        """Test confidence > 1.0 raises ValidationError."""
        from src.mcp.models import VendorEntity

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            VendorEntity(
                entity_id="vendor_999",
                name="BadCorp",
                entity_type="COMPANY",
                confidence=1.5,
            )

    def test_invalid_entity_snippet_too_long(self) -> None:
        """Test snippet exceeding 200 chars raises ValidationError."""
        from src.mcp.models import VendorEntity

        long_snippet = "A" * 201
        with pytest.raises(ValidationError, match="at most 200 characters"):
            VendorEntity(
                entity_id="vendor_999",
                name="BadCorp",
                entity_type="COMPANY",
                confidence=0.8,
                snippet=long_snippet,
            )


class TestVendorRelationship:
    """Test VendorRelationship model validation (5+ cases)."""

    def test_valid_relationship_minimal(self) -> None:
        """Test valid VendorRelationship with required fields."""
        from src.mcp.models import VendorRelationship

        rel = VendorRelationship(
            source_id="vendor_1",
            target_id="vendor_2",
            relationship_type="PARTNER",
        )
        assert rel.source_id == "vendor_1"
        assert rel.target_id == "vendor_2"
        assert rel.relationship_type == "PARTNER"

    def test_valid_relationship_with_metadata(self) -> None:
        """Test VendorRelationship with metadata dict."""
        from src.mcp.models import VendorRelationship

        metadata = {"strength": 0.9, "established": "2020-01-01"}
        rel = VendorRelationship(
            source_id="vendor_1",
            target_id="vendor_2",
            relationship_type="COMPETITOR",
            metadata=metadata,
        )
        assert rel.metadata == metadata

    def test_valid_relationship_optional_metadata_none(self) -> None:
        """Test VendorRelationship with no metadata."""
        from src.mcp.models import VendorRelationship

        rel = VendorRelationship(
            source_id="vendor_1",
            target_id="vendor_2",
            relationship_type="SUPPLIER",
            metadata=None,
        )
        assert rel.metadata is None

    @pytest.mark.parametrize(
        "rel_type",
        [
            "PARTNER",
            "COMPETITOR",
            "SUPPLIER",
            "CUSTOMER",
            "OWNER",
        ],
    )
    def test_valid_relationship_types(self, rel_type: str) -> None:
        """Test various relationship types."""
        from src.mcp.models import VendorRelationship

        rel = VendorRelationship(
            source_id="vendor_1",
            target_id="vendor_2",
            relationship_type=rel_type,
        )
        assert rel.relationship_type == rel_type


class TestVendorStatistics:
    """Test VendorStatistics model validation (8+ cases)."""

    # ========================================================================
    # Valid Statistics Cases
    # ========================================================================

    def test_valid_statistics_empty(self) -> None:
        """Test VendorStatistics with zero entities and relationships."""
        from src.mcp.models import VendorStatistics

        stats = VendorStatistics(
            entity_count=0,
            relationship_count=0,
        )
        assert stats.entity_count == 0
        assert stats.relationship_count == 0

    def test_valid_statistics_with_distributions(self) -> None:
        """Test VendorStatistics with entity and relationship type counts."""
        from src.mcp.models import VendorStatistics

        entity_types = {"COMPANY": 50, "PERSON": 25, "GPE": 10}
        rel_types = {"PARTNER": 15, "COMPETITOR": 10}
        stats = VendorStatistics(
            entity_count=85,
            relationship_count=25,
            entity_type_distribution=entity_types,
            relationship_type_distribution=rel_types,
        )
        assert stats.entity_count == 85
        assert stats.relationship_count == 25
        assert stats.entity_type_distribution == entity_types
        assert stats.relationship_type_distribution == rel_types

    def test_valid_statistics_large_counts(self) -> None:
        """Test VendorStatistics with very large counts (1M+)."""
        from src.mcp.models import VendorStatistics

        stats = VendorStatistics(
            entity_count=1000000,
            relationship_count=5000000,
        )
        assert stats.entity_count == 1000000
        assert stats.relationship_count == 5000000

    def test_valid_statistics_only_entity_types(self) -> None:
        """Test VendorStatistics with only entity type distribution."""
        from src.mcp.models import VendorStatistics

        entity_types = {"PRODUCT": 25, "SERVICE": 15}
        stats = VendorStatistics(
            entity_count=40,
            relationship_count=10,
            entity_type_distribution=entity_types,
        )
        assert stats.entity_type_distribution == entity_types
        assert stats.relationship_type_distribution is None or stats.relationship_type_distribution == {}

    # ========================================================================
    # Invalid Statistics Cases
    # ========================================================================

    def test_invalid_statistics_negative_entity_count(self) -> None:
        """Test entity_count < 0 raises ValidationError."""
        from src.mcp.models import VendorStatistics

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            VendorStatistics(
                entity_count=-1,
                relationship_count=0,
            )

    def test_invalid_statistics_negative_relationship_count(self) -> None:
        """Test relationship_count < 0 raises ValidationError."""
        from src.mcp.models import VendorStatistics

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            VendorStatistics(
                entity_count=10,
                relationship_count=-5,
            )

    def test_invalid_statistics_invalid_type_distribution(self) -> None:
        """Test non-dict type distribution raises ValidationError."""
        from src.mcp.models import VendorStatistics

        with pytest.raises(ValidationError):
            VendorStatistics(
                entity_count=10,
                relationship_count=5,
                entity_type_distribution="not_a_dict",  # type: ignore[arg-type]
            )


class TestVendorInfoIDs:
    """Test VendorInfoIDs response model (4+ cases)."""

    def test_vendor_info_ids_valid_minimal(self) -> None:
        """Test valid VendorInfoIDs with required fields."""
        from src.mcp.models import VendorInfoIDs

        response = VendorInfoIDs(
            vendor_name="Acme Corp",
            entity_ids=["vendor_1", "vendor_2"],
        )
        assert response.vendor_name == "Acme Corp"
        assert len(response.entity_ids) == 2

    def test_vendor_info_ids_with_relationship_ids(self) -> None:
        """Test VendorInfoIDs with relationship IDs."""
        from src.mcp.models import VendorInfoIDs

        response = VendorInfoIDs(
            vendor_name="TechCorp",
            entity_ids=["e1", "e2", "e3"],
            relationship_ids=["r1", "r2"],
        )
        assert len(response.entity_ids) == 3
        assert len(response.relationship_ids) == 2

    def test_vendor_info_ids_empty_ids(self) -> None:
        """Test VendorInfoIDs with empty ID lists."""
        from src.mcp.models import VendorInfoIDs

        response = VendorInfoIDs(
            vendor_name="UnknownVendor",
            entity_ids=[],
            relationship_ids=[],
        )
        assert len(response.entity_ids) == 0
        assert len(response.relationship_ids) == 0


class TestVendorInfoMetadata:
    """Test VendorInfoMetadata response model (5+ cases)."""

    def test_vendor_info_metadata_valid_minimal(self) -> None:
        """Test valid VendorInfoMetadata with required fields."""
        from src.mcp.models import VendorInfoMetadata, VendorStatistics

        stats = VendorStatistics(entity_count=10, relationship_count=5)
        response = VendorInfoMetadata(
            vendor_name="Acme",
            statistics=stats,
        )
        assert response.vendor_name == "Acme"
        assert response.statistics.entity_count == 10

    def test_vendor_info_metadata_with_all_fields(self) -> None:
        """Test VendorInfoMetadata with all optional fields."""
        from src.mcp.models import (
            VendorInfoMetadata,
            VendorStatistics,
            VendorEntity,
        )

        stats = VendorStatistics(entity_count=20, relationship_count=10)
        entities = [
            VendorEntity(
                entity_id="e1",
                name="Entity1",
                entity_type="COMPANY",
                confidence=0.9,
            )
        ]
        response = VendorInfoMetadata(
            vendor_name="TechCorp",
            statistics=stats,
            top_entities=entities,
            last_updated="2025-11-09T00:00:00Z",
        )
        assert response.vendor_name == "TechCorp"
        assert len(response.top_entities) == 1

    def test_vendor_info_metadata_optional_fields_none(self) -> None:
        """Test VendorInfoMetadata with optional fields as None."""
        from src.mcp.models import VendorInfoMetadata, VendorStatistics

        stats = VendorStatistics(entity_count=5, relationship_count=2)
        response = VendorInfoMetadata(
            vendor_name="SimpleCorp",
            statistics=stats,
            top_entities=None,
            last_updated=None,
        )
        assert response.top_entities is None
        assert response.last_updated is None


class TestVendorInfoPreview:
    """Test VendorInfoPreview response model (5+ cases)."""

    def test_vendor_info_preview_valid_basic(self) -> None:
        """Test valid VendorInfoPreview with basic data."""
        from src.mcp.models import (
            VendorInfoPreview,
            VendorEntity,
            VendorRelationship,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=5, relationship_count=2)
        entities = [
            VendorEntity(
                entity_id="e1",
                name="Company1",
                entity_type="COMPANY",
                confidence=0.9,
            )
        ]
        rels = [
            VendorRelationship(
                source_id="e1",
                target_id="e2",
                relationship_type="PARTNER",
            )
        ]
        response = VendorInfoPreview(
            vendor_name="Preview Corp",
            entities=entities,
            relationships=rels,
            statistics=stats,
        )
        assert response.vendor_name == "Preview Corp"
        assert len(response.entities) == 1
        assert len(response.relationships) == 1

    def test_vendor_info_preview_max_entities(self) -> None:
        """Test VendorInfoPreview with exactly 5 entities."""
        from src.mcp.models import (
            VendorInfoPreview,
            VendorEntity,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=5, relationship_count=0)
        entities = [
            VendorEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                entity_type="COMPANY",
                confidence=0.9,
            )
            for i in range(5)
        ]
        response = VendorInfoPreview(
            vendor_name="Test",
            entities=entities,
            relationships=[],
            statistics=stats,
        )
        assert len(response.entities) == 5

    def test_vendor_info_preview_entity_count_validation(self) -> None:
        """Test that VendorInfoPreview enforces max 5 entities."""
        from src.mcp.models import (
            VendorInfoPreview,
            VendorEntity,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=6, relationship_count=0)
        entities = [
            VendorEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                entity_type="COMPANY",
                confidence=0.9,
            )
            for i in range(6)
        ]
        with pytest.raises(ValidationError, match="at most 5"):
            VendorInfoPreview(
                vendor_name="Test",
                entities=entities,
                relationships=[],
                statistics=stats,
            )


class TestVendorInfoFull:
    """Test VendorInfoFull response model (6+ cases)."""

    def test_vendor_info_full_valid_basic(self) -> None:
        """Test valid VendorInfoFull with all data."""
        from src.mcp.models import (
            VendorInfoFull,
            VendorEntity,
            VendorRelationship,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=10, relationship_count=5)
        entities = [
            VendorEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                entity_type="COMPANY",
                confidence=0.9,
            )
            for i in range(10)
        ]
        rels = [
            VendorRelationship(
                source_id=f"e{i}",
                target_id=f"e{i+1}",
                relationship_type="PARTNER",
            )
            for i in range(5)
        ]
        response = VendorInfoFull(
            vendor_name="Full Corp",
            entities=entities,
            relationships=rels,
            statistics=stats,
        )
        assert response.vendor_name == "Full Corp"
        assert len(response.entities) == 10
        assert len(response.relationships) == 5

    def test_vendor_info_full_max_entities(self) -> None:
        """Test VendorInfoFull with exactly 100 entities."""
        from src.mcp.models import (
            VendorInfoFull,
            VendorEntity,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=100, relationship_count=0)
        entities = [
            VendorEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                entity_type="COMPANY",
                confidence=0.9,
            )
            for i in range(100)
        ]
        response = VendorInfoFull(
            vendor_name="Large Corp",
            entities=entities,
            relationships=[],
            statistics=stats,
        )
        assert len(response.entities) == 100

    def test_vendor_info_full_max_relationships(self) -> None:
        """Test VendorInfoFull with exactly 500 relationships."""
        from src.mcp.models import (
            VendorInfoFull,
            VendorRelationship,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=100, relationship_count=500)
        rels = [
            VendorRelationship(
                source_id=f"e{i % 100}",
                target_id=f"e{(i+1) % 100}",
                relationship_type="PARTNER",
            )
            for i in range(500)
        ]
        response = VendorInfoFull(
            vendor_name="Connected Corp",
            entities=[],
            relationships=rels,
            statistics=stats,
        )
        assert len(response.relationships) == 500

    def test_vendor_info_full_entity_count_exceeded(self) -> None:
        """Test that VendorInfoFull enforces max 100 entities."""
        from src.mcp.models import (
            VendorInfoFull,
            VendorEntity,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=101, relationship_count=0)
        entities = [
            VendorEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                entity_type="COMPANY",
                confidence=0.9,
            )
            for i in range(101)
        ]
        with pytest.raises(ValidationError, match="at most 100"):
            VendorInfoFull(
                vendor_name="Too Large",
                entities=entities,
                relationships=[],
                statistics=stats,
            )

    def test_vendor_info_full_relationship_count_exceeded(self) -> None:
        """Test that VendorInfoFull enforces max 500 relationships."""
        from src.mcp.models import (
            VendorInfoFull,
            VendorRelationship,
            VendorStatistics,
        )

        stats = VendorStatistics(entity_count=10, relationship_count=501)
        rels = [
            VendorRelationship(
                source_id=f"e{i % 10}",
                target_id=f"e{(i+1) % 10}",
                relationship_type="PARTNER",
            )
            for i in range(501)
        ]
        with pytest.raises(ValidationError, match="at most 500"):
            VendorInfoFull(
                vendor_name="Too Connected",
                entities=[],
                relationships=rels,
                statistics=stats,
            )


class TestAuthenticationError:
    """Test AuthenticationError model (3+ cases)."""

    def test_authentication_error_with_details(self) -> None:
        """Test AuthenticationError with error details."""
        from src.mcp.models import AuthenticationError

        error = AuthenticationError(
            error_code="AUTH_001",
            message="Invalid API key",
            details="API key expired on 2025-11-01",
        )
        assert error.error_code == "AUTH_001"
        assert error.message == "Invalid API key"
        assert error.details == "API key expired on 2025-11-01"

    def test_authentication_error_minimal(self) -> None:
        """Test AuthenticationError with only required fields."""
        from src.mcp.models import AuthenticationError

        error = AuthenticationError(
            error_code="AUTH_002",
            message="Access denied",
        )
        assert error.error_code == "AUTH_002"
        assert error.message == "Access denied"
        assert error.details is None or error.details == ""


class TestAuthenticationConfig:
    """Test AuthenticationConfig model (4+ cases)."""

    def test_authentication_config_defaults(self) -> None:
        """Test AuthenticationConfig with default values."""
        from src.mcp.models import AuthenticationConfig

        config = AuthenticationConfig()
        assert config.max_auth_attempts >= 1
        assert config.token_expiry_seconds >= 1
        assert config.rate_limit_per_minute >= 1

    def test_authentication_config_custom_limits(self) -> None:
        """Test AuthenticationConfig with custom values."""
        from src.mcp.models import AuthenticationConfig

        config = AuthenticationConfig(
            max_auth_attempts=10,
            token_expiry_seconds=3600,
            rate_limit_per_minute=100,
        )
        assert config.max_auth_attempts == 10
        assert config.token_expiry_seconds == 3600
        assert config.rate_limit_per_minute == 100

    def test_authentication_config_invalid_negative_limits(self) -> None:
        """Test that negative limits raise ValidationError."""
        from src.mcp.models import AuthenticationConfig

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AuthenticationConfig(
                max_auth_attempts=-1,
                token_expiry_seconds=3600,
                rate_limit_per_minute=100,
            )
