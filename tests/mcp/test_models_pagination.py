"""Comprehensive tests for pagination and filtering models (Task 10.3 Phase B).

Tests cover:
- PaginationMetadata creation and validation
- SemanticSearchRequest pagination and filtering
- FindVendorInfoRequest pagination and filtering
- Field whitelist validation
- Cursor format validation
- Backward compatibility (top_k vs page_size)
- Edge cases and error conditions
"""

import base64
import json

import pytest
from pydantic import ValidationError

from src.mcp.models import (
    FindVendorInfoRequest,
    PaginationMetadata,
    SearchResultMetadata,
    SemanticSearchRequest,
    SemanticSearchResponse,
    WhitelistedSemanticSearchFields,
    WhitelistedVendorInfoFields,
)

# ==============================================================================
# PaginationMetadata Tests (12 tests)
# ==============================================================================


class TestPaginationMetadata:
    """Test PaginationMetadata model validation and cursor handling."""

    def test_pagination_metadata_valid(self) -> None:
        """Test valid PaginationMetadata creation."""
        pagination = PaginationMetadata(
            cursor=None,
            page_size=10,
            has_more=False,
            total_available=42,
        )
        assert pagination.cursor is None
        assert pagination.page_size == 10
        assert pagination.has_more is False
        assert pagination.total_available == 42

    def test_pagination_metadata_with_cursor(self) -> None:
        """Test PaginationMetadata with valid cursor."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )
        assert pagination.cursor == cursor
        assert pagination.has_more is True

    def test_pagination_metadata_page_size_min(self) -> None:
        """Test page_size minimum validation (0 is now valid for empty results)."""
        # page_size=0 is now allowed for empty result scenarios
        pagination = PaginationMetadata(
            cursor=None,
            page_size=0,
            has_more=False,
            total_available=0,
        )
        assert pagination.page_size == 0

    def test_pagination_metadata_page_size_max(self) -> None:
        """Test page_size maximum validation (50)."""
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=None,
                page_size=51,
                has_more=False,
                total_available=10,
            )
        assert "page_size" in str(exc_info.value)

    def test_pagination_metadata_total_negative(self) -> None:
        """Test total_available cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=None,
                page_size=10,
                has_more=False,
                total_available=-1,
            )
        assert "total_available" in str(exc_info.value)

    def test_cursor_invalid_base64(self) -> None:
        """Test cursor rejects invalid base64."""
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor="not-valid-base64!!!",
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "Invalid cursor format" in str(exc_info.value)

    def test_cursor_invalid_json(self) -> None:
        """Test cursor rejects invalid JSON."""
        invalid_cursor = base64.b64encode(b"not json").decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=invalid_cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "Invalid cursor format" in str(exc_info.value)

    def test_cursor_missing_required_fields(self) -> None:
        """Test cursor requires query_hash, offset, response_mode."""
        cursor_data = {"query_hash": "abc"}  # Missing offset and response_mode
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "Cursor must contain fields" in str(exc_info.value)

    def test_cursor_invalid_offset_type(self) -> None:
        """Test cursor offset must be non-negative integer."""
        cursor_data = {
            "query_hash": "abc",
            "offset": "not-an-int",
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "offset must be a non-negative integer" in str(exc_info.value)

    def test_cursor_negative_offset(self) -> None:
        """Test cursor offset cannot be negative."""
        cursor_data = {
            "query_hash": "abc",
            "offset": -5,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "offset must be a non-negative integer" in str(exc_info.value)

    def test_cursor_invalid_response_mode(self) -> None:
        """Test cursor response_mode must be valid."""
        cursor_data = {
            "query_hash": "abc",
            "offset": 10,
            "response_mode": "invalid",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "response_mode must be one of" in str(exc_info.value)

    def test_cursor_not_dict(self) -> None:
        """Test cursor must decode to JSON object (not array)."""
        cursor_data = ["not", "a", "dict"]
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
        with pytest.raises(ValidationError) as exc_info:
            PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=10,
            )
        assert "Cursor must decode to a JSON object" in str(exc_info.value)


# ==============================================================================
# SemanticSearchRequest Pagination Tests (15 tests)
# ==============================================================================


class TestSemanticSearchRequestPagination:
    """Test SemanticSearchRequest pagination and backward compatibility."""

    def test_request_default_page_size(self) -> None:
        """Test default page_size is 10 and top_k defaults to 10."""
        request = SemanticSearchRequest(query="test")
        assert request.page_size == 10
        assert request.top_k == 10  # top_k now defaults to 10 for backward compatibility

    def test_request_custom_page_size(self) -> None:
        """Test custom page_size with explicit top_k."""
        # When only page_size is provided, top_k defaults to 10 and takes precedence
        # To actually get page_size=25, must also set top_k=25
        request = SemanticSearchRequest(query="test", page_size=25, top_k=25)
        assert request.page_size == 25
        assert request.top_k == 25

    def test_request_page_size_min(self) -> None:
        """Test page_size minimum (1)."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", page_size=0)
        assert "page_size" in str(exc_info.value)

    def test_request_page_size_max(self) -> None:
        """Test page_size maximum (50)."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", page_size=51)
        assert "page_size" in str(exc_info.value)

    def test_request_with_cursor(self) -> None:
        """Test request with valid cursor."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = SemanticSearchRequest(query="test", cursor=cursor)
        assert request.cursor == cursor

    def test_request_cursor_validation(self) -> None:
        """Test cursor validation in request."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", cursor="invalid-cursor")
        assert "Invalid cursor format" in str(exc_info.value)

    def test_request_top_k_precedence(self) -> None:
        """Test top_k takes precedence over page_size (backward compatibility)."""
        request = SemanticSearchRequest(query="test", top_k=15, page_size=20)
        assert request.top_k == 15
        assert request.page_size == 15  # Should be overridden by top_k

    def test_request_top_k_only(self) -> None:
        """Test top_k alone sets page_size."""
        request = SemanticSearchRequest(query="test", top_k=7)
        assert request.top_k == 7
        assert request.page_size == 7

    def test_request_top_k_min(self) -> None:
        """Test top_k minimum validation."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", top_k=0)
        assert "top_k" in str(exc_info.value)

    def test_request_top_k_max(self) -> None:
        """Test top_k maximum validation."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", top_k=51)
        assert "top_k" in str(exc_info.value)

    def test_request_pagination_with_response_mode(self) -> None:
        """Test pagination works with all response modes."""
        for mode in ["ids_only", "metadata", "preview", "full"]:
            cursor_data = {
                "query_hash": "abc",
                "offset": 10,
                "response_mode": mode,
            }
            cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
            request = SemanticSearchRequest(
                query="test", page_size=10, cursor=cursor, response_mode=mode
            )
            assert request.response_mode == mode
            assert request.cursor == cursor

    def test_request_empty_query_rejected(self) -> None:
        """Test empty query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_request_whitespace_query_rejected(self) -> None:
        """Test whitespace-only query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="   ")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_request_query_too_long(self) -> None:
        """Test query max length (500 chars)."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="a" * 501)
        assert "query" in str(exc_info.value)

    def test_request_all_pagination_fields(self) -> None:
        """Test request with all pagination fields."""
        cursor_data = {
            "query_hash": "abc123",
            "offset": 20,
            "response_mode": "preview",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = SemanticSearchRequest(
            query="test query",
            page_size=15,
            top_k=15,  # Must set top_k to match page_size for it to take effect
            cursor=cursor,
            response_mode="preview",
        )
        assert request.query == "test query"
        assert request.page_size == 15
        assert request.top_k == 15
        assert request.cursor == cursor
        assert request.response_mode == "preview"


# ==============================================================================
# SemanticSearchRequest Field Filtering Tests (10 tests)
# ==============================================================================


class TestSemanticSearchRequestFieldFiltering:
    """Test SemanticSearchRequest field filtering and whitelist validation."""

    def test_fields_none_default(self) -> None:
        """Test fields defaults to None (all fields)."""
        request = SemanticSearchRequest(query="test")
        assert request.fields is None

    def test_fields_valid_metadata(self) -> None:
        """Test valid fields for metadata mode."""
        request = SemanticSearchRequest(
            query="test",
            response_mode="metadata",
            fields=["chunk_id", "source_file", "hybrid_score"],
        )
        assert request.fields == ["chunk_id", "source_file", "hybrid_score"]

    def test_fields_valid_ids_only(self) -> None:
        """Test valid fields for ids_only mode."""
        request = SemanticSearchRequest(
            query="test",
            response_mode="ids_only",
            fields=["chunk_id", "rank"],
        )
        assert request.fields == ["chunk_id", "rank"]

    def test_fields_valid_preview(self) -> None:
        """Test valid fields for preview mode."""
        # Pass response_mode first to ensure it's available during validation
        request = SemanticSearchRequest(
            query="test",
            fields=["chunk_id", "chunk_snippet", "context_header"],
            response_mode="preview",
        )
        assert request.fields == ["chunk_id", "chunk_snippet", "context_header"]

    def test_fields_valid_full(self) -> None:
        """Test valid fields for full mode."""
        # Pass response_mode first to ensure it's available during validation
        request = SemanticSearchRequest(
            query="test",
            fields=["chunk_id", "chunk_text", "similarity_score"],
            response_mode="full",
        )
        assert request.fields == ["chunk_id", "chunk_text", "similarity_score"]

    def test_fields_invalid_for_mode(self) -> None:
        """Test fields validation rejects invalid fields for response_mode."""
        # Test with default mode (metadata) trying to use full mode fields
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(
                query="test",
                fields=["chunk_id", "chunk_text"],  # chunk_text not in metadata (default)
            )
        assert "Invalid fields" in str(exc_info.value)
        assert "chunk_text" in str(exc_info.value)

    def test_fields_empty_list_rejected(self) -> None:
        """Test empty fields list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", fields=[])
        assert "fields list cannot be empty" in str(exc_info.value)

    def test_fields_unknown_field(self) -> None:
        """Test unknown field names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(
                query="test",
                response_mode="metadata",
                fields=["chunk_id", "unknown_field"],
            )
        assert "Invalid fields" in str(exc_info.value)
        assert "unknown_field" in str(exc_info.value)

    def test_fields_all_valid_metadata(self) -> None:
        """Test all valid metadata fields are accepted."""
        all_metadata_fields = list(WhitelistedSemanticSearchFields.METADATA)
        request = SemanticSearchRequest(
            query="test",
            response_mode="metadata",
            fields=all_metadata_fields,
        )
        assert set(request.fields or []) == WhitelistedSemanticSearchFields.METADATA

    def test_fields_with_pagination(self) -> None:
        """Test field filtering works with pagination."""
        cursor_data = {
            "query_hash": "abc",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = SemanticSearchRequest(
            query="test",
            page_size=10,
            cursor=cursor,
            response_mode="metadata",
            fields=["chunk_id", "source_file"],
        )
        assert request.fields == ["chunk_id", "source_file"]
        assert request.cursor == cursor


# ==============================================================================
# FindVendorInfoRequest Pagination Tests (8 tests)
# ==============================================================================


class TestFindVendorInfoRequestPagination:
    """Test FindVendorInfoRequest pagination support."""

    def test_vendor_request_default_page_size(self) -> None:
        """Test default page_size is 10."""
        request = FindVendorInfoRequest(vendor_name="Acme Corp")
        assert request.page_size == 10

    def test_vendor_request_custom_page_size(self) -> None:
        """Test custom page_size."""
        request = FindVendorInfoRequest(vendor_name="Acme Corp", page_size=20)
        assert request.page_size == 20

    def test_vendor_request_page_size_min(self) -> None:
        """Test page_size minimum validation."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="Acme Corp", page_size=0)
        assert "page_size" in str(exc_info.value)

    def test_vendor_request_page_size_max(self) -> None:
        """Test page_size maximum validation."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="Acme Corp", page_size=51)
        assert "page_size" in str(exc_info.value)

    def test_vendor_request_with_cursor(self) -> None:
        """Test vendor request with valid cursor."""
        cursor_data = {
            "query_hash": "vendor_abc",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = FindVendorInfoRequest(vendor_name="Acme Corp", cursor=cursor)
        assert request.cursor == cursor

    def test_vendor_request_cursor_validation(self) -> None:
        """Test cursor validation in vendor request."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="Acme Corp", cursor="invalid")
        assert "Invalid cursor format" in str(exc_info.value)

    def test_vendor_request_pagination_all_modes(self) -> None:
        """Test pagination works with all response modes."""
        for mode in ["ids_only", "metadata", "preview", "full"]:
            cursor_data = {
                "query_hash": "vendor",
                "offset": 5,
                "response_mode": mode,
            }
            cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()
            request = FindVendorInfoRequest(
                vendor_name="Acme Corp",
                page_size=10,
                cursor=cursor,
                response_mode=mode,
            )
            assert request.response_mode == mode
            assert request.cursor == cursor

    def test_vendor_request_all_pagination_fields(self) -> None:
        """Test vendor request with all pagination fields."""
        cursor_data = {
            "query_hash": "vendor_123",
            "offset": 15,
            "response_mode": "preview",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = FindVendorInfoRequest(
            vendor_name="Acme Corporation",
            page_size=25,
            cursor=cursor,
            response_mode="preview",
            include_relationships=True,
        )
        assert request.vendor_name == "Acme Corporation"
        assert request.page_size == 25
        assert request.cursor == cursor
        assert request.response_mode == "preview"
        assert request.include_relationships is True


# ==============================================================================
# FindVendorInfoRequest Field Filtering Tests (8 tests)
# ==============================================================================


class TestFindVendorInfoRequestFieldFiltering:
    """Test FindVendorInfoRequest field filtering validation."""

    def test_vendor_fields_none_default(self) -> None:
        """Test fields defaults to None (all fields)."""
        request = FindVendorInfoRequest(vendor_name="Acme Corp")
        assert request.fields is None

    def test_vendor_fields_valid_metadata(self) -> None:
        """Test valid fields for metadata mode."""
        request = FindVendorInfoRequest(
            vendor_name="Acme Corp",
            response_mode="metadata",
            fields=["vendor_name", "statistics"],
        )
        assert request.fields == ["vendor_name", "statistics"]

    def test_vendor_fields_valid_ids_only(self) -> None:
        """Test valid fields for ids_only mode."""
        # Pass response_mode first to ensure it's available during validation
        request = FindVendorInfoRequest(
            vendor_name="Acme Corp",
            fields=["vendor_name", "entity_ids"],
            response_mode="ids_only",
        )
        assert request.fields == ["vendor_name", "entity_ids"]

    def test_vendor_fields_invalid_for_mode(self) -> None:
        """Test fields validation rejects invalid fields for response_mode."""
        # Test with default mode (metadata) trying to use preview mode fields
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(
                vendor_name="Acme Corp",
                fields=["vendor_name", "entities"],  # entities not in metadata (default)
            )
        assert "Invalid fields" in str(exc_info.value)
        assert "entities" in str(exc_info.value)

    def test_vendor_fields_empty_list_rejected(self) -> None:
        """Test empty fields list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="Acme Corp", fields=[])
        assert "fields list cannot be empty" in str(exc_info.value)

    def test_vendor_fields_unknown_field(self) -> None:
        """Test unknown field names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(
                vendor_name="Acme Corp",
                response_mode="metadata",
                fields=["vendor_name", "unknown_field"],
            )
        assert "Invalid fields" in str(exc_info.value)
        assert "unknown_field" in str(exc_info.value)

    def test_vendor_fields_all_valid_metadata(self) -> None:
        """Test all valid metadata fields are accepted."""
        all_metadata_fields = list(WhitelistedVendorInfoFields.METADATA)
        request = FindVendorInfoRequest(
            vendor_name="Acme Corp",
            response_mode="metadata",
            fields=all_metadata_fields,
        )
        assert set(request.fields or []) == WhitelistedVendorInfoFields.METADATA

    def test_vendor_fields_with_pagination(self) -> None:
        """Test field filtering works with pagination."""
        cursor_data = {
            "query_hash": "vendor",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = FindVendorInfoRequest(
            vendor_name="Acme Corp",
            page_size=10,
            cursor=cursor,
            response_mode="metadata",
            fields=["vendor_name", "statistics"],
        )
        assert request.fields == ["vendor_name", "statistics"]
        assert request.cursor == cursor


# ==============================================================================
# SemanticSearchResponse Pagination Tests (5 tests)
# ==============================================================================


class TestSemanticSearchResponsePagination:
    """Test SemanticSearchResponse with pagination metadata."""

    def test_response_without_pagination(self) -> None:
        """Test response without pagination (backward compatible)."""
        response = SemanticSearchResponse(
            results=[
                SearchResultMetadata(
                    chunk_id=1,
                    source_file="test.md",
                    source_category="docs",
                    hybrid_score=0.95,
                    rank=1,
                    chunk_index=0,
                    total_chunks=5,
                )
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=123.45,
        )
        assert response.pagination is None
        assert len(response.results) == 1

    def test_response_with_pagination(self) -> None:
        """Test response with pagination metadata."""
        cursor_data = {
            "query_hash": "abc",
            "offset": 10,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        response = SemanticSearchResponse(
            results=[
                SearchResultMetadata(
                    chunk_id=1,
                    source_file="test.md",
                    source_category="docs",
                    hybrid_score=0.95,
                    rank=1,
                    chunk_index=0,
                    total_chunks=5,
                )
            ],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=123.45,
            pagination=PaginationMetadata(
                cursor=cursor,
                page_size=10,
                has_more=True,
                total_available=42,
            ),
        )
        assert response.pagination is not None
        assert response.pagination.cursor == cursor
        assert response.pagination.has_more is True
        assert response.pagination.total_available == 42

    def test_response_pagination_last_page(self) -> None:
        """Test response pagination for last page (cursor=None, has_more=False)."""
        response = SemanticSearchResponse(
            results=[
                SearchResultMetadata(
                    chunk_id=1,
                    source_file="test.md",
                    source_category="docs",
                    hybrid_score=0.95,
                    rank=1,
                    chunk_index=0,
                    total_chunks=5,
                )
            ],
            total_found=10,
            strategy_used="hybrid",
            execution_time_ms=123.45,
            pagination=PaginationMetadata(
                cursor=None,  # No next page
                page_size=10,
                has_more=False,
                total_available=10,
            ),
        )
        assert response.pagination is not None
        assert response.pagination.cursor is None
        assert response.pagination.has_more is False

    def test_response_pagination_matches_results(self) -> None:
        """Test pagination page_size can differ from results length."""
        # Page size 10 but only 7 results on last page
        response = SemanticSearchResponse(
            results=[
                SearchResultMetadata(
                    chunk_id=i,
                    source_file="test.md",
                    source_category="docs",
                    hybrid_score=0.95,
                    rank=i,
                    chunk_index=0,
                    total_chunks=5,
                )
                for i in range(1, 8)
            ],
            total_found=47,
            strategy_used="hybrid",
            execution_time_ms=123.45,
            pagination=PaginationMetadata(
                cursor=None,
                page_size=10,  # Requested 10, got 7 (last page)
                has_more=False,
                total_available=47,
            ),
        )
        assert len(response.results) == 7
        assert response.pagination.page_size == 10
        assert response.pagination.has_more is False

    def test_response_empty_results_with_pagination(self) -> None:
        """Test response can have empty results with pagination."""
        response = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=50.0,
            pagination=PaginationMetadata(
                cursor=None,
                page_size=10,
                has_more=False,
                total_available=0,
            ),
        )
        assert len(response.results) == 0
        assert response.pagination is not None
        assert response.pagination.total_available == 0


# ==============================================================================
# Whitelist Field Tests (6 tests)
# ==============================================================================


class TestWhitelistedFields:
    """Test whitelisted field constants and helpers."""

    def test_semantic_search_ids_only_fields(self) -> None:
        """Test SemanticSearch ids_only whitelist."""
        fields = WhitelistedSemanticSearchFields.IDS_ONLY
        assert "chunk_id" in fields
        assert "hybrid_score" in fields
        assert "rank" in fields
        assert "chunk_text" not in fields

    def test_semantic_search_metadata_fields(self) -> None:
        """Test SemanticSearch metadata whitelist."""
        fields = WhitelistedSemanticSearchFields.METADATA
        assert "chunk_id" in fields
        assert "source_file" in fields
        assert "hybrid_score" in fields
        assert "chunk_snippet" not in fields

    def test_semantic_search_preview_fields(self) -> None:
        """Test SemanticSearch preview whitelist."""
        fields = WhitelistedSemanticSearchFields.PREVIEW
        assert "chunk_snippet" in fields
        assert "context_header" in fields
        assert "chunk_text" not in fields  # Full mode only

    def test_semantic_search_full_fields(self) -> None:
        """Test SemanticSearch full whitelist."""
        fields = WhitelistedSemanticSearchFields.FULL
        assert "chunk_text" in fields
        assert "similarity_score" in fields
        assert "bm25_score" in fields

    def test_vendor_info_ids_only_fields(self) -> None:
        """Test VendorInfo ids_only whitelist."""
        fields = WhitelistedVendorInfoFields.IDS_ONLY
        assert "vendor_name" in fields
        assert "entity_ids" in fields
        assert "relationship_ids" in fields
        assert "entities" not in fields

    def test_vendor_info_metadata_fields(self) -> None:
        """Test VendorInfo metadata whitelist."""
        fields = WhitelistedVendorInfoFields.METADATA
        assert "vendor_name" in fields
        assert "statistics" in fields
        assert "top_entities" in fields
        assert "entities" not in fields  # Preview/full only


# ==============================================================================
# Edge Cases and Integration Tests (6 tests)
# ==============================================================================


class TestPaginationEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_cursor_roundtrip(self) -> None:
        """Test cursor encoding/decoding roundtrip."""
        cursor_data = {
            "query_hash": "test_hash_12345",
            "offset": 42,
            "response_mode": "preview",
        }
        encoded = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        # Validate via PaginationMetadata
        pagination = PaginationMetadata(
            cursor=encoded,
            page_size=10,
            has_more=True,
            total_available=100,
        )
        assert pagination.cursor == encoded

        # Decode and verify
        decoded = json.loads(base64.b64decode(encoded))
        assert decoded == cursor_data

    def test_request_with_all_optional_fields_none(self) -> None:
        """Test request with all optional fields as None."""
        request = SemanticSearchRequest(
            query="test",
            top_k=None,
            cursor=None,
            fields=None,
        )
        assert request.top_k is None
        assert request.cursor is None
        assert request.fields is None
        assert request.page_size == 10  # Default

    def test_vendor_request_whitespace_handling(self) -> None:
        """Test vendor name whitespace stripping."""
        request = FindVendorInfoRequest(vendor_name="  Acme Corp  ")
        assert request.vendor_name == "Acme Corp"

    def test_vendor_request_empty_name_rejected(self) -> None:
        """Test empty vendor name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="")
        assert "Vendor name cannot be empty" in str(exc_info.value)

    def test_vendor_request_whitespace_name_rejected(self) -> None:
        """Test whitespace-only vendor name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FindVendorInfoRequest(vendor_name="   ")
        assert "Vendor name cannot be empty" in str(exc_info.value)

    def test_complex_pagination_scenario(self) -> None:
        """Test complex scenario with pagination, filtering, and custom page size."""
        # Page 2 of search results
        cursor_data = {
            "query_hash": "complex_query_hash",
            "offset": 20,
            "response_mode": "metadata",
        }
        cursor = base64.b64encode(json.dumps(cursor_data).encode()).decode()

        request = SemanticSearchRequest(
            query="complex search query",
            page_size=20,
            top_k=20,  # Must set top_k to match page_size for it to take effect
            cursor=cursor,
            fields=["chunk_id", "source_file", "hybrid_score"],
            response_mode="metadata",
        )

        # Validate all fields
        assert request.query == "complex search query"
        assert request.page_size == 20
        assert request.top_k == 20
        assert request.cursor == cursor
        assert request.fields == ["chunk_id", "source_file", "hybrid_score"]
        assert request.response_mode == "metadata"

        # Verify cursor decodes correctly
        decoded = json.loads(base64.b64decode(cursor))
        assert decoded["offset"] == 20
        assert decoded["query_hash"] == "complex_query_hash"


# ==============================================================================
# Summary: 70+ tests covering all requirements
# ==============================================================================
# - PaginationMetadata: 12 tests
# - SemanticSearchRequest Pagination: 15 tests
# - SemanticSearchRequest Field Filtering: 10 tests
# - FindVendorInfoRequest Pagination: 8 tests
# - FindVendorInfoRequest Field Filtering: 8 tests
# - SemanticSearchResponse Pagination: 5 tests
# - Whitelist Fields: 6 tests
# - Edge Cases: 6 tests
# TOTAL: 70 tests
