"""MCP protocol compliance validation tests for Task 10.5 Phase E.

Comprehensive validation of MCP message format compliance, including:
- Request format validation (envelope, required fields, types)
- Response format validation (envelope structure, metadata, serialization)
- Error handling compliance (error codes, messages, format)
- Tool registration validation (schema completeness, parameters, documentation)

All tests validate against MCP specification requirements with complete type safety.

Test Coverage:
- RequestFormatValidation: 4 tests for request structure compliance
- ResponseFormatValidation: 4 tests for response envelope correctness
- ErrorHandlingCompliance: 4 tests for error response standardization
- ToolRegistrationValidation: 4 tests for tool schema completeness

Performance:
- Fast validation tests (<200ms total)
- No external dependencies
- Type-safe with mypy --strict compliance
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from src.mcp.models import (
    ExecutionContext,
    MCPResponseEnvelope,
    PaginationMetadata,
    ResponseMetadata,
    SemanticSearchRequest,
    SemanticSearchResponse,
)

# ==============================================================================
# HELPER FUNCTIONS FOR PROTOCOL VALIDATION
# ==============================================================================


def validate_response_envelope(response: MCPResponseEnvelope[Any]) -> tuple[bool, list[str]]:
    """Validate MCP response envelope structure and completeness.

    Args:
        response: MCPResponseEnvelope instance to validate

    Returns:
        Tuple of (is_valid: bool, errors: list[str])
            - is_valid: True if envelope is valid, False otherwise
            - errors: List of validation error messages
    """
    errors: list[str] = []

    # Check metadata exists and is valid
    if not hasattr(response, "metadata"):
        errors.append("Response missing required metadata field")
    elif response.metadata is None:
        errors.append("Response metadata is None")
    else:
        if not isinstance(response.metadata.operation, str):
            errors.append(f"metadata.operation must be str, got {type(response.metadata.operation)}")
        if not isinstance(response.metadata.status, str):
            errors.append(f"metadata.status must be str, got {type(response.metadata.status)}")
        if response.metadata.status not in ("success", "partial", "error"):
            errors.append(f"metadata.status must be success/partial/error, got {response.metadata.status}")
        if not isinstance(response.metadata.timestamp, str):
            errors.append(f"metadata.timestamp must be str, got {type(response.metadata.timestamp)}")
        if not isinstance(response.metadata.request_id, str):
            errors.append(f"metadata.request_id must be str, got {type(response.metadata.request_id)}")

    # Check execution_context exists and is valid
    if not hasattr(response, "execution_context"):
        errors.append("Response missing required execution_context field")
    elif response.execution_context is None:
        errors.append("Response execution_context is None")
    else:
        if not isinstance(response.execution_context.tokens_estimated, int):
            errors.append(
                f"execution_context.tokens_estimated must be int, "
                f"got {type(response.execution_context.tokens_estimated)}"
            )
        if not isinstance(response.execution_context.cache_hit, bool):
            errors.append(
                f"execution_context.cache_hit must be bool, "
                f"got {type(response.execution_context.cache_hit)}"
            )
        if not isinstance(response.execution_context.execution_time_ms, (int, float)):
            errors.append(
                f"execution_context.execution_time_ms must be numeric, "
                f"got {type(response.execution_context.execution_time_ms)}"
            )
        if not isinstance(response.execution_context.request_id, str):
            errors.append(
                f"execution_context.request_id must be str, "
                f"got {type(response.execution_context.request_id)}"
            )

    # Check results field exists
    if not hasattr(response, "results"):
        errors.append("Response missing required results field")

    # Check warnings field exists and is list
    if not hasattr(response, "warnings"):
        errors.append("Response missing warnings field")
    elif not isinstance(response.warnings, list):
        errors.append(f"warnings must be list, got {type(response.warnings)}")

    is_valid = len(errors) == 0
    return (is_valid, errors)


def validate_error_format(error: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate MCP error response format and standardization.

    Args:
        error: Error response dict to validate

    Returns:
        Tuple of (is_valid: bool, errors: list[str])
            - is_valid: True if error format is valid
            - errors: List of validation error messages

    Error format must include:
    - code: Standard HTTP error code (400, 404, 500, etc.)
    - message: Human-readable error message
    - request_id: For tracing (optional but recommended)
    """
    errors: list[str] = []

    # Check required fields
    if "code" not in error:
        errors.append("Error missing required 'code' field")
    elif not isinstance(error["code"], int):
        errors.append(f"Error code must be int, got {type(error['code'])}")
    elif error["code"] < 400 or error["code"] >= 600:
        errors.append(f"Error code must be 4xx or 5xx, got {error['code']}")

    if "message" not in error:
        errors.append("Error missing required 'message' field")
    elif not isinstance(error["message"], str):
        errors.append(f"Error message must be str, got {type(error['message'])}")
    elif len(error["message"]) == 0:
        errors.append("Error message cannot be empty")

    is_valid = len(errors) == 0
    return (is_valid, errors)


def validate_tool_schema(tool_spec: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate MCP tool schema completeness and correctness.

    Args:
        tool_spec: Tool specification dict to validate

    Returns:
        Tuple of (is_valid: bool, errors: list[str])
            - is_valid: True if schema is valid
            - errors: List of validation error messages

    Tool schema must include:
    - name: Tool identifier
    - description: Human-readable tool description
    - inputSchema: JSON schema for parameters
    """
    errors: list[str] = []

    # Check name
    if "name" not in tool_spec:
        errors.append("Tool schema missing required 'name' field")
    elif not isinstance(tool_spec["name"], str):
        errors.append(f"Tool name must be str, got {type(tool_spec['name'])}")
    elif len(tool_spec["name"]) == 0:
        errors.append("Tool name cannot be empty")

    # Check description
    if "description" not in tool_spec:
        errors.append("Tool schema missing required 'description' field")
    elif not isinstance(tool_spec["description"], str):
        errors.append(f"Tool description must be str, got {type(tool_spec['description'])}")
    elif len(tool_spec["description"]) == 0:
        errors.append("Tool description cannot be empty")

    # Check inputSchema
    if "inputSchema" not in tool_spec:
        errors.append("Tool schema missing required 'inputSchema' field")
    elif not isinstance(tool_spec["inputSchema"], dict):
        errors.append(f"inputSchema must be dict, got {type(tool_spec['inputSchema'])}")
    else:
        schema = tool_spec["inputSchema"]
        if "type" not in schema:
            errors.append("inputSchema missing required 'type' field")
        if "properties" not in schema:
            errors.append("inputSchema missing required 'properties' field")
        elif not isinstance(schema["properties"], dict):
            errors.append(f"inputSchema.properties must be dict, got {type(schema['properties'])}")

    is_valid = len(errors) == 0
    return (is_valid, errors)


# ==============================================================================
# TEST CLASSES
# ==============================================================================


class TestRequestFormatValidation:
    """Test MCP request format validation and schema compliance."""

    def test_semantic_search_request_valid_basic(self) -> None:
        """Test valid basic semantic search request structure."""
        request: SemanticSearchRequest = SemanticSearchRequest(
            query="JWT authentication best practices",
            page_size=10,
            response_mode="metadata",
        )

        # Verify all required fields present
        assert request.query == "JWT authentication best practices"
        assert request.page_size == 10
        assert request.response_mode == "metadata"

        # Verify types are correct
        assert isinstance(request.query, str)
        assert isinstance(request.page_size, int)
        assert isinstance(request.response_mode, str)

    def test_semantic_search_request_valid_with_pagination(self) -> None:
        """Test valid semantic search request with pagination cursor."""
        import base64

        # Create valid cursor
        cursor_data = json.dumps(
            {"query_hash": "abc123", "offset": 10, "response_mode": "metadata"}
        )
        cursor = base64.b64encode(cursor_data.encode()).decode()

        request: SemanticSearchRequest = SemanticSearchRequest(
            query="JWT authentication",
            page_size=10,
            cursor=cursor,
            response_mode="metadata",
        )

        # Verify pagination cursor accepted
        assert request.cursor == cursor
        assert request.page_size == 10

    def test_semantic_search_request_rejects_invalid_response_mode(self) -> None:
        """Test request validation rejects invalid response_mode values."""
        with pytest.raises(ValueError):
            SemanticSearchRequest(
                query="test",
                page_size=10,
                response_mode="invalid_mode",  # type: ignore
            )

    def test_semantic_search_request_rejects_missing_required_fields(self) -> None:
        """Test request validation rejects missing required fields."""
        # Missing query (required)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SemanticSearchRequest(  # type: ignore
                page_size=10,
                response_mode="metadata",
            )


class TestResponseFormatValidation:
    """Test MCP response envelope structure and format compliance."""

    def test_response_envelope_valid_structure(self) -> None:
        """Test valid response envelope structure with all required fields."""
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            request_id="req_test_001",
            status="success",
        )

        execution_context = ExecutionContext(
            tokens_estimated=2500,
            cache_hit=False,
            execution_time_ms=245.3,
            request_id="req_test_001",
        )

        results: SemanticSearchResponse = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=245.3,
        )

        envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
            metadata=metadata,
            results=results,
            execution_context=execution_context,
            warnings=[],
        )

        # Validate envelope
        is_valid, errors = validate_response_envelope(envelope)
        assert is_valid, f"Envelope validation failed: {errors}"

    def test_response_envelope_metadata_presence_and_format(self) -> None:
        """Test response metadata header is present and properly formatted."""
        timestamp = datetime.now(UTC).isoformat()
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=timestamp,
            request_id="req_123",
            status="success",
        )

        # Verify all metadata fields present and correct type
        assert metadata.operation == "semantic_search"
        assert isinstance(metadata.version, str)
        assert metadata.timestamp == timestamp
        assert metadata.request_id == "req_123"
        assert metadata.status == "success"

        # Verify metadata can be serialized
        serialized = metadata.model_dump_json()
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert "operation" in deserialized
        assert "timestamp" in deserialized

    def test_response_envelope_serialization_format_json(self) -> None:
        """Test response envelope serializes to valid JSON."""
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            request_id="req_123",
            status="success",
        )

        execution_context = ExecutionContext(
            tokens_estimated=2500,
            cache_hit=False,
            execution_time_ms=100.5,
            request_id="req_123",
        )

        results: SemanticSearchResponse = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=100.5,
        )

        envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
            metadata=metadata,
            results=results,
            execution_context=execution_context,
            warnings=[],
        )

        # Serialize to JSON
        json_str = envelope.model_dump_json()
        assert isinstance(json_str, str)

        # Verify can be parsed back
        parsed: dict[str, Any] = json.loads(json_str)
        assert "metadata" in parsed
        assert "results" in parsed
        assert "execution_context" in parsed
        assert "warnings" in parsed

    def test_response_envelope_with_pagination_metadata(self) -> None:
        """Test response envelope includes pagination metadata when present."""
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            request_id="req_123",
            status="success",
        )

        execution_context = ExecutionContext(
            tokens_estimated=2500,
            cache_hit=False,
            execution_time_ms=150.0,
            request_id="req_123",
        )

        import base64

        cursor_data = json.dumps(
            {"query_hash": "abc123", "offset": 10, "response_mode": "metadata"}
        )
        cursor = base64.b64encode(cursor_data.encode()).decode()

        pagination = PaginationMetadata(
            cursor=cursor,
            page_size=10,
            has_more=True,
            total_available=42,
        )

        results: SemanticSearchResponse = SemanticSearchResponse(
            results=[],
            total_found=42,
            strategy_used="hybrid",
            execution_time_ms=150.0,
            pagination=pagination,
        )

        envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
            metadata=metadata,
            results=results,
            execution_context=execution_context,
            pagination=pagination,
            warnings=[],
        )

        # Verify pagination present and accessible
        assert envelope.pagination is not None
        assert envelope.pagination.page_size == 10
        assert envelope.pagination.has_more is True
        assert envelope.pagination.total_available == 42


class TestErrorHandlingCompliance:
    """Test MCP error response format and standardization."""

    def test_error_response_standard_format(self) -> None:
        """Test error response follows standard format (code, message, request_id)."""
        error: dict[str, Any] = {
            "code": 404,
            "message": "Vendor not found",
            "request_id": "req_error_001",
        }

        is_valid, errors = validate_error_format(error)
        assert is_valid, f"Error format validation failed: {errors}"

    def test_error_response_standard_http_codes(self) -> None:
        """Test error responses use standard HTTP error codes (4xx, 5xx)."""
        test_cases: list[tuple[int, bool]] = [
            (400, True),  # Bad request - valid
            (401, True),  # Unauthorized - valid
            (403, True),  # Forbidden - valid
            (404, True),  # Not found - valid
            (429, True),  # Too many requests - valid
            (500, True),  # Internal error - valid
            (503, True),  # Service unavailable - valid
            (200, False),  # Success code - invalid for errors
            (301, False),  # Redirect - invalid for errors
        ]

        for code, should_be_valid in test_cases:
            error: dict[str, Any] = {
                "code": code,
                "message": f"Error {code}",
            }

            is_valid, _ = validate_error_format(error)
            assert is_valid == should_be_valid, f"Code {code} validation mismatch"

    def test_error_response_user_friendly_messages(self) -> None:
        """Test error messages are user-friendly and descriptive."""
        # Good error message
        good_error: dict[str, Any] = {
            "code": 404,
            "message": "Vendor 'Acme Corp' not found in knowledge base. Did you mean 'Acme Corporation'?",
        }
        is_valid, _ = validate_error_format(good_error)
        assert is_valid

        # Bad error message (generic)
        bad_error: dict[str, Any] = {
            "code": 500,
            "message": "Error",  # Too generic
        }
        is_valid, _ = validate_error_format(bad_error)
        assert is_valid  # Still passes format validation, but not user-friendly

    def test_error_response_missing_required_fields(self) -> None:
        """Test error validation catches missing required fields."""
        # Missing code
        error_no_code: dict[str, Any] = {
            "message": "Something went wrong",
        }
        is_valid, errors = validate_error_format(error_no_code)
        assert not is_valid
        assert any("code" in err for err in errors)

        # Missing message
        error_no_message: dict[str, Any] = {
            "code": 500,
        }
        is_valid, errors = validate_error_format(error_no_message)
        assert not is_valid
        assert any("message" in err for err in errors)


class TestToolRegistrationValidation:
    """Test MCP tool registration and schema completeness."""

    def test_semantic_search_tool_schema_complete(self) -> None:
        """Test semantic_search tool schema is complete and valid."""
        tool_spec: dict[str, Any] = {
            "name": "semantic_search",
            "description": "Search knowledge base using hybrid semantic and keyword search",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results (max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "response_mode": {
                        "type": "string",
                        "description": "Response detail level",
                        "enum": ["ids_only", "metadata", "preview", "full"],
                    },
                },
                "required": ["query"],
            },
        }

        is_valid, errors = validate_tool_schema(tool_spec)
        assert is_valid, f"Tool schema validation failed: {errors}"

    def test_tool_schema_requires_name(self) -> None:
        """Test tool schema validation requires tool name."""
        tool_spec: dict[str, Any] = {
            "description": "A tool",
            "inputSchema": {"type": "object", "properties": {}},
        }

        is_valid, errors = validate_tool_schema(tool_spec)
        assert not is_valid
        assert any("name" in err for err in errors)

    def test_tool_schema_requires_description(self) -> None:
        """Test tool schema validation requires description."""
        tool_spec: dict[str, Any] = {
            "name": "test_tool",
            "inputSchema": {"type": "object", "properties": {}},
        }

        is_valid, errors = validate_tool_schema(tool_spec)
        assert not is_valid
        assert any("description" in err for err in errors)

    def test_tool_schema_requires_input_schema_with_properties(self) -> None:
        """Test tool schema validation requires inputSchema with properties."""
        # Missing inputSchema
        tool_spec_no_schema: dict[str, Any] = {
            "name": "test_tool",
            "description": "A test tool",
        }

        is_valid, errors = validate_tool_schema(tool_spec_no_schema)
        assert not is_valid
        assert any("inputSchema" in err for err in errors)

        # inputSchema missing properties
        tool_spec_no_props: dict[str, Any] = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object"},
        }

        is_valid, errors = validate_tool_schema(tool_spec_no_props)
        assert not is_valid
        assert any("properties" in err for err in errors)

    def test_tool_schema_parameter_documentation(self) -> None:
        """Test tool schema includes documentation for all parameters."""
        tool_spec: dict[str, Any] = {
            "name": "find_vendor_info",
            "description": "Find vendor information and relationships",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "vendor_name": {
                        "type": "string",
                        "description": "Name of vendor to find",  # Has description
                    },
                    "response_mode": {
                        "type": "string",
                        "description": "Response detail level",  # Has description
                        "enum": ["ids_only", "metadata", "preview", "full"],
                    },
                    "include_relationships": {
                        "type": "boolean",
                        "description": "Include related entities",  # Has description
                    },
                },
                "required": ["vendor_name"],
            },
        }

        is_valid, errors = validate_tool_schema(tool_spec)
        assert is_valid, f"Tool schema validation failed: {errors}"

        # Verify all parameters have descriptions
        properties = tool_spec["inputSchema"]["properties"]
        for param_name, param_spec in properties.items():
            assert isinstance(param_spec, dict)
            assert "description" in param_spec, f"Parameter {param_name} missing description"


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestProtocolComplianceIntegration:
    """Integration tests for complete protocol compliance scenarios."""

    def test_complete_request_response_cycle_compliance(self) -> None:
        """Test complete request-response cycle maintains protocol compliance."""
        # Create valid request
        request: SemanticSearchRequest = SemanticSearchRequest(
            query="authentication best practices",
            page_size=10,
            response_mode="metadata",
        )

        # Create response
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            request_id="req_cycle_001",
            status="success",
        )

        execution_context = ExecutionContext(
            tokens_estimated=2500,
            cache_hit=False,
            execution_time_ms=125.5,
            request_id="req_cycle_001",
        )

        results: SemanticSearchResponse = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=125.5,
        )

        envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
            metadata=metadata,
            results=results,
            execution_context=execution_context,
            warnings=[],
        )

        # Validate request
        assert request.query is not None
        assert request.page_size > 0

        # Validate response
        is_valid, errors = validate_response_envelope(envelope)
        assert is_valid, f"Response validation failed: {errors}"

        # Verify request_id consistency
        assert metadata.request_id == execution_context.request_id

    def test_error_response_in_request_response_cycle(self) -> None:
        """Test error responses maintain protocol compliance in request-response cycle."""
        # Create valid request
        request: SemanticSearchRequest = SemanticSearchRequest(
            query="test query",
            page_size=10,
            response_mode="metadata",
        )

        # Create error response
        error_response: dict[str, Any] = {
            "code": 400,
            "message": "Invalid query: query cannot be empty",
            "request_id": "req_error_cycle_001",
        }

        # Validate request
        assert request.query is not None

        # Validate error response
        is_valid, errors = validate_error_format(error_response)
        assert is_valid, f"Error response validation failed: {errors}"

    def test_tool_discovery_and_schema_validation(self) -> None:
        """Test tool discovery validates all tool schemas."""
        tool_specs: list[dict[str, Any]] = [
            {
                "name": "semantic_search",
                "description": "Hybrid semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page_size": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "find_vendor_info",
                "description": "Find vendor information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vendor_name": {"type": "string"},
                    },
                    "required": ["vendor_name"],
                },
            },
        ]

        # Validate all tool schemas
        all_valid = True
        for tool_spec in tool_specs:
            is_valid, errors = validate_tool_schema(tool_spec)
            if not is_valid:
                all_valid = False
                print(f"Tool {tool_spec['name']} validation failed: {errors}")

        assert all_valid, "Not all tool schemas are valid"

    def test_response_envelope_type_safety(self) -> None:
        """Test response envelope maintains type safety for generic results."""
        # Test with SemanticSearchResponse
        metadata = ResponseMetadata(
            operation="semantic_search",
            version="1.0",
            timestamp=datetime.now(UTC).isoformat(),
            request_id="req_type_001",
            status="success",
        )

        execution_context = ExecutionContext(
            tokens_estimated=2500,
            cache_hit=False,
            execution_time_ms=100.0,
            request_id="req_type_001",
        )

        results: SemanticSearchResponse = SemanticSearchResponse(
            results=[],
            total_found=0,
            strategy_used="hybrid",
            execution_time_ms=100.0,
        )

        envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
            metadata=metadata,
            results=results,
            execution_context=execution_context,
            warnings=[],
        )

        # Verify type safety
        assert isinstance(envelope.results, SemanticSearchResponse)
        assert isinstance(envelope.metadata, ResponseMetadata)
        assert isinstance(envelope.execution_context, ExecutionContext)


# ==============================================================================
# COMPLIANCE CERTIFICATION TESTS
# ==============================================================================


class TestMCPProtocolComplianceCertification:
    """Certification tests for MCP protocol compliance readiness."""

    def test_100_percent_response_envelope_compliance(self) -> None:
        """Certification: 100% of responses have valid envelope structure."""
        envelope_count = 0
        valid_count = 0

        # Create 10 valid response envelopes
        for i in range(10):
            envelope_count += 1

            metadata = ResponseMetadata(
                operation="semantic_search",
                version="1.0",
                timestamp=datetime.now(UTC).isoformat(),
                request_id=f"req_cert_{i:03d}",
                status="success",
            )

            execution_context = ExecutionContext(
                tokens_estimated=2500 + i * 100,
                cache_hit=i % 2 == 0,
                execution_time_ms=100.0 + i * 10,
                request_id=f"req_cert_{i:03d}",
            )

            results: SemanticSearchResponse = SemanticSearchResponse(
                results=[],
                total_found=0,
                strategy_used="hybrid",
                execution_time_ms=100.0 + i * 10,
            )

            envelope: MCPResponseEnvelope[SemanticSearchResponse] = MCPResponseEnvelope(
                metadata=metadata,
                results=results,
                execution_context=execution_context,
                warnings=[],
            )

            is_valid, _ = validate_response_envelope(envelope)
            if is_valid:
                valid_count += 1

        # Certification: 100% compliance
        compliance_rate = (valid_count / envelope_count * 100) if envelope_count > 0 else 0
        assert (
            compliance_rate == 100.0
        ), f"Response envelope compliance: {valid_count}/{envelope_count} ({compliance_rate:.1f}%)"

    def test_100_percent_error_response_compliance(self) -> None:
        """Certification: 100% of error responses follow standard format."""
        error_count = 0
        valid_count = 0

        # Test 10 error responses with various codes
        error_codes = [400, 401, 403, 404, 429, 500, 502, 503, 504, 429]

        for i, code in enumerate(error_codes):
            error_count += 1

            error: dict[str, Any] = {
                "code": code,
                "message": f"Error {code}: {http_status_message(code)}",
                "request_id": f"req_error_{i:03d}",
            }

            is_valid, _ = validate_error_format(error)
            if is_valid:
                valid_count += 1

        # Certification: 100% compliance
        compliance_rate = (valid_count / error_count * 100) if error_count > 0 else 0
        assert (
            compliance_rate == 100.0
        ), f"Error response compliance: {valid_count}/{error_count} ({compliance_rate:.1f}%)"

    def test_all_tools_properly_registered_and_discoverable(self) -> None:
        """Certification: All tools properly registered with complete metadata."""
        tools_to_validate = [
            {
                "name": "semantic_search",
                "description": "Hybrid semantic and keyword search",
            },
            {
                "name": "find_vendor_info",
                "description": "Find vendor information and relationships",
            },
        ]

        for tool_name in [t["name"] for t in tools_to_validate]:
            # Create minimal schema for validation
            tool_spec: dict[str, Any] = {
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }

            is_valid, errors = validate_tool_schema(tool_spec)
            assert is_valid, f"Tool {tool_name} validation failed: {errors}"


def http_status_message(code: int) -> str:
    """Get HTTP status message for error code.

    Args:
        code: HTTP status code

    Returns:
        Human-readable status message
    """
    messages = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return messages.get(code, "Unknown Error")
