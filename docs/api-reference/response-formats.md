# API Reference: Response Formats

## Table of Contents

1. [Overview](#overview)
2. [Response Envelope Structure](#response-envelope-structure)
3. [Response Modes](#response-modes)
4. [Field Definitions](#field-definitions)
5. [Metadata Components](#metadata-components)
6. [Execution Context](#execution-context)
7. [Warning System](#warning-system)
8. [JSON Schema Reference](#json-schema-reference)
9. [Type Definitions](#type-definitions)

## Overview

This document provides a complete API reference for the response formatting system used by the BMCIS Knowledge MCP server. All responses follow a consistent envelope structure with progressive disclosure modes for optimal token usage.

### Response Format Philosophy

- **Consistent Structure**: All responses use `MCPResponseEnvelope`
- **Progressive Disclosure**: Four levels of detail (ids_only → metadata → preview → full)
- **Type Safety**: Full Pydantic validation with mypy-strict compatibility
- **Token Efficiency**: Field-level control over response size
- **Desktop Optimized**: Designed for Claude Desktop integration

## Response Envelope Structure

### MCPResponseEnvelope[T]

The generic wrapper for all MCP tool responses.

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class MCPResponseEnvelope(BaseModel, Generic[T]):
    """
    Standard envelope for all MCP responses
    Token overhead: ~150-300 tokens per response
    """
    _metadata: ResponseMetadata
    results: T  # Generic type, varies by tool
    pagination: Optional[PaginationMetadata]
    execution_context: ExecutionContext
    warnings: List[ResponseWarning]
```

### Complete JSON Structure

```json
{
    "_metadata": {
        "operation": "semantic_search",
        "version": "1.0.0",
        "timestamp": "2025-11-09T10:30:00Z",
        "request_id": "req_7f3a2b1c",
        "status": "success",
        "message": null
    },
    "results": [...],  // Tool-specific results
    "pagination": {
        "cursor": "eyJxdWVyeV9oYXNoIjoiYWJjIiwib2Zmc2V0IjoxMH0=",
        "page_size": 10,
        "has_more": true,
        "total_available": 42
    },
    "execution_context": {
        "tokens_estimated": 2450,
        "tokens_used": 2523,
        "cache_hit": true,
        "execution_time_ms": 156.3,
        "request_id": "req_7f3a2b1c"
    },
    "warnings": []
}
```

## Response Modes

### Semantic Search Response Modes

Each mode provides different levels of detail with corresponding token costs:

#### 1. IDs Only Mode

**Model**: `SearchResultIDs`
**Token Budget**: ~10 tokens per result
**Fields Available**: `chunk_id`, `hybrid_score`, `rank`

```json
{
    "chunk_id": 123,
    "hybrid_score": 0.85,
    "rank": 1
}
```

#### 2. Metadata Mode (Default)

**Model**: `SearchResultMetadata`
**Token Budget**: ~200 tokens per result
**Fields Available**: All from IDs Only plus `source_file`, `source_category`, `chunk_index`, `total_chunks`

```json
{
    "chunk_id": 123,
    "source_file": "docs/auth/jwt.md",
    "source_category": "security",
    "hybrid_score": 0.85,
    "rank": 1,
    "chunk_index": 3,
    "total_chunks": 15
}
```

#### 3. Preview Mode

**Model**: `SearchResultPreview`
**Token Budget**: ~500 tokens per result
**Fields Available**: All from Metadata plus `chunk_snippet`, `context_header`

```json
{
    "chunk_id": 123,
    "source_file": "docs/auth/jwt.md",
    "source_category": "security",
    "hybrid_score": 0.85,
    "rank": 1,
    "chunk_index": 3,
    "total_chunks": 15,
    "chunk_snippet": "JWT authentication provides stateless...",
    "context_header": "JWT Guide > Implementation"
}
```

#### 4. Full Mode

**Model**: `SearchResultFull`
**Token Budget**: ~1500+ tokens per result
**Fields Available**: Complete chunk content and all metadata

```json
{
    "chunk_id": 123,
    "chunk_text": "Complete chunk content here...",
    "similarity_score": 0.82,
    "bm25_score": 0.88,
    "hybrid_score": 0.85,
    "rank": 1,
    "score_type": "hybrid",
    "source_file": "docs/auth/jwt.md",
    "source_category": "security",
    "context_header": "JWT Guide > Implementation",
    "chunk_index": 3,
    "total_chunks": 15,
    "chunk_token_count": 512
}
```

### Vendor Info Response Modes

Similar progressive disclosure for vendor information:

#### 1. VendorInfoIDs

**Token Budget**: ~100-500 tokens
**Fields**: `vendor_name`, `entity_ids`, `relationship_ids`

```json
{
    "vendor_name": "Acme Corporation",
    "entity_ids": ["e_001", "e_002", "e_003"],
    "relationship_ids": ["r_001", "r_002"]
}
```

#### 2. VendorInfoMetadata

**Token Budget**: ~2-4K tokens
**Fields**: `vendor_name`, `statistics`, `top_entities`, `last_updated`

```json
{
    "vendor_name": "Acme Corporation",
    "statistics": {
        "entity_count": 85,
        "relationship_count": 25,
        "entity_type_distribution": {"COMPANY": 50, "PERSON": 25},
        "relationship_type_distribution": {"PARTNER": 15, "COMPETITOR": 10}
    },
    "top_entities": [...],
    "last_updated": "2025-11-09T10:00:00Z"
}
```

#### 3. VendorInfoPreview

**Token Budget**: ~5-10K tokens
**Constraints**: Max 5 entities, max 5 relationships

```json
{
    "vendor_name": "Acme Corporation",
    "entities": [/* max 5 */],
    "relationships": [/* max 5 */],
    "statistics": {...}
}
```

#### 4. VendorInfoFull

**Token Budget**: ~10-50K+ tokens
**Constraints**: Max 100 entities, max 500 relationships

```json
{
    "vendor_name": "Acme Corporation",
    "entities": [/* max 100 */],
    "relationships": [/* max 500 */],
    "statistics": {...}
}
```

## Field Definitions

### Common Fields

| Field | Type | Description | Modes Available |
|-------|------|-------------|-----------------|
| `chunk_id` | `int` | Unique chunk identifier | All |
| `hybrid_score` | `float` | Combined relevance score (0.0-1.0) | All |
| `rank` | `int` | Result rank (1-based) | All |
| `source_file` | `str` | Source file path | metadata, preview, full |
| `source_category` | `str?` | Document category | metadata, preview, full |
| `chunk_index` | `int` | Position in document | metadata, preview, full |
| `total_chunks` | `int` | Total chunks in document | metadata, preview, full |

### Extended Fields

| Field | Type | Description | Modes Available |
|-------|------|-------------|-----------------|
| `chunk_snippet` | `str` | First 200 chars of content | preview |
| `context_header` | `str` | Hierarchical context path | preview, full |
| `chunk_text` | `str` | Complete chunk content | full |
| `similarity_score` | `float` | Vector similarity (0.0-1.0) | full |
| `bm25_score` | `float` | BM25 relevance (0.0-1.0) | full |
| `score_type` | `str` | Scoring method used | full |
| `chunk_token_count` | `int` | Token count in chunk | full |

## Metadata Components

### ResponseMetadata

Core metadata for every response:

```python
class ResponseMetadata(BaseModel):
    operation: str  # Tool operation name
    version: str  # API version (e.g., "1.0.0")
    timestamp: str  # ISO 8601 with timezone
    request_id: str  # Unique request identifier
    status: Literal["success", "partial", "error"]
    message: Optional[str]  # Status message
```

**Validation Rules**:
- `timestamp` must be ISO 8601 format with timezone
- `status` must be one of: success, partial, error
- `request_id` should be unique per request

### PaginationMetadata

Cursor-based pagination information:

```python
class PaginationMetadata(BaseModel):
    cursor: str  # Base64-encoded JSON cursor
    page_size: int  # Results per page (1-50)
    has_more: bool  # More results available
    total_available: Optional[int]  # Total results (if known)
```

**Cursor Structure** (decoded):
```json
{
    "query_hash": "abc123def456",
    "offset": 10,
    "response_mode": "metadata"
}
```

### ConfidenceScore

Multi-dimensional confidence assessment:

```python
class ConfidenceScore(BaseModel):
    score_reliability: float  # Score confidence (0.0-1.0)
    source_quality: float  # Source rating (0.0-1.0)
    recency: float  # Content freshness (0.0-1.0)
```

**Usage Notes**:
- All fields must be populated if model is used
- Use `None` for entire model if confidence not available
- Higher scores indicate higher confidence

### RankingContext

Result ranking explanation:

```python
class RankingContext(BaseModel):
    percentile: int  # Position in results (0-100)
    explanation: str  # Human-readable explanation
    score_method: str  # vector/bm25/hybrid
```

**Example Explanations**:
- "Top-tier result: Strong semantic and keyword match"
- "High relevance: Strong keyword match"
- "Moderate match: Some keyword overlap"

### DeduplicationInfo

Duplicate detection information:

```python
class DeduplicationInfo(BaseModel):
    is_duplicate: bool  # Duplicate of higher-ranked result
    similar_chunk_ids: List[int]  # Similar chunks (>0.8 similarity)
    confidence: float  # Dedup confidence (0.0-1.0)
```

## Execution Context

Performance and token accounting:

```python
class ExecutionContext(BaseModel):
    tokens_estimated: int  # Estimated token count
    tokens_used: Optional[int]  # Actual tokens (if available)
    cache_hit: bool  # Result from cache
    execution_time_ms: float  # Execution time
    request_id: str  # Matches ResponseMetadata.request_id
```

**Validation Rules**:
- `tokens_used` cannot exceed `tokens_estimated` by more than 10%
- `execution_time_ms` must be positive
- `request_id` must match the response metadata

## Warning System

### ResponseWarning

Actionable warnings for clients:

```python
class ResponseWarning(BaseModel):
    level: Literal["info", "warning", "error"]
    code: str  # SCREAMING_SNAKE_CASE
    message: str  # Human-readable message
    suggestion: Optional[str]  # Remediation action
```

### Standard Warning Codes

| Code | Level | Description | Suggestion |
|------|-------|-------------|------------|
| `TOKEN_LIMIT_WARNING` | warning | Approaching token limit | Use metadata mode or reduce page_size |
| `TOKEN_LIMIT_EXCEEDED` | error | Exceeded token limit | Use ids_only mode |
| `CACHE_MISS_SLOW` | info | Cache miss, slower response | Query will be cached for next time |
| `LOW_QUALITY_RESULTS` | info | No high-confidence results | Try different keywords |
| `PARTIAL_RESULTS` | warning | Some results omitted | Reduce page_size for complete results |
| `DEPRECATED_PARAMETER` | warning | Using deprecated parameter | Use suggested alternative |

## JSON Schema Reference

### Request Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "Search query"
        },
        "response_mode": {
            "type": "string",
            "enum": ["ids_only", "metadata", "preview", "full"],
            "default": "metadata",
            "description": "Response detail level"
        },
        "page_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 10,
            "description": "Results per page"
        },
        "cursor": {
            "type": "string",
            "description": "Pagination cursor (base64 JSON)"
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Field whitelist"
        }
    },
    "required": ["query"]
}
```

### Response Schema (Envelope)

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "_metadata": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "version": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "request_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["success", "partial", "error"]
                },
                "message": {"type": ["string", "null"]}
            },
            "required": ["operation", "version", "timestamp", "request_id", "status"]
        },
        "results": {
            "type": "array",
            "description": "Tool-specific results"
        },
        "pagination": {
            "type": ["object", "null"],
            "properties": {
                "cursor": {"type": "string"},
                "page_size": {"type": "integer"},
                "has_more": {"type": "boolean"},
                "total_available": {"type": ["integer", "null"]}
            }
        },
        "execution_context": {
            "type": "object",
            "properties": {
                "tokens_estimated": {"type": "integer"},
                "tokens_used": {"type": ["integer", "null"]},
                "cache_hit": {"type": "boolean"},
                "execution_time_ms": {"type": "number"},
                "request_id": {"type": "string"}
            },
            "required": ["tokens_estimated", "cache_hit", "execution_time_ms", "request_id"]
        },
        "warnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["info", "warning", "error"]
                    },
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "suggestion": {"type": ["string", "null"]}
                },
                "required": ["level", "code", "message"]
            }
        }
    },
    "required": ["_metadata", "results", "execution_context", "warnings"]
}
```

## Type Definitions

### Python Type Aliases

```python
from typing import Union, List, Optional, Literal, TypeVar, Generic

# Response mode literals
ResponseMode = Literal["ids_only", "metadata", "preview", "full"]

# Status literals
ResponseStatus = Literal["success", "partial", "error"]

# Warning level literals
WarningLevel = Literal["info", "warning", "error"]

# Generic result type
T = TypeVar('T')

# Result union types for semantic search
SemanticSearchResult = Union[
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SearchResultFull
]

# Result union types for vendor info
VendorInfoResult = Union[
    VendorInfoIDs,
    VendorInfoMetadata,
    VendorInfoPreview,
    VendorInfoFull
]

# Complete response types
SemanticSearchResponseType = MCPResponseEnvelope[List[SemanticSearchResult]]
VendorInfoResponseType = MCPResponseEnvelope[VendorInfoResult]
```

### TypeScript Type Definitions

```typescript
// Response mode type
type ResponseMode = "ids_only" | "metadata" | "preview" | "full";

// Response status type
type ResponseStatus = "success" | "partial" | "error";

// Warning level type
type WarningLevel = "info" | "warning" | "error";

// Response metadata interface
interface ResponseMetadata {
    operation: string;
    version: string;
    timestamp: string;
    request_id: string;
    status: ResponseStatus;
    message?: string;
}

// Pagination metadata interface
interface PaginationMetadata {
    cursor: string;
    page_size: number;
    has_more: boolean;
    total_available?: number;
}

// Execution context interface
interface ExecutionContext {
    tokens_estimated: number;
    tokens_used?: number;
    cache_hit: boolean;
    execution_time_ms: number;
    request_id: string;
}

// Response warning interface
interface ResponseWarning {
    level: WarningLevel;
    code: string;
    message: string;
    suggestion?: string;
}

// Generic response envelope
interface MCPResponseEnvelope<T> {
    _metadata: ResponseMetadata;
    results: T;
    pagination?: PaginationMetadata;
    execution_context: ExecutionContext;
    warnings: ResponseWarning[];
}

// Semantic search result types
type SemanticSearchResult =
    | SearchResultIDs
    | SearchResultMetadata
    | SearchResultPreview
    | SearchResultFull;

// Complete semantic search response
type SemanticSearchResponse = MCPResponseEnvelope<SemanticSearchResult[]>;
```

## Field Whitelisting

### Semantic Search Field Whitelist

Fields available by response mode:

```python
class WhitelistedSemanticSearchFields:
    IDS_ONLY = frozenset(["chunk_id", "hybrid_score", "rank"])

    METADATA = frozenset([
        "chunk_id", "source_file", "source_category",
        "hybrid_score", "rank", "chunk_index", "total_chunks"
    ])

    PREVIEW = frozenset([
        "chunk_id", "source_file", "source_category",
        "hybrid_score", "rank", "chunk_index", "total_chunks",
        "chunk_snippet", "context_header"
    ])

    FULL = frozenset([
        "chunk_id", "chunk_text", "similarity_score",
        "bm25_score", "hybrid_score", "rank", "score_type",
        "source_file", "source_category", "context_header",
        "chunk_index", "total_chunks", "chunk_token_count"
    ])
```

### Vendor Info Field Whitelist

Fields available by response mode:

```python
class WhitelistedVendorInfoFields:
    IDS_ONLY = frozenset([
        "vendor_name", "entity_ids", "relationship_ids"
    ])

    METADATA = frozenset([
        "vendor_name", "statistics", "top_entities", "last_updated"
    ])

    PREVIEW = frozenset([
        "vendor_name", "entities", "relationships", "statistics"
    ])

    FULL = frozenset([
        "vendor_name", "entities", "relationships", "statistics"
    ])
```

## Usage Examples

### Example 1: Basic Search with Metadata

**Request**:
```python
response = semantic_search(
    query="JWT authentication",
    response_mode="metadata",
    page_size=10
)
```

**Response**:
```json
{
    "_metadata": {
        "operation": "semantic_search",
        "version": "1.0.0",
        "timestamp": "2025-11-09T10:30:00Z",
        "request_id": "req_abc123",
        "status": "success"
    },
    "results": [
        {
            "chunk_id": 123,
            "source_file": "docs/auth/jwt.md",
            "source_category": "security",
            "hybrid_score": 0.85,
            "rank": 1,
            "chunk_index": 3,
            "total_chunks": 15
        }
    ],
    "pagination": null,
    "execution_context": {
        "tokens_estimated": 2000,
        "tokens_used": 2050,
        "cache_hit": false,
        "execution_time_ms": 156.3,
        "request_id": "req_abc123"
    },
    "warnings": []
}
```

### Example 2: Paginated Search

**Request**:
```python
# First page
page1 = semantic_search(
    query="authentication",
    response_mode="ids_only",
    page_size=20
)

# Next page
page2 = semantic_search(
    query="authentication",
    response_mode="ids_only",
    page_size=20,
    cursor=page1["pagination"]["cursor"]
)
```

### Example 3: Field-Filtered Response

**Request**:
```python
response = semantic_search(
    query="OAuth",
    response_mode="metadata",
    fields=["chunk_id", "source_file", "hybrid_score"]
)
```

**Response** (only requested fields):
```json
{
    "results": [
        {
            "chunk_id": 456,
            "source_file": "docs/oauth/guide.md",
            "hybrid_score": 0.78
        }
    ]
}
```

## Error Responses

### Token Limit Error

```json
{
    "_metadata": {
        "operation": "semantic_search",
        "status": "error",
        "message": "Response size exceeds token limit"
    },
    "results": [],
    "warnings": [
        {
            "level": "error",
            "code": "TOKEN_LIMIT_EXCEEDED",
            "message": "Response would use 25000 tokens, limit is 15000",
            "suggestion": "Use 'metadata' mode or reduce page_size to 5"
        }
    ]
}
```

### Invalid Field Error

```json
{
    "_metadata": {
        "status": "error",
        "message": "Invalid fields for response_mode"
    },
    "error": {
        "code": "INVALID_FIELDS",
        "message": "Fields ['chunk_text'] not available in 'metadata' mode",
        "allowed_fields": ["chunk_id", "source_file", "hybrid_score", ...]
    }
}
```

## Conclusion

This API reference provides complete documentation for the response formatting system. The consistent envelope structure, progressive disclosure modes, and comprehensive metadata ensure efficient, type-safe integration with Claude Desktop and other MCP clients.

For implementation guides and best practices, see:
- [Response Formatting Guide](../guides/response-formatting-guide.md)
- [Claude Desktop Optimization](../guides/claude-desktop-optimization.md)
- [MCP Tools API Reference](./mcp-tools.md)