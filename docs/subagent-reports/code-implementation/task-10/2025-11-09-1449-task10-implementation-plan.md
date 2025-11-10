# Task 10: FastMCP Server Integration - Comprehensive Implementation Plan

**Date**: 2025-11-09 14:49
**Branch**: `feat/task-10-fastmcp-integration`
**Context**: DEVELOPMENT (FastMCP Server Implementation)
**Author**: Claude Code (Planning Agent)
**Status**: IMPLEMENTATION PLAN READY

---

## Executive Summary

This document provides a comprehensive implementation plan for Task 10 (FastMCP Server Integration), which exposes the production-ready knowledge graph system (310 tests passing, 78-92% coverage) via FastMCP server for Claude Desktop integration.

**Key Objectives**:
- Implement FastMCP server with `semantic_search` and `find_vendor_info` tools
- Achieve <500ms P95 latency for all MCP tool calls
- Support progressive disclosure pattern (metadata-only → full-content variants)
- Design for future code execution integration (90-95% token reduction)
- Maintain 100% mypy compliance and comprehensive test coverage

**Estimated Effort**: 6-8 hours (MVP), +2-3 hours (code execution readiness)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tool Interface Specifications](#tool-interface-specifications)
3. [Detailed Subtask Breakdown](#detailed-subtask-breakdown)
4. [Code Structure](#code-structure)
5. [Testing Strategy](#testing-strategy)
6. [Progressive Disclosure Design](#progressive-disclosure-design)
7. [Code Execution Integration Roadmap](#code-execution-integration-roadmap)
8. [Implementation Timeline](#implementation-timeline)
9. [Success Criteria](#success-criteria)
10. [Next Steps](#next-steps)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Desktop                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP Protocol (stdio)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastMCP Server (Task 10)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Tool Registry│  │ Auth Layer   │  │ Response Formatter │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Direct function calls
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Knowledge Graph System (Task 7, 9)                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ HybridSearch     │  │ GraphService     │  │ QueryRouter  │  │
│  │ (vector + BM25)  │  │ (entity/rel)     │  │ (strategy)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ psycopg connection pool
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector (Production DB)              │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Type-Safe API Development**: Complete Pydantic models with mypy compliance
2. **Progressive Disclosure**: Support metadata-only → full-content retrieval patterns
3. **Performance First**: <500ms P95 latency, async-first design
4. **Security by Design**: API key validation, input sanitization, output filtering
5. **Code Execution Ready**: Tool interfaces designed for future sandbox integration

---

## Tool Interface Specifications

### Tool 1: semantic_search

**Purpose**: Hybrid semantic search combining vector similarity and BM25 keyword matching.

#### Input Schema (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class SemanticSearchRequest(BaseModel):
    """Request schema for semantic_search tool.

    MCP Tool Parameters:
    - query: Search query string (required)
    - top_k: Number of results to return (default: 10, max: 50)
    - strategy: Search strategy (auto/vector/bm25/hybrid)
    - min_score: Minimum relevance score threshold (0.0-1.0)
    - detail_level: Progressive disclosure level (metadata/full)
    """
    query: str = Field(
        ...,
        description="Search query (natural language or keywords)",
        min_length=1,
        max_length=500
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=50
    )
    strategy: Literal["auto", "vector", "bm25", "hybrid"] = Field(
        default="auto",
        description="Search strategy: auto (intelligent routing), vector (semantic), bm25 (keyword), hybrid (combined)"
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum relevance score threshold (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    detail_level: Literal["metadata", "full"] = Field(
        default="full",
        description="Response detail level: metadata (IDs + scores only, ~50 tokens/result), full (complete content, ~1500 tokens/result)"
    )
```

#### Output Schema (Pydantic)

```python
from typing import List, Optional
from uuid import UUID

class SearchResultMetadata(BaseModel):
    """Minimal search result for progressive disclosure (Level 0).

    Token Budget: ~50 tokens per result
    Use Case: Quick overview, relevance filtering, count estimation
    """
    chunk_id: int = Field(..., description="Unique chunk identifier")
    entity_id: Optional[UUID] = Field(None, description="Associated entity UUID (if available)")
    source_file: str = Field(..., description="Source file path")
    entity_type: Optional[str] = Field(None, description="Entity type (vendor/library/framework)")
    hybrid_score: float = Field(..., description="Relevance score (0.0-1.0)")
    rank: int = Field(..., description="Result rank (1-based)")

class SearchResultFull(SearchResultMetadata):
    """Full search result with content (Level 1).

    Token Budget: ~1500 tokens per result
    Use Case: Deep analysis, implementation details, code context
    """
    chunk_text: str = Field(..., description="Full chunk content")
    context_header: Optional[str] = Field(None, description="Context header (file structure)")
    vendor_names: List[str] = Field(default_factory=list, description="Detected vendor names")
    section_title: Optional[str] = Field(None, description="Section title (if available)")

class SemanticSearchResponse(BaseModel):
    """Response schema for semantic_search tool.

    Supports progressive disclosure:
    - metadata: List[SearchResultMetadata] (50 tokens/result)
    - full: List[SearchResultFull] (1500 tokens/result)

    Token Reduction Example:
    - Traditional (full, 10 results): ~15,000 tokens
    - Progressive (metadata, 10 results): ~500 tokens (97% reduction)
    - Selective (metadata 10 + full 3): ~5,000 tokens (67% reduction)
    """
    results: List[SearchResultMetadata] | List[SearchResultFull] = Field(
        ...,
        description="Search results (metadata or full based on detail_level)"
    )
    total_found: int = Field(..., description="Total matching results before top_k limit")
    strategy_used: str = Field(..., description="Actual search strategy used (auto-routing result)")
    query_analysis: Optional[str] = Field(None, description="Query routing explanation (if strategy=auto)")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
```

#### Performance Targets

- **Metadata response**: <200ms P95, ~500 tokens for 10 results
- **Full response**: <500ms P95, ~15,000 tokens for 10 results
- **Hybrid search**: <300ms P50, <500ms P95

---

### Tool 2: find_vendor_info

**Purpose**: Retrieve entity-centric information about vendors, libraries, or frameworks from knowledge graph.

#### Input Schema (Pydantic)

```python
class VendorInfoRequest(BaseModel):
    """Request schema for find_vendor_info tool.

    MCP Tool Parameters:
    - vendor_filter: Vendor name or pattern (required)
    - entity_type: Entity type filter (optional)
    - include_relationships: Include related entities (default: True)
    - detail_level: Progressive disclosure level (metadata/full)
    """
    vendor_filter: str = Field(
        ...,
        description="Vendor name or pattern (e.g., 'OpenAI', 'Anthropic', 'Google')",
        min_length=1,
        max_length=100
    )
    entity_type: Optional[Literal["vendor", "library", "framework", "service"]] = Field(
        None,
        description="Filter by entity type (optional)"
    )
    include_relationships: bool = Field(
        default=True,
        description="Include related entities (libraries, frameworks used by vendor)"
    )
    detail_level: Literal["metadata", "full"] = Field(
        default="full",
        description="Response detail level: metadata (counts + IDs), full (complete entities)"
    )
```

#### Output Schema (Pydantic)

```python
class EntityMetadata(BaseModel):
    """Minimal entity metadata for progressive disclosure.

    Token Budget: ~30 tokens per entity
    """
    entity_id: UUID = Field(..., description="Entity UUID")
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    mention_count: int = Field(..., description="Number of times mentioned")

class EntityFull(EntityMetadata):
    """Full entity information.

    Token Budget: ~200 tokens per entity
    """
    description: Optional[str] = Field(None, description="Entity description")
    source_chunks: List[int] = Field(default_factory=list, description="Associated chunk IDs")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")

class RelationshipInfo(BaseModel):
    """Entity relationship information."""
    source_id: UUID = Field(..., description="Source entity UUID")
    target_id: UUID = Field(..., description="Target entity UUID")
    relationship_type: str = Field(..., description="Relationship type")
    source_name: str = Field(..., description="Source entity name")
    target_name: str = Field(..., description="Target entity name")

class VendorInfoResponse(BaseModel):
    """Response schema for find_vendor_info tool.

    Supports progressive disclosure:
    - metadata: Entity counts + IDs only (~100 tokens)
    - full: Complete entity details + relationships (~2000 tokens)
    """
    entities: List[EntityMetadata] | List[EntityFull] = Field(
        ...,
        description="Matching entities (metadata or full based on detail_level)"
    )
    relationships: List[RelationshipInfo] = Field(
        default_factory=list,
        description="Entity relationships (if include_relationships=True)"
    )
    total_entities: int = Field(..., description="Total matching entities")
    total_relationships: int = Field(..., description="Total relationships found")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
```

---

### Tool 3: get_full_content (Future Enhancement)

**Purpose**: Fetch full content for specific chunk IDs (progressive disclosure follow-up).

#### Input Schema (Pydantic)

```python
class GetFullContentRequest(BaseModel):
    """Request schema for get_full_content tool.

    Use Case: After receiving metadata-only results from semantic_search,
    selectively fetch full content for top 2-3 relevant results.

    Token Reduction: 500 tokens (metadata 10) + 4500 tokens (full 3) = 5000 tokens
    vs. 15,000 tokens (full 10) = 67% reduction
    """
    chunk_ids: List[int] = Field(
        ...,
        description="List of chunk IDs to fetch full content for",
        min_length=1,
        max_length=10
    )

class GetFullContentResponse(BaseModel):
    """Response schema for get_full_content tool."""
    chunks: List[SearchResultFull] = Field(..., description="Full chunk content")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
```

---

## Detailed Subtask Breakdown

### Subtask 10.1: FastMCP Server Setup (2-3 hours)

**Objective**: Initialize FastMCP server with tool registry and configuration.

#### Tasks

1. **Install FastMCP dependency** (5 min):
   ```bash
   poetry add fastmcp
   # or
   pip install fastmcp
   ```

2. **Create server initialization module** (30 min):
   - File: `src/mcp/server.py`
   - Initialize FastMCP app
   - Configure logging (StructuredLogger integration)
   - Initialize database pool (DatabasePool from core.database)
   - Initialize HybridSearch and KnowledgeGraphService
   - Register tools with FastMCP

3. **Define tool registry** (30 min):
   - File: `src/mcp/tools/__init__.py`
   - Export all tool handlers
   - Tool metadata (name, description, schema)
   - FastMCP @tool decorators

4. **Configuration management** (20 min):
   - File: `src/mcp/config.py`
   - MCPConfig Pydantic model
   - Environment variables (API_KEY, DB_URL, LOG_LEVEL)
   - Async settings (pool size, timeout, backpressure limits)

5. **Testing** (30 min):
   - File: `tests/mcp/test_server_init.py`
   - Test server initialization
   - Test tool registration
   - Test config loading
   - Test database pool initialization

#### Deliverables

- FastMCP server running with tool registry
- Configuration management
- Unit tests for initialization
- Logging integration

#### Success Criteria

- Server starts without errors
- All tools registered successfully
- Database pool initializes correctly
- 100% mypy compliance

---

### Subtask 10.2: semantic_search Tool Implementation (2-3 hours)

**Objective**: Implement semantic_search tool with progressive disclosure support.

#### Tasks

1. **Request handler** (45 min):
   - File: `src/mcp/tools/semantic_search.py`
   - Implement `semantic_search_handler()` function
   - Validate SemanticSearchRequest with Pydantic
   - Call HybridSearch.search() with parameters
   - Format response based on detail_level

2. **Response formatting** (45 min):
   - File: `src/mcp/response/formatter.py`
   - `format_search_results()` function
   - Convert SearchResult → SearchResultMetadata
   - Convert SearchResult → SearchResultFull
   - Apply detail_level filtering

3. **Error handling** (30 min):
   - Invalid query handling (empty, too long)
   - Database connection errors
   - Search timeout handling (>2s)
   - Pydantic validation errors

4. **Testing** (60 min):
   - File: `tests/mcp/test_semantic_search.py`
   - Test metadata-only responses
   - Test full-content responses
   - Test search strategies (vector, bm25, hybrid, auto)
   - Test error cases (invalid query, timeout)
   - Test performance benchmarks (<500ms)

#### Deliverables

- Functional semantic_search tool
- Progressive disclosure support
- Error handling
- Comprehensive tests

#### Success Criteria

- Tool callable from Claude Desktop
- Metadata responses <200ms P95
- Full responses <500ms P95
- 100% mypy compliance
- >80% test coverage

---

### Subtask 10.3: find_vendor_info Tool Implementation (1.5-2 hours)

**Objective**: Implement find_vendor_info tool with entity relationship traversal.

#### Tasks

1. **Request handler** (45 min):
   - File: `src/mcp/tools/vendor_search.py`
   - Implement `find_vendor_info_handler()` function
   - Validate VendorInfoRequest with Pydantic
   - Query KnowledgeGraphService for entities
   - Traverse relationships if requested

2. **Entity formatting** (30 min):
   - File: `src/mcp/response/entity_formatter.py`
   - `format_entities()` function
   - Convert Entity → EntityMetadata
   - Convert Entity → EntityFull
   - Format relationships

3. **Testing** (45 min):
   - File: `tests/mcp/test_vendor_search.py`
   - Test vendor filtering
   - Test entity type filtering
   - Test relationship traversal
   - Test detail_level variants
   - Test error cases

#### Deliverables

- Functional find_vendor_info tool
- Relationship traversal
- Progressive disclosure support
- Comprehensive tests

#### Success Criteria

- Tool callable from Claude Desktop
- Response time <300ms P95
- Relationship traversal works correctly
- 100% mypy compliance

---

### Subtask 10.4: Authentication & Security (1 hour)

**Objective**: Implement API key validation and request sanitization.

#### Tasks

1. **API key validation** (30 min):
   - File: `src/mcp/auth/api_key.py`
   - `validate_api_key()` function
   - Environment variable API_KEY
   - FastMCP middleware integration
   - Error responses for invalid keys

2. **Input sanitization** (15 min):
   - File: `src/mcp/auth/sanitizer.py`
   - Query string validation (no SQL injection)
   - Parameter bounds checking
   - XSS prevention (output escaping)

3. **Testing** (15 min):
   - File: `tests/mcp/test_auth.py`
   - Test valid API key
   - Test invalid API key
   - Test missing API key
   - Test injection attempts

#### Deliverables

- API key validation
- Input sanitization
- Security tests

#### Success Criteria

- Invalid API keys rejected
- Injection attempts blocked
- No security vulnerabilities
- 100% mypy compliance

---

### Subtask 10.5: End-to-End Testing & Performance Validation (2 hours)

**Objective**: Validate complete MCP integration with Claude Desktop.

#### Tasks

1. **E2E test setup** (30 min):
   - File: `tests/mcp/test_integration_e2e.py`
   - Configure Claude Desktop MCP client
   - Test server startup
   - Test tool discovery

2. **Functional E2E tests** (45 min):
   - Test semantic_search from Claude Desktop
   - Test find_vendor_info from Claude Desktop
   - Test progressive disclosure workflow
   - Test error handling end-to-end

3. **Performance benchmarks** (30 min):
   - File: `tests/mcp/test_performance.py`
   - Latency benchmarks (P50, P95, P99)
   - Token count validation
   - Throughput testing (concurrent requests)

4. **Documentation** (15 min):
   - File: `docs/mcp-server-usage.md`
   - Configuration instructions
   - Claude Desktop integration guide
   - Example queries
   - Troubleshooting

#### Deliverables

- E2E tests passing
- Performance benchmarks met
- Documentation complete
- Ready for production deployment

#### Success Criteria

- All tools work in Claude Desktop
- Latency targets met (<500ms P95)
- Token reduction validated (metadata vs. full)
- Documentation complete

---

## Code Structure

### Directory Layout

```
src/mcp/
├── __init__.py                      # Package initialization
├── server.py                        # FastMCP server initialization (100 lines)
├── config.py                        # Configuration management (50 lines)
│
├── tools/                           # Tool handlers
│   ├── __init__.py                  # Tool registry exports
│   ├── semantic_search.py           # semantic_search handler (150 lines)
│   ├── vendor_search.py             # find_vendor_info handler (120 lines)
│   └── handlers.py                  # Common handler utilities (80 lines)
│
├── auth/                            # Authentication & security
│   ├── __init__.py
│   ├── api_key.py                   # API key validation (60 lines)
│   └── sanitizer.py                 # Input sanitization (40 lines)
│
├── response/                        # Response formatting
│   ├── __init__.py
│   ├── formatter.py                 # Search result formatting (100 lines)
│   └── entity_formatter.py          # Entity formatting (80 lines)
│
└── schemas/                         # Pydantic schemas
    ├── __init__.py
    ├── requests.py                  # Request schemas (150 lines)
    └── responses.py                 # Response schemas (150 lines)

tests/mcp/
├── __init__.py
├── test_server_init.py              # Server initialization tests (80 lines)
├── test_semantic_search.py          # semantic_search tool tests (150 lines)
├── test_vendor_search.py            # find_vendor_info tool tests (120 lines)
├── test_auth.py                     # Authentication tests (60 lines)
├── test_integration_e2e.py          # E2E tests (100 lines)
└── test_performance.py              # Performance benchmarks (80 lines)

docs/
└── mcp-server-usage.md              # Usage documentation (200 lines)
```

**Total Lines of Code**: ~1,600 lines (implementation + tests)

---

## Testing Strategy

### Test Pyramid

```
                    /\
                   /  \
                  / E2E \              10% - 2 E2E tests (100 lines)
                 /______\
                /        \
               /Integration\           20% - 5 integration tests (200 lines)
              /____________\
             /              \
            /  Unit Tests    \         70% - 15 unit tests (300 lines)
           /__________________\
```

### Coverage Requirements

- **Overall Coverage**: >85%
- **Critical Paths**: 100% (tool handlers, auth, formatting)
- **Error Handling**: 100% (all error paths tested)
- **Performance Tests**: Required for all tools

### Test Categories

#### Unit Tests (70% of tests)

1. **Server Initialization**:
   - Test FastMCP app creation
   - Test tool registration
   - Test config loading
   - Test database pool initialization

2. **Tool Handlers**:
   - Test semantic_search with mock HybridSearch
   - Test find_vendor_info with mock KnowledgeGraphService
   - Test parameter validation
   - Test error handling

3. **Response Formatting**:
   - Test metadata-only formatting
   - Test full-content formatting
   - Test entity formatting
   - Test relationship formatting

4. **Authentication**:
   - Test API key validation
   - Test input sanitization
   - Test injection attempts

#### Integration Tests (20% of tests)

1. **Database Integration**:
   - Test semantic_search with real database
   - Test find_vendor_info with real database
   - Test concurrent requests
   - Test connection pool behavior

2. **Cross-Component Integration**:
   - Test tool → HybridSearch → database
   - Test tool → KnowledgeGraphService → database
   - Test progressive disclosure workflow

#### E2E Tests (10% of tests)

1. **Claude Desktop Integration**:
   - Test tool discovery
   - Test semantic_search from Claude Desktop
   - Test find_vendor_info from Claude Desktop

2. **Performance Benchmarks**:
   - Test latency targets (P50, P95, P99)
   - Test token counts (metadata vs. full)
   - Test throughput (concurrent requests)

---

## Progressive Disclosure Design

### Token Reduction Analysis

#### Traditional Approach (No Progressive Disclosure)

```
Query: "enterprise authentication patterns"
Results: 10 chunks × 1,500 tokens each = 15,000 tokens

Total: 15,000 tokens
Cost: $0.045 (Claude Sonnet input pricing)
Latency: 500ms
```

#### Progressive Disclosure Approach (Level 0 → Level 1)

```
Step 1: Metadata-only request
Query: "enterprise authentication patterns"
Results: 10 chunks × 50 tokens each = 500 tokens
Latency: 200ms

Step 2: Selective full-content request
Selected: Top 3 chunks × 1,500 tokens each = 4,500 tokens
Latency: 150ms

Total: 5,000 tokens (67% reduction)
Cost: $0.015 (67% cost reduction)
Latency: 350ms (30% faster)
```

### Implementation Strategy

#### Phase 1: MVP (Task 10)

- Single `detail_level` parameter (metadata/full)
- No multi-level progressive disclosure
- No caching between requests
- Simple, straightforward implementation

**Benefits**:
- Fastest implementation (6-8 hours)
- Validates FastMCP integration
- Demonstrates token reduction (67%)
- Production-ready baseline

#### Phase 2: Code Execution Integration (Future)

- Multi-level progressive disclosure (Level 0-3)
- `get_full_content()` tool for selective retrieval
- Result caching for efficiency
- Code execution sandbox support

**Benefits**:
- 90-95% token reduction (vs. 67% in Phase 1)
- Fully automated progressive disclosure
- Cost reduction: $0.045 → $0.005
- Supports complex agent workflows

---

## Code Execution Integration Roadmap

### Design for Future Compatibility

The Task 10 implementation should be designed to support future code execution integration without refactoring:

#### 1. Tool Interface Stability

**Current Design** (Task 10):
```python
@tool
def semantic_search(
    query: str,
    top_k: int = 10,
    strategy: str = "auto",
    detail_level: str = "full"
) -> SemanticSearchResponse:
    ...
```

**Future Design** (Code Execution):
```python
# SAME INTERFACE - no breaking changes
@tool
def semantic_search(
    query: str,
    top_k: int = 10,
    strategy: str = "auto",
    detail_level: str = "metadata"  # Default changes to "metadata"
) -> SemanticSearchResponse:
    ...
```

**Key Point**: Interface remains identical, only default changes.

#### 2. Progressive Disclosure Architecture

**Current Architecture** (Task 10):
```
Claude Desktop
    ↓ (MCP call with detail_level="full")
FastMCP Server
    ↓ (fetch full content)
Knowledge Graph
    ↓
PostgreSQL
```

**Future Architecture** (Code Execution):
```
Claude Desktop + Code Execution Sandbox
    ↓ (Python code: results = semantic_search(query, detail_level="metadata"))
    ↓ (Python code: analyze metadata, select top 3)
    ↓ (Python code: full = get_full_content([id1, id2, id3]))
FastMCP Server
    ↓ (2 MCP calls: metadata + selective full)
Knowledge Graph
    ↓
PostgreSQL
```

**Key Point**: Same MCP tools, different orchestration layer.

#### 3. Response Schema Compatibility

**Current Schema** (Task 10):
```python
class SemanticSearchResponse(BaseModel):
    results: List[SearchResultMetadata] | List[SearchResultFull]
    total_found: int
    strategy_used: str
    query_analysis: Optional[str]
    execution_time_ms: float
```

**Future Schema** (Code Execution):
```python
# SAME SCHEMA - no changes needed
class SemanticSearchResponse(BaseModel):
    results: List[SearchResultMetadata] | List[SearchResultFull]
    total_found: int
    strategy_used: str
    query_analysis: Optional[str]
    execution_time_ms: float
    # Optional future additions (backward compatible):
    # cache_hit: Optional[bool] = None
    # result_ids: Optional[List[int]] = None
```

**Key Point**: Schema is forward-compatible with optional fields.

### Integration Phases

#### Phase 1: MVP (Task 10) - 6-8 hours

**Scope**:
- FastMCP server with 2 tools (semantic_search, find_vendor_info)
- Single detail_level parameter (metadata/full)
- No caching, no multi-level disclosure
- Direct integration with HybridSearch and KnowledgeGraphService

**Deliverables**:
- Working MCP server
- Claude Desktop integration
- 67% token reduction demonstrated
- Performance benchmarks met

#### Phase 2: Code Execution Integration - 12-16 hours (future)

**Scope**:
- Add `get_full_content()` tool
- Add result caching (Redis/in-memory)
- Add code execution sandbox
- Add multi-level progressive disclosure
- Update default detail_level to "metadata"

**Deliverables**:
- Code execution sandbox operational
- 90-95% token reduction achieved
- Cost reduction: $0.045 → $0.005
- Agent adoption >50% within 90 days

#### Phase 3: Production Optimization - 8-10 hours (future)

**Scope**:
- Add monitoring and observability
- Add rate limiting and quotas
- Add cache warming strategies
- Add adaptive detail_level selection

**Deliverables**:
- Production-grade reliability (99.9%)
- Cost optimization (<$0.01 per query)
- Performance optimization (<200ms P95)

---

## Implementation Timeline

### Day 1 (3-4 hours)

**Morning Session** (2-2.5 hours):
- ✅ Subtask 10.1a: Install FastMCP, create server initialization (1 hour)
- ✅ Subtask 10.1b: Define Pydantic schemas (requests.py, responses.py) (1-1.5 hours)

**Afternoon Session** (1-1.5 hours):
- ✅ Subtask 10.2a: Implement semantic_search handler (1-1.5 hours)

**Deliverables**:
- FastMCP server running
- Pydantic schemas defined
- semantic_search tool callable (no tests yet)

---

### Day 2 (2-3 hours)

**Morning Session** (1.5-2 hours):
- ✅ Subtask 10.3a: Implement find_vendor_info handler (1-1.5 hours)
- ✅ Subtask 10.4a: Implement API key validation (0.5 hour)

**Afternoon Session** (0.5-1 hour):
- ✅ Subtask 10.4b: Response formatting and error handling (0.5-1 hour)

**Deliverables**:
- Both tools callable
- API key validation working
- Error handling implemented

---

### Day 3 (1-2 hours)

**Testing & Validation Session** (1-2 hours):
- ✅ Subtask 10.5a: Unit tests for all tools (0.5-1 hour)
- ✅ Subtask 10.5b: Integration tests with real database (0.5 hour)
- ✅ Subtask 10.5c: E2E test with Claude Desktop (0.5 hour)

**Deliverables**:
- All tests passing
- E2E integration validated
- Performance benchmarks met
- Documentation complete

---

### Total Estimated Time: 6-9 hours

- **Day 1**: 3-4 hours (server setup + semantic_search)
- **Day 2**: 2-3 hours (find_vendor_info + auth + formatting)
- **Day 3**: 1-2 hours (testing + validation)

**Buffer**: 1-2 hours for unexpected issues, debugging, documentation

---

## Success Criteria

### Functional Requirements

- ✅ FastMCP server starts without errors
- ✅ `semantic_search` tool callable from Claude Desktop
- ✅ `find_vendor_info` tool callable from Claude Desktop
- ✅ Progressive disclosure (metadata/full) working
- ✅ API key validation enforced
- ✅ Error handling comprehensive
- ✅ 100% mypy compliance

### Performance Requirements

- ✅ semantic_search (metadata): <200ms P95
- ✅ semantic_search (full): <500ms P95
- ✅ find_vendor_info: <300ms P95
- ✅ Server startup: <5 seconds
- ✅ Concurrent requests: 10 RPS sustained

### Quality Requirements

- ✅ Test coverage: >85% overall, 100% critical paths
- ✅ All tests passing (unit, integration, E2E)
- ✅ No security vulnerabilities
- ✅ No type errors (mypy --strict)
- ✅ Documentation complete

### Token Reduction Validation

- ✅ Metadata response: ~50 tokens/result (vs. 1,500 baseline)
- ✅ Full response: ~1,500 tokens/result (baseline)
- ✅ Progressive workflow: ~5,000 tokens (vs. 15,000 baseline, 67% reduction)

---

## Next Steps

### Immediate Actions (Today)

1. **Review this plan** (10 min):
   - Confirm scope and timeline
   - Clarify any questions
   - Get approval to proceed

2. **Start implementation** (3-4 hours):
   - Begin Subtask 10.1a: FastMCP server initialization
   - Begin Subtask 10.1b: Define Pydantic schemas
   - Begin Subtask 10.2a: Implement semantic_search handler

3. **Commit frequently** (every 30-50 lines):
   - Follow micro-commit discipline
   - Use task-master to track progress
   - Update subtask status after each milestone

### Tomorrow

1. **Complete remaining tools** (2-3 hours):
   - Subtask 10.3: find_vendor_info
   - Subtask 10.4: Auth + formatting

2. **Testing & validation** (1-2 hours):
   - Subtask 10.5: E2E tests + performance

3. **Documentation** (30 min):
   - Complete mcp-server-usage.md
   - Update CLAUDE.md with MCP context

### Day 3

1. **Production readiness** (1 hour):
   - Final E2E validation with Claude Desktop
   - Performance benchmark verification
   - Security audit

2. **Task completion** (30 min):
   - Mark Task 10 as DONE
   - Create PR for review
   - Update session handoff

---

## Recommended First Steps (Immediate)

### Step 1: Install FastMCP (5 min)

```bash
# Add FastMCP to dependencies
poetry add fastmcp
# or
pip install fastmcp

# Verify installation
python -c "import fastmcp; print(fastmcp.__version__)"
```

### Step 2: Create Basic Server Structure (15 min)

```bash
# Create directory structure
mkdir -p src/mcp/tools src/mcp/auth src/mcp/response src/mcp/schemas
touch src/mcp/__init__.py
touch src/mcp/server.py
touch src/mcp/config.py
touch src/mcp/tools/__init__.py
touch src/mcp/schemas/requests.py
touch src/mcp/schemas/responses.py
```

### Step 3: Define Core Pydantic Schemas (30 min)

Start with `src/mcp/schemas/requests.py`:
- SemanticSearchRequest
- VendorInfoRequest

Then `src/mcp/schemas/responses.py`:
- SearchResultMetadata
- SearchResultFull
- SemanticSearchResponse
- EntityMetadata
- EntityFull
- VendorInfoResponse

### Step 4: Implement Server Initialization (30 min)

File: `src/mcp/server.py`

Key components:
- Initialize FastMCP app
- Initialize DatabasePool
- Initialize HybridSearch
- Initialize KnowledgeGraphService
- Register tools
- Configure logging

### Step 5: Implement semantic_search Handler (1 hour)

File: `src/mcp/tools/semantic_search.py`

Key components:
- @tool decorator
- Request validation (Pydantic)
- Call HybridSearch.search()
- Format response based on detail_level
- Error handling

---

## Appendix: Key Code Snippets

### FastMCP Server Initialization Template

```python
"""FastMCP server initialization for BMCIS Knowledge MCP.

Type-safe MCP server exposing semantic_search and find_vendor_info tools
to Claude Desktop. Supports progressive disclosure for token efficiency.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP
from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.knowledge_graph.service_factory import create_knowledge_graph_service
from src.search.hybrid_search import HybridSearch

logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Initialize FastMCP app
mcp = FastMCP("bmcis-knowledge-mcp")

# Global state (initialized on startup)
_db_pool: DatabasePool | None = None
_hybrid_search: HybridSearch | None = None
_graph_service: Any | None = None


def initialize_server() -> None:
    """Initialize server dependencies (database, search, graph).

    Called on server startup. Initializes:
    - DatabasePool (PostgreSQL connection pool)
    - HybridSearch (vector + BM25 search)
    - KnowledgeGraphService (entity/relationship queries)

    Raises:
        RuntimeError: If initialization fails
    """
    global _db_pool, _hybrid_search, _graph_service

    logger.info("Initializing BMCIS Knowledge MCP server...")

    # Load settings
    settings = get_settings()

    # Initialize database pool
    _db_pool = DatabasePool()
    logger.info("Database pool initialized")

    # Initialize hybrid search
    _hybrid_search = HybridSearch(
        db_pool=_db_pool,
        settings=settings,
        logger=logger
    )
    logger.info("HybridSearch initialized")

    # Initialize knowledge graph service
    _graph_service = create_knowledge_graph_service(_db_pool)
    logger.info("KnowledgeGraphService initialized")

    logger.info("BMCIS Knowledge MCP server ready")


def get_hybrid_search() -> HybridSearch:
    """Get initialized HybridSearch instance."""
    if _hybrid_search is None:
        raise RuntimeError("Server not initialized. Call initialize_server() first.")
    return _hybrid_search


def get_graph_service() -> Any:
    """Get initialized KnowledgeGraphService instance."""
    if _graph_service is None:
        raise RuntimeError("Server not initialized. Call initialize_server() first.")
    return _graph_service


# Tool imports (register tools with FastMCP)
from src.mcp.tools import semantic_search, find_vendor_info

# Initialize on module load
initialize_server()
```

### semantic_search Tool Handler Template

```python
"""semantic_search tool implementation for FastMCP.

Hybrid semantic search combining vector similarity and BM25 keyword matching.
Supports progressive disclosure (metadata/full) for token efficiency.
"""

from __future__ import annotations

import time
from typing import List

from src.mcp.schemas.requests import SemanticSearchRequest
from src.mcp.schemas.responses import (
    SemanticSearchResponse,
    SearchResultMetadata,
    SearchResultFull,
)
from src.mcp.server import get_hybrid_search, mcp
from src.search.results import SearchResult


def format_metadata_result(result: SearchResult) -> SearchResultMetadata:
    """Convert SearchResult to metadata-only format."""
    return SearchResultMetadata(
        chunk_id=result.chunk_id,
        entity_id=None,  # TODO: Add entity_id to SearchResult
        source_file=result.source_file,
        entity_type=None,  # TODO: Extract from result
        hybrid_score=result.hybrid_score,
        rank=result.rank,
    )


def format_full_result(result: SearchResult) -> SearchResultFull:
    """Convert SearchResult to full-content format."""
    return SearchResultFull(
        chunk_id=result.chunk_id,
        entity_id=None,  # TODO: Add entity_id to SearchResult
        source_file=result.source_file,
        entity_type=None,  # TODO: Extract from result
        hybrid_score=result.hybrid_score,
        rank=result.rank,
        chunk_text=result.chunk_text,
        context_header=result.context_header,
        vendor_names=result.vendor_names,
        section_title=result.section_title,
    )


@mcp.tool()
def semantic_search(
    query: str,
    top_k: int = 10,
    strategy: str = "auto",
    min_score: float = 0.0,
    detail_level: str = "full",
) -> SemanticSearchResponse:
    """Hybrid semantic search combining vector similarity and BM25 keyword matching.

    Args:
        query: Search query (natural language or keywords)
        top_k: Number of results to return (1-50)
        strategy: Search strategy (auto/vector/bm25/hybrid)
        min_score: Minimum relevance score threshold (0.0-1.0)
        detail_level: Response detail level (metadata/full)

    Returns:
        SemanticSearchResponse with results, metadata, and timing

    Example:
        # Metadata-only (fast, token-efficient)
        >>> semantic_search("JWT authentication", detail_level="metadata")

        # Full content (slower, more tokens)
        >>> semantic_search("JWT authentication", detail_level="full")
    """
    # Validate request
    request = SemanticSearchRequest(
        query=query,
        top_k=top_k,
        strategy=strategy,  # type: ignore[arg-type]
        min_score=min_score,
        detail_level=detail_level,  # type: ignore[arg-type]
    )

    # Execute search
    start_time = time.time()
    hybrid_search = get_hybrid_search()

    results: List[SearchResult] = hybrid_search.search(
        query=request.query,
        top_k=request.top_k,
        strategy=request.strategy if request.strategy != "auto" else None,
        min_score=request.min_score,
    )

    execution_time_ms = (time.time() - start_time) * 1000

    # Format results based on detail_level
    if request.detail_level == "metadata":
        formatted_results = [format_metadata_result(r) for r in results]
    else:
        formatted_results = [format_full_result(r) for r in results]

    return SemanticSearchResponse(
        results=formatted_results,  # type: ignore[arg-type]
        total_found=len(results),
        strategy_used=request.strategy,
        query_analysis=None,  # TODO: Add query analysis
        execution_time_ms=execution_time_ms,
    )
```

---

## Document Status

**Status**: COMPLETE - Ready for implementation
**Review Status**: Pending approval
**Last Updated**: 2025-11-09 14:49
**Next Review**: After Day 1 implementation

---

**Generated with Claude Code**

Co-Authored-By: Claude <noreply@anthropic.com>
