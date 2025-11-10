# Task 10 + Code Execution MCP: Comprehensive Architecture Analysis

**Document Metadata**
- **Date**: 2025-11-09
- **Time**: 14:49
- **Scope**: Task 10 FastMCP Integration + Future Code Execution MCP Integration
- **Branch**: feat/task-10-fastmcp-integration
- **Author**: Architecture Review Subagent

---

## Executive Summary

This document provides a comprehensive architectural design for **Task 10 (FastMCP Server Integration)** with forward compatibility for the **Code Execution MCP** system described in `PRD_CODE_EXECUTION_WITH_MCP.md`. The analysis reveals a strategic opportunity to implement progressive disclosure patterns in Task 10 that will enable seamless integration with code execution capabilities, achieving 90-95% token reduction while maintaining architectural simplicity.

**Key Recommendation**: Implement Task 10 with **tiered response modes** (metadata-only, metadata+preview, full-content) to support both traditional FastMCP usage AND future code execution integration without architectural rework.

**Impact Assessment**: HIGH - This decision affects API design, caching strategy, response formats, and future extensibility.

---

## Table of Contents

1. [Context & Requirements](#1-context--requirements)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Component Design](#3-component-design)
4. [Integration Patterns](#4-integration-patterns)
5. [Data Flow Analysis](#5-data-flow-analysis)
6. [Progressive Disclosure Design](#6-progressive-disclosure-design)
7. [Tool Interface Specifications](#7-tool-interface-specifications)
8. [Security Architecture](#8-security-architecture)
9. [Caching Strategy](#9-caching-strategy)
10. [Decision Matrix: MVP vs Full Integration](#10-decision-matrix-mvp-vs-full-integration)
11. [Risk Assessment](#11-risk-assessment)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Recommendations](#13-recommendations)

---

## 1. Context & Requirements

### 1.1 Task 10 Requirements (FastMCP Server)

**Primary Goal**: Expose BMCIS Knowledge Graph to Claude Desktop via FastMCP server.

**Core Tools**:
- `semantic_search`: Hybrid search (BM25 + vector) with boosting and reranking
- `find_vendor_info`: Vendor-specific entity and relationship queries

**Performance Targets**:
- Latency: <500ms p95 for search operations
- Throughput: Support multiple concurrent Claude Desktop sessions
- Reliability: >99% uptime with graceful degradation

**Integration Points**:
- Existing search infrastructure (`src/search/hybrid_search.py`)
- Knowledge graph service (`src/knowledge_graph/graph_service.py`)
- LRU cache layer (`src/knowledge_graph/cache.py`)

### 1.2 Code Execution MCP Opportunity

**PRD Key Insights**:
- **Token Reduction**: 90-95% reduction via progressive disclosure (150K → 7.5-15K tokens)
- **Progressive Disclosure Pattern**: Metadata first → selective full content
- **Performance**: 3-4x latency improvement (1,200ms → 300-500ms)
- **Architecture**: Code execution sandbox calls MCP tools in-memory (no round-trips)

**Tiered Disclosure Levels** (from PRD):
- **Level 0**: IDs only (100-500 tokens) - existence checks, counts
- **Level 1**: Metadata + signatures (2K-4K tokens) - understanding what exists
- **Level 2**: Metadata + truncated content (5K-10K tokens) - implementation approach
- **Level 3**: Full content (10K-50K+ tokens) - deep analysis

**Integration Question**: Can Task 10 implement these response tiers NOW to avoid architectural refactoring later?

### 1.3 Existing Infrastructure Analysis

**Search System** (`src/search/`):
- **HybridSearch**: BM25 + vector with RRF fusion ✅
- **CrossEncoderReranker**: Semantic reranking ✅
- **BoostingSystem**: Multi-factor boosting (vendor, recency, entity_type) ✅
- **FilterExpression**: Complex filtering logic ✅
- **QueryCache**: Redis-compatible caching layer ✅

**Knowledge Graph** (`src/knowledge_graph/`):
- **KnowledgeGraphService**: Entity/relationship queries with LRU cache ✅
- **QueryRepository**: Optimized 1-hop, 2-hop traversals (P95 <10ms) ✅
- **Cache**: LRU implementation with configurable eviction ✅

**Gap Analysis**:
- ❌ No tiered response formatting (only full content)
- ❌ No metadata-only extraction capability
- ❌ No progressive disclosure API design
- ⚠️ Caching layer doesn't separate metadata from content

---

## 2. System Architecture Overview

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Claude Desktop / Agent                           │
│                                                                         │
│  Traditional Usage:          Code Execution Usage:                     │
│  - Direct tool calls         - execute_code tool                       │
│  - Full responses            - Calls MCP tools in-memory               │
│  - Higher token cost         - Progressive disclosure                  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastMCP Server (Task 10)                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                  MCP Tool Registry                           │     │
│  │  - semantic_search (w/ response_mode parameter)             │     │
│  │  - find_vendor_info (w/ include_content parameter)          │     │
│  │  - [future] execute_code                                    │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                     │                                  │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              Request Handler Layer                           │     │
│  │  - Input validation & authentication                         │     │
│  │  - Response mode selection (metadata/preview/full)           │     │
│  │  - Error handling & rate limiting                            │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                     │                                  │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │            Response Formatter Layer                          │     │
│  │  - Metadata extraction                                       │     │
│  │  - Content truncation/preview generation                     │     │
│  │  - Token counting & optimization                             │     │
│  │  - Claude Desktop format compliance                          │     │
│  └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                               │
│                                                                         │
│  ┌──────────────────────┐              ┌──────────────────────┐        │
│  │  Search Orchestrator │              │  Graph Query Service │        │
│  │  - HybridSearch      │              │  - Entity lookups    │        │
│  │  - Boosting          │              │  - Relationship      │        │
│  │  - Reranking         │              │    traversal         │        │
│  │  - Filtering         │              │  - Vendor filtering  │        │
│  └──────────────────────┘              └──────────────────────┘        │
│           │                                      │                     │
│           └──────────────┬───────────────────────┘                     │
│                          ▼                                             │
│                 ┌─────────────────┐                                    │
│                 │  Cache Layer    │                                    │
│                 │  - Query cache  │                                    │
│                 │  - Entity cache │                                    │
│                 │  - Metadata     │                                    │
│                 │    separation   │                                    │
│                 └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                     │
│                                                                         │
│  ┌──────────────────────┐              ┌──────────────────────┐        │
│  │  PostgreSQL          │              │  Vector Store        │        │
│  │  - Knowledge graph   │              │  - Embeddings        │        │
│  │  - Entities          │              │  - Similarity search │        │
│  │  - Relationships     │              │                      │        │
│  │  - Metadata          │              │                      │        │
│  └──────────────────────┘              └──────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dual-Path Architecture: Traditional vs Code Execution

```
TRADITIONAL FASTMCP PATH (Higher tokens, simpler usage)
========================================================
1. Claude Desktop → semantic_search(query, top_k=10)
2. FastMCP → HybridSearch → full results with content
3. Response: 10 full documents (~30K-50K tokens)
4. Claude analyzes, makes follow-up calls if needed
5. Total: 4-5 round trips, 150K tokens, 1,200ms

CODE EXECUTION PATH (Lower tokens, requires code)
==================================================
1. Claude Desktop → execute_code(python_code)
2. Code executes IN SANDBOX (no round trips):
   a. semantic_search(query, response_mode="metadata", top_k=50)
      → Returns: IDs + titles + types (2K tokens)
   b. Analyze metadata, select top 3 relevant
   c. semantic_search(query, response_mode="full", filter_ids=[...])
      → Returns: Full content for 3 docs (12K tokens)
3. Total: 1 round trip, 14K tokens, 350ms

KEY DIFFERENCE: Code execution eliminates round-trip overhead and enables
intelligent filtering BEFORE fetching full content.
```

### 2.3 Architecture Principles

1. **Progressive Disclosure**: Support metadata → preview → full content tiers
2. **Backward Compatibility**: Traditional full-response mode remains default
3. **Forward Compatibility**: Design enables seamless code execution integration
4. **Performance First**: Metadata responses must be <100ms, full <500ms
5. **Separation of Concerns**: Response formatting is independent of business logic
6. **Security by Design**: Authentication, rate limiting, input validation at boundary

---

## 3. Component Design

### 3.1 FastMCP Server Core

**Responsibilities**:
- Tool registration and discovery
- Request routing and dispatch
- Authentication and authorization
- Error handling and logging
- Health monitoring

**Technology**: FastMCP (Node.js) wrapping Python backend via subprocess/HTTP

**Key Files** (to be created):
```
src/mcp_server/
├── __init__.py
├── server.py              # FastMCP server initialization
├── tools.py               # Tool definitions and registry
├── handlers.py            # Request handlers
├── auth.py                # API key validation
├── errors.py              # Error response formatting
└── config.py              # Server configuration
```

**Configuration**:
```python
@dataclass
class MCPServerConfig:
    host: str = "localhost"
    port: int = 3000
    api_key: Optional[str] = None  # For authentication
    rate_limit_rpm: int = 100      # Requests per minute
    max_response_size: int = 10_000_000  # 10MB
    timeout_seconds: int = 30
    enable_metadata_mode: bool = True  # Progressive disclosure
```

### 3.2 Response Formatter Layer

**NEW COMPONENT** - Critical for progressive disclosure

**Responsibilities**:
- Extract metadata from search results
- Generate previews/truncations
- Count tokens and optimize responses
- Format for Claude Desktop compatibility

**Key Classes**:

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class ResponseMode(Enum):
    """Progressive disclosure tiers"""
    IDS_ONLY = "ids_only"           # 100-500 tokens
    METADATA = "metadata"            # 2K-4K tokens
    PREVIEW = "preview"              # 5K-10K tokens
    FULL = "full"                    # 10K-50K+ tokens

@dataclass
class SearchResultMetadata:
    """Compact metadata representation"""
    id: str
    title: str
    entity_type: str
    vendor: Optional[str]
    source_file: str
    score: float
    created_at: str

    def to_dict(self) -> dict:
        """Serialize for JSON response"""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.entity_type,
            "vendor": self.vendor,
            "file": self.source_file,
            "score": round(self.score, 3),
            "created": self.created_at
        }

@dataclass
class SearchResultPreview:
    """Metadata + truncated content"""
    metadata: SearchResultMetadata
    preview_text: str  # First 200 chars
    preview_tokens: int

@dataclass
class SearchResultFull:
    """Complete result with full content"""
    metadata: SearchResultMetadata
    content: str
    tokens: int
    relationships: Optional[List[dict]] = None

class ResponseFormatter:
    """Format search results based on response mode"""

    def __init__(self, max_preview_chars: int = 200):
        self.max_preview_chars = max_preview_chars

    def format_results(
        self,
        results: List[SearchResult],
        mode: ResponseMode,
        include_relationships: bool = False
    ) -> dict:
        """
        Format results based on progressive disclosure tier.

        Args:
            results: Raw search results from HybridSearch
            mode: ResponseMode enum (ids/metadata/preview/full)
            include_relationships: Add related entities (graph traversal)

        Returns:
            Formatted response dict ready for MCP serialization
        """
        if mode == ResponseMode.IDS_ONLY:
            return self._format_ids(results)
        elif mode == ResponseMode.METADATA:
            return self._format_metadata(results)
        elif mode == ResponseMode.PREVIEW:
            return self._format_preview(results)
        else:  # FULL
            return self._format_full(results, include_relationships)

    def _format_ids(self, results: List[SearchResult]) -> dict:
        """IDs only - minimal token usage"""
        return {
            "mode": "ids_only",
            "count": len(results),
            "ids": [r.chunk_id for r in results],
            "estimated_tokens": len(results) * 10  # ~10 tokens per ID
        }

    def _format_metadata(self, results: List[SearchResult]) -> dict:
        """Metadata without content - ~200-400 tokens per result"""
        metadata = [
            SearchResultMetadata(
                id=str(r.chunk_id),
                title=r.title or r.source_file,
                entity_type=r.entity_type or "unknown",
                vendor=r.vendor,
                source_file=r.source_file,
                score=r.hybrid_score,
                created_at=r.created_at.isoformat() if r.created_at else None
            ).to_dict()
            for r in results
        ]

        return {
            "mode": "metadata",
            "count": len(results),
            "results": metadata,
            "estimated_tokens": len(results) * 300  # ~300 tokens per metadata entry
        }

    def _format_preview(self, results: List[SearchResult]) -> dict:
        """Metadata + truncated content preview"""
        previews = []
        total_tokens = 0

        for r in results:
            metadata = SearchResultMetadata(...)  # Same as above
            preview_text = r.content[:self.max_preview_chars] + "..."
            preview_tokens = len(preview_text) // 4  # Rough estimate

            previews.append({
                "metadata": metadata.to_dict(),
                "preview": preview_text,
                "preview_tokens": preview_tokens,
                "has_more": len(r.content) > self.max_preview_chars
            })
            total_tokens += preview_tokens

        return {
            "mode": "preview",
            "count": len(results),
            "results": previews,
            "estimated_tokens": total_tokens
        }

    def _format_full(
        self,
        results: List[SearchResult],
        include_relationships: bool
    ) -> dict:
        """Full content with optional relationship data"""
        full_results = []
        total_tokens = 0

        for r in results:
            metadata = SearchResultMetadata(...)
            content_tokens = len(r.content) // 4

            result_dict = {
                "metadata": metadata.to_dict(),
                "content": r.content,
                "tokens": content_tokens
            }

            if include_relationships:
                # Query knowledge graph for related entities
                result_dict["relationships"] = self._get_relationships(r.chunk_id)

            full_results.append(result_dict)
            total_tokens += content_tokens

        return {
            "mode": "full",
            "count": len(results),
            "results": full_results,
            "estimated_tokens": total_tokens
        }
```

### 3.3 Request Handler Layer

**Responsibilities**:
- Parse and validate MCP requests
- Authenticate API keys
- Route to business logic
- Apply response formatting
- Handle errors gracefully

**Key Implementation**:

```python
from typing import Optional, Any
from fastmcp import FastMCP
from src.search.hybrid_search import HybridSearch
from src.knowledge_graph.graph_service import KnowledgeGraphService

class MCPRequestHandler:
    """Handle MCP tool invocations"""

    def __init__(
        self,
        hybrid_search: HybridSearch,
        graph_service: KnowledgeGraphService,
        formatter: ResponseFormatter,
        config: MCPServerConfig
    ):
        self.search = hybrid_search
        self.graph = graph_service
        self.formatter = formatter
        self.config = config

    async def handle_semantic_search(
        self,
        query: str,
        top_k: int = 10,
        response_mode: str = "full",  # "ids_only"|"metadata"|"preview"|"full"
        vendor_filter: Optional[str] = None,
        min_score: float = 0.0,
        include_relationships: bool = False
    ) -> dict:
        """
        Semantic search with progressive disclosure support.

        Args:
            query: Search query string
            top_k: Number of results to return
            response_mode: Progressive disclosure tier
            vendor_filter: Optional vendor name filter
            min_score: Minimum relevance score threshold
            include_relationships: Include related entities (graph traversal)

        Returns:
            Formatted search results based on response_mode

        Raises:
            ValueError: Invalid parameters
            AuthenticationError: Invalid API key
            RateLimitError: Too many requests
        """
        # Validate inputs
        self._validate_search_params(query, top_k, response_mode)

        # Execute search
        results = await self.search.search(
            query=query,
            top_k=top_k,
            strategy="hybrid",  # Always use hybrid for best quality
            min_score=min_score
        )

        # Apply vendor filter if specified
        if vendor_filter:
            results = [r for r in results if r.vendor == vendor_filter]

        # Format based on response mode
        mode_enum = ResponseMode(response_mode)
        formatted = self.formatter.format_results(
            results=results,
            mode=mode_enum,
            include_relationships=include_relationships
        )

        return formatted

    async def handle_find_vendor_info(
        self,
        vendor_name: str,
        include_content: bool = False,  # Progressive disclosure flag
        max_results: int = 50
    ) -> dict:
        """
        Find vendor-specific entities and relationships.

        Args:
            vendor_name: Vendor identifier
            include_content: If True, include full content; else metadata only
            max_results: Maximum entities to return

        Returns:
            Vendor entities and relationships
        """
        # Query knowledge graph for vendor entities
        entities = await self.graph.find_entities_by_vendor(
            vendor=vendor_name,
            limit=max_results
        )

        # Format based on include_content flag
        if include_content:
            mode = ResponseMode.FULL
        else:
            mode = ResponseMode.METADATA

        formatted = self.formatter.format_results(
            results=entities,
            mode=mode,
            include_relationships=True  # Always include for vendor info
        )

        return formatted

    def _validate_search_params(
        self,
        query: str,
        top_k: int,
        response_mode: str
    ) -> None:
        """Validate search parameters"""
        if not query or len(query) < 2:
            raise ValueError("Query must be at least 2 characters")

        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")

        valid_modes = {"ids_only", "metadata", "preview", "full"}
        if response_mode not in valid_modes:
            raise ValueError(f"Invalid response_mode: {response_mode}")
```

### 3.4 Authentication & Security

**API Key Validation**:
```python
class APIKeyAuth:
    """Simple API key authentication for MCP server"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MCP_API_KEY")

    def validate(self, request_key: Optional[str]) -> bool:
        """Validate API key from request"""
        if not self.api_key:
            return True  # No auth required if not configured

        return request_key == self.api_key

    def require_auth(self, func):
        """Decorator to require authentication"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_key = kwargs.get("api_key")
            if not self.validate(request_key):
                raise AuthenticationError("Invalid API key")
            return await func(*args, **kwargs)
        return wrapper
```

**Rate Limiting**:
```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests_per_minute: int = 100):
        self.rpm = requests_per_minute
        self.buckets = defaultdict(list)  # client_id → [timestamp, ...]

    def check_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove old timestamps
        self.buckets[client_id] = [
            ts for ts in self.buckets[client_id]
            if ts > cutoff
        ]

        # Check limit
        if len(self.buckets[client_id]) >= self.rpm:
            return False

        # Add current request
        self.buckets[client_id].append(now)
        return True
```

---

## 4. Integration Patterns

### 4.1 Traditional FastMCP Integration

**Use Case**: Direct tool calls from Claude Desktop

**Flow**:
```python
# Claude Desktop makes tool call
request = {
    "tool": "semantic_search",
    "arguments": {
        "query": "JWT authentication best practices",
        "top_k": 10,
        "response_mode": "full"  # Default for traditional usage
    }
}

# FastMCP server processes
response = await handler.handle_semantic_search(**request["arguments"])

# Response structure (full mode)
{
    "mode": "full",
    "count": 10,
    "results": [
        {
            "metadata": {
                "id": "chunk_123",
                "title": "JWT Implementation Guide",
                "type": "documentation",
                "vendor": "openai",
                "file": "docs/auth/jwt.md",
                "score": 0.892,
                "created": "2024-11-01T12:00:00Z"
            },
            "content": "# JWT Authentication\n\nJSON Web Tokens (JWT)...",
            "tokens": 1250
        },
        # ... 9 more results
    ],
    "estimated_tokens": 12500
}
```

**Token Usage**: ~12,500 tokens for 10 results

### 4.2 Code Execution MCP Integration

**Use Case**: Agent writes Python code that calls MCP tools in-memory

**Flow**:
```python
# Agent submits code to execute_code tool
code = """
from src.mcp_server.client import MCPClient

# Create in-memory client (no network overhead)
client = MCPClient()

# Step 1: Get metadata for 50 results (2K tokens)
metadata_results = await client.semantic_search(
    query="JWT authentication best practices",
    top_k=50,
    response_mode="metadata"  # Only IDs, titles, scores
)

# Step 2: Analyze metadata in Python (zero tokens!)
relevant_ids = [
    r["metadata"]["id"]
    for r in metadata_results["results"]
    if r["metadata"]["vendor"] == "openai"
    and r["metadata"]["score"] > 0.7
][:3]  # Top 3 OpenAI docs

# Step 3: Fetch full content for selected IDs (12K tokens)
full_results = await client.semantic_search(
    query="JWT authentication",
    response_mode="full",
    filter_ids=relevant_ids  # Only fetch these 3
)

# Return results (assigned to __result__)
__result__ = full_results
"""

# Execution sandbox runs code
result = await executor.execute(code)

# Total tokens: 2K (metadata) + 12K (3 full docs) = 14K tokens
# vs Traditional: 50 full docs = ~62K tokens
# Savings: 77% reduction
```

**Key Difference**: Code execution enables **filtering before fetching full content**, which is impossible with traditional tool calling.

### 4.3 Hybrid Pattern: Progressive Refinement

**Use Case**: Agent iteratively refines search with metadata → preview → full

**Flow**:
```python
# Iteration 1: Get metadata for broad search
initial = await semantic_search(
    query="authentication",
    top_k=100,
    response_mode="metadata"
)

# Agent analyzes, narrows to specific types
jwt_candidates = filter_by_type(initial, "jwt")

# Iteration 2: Get previews for candidates
previews = await semantic_search(
    query="JWT refresh token rotation",
    response_mode="preview",
    filter_ids=jwt_candidates
)

# Agent selects best matches
final_ids = select_top_3(previews)

# Iteration 3: Get full content for final selection
final = await semantic_search(
    response_mode="full",
    filter_ids=final_ids
)

# Total: 4K (metadata) + 8K (previews) + 12K (full) = 24K tokens
# vs Traditional: 3 searches × 30K = 90K tokens
# Savings: 73% reduction
```

---

## 5. Data Flow Analysis

### 5.1 Request Flow: Traditional FastMCP

```
┌──────────────┐
│Claude Desktop│
└──────┬───────┘
       │ semantic_search(query, top_k=10, mode="full")
       ▼
┌─────────────────────────────────────────────┐
│          FastMCP Server                     │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Validate request & authenticate   │  │
│  │ 2. Parse arguments                   │  │
│  │ 3. Route to handler                  │  │
│  └──────────┬───────────────────────────┘  │
└─────────────┼──────────────────────────────┘
              ▼
┌─────────────────────────────────────────────┐
│      Request Handler                        │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Call HybridSearch.search()        │  │
│  │ 2. Apply filters (vendor, score)     │  │
│  │ 3. Pass to ResponseFormatter         │  │
│  └──────────┬───────────────────────────┘  │
└─────────────┼──────────────────────────────┘
              ▼
┌─────────────────────────────────────────────┐
│      HybridSearch                           │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Check query cache (if enabled)    │  │
│  │ 2. Execute BM25 + Vector in parallel │  │
│  │ 3. RRF fusion                        │  │
│  │ 4. Apply boosting                    │  │
│  │ 5. Reranking (if enabled)            │  │
│  │ 6. Return SearchResult[]             │  │
│  └──────────┬───────────────────────────┘  │
└─────────────┼──────────────────────────────┘
              ▼
┌─────────────────────────────────────────────┐
│      ResponseFormatter                      │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Extract metadata from results     │  │
│  │ 2. Include full content (mode=full)  │  │
│  │ 3. Add relationships (if requested)  │  │
│  │ 4. Count tokens                      │  │
│  │ 5. Format for MCP                    │  │
│  └──────────┬───────────────────────────┘  │
└─────────────┼──────────────────────────────┘
              │
              ▼
       ┌────────────┐
       │   Cache    │ ← Update query cache with results
       └────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│  MCP Response (JSON)                         │
│  {                                           │
│    "mode": "full",                           │
│    "count": 10,                              │
│    "results": [...],  ← Full content         │
│    "estimated_tokens": 12500                 │
│  }                                           │
└──────────────────────────────────────────────┘
              │
              ▼
       ┌──────────────┐
       │Claude Desktop│
       └──────────────┘
```

**Latency Breakdown**:
- Request validation: 5ms
- HybridSearch execution: 250-400ms
  - BM25 search: 50ms
  - Vector search: 100ms
  - RRF fusion: 50ms
  - Boosting: 10ms
  - Reranking: 150ms (optional)
- Response formatting: 20ms
- Network overhead: 50ms
- **Total**: 325-475ms (within <500ms target)

### 5.2 Request Flow: Code Execution Path

```
┌──────────────┐
│Claude Desktop│
└──────┬───────┘
       │ execute_code(python_code)
       ▼
┌─────────────────────────────────────────────┐
│       Code Execution Sandbox (Future)       │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Validate code (AST analysis)      │  │
│  │ 2. Spawn subprocess                  │  │
│  │ 3. Execute code with MCP client      │  │
│  └──────────┬───────────────────────────┘  │
└─────────────┼──────────────────────────────┘
              │
              │ WITHIN SUBPROCESS (no network calls):
              │
              ├─► Call 1: semantic_search(mode="metadata", top_k=50)
              │   ├─► HybridSearch → 50 results
              │   ├─► ResponseFormatter → metadata only
              │   └─► Return 2K tokens
              │
              ├─► Python filtering: select top 3 IDs
              │   └─► Zero tokens (pure computation)
              │
              └─► Call 2: semantic_search(mode="full", filter_ids=[...])
                  ├─► HybridSearch → 3 results (filtered)
                  ├─► ResponseFormatter → full content
                  └─► Return 12K tokens

              Total in-subprocess: 14K tokens
              Total round-trips: 1 (execute_code)

              ▼
       ┌──────────────┐
       │Claude Desktop│ ← Receives 14K tokens (vs 62K traditional)
       └──────────────┘
```

**Latency Breakdown**:
- Code validation: 50ms
- Subprocess spawn: 100ms (one-time)
- Call 1 (metadata): 200ms (no reranking needed)
- Python filtering: 5ms
- Call 2 (full, 3 results): 100ms (smaller dataset)
- Network overhead: 50ms
- **Total**: 505ms

**Note**: Slightly higher latency, but 77% token reduction compensates with cost savings and context efficiency.

---

## 6. Progressive Disclosure Design

### 6.1 Response Mode Specifications

**Level 0: IDs Only** (`response_mode="ids_only"`)

**Use Case**: Existence checks, counting, cache validation

**Response Structure**:
```json
{
    "mode": "ids_only",
    "count": 50,
    "ids": ["chunk_123", "chunk_456", ...],
    "estimated_tokens": 500
}
```

**Token Budget**: 10 tokens per ID = 500 tokens for 50 results

**Performance**: <50ms (no content retrieval)

---

**Level 1: Metadata** (`response_mode="metadata"`)

**Use Case**: Understanding what exists, filtering candidates, initial analysis

**Response Structure**:
```json
{
    "mode": "metadata",
    "count": 50,
    "results": [
        {
            "id": "chunk_123",
            "title": "JWT Authentication Guide",
            "type": "documentation",
            "vendor": "openai",
            "file": "docs/auth/jwt.md",
            "score": 0.892,
            "created": "2024-11-01T12:00:00Z"
        },
        ...
    ],
    "estimated_tokens": 2000
}
```

**Token Budget**: ~40 tokens per result × 50 = 2,000 tokens

**Performance**: <100ms (metadata is cached separately)

**Implementation**:
```python
def _extract_metadata(self, result: SearchResult) -> dict:
    """Extract metadata without content"""
    return {
        "id": str(result.chunk_id),
        "title": result.title or result.source_file,
        "type": result.entity_type or "unknown",
        "vendor": result.vendor,
        "file": result.source_file,
        "score": round(result.hybrid_score, 3),
        "created": result.created_at.isoformat() if result.created_at else None
    }
```

---

**Level 2: Preview** (`response_mode="preview"`)

**Use Case**: Understanding implementation approach without full details

**Response Structure**:
```json
{
    "mode": "preview",
    "count": 10,
    "results": [
        {
            "metadata": { ... },
            "preview": "# JWT Authentication\n\nJSON Web Tokens provide stateless authentication by encoding user claims...",
            "preview_tokens": 50,
            "has_more": true
        },
        ...
    ],
    "estimated_tokens": 5000
}
```

**Token Budget**: ~500 tokens per result × 10 = 5,000 tokens

**Performance**: <200ms (fetches first N chars from DB)

**Implementation**:
```python
def _generate_preview(self, result: SearchResult, max_chars: int = 200) -> dict:
    """Generate truncated content preview"""
    preview_text = result.content[:max_chars]
    if len(result.content) > max_chars:
        preview_text += "..."

    return {
        "metadata": self._extract_metadata(result),
        "preview": preview_text,
        "preview_tokens": len(preview_text) // 4,  # Rough estimate
        "has_more": len(result.content) > max_chars
    }
```

---

**Level 3: Full Content** (`response_mode="full"`)

**Use Case**: Deep analysis, refactoring, implementation

**Response Structure**:
```json
{
    "mode": "full",
    "count": 3,
    "results": [
        {
            "metadata": { ... },
            "content": "# JWT Authentication\n\n[full document content]...",
            "tokens": 4200,
            "relationships": [  // Optional
                {
                    "entity_id": "entity_789",
                    "entity_name": "OAuth2Flow",
                    "relationship_type": "RELATED_TO",
                    "confidence": 0.85
                }
            ]
        },
        ...
    ],
    "estimated_tokens": 12600
}
```

**Token Budget**: ~4,000 tokens per result × 3 = 12,000 tokens

**Performance**: <500ms (includes optional relationship traversal)

---

### 6.2 Progressive Disclosure Workflow Example

**Scenario**: Agent researching "JWT refresh token best practices in Node.js"

**Traditional Approach** (no progressive disclosure):
```
1. semantic_search("JWT refresh token Node.js", top_k=10)
   → Returns: 10 full docs with content
   → Tokens: ~30K
   → Agent realizes only 2 are relevant, wasted 24K tokens

Total: 30K tokens, 1 round trip
```

**Progressive Approach**:
```
1. semantic_search("JWT refresh token Node.js", top_k=50, mode="metadata")
   → Returns: Metadata for 50 candidates
   → Tokens: 2K
   → Agent filters: vendor=node.js, type=best_practices, score>0.7
   → Narrows to 8 candidates

2. semantic_search(..., filter_ids=[...], mode="preview")
   → Returns: Previews for 8 candidates
   → Tokens: 4K
   → Agent selects top 2 based on preview content

3. semantic_search(..., filter_ids=[...], mode="full", include_relationships=true)
   → Returns: Full content + related entities for 2 docs
   → Tokens: 10K

Total: 16K tokens, 3 round trips (traditional path)
   OR: 16K tokens, 1 round trip (code execution path)
```

**Savings**: 47% token reduction (30K → 16K)

---

## 7. Tool Interface Specifications

### 7.1 `semantic_search` Tool Definition

**MCP Tool Registration**:
```python
{
    "name": "semantic_search",
    "description": """
    Perform hybrid semantic search across BMCIS knowledge base using BM25 + vector
    similarity with intelligent boosting and reranking.

    Supports progressive disclosure via response_mode parameter:
    - "ids_only": Return only document IDs (minimal tokens)
    - "metadata": Return metadata without content (compact)
    - "preview": Return metadata + truncated content preview
    - "full": Return complete content with optional relationships

    Use "metadata" mode for initial exploration, then fetch "full" content
    for selected results to minimize token usage.
    """,
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (keywords or natural language)",
                "minLength": 2
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-100)",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "response_mode": {
                "type": "string",
                "enum": ["ids_only", "metadata", "preview", "full"],
                "default": "full",
                "description": "Progressive disclosure tier"
            },
            "vendor_filter": {
                "type": "string",
                "description": "Filter results by vendor name",
                "optional": true
            },
            "min_score": {
                "type": "number",
                "description": "Minimum relevance score (0.0-1.0)",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "include_relationships": {
                "type": "boolean",
                "description": "Include related entities from knowledge graph",
                "default": false
            },
            "filter_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Only search within these document IDs",
                "optional": true
            }
        },
        "required": ["query"]
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "mode": {"type": "string"},
            "count": {"type": "integer"},
            "results": {"type": "array"},
            "estimated_tokens": {"type": "integer"}
        }
    }
}
```

**Usage Examples**:

```python
# Example 1: Traditional full search
await semantic_search(
    query="JWT authentication",
    top_k=10,
    response_mode="full"
)

# Example 2: Progressive disclosure - metadata first
metadata = await semantic_search(
    query="JWT authentication",
    top_k=50,
    response_mode="metadata"
)

# Example 3: Fetch full content for selected IDs
full = await semantic_search(
    query="JWT authentication",
    response_mode="full",
    filter_ids=["chunk_123", "chunk_456", "chunk_789"],
    include_relationships=true
)

# Example 4: Vendor-specific preview
openai_previews = await semantic_search(
    query="API rate limiting",
    vendor_filter="openai",
    response_mode="preview",
    top_k=20
)
```

### 7.2 `find_vendor_info` Tool Definition

**MCP Tool Registration**:
```python
{
    "name": "find_vendor_info",
    "description": """
    Find all entities and relationships for a specific vendor in the knowledge graph.

    Returns vendor-specific documentation, code samples, patterns, and entity
    relationships with optional full content based on include_content flag.
    """,
    "inputSchema": {
        "type": "object",
        "properties": {
            "vendor_name": {
                "type": "string",
                "description": "Vendor identifier (e.g., 'openai', 'anthropic')"
            },
            "include_content": {
                "type": "boolean",
                "description": "Include full content (true) or metadata only (false)",
                "default": false
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum entities to return (1-100)",
                "default": 50,
                "minimum": 1,
                "maximum": 100
            },
            "entity_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by entity types",
                "optional": true
            }
        },
        "required": ["vendor_name"]
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "mode": {"type": "string"},
            "entity_count": {"type": "integer"},
            "entities": {"type": "array"},
            "relationships": {"type": "array"},
            "estimated_tokens": {"type": "integer"}
        }
    }
}
```

**Usage Examples**:

```python
# Example 1: Metadata only (compact)
openai_metadata = await find_vendor_info(
    vendor_name="openai",
    include_content=false,
    max_results=50
)
# Returns: Entity metadata + relationship graph
# Tokens: ~2K-3K

# Example 2: Full content for deep analysis
openai_full = await find_vendor_info(
    vendor_name="openai",
    include_content=true,
    max_results=20,
    entity_types=["api_reference", "best_practices"]
)
# Returns: Full content for 20 entities + relationships
# Tokens: ~30K-40K

# Example 3: Entity type filtering
auth_docs = await find_vendor_info(
    vendor_name="anthropic",
    include_content=false,
    entity_types=["authentication", "security"]
)
```

---

## 8. Security Architecture

### 8.1 Security Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                     Security Perimeter                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │             Public FastMCP Interface                  │ │
│  │  - API key validation (required)                      │ │
│  │  - Rate limiting (100 req/min default)                │ │
│  │  - Input sanitization & validation                    │ │
│  │  - Request size limits (10MB max)                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │          Request Processing Layer                     │ │
│  │  - Query parameter validation                         │ │
│  │  - SQL injection prevention (parameterized queries)   │ │
│  │  - Response size enforcement                          │ │
│  │  - Token counting & budget checks                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │         Business Logic Layer (Trusted)                │ │
│  │  - HybridSearch execution                             │ │
│  │  - Knowledge graph queries                            │ │
│  │  - Cache access                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │            Data Layer (Isolated)                      │ │
│  │  - PostgreSQL connection pooling                      │ │
│  │  - Read-only access for search operations             │ │
│  │  - Audit logging for all queries                      │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

FUTURE: Code Execution Sandbox
┌─────────────────────────────────────────────────────────────┐
│            Execution Sandbox (Additional Layer)             │
│  - Subprocess isolation                                     │
│  - Resource limits (CPU, memory, time)                      │
│  - AST-based code validation                                │
│  - Restricted module imports (whitelist)                    │
│  - No network access from sandbox                           │
│  - Calls MCP tools via internal API (not network)           │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Threat Model & Mitigations

**Threat 1: Unauthorized Access**
- **Attack**: Brute-force API keys, credential theft
- **Mitigation**:
  - API key validation with bcrypt hashing
  - Rate limiting (100 req/min, configurable)
  - Request logging with IP tracking
  - Optional IP whitelist

**Threat 2: Denial of Service**
- **Attack**: Excessive requests, large result sets, expensive queries
- **Mitigation**:
  - Rate limiting per client
  - Response size limits (10MB max)
  - Query timeout enforcement (30s max)
  - Circuit breaker for downstream services
  - Query cache to reduce DB load

**Threat 3: Data Exfiltration**
- **Attack**: Harvest all documents via progressive searches
- **Mitigation**:
  - Rate limiting prevents bulk extraction
  - Audit logging tracks all queries
  - Anomaly detection for unusual patterns
  - Max results per query (100 docs)

**Threat 4: SQL Injection**
- **Attack**: Malicious query strings with SQL code
- **Mitigation**:
  - Parameterized queries (ONLY)
  - Input validation & sanitization
  - ORM/query builder usage (psycopg3)
  - Read-only DB user permissions

**Threat 5: Code Execution Sandbox Escape** (Future)
- **Attack**: Malicious Python code breaks isolation
- **Mitigation**:
  - Subprocess-based isolation (OS-level)
  - AST analysis blocks dangerous patterns
  - Module whitelist (no os, subprocess, socket)
  - Resource limits (CPU, memory, time)
  - 100% timeout reliability (SIGKILL)

### 8.3 Security Configuration

```python
@dataclass
class SecurityConfig:
    """Security settings for MCP server"""

    # Authentication
    api_key: Optional[str] = None  # If None, no auth required
    api_key_header: str = "X-API-Key"

    # Rate Limiting
    rate_limit_rpm: int = 100  # Requests per minute
    rate_limit_enabled: bool = True

    # Request Limits
    max_query_length: int = 1000  # Characters
    max_results: int = 100  # Per query
    max_response_size: int = 10_000_000  # 10MB
    query_timeout_seconds: int = 30

    # Code Execution (Future)
    enable_code_execution: bool = False
    sandbox_memory_mb: int = 512
    sandbox_timeout_seconds: int = 30
    allowed_modules: List[str] = field(default_factory=lambda: [
        "json", "datetime", "re", "math", "typing"
    ])

    # Audit Logging
    log_all_queries: bool = True
    log_retention_days: int = 90

    # IP Filtering
    ip_whitelist: Optional[List[str]] = None
```

---

## 9. Caching Strategy

### 9.1 Current Cache Architecture

**Existing Caches**:

1. **Entity Cache** (`KnowledgeGraphCache`)
   - Type: LRU in-memory
   - Scope: Individual entities
   - TTL: No expiration (LRU eviction only)
   - Size: Configurable (default: 1000 entities)

2. **Query Cache** (`src/search/query_cache.py`)
   - Type: Redis-compatible
   - Scope: Full search results
   - TTL: Configurable (default: 1 hour)
   - Key: Hash of (query, top_k, filters)

**Gap**: No separation of metadata vs. full content in cache

### 9.2 Enhanced Caching for Progressive Disclosure

**Proposal**: Tiered cache entries

**Level 1: Metadata Cache**
```python
cache_key = f"metadata:{query_hash}:{top_k}"
cached_metadata = [
    {
        "id": "chunk_123",
        "title": "...",
        "type": "...",
        "vendor": "...",
        "score": 0.89,
        "created": "..."
    },
    ...
]
# Size: ~50 bytes per entry × 50 = 2.5KB
# TTL: 1 hour (high hit rate expected)
```

**Level 2: Preview Cache**
```python
cache_key = f"preview:{query_hash}:{top_k}"
cached_previews = [
    {
        "metadata": {...},
        "preview": "First 200 chars...",
        "has_more": true
    },
    ...
]
# Size: ~300 bytes per entry × 10 = 3KB
# TTL: 1 hour
```

**Level 3: Full Content Cache**
```python
cache_key = f"full:{query_hash}:{top_k}"
cached_full = [
    {
        "metadata": {...},
        "content": "Full content...",
        "tokens": 4200
    },
    ...
]
# Size: ~15KB per entry × 10 = 150KB
# TTL: 30 minutes (lower due to size)
```

**Cache Hit Logic**:
```python
async def get_search_results(
    query: str,
    top_k: int,
    response_mode: ResponseMode
) -> dict:
    """Get results with cache hierarchy"""

    # Try metadata cache first (cheapest)
    metadata_key = f"metadata:{hash(query)}:{top_k}"
    metadata = await cache.get(metadata_key)

    if response_mode == ResponseMode.METADATA and metadata:
        return {"mode": "metadata", "results": metadata}

    # Try preview cache
    if response_mode == ResponseMode.PREVIEW:
        preview_key = f"preview:{hash(query)}:{top_k}"
        previews = await cache.get(preview_key)
        if previews:
            return {"mode": "preview", "results": previews}

    # Try full cache
    if response_mode == ResponseMode.FULL:
        full_key = f"full:{hash(query)}:{top_k}"
        full = await cache.get(full_key)
        if full:
            return {"mode": "full", "results": full}

    # Cache miss - execute search
    results = await execute_search(query, top_k)

    # Populate all cache tiers
    await cache.set(metadata_key, extract_metadata(results), ttl=3600)
    await cache.set(preview_key, generate_previews(results), ttl=3600)
    await cache.set(full_key, results, ttl=1800)

    return format_results(results, response_mode)
```

### 9.3 Cache Invalidation Strategy

**Triggers**:
1. Document updates/deletes → Invalidate all caches for that document
2. Reindexing → Flush all query caches
3. Manual purge → Admin endpoint for cache clearing

**Implementation**:
```python
async def invalidate_document_caches(document_id: str):
    """Invalidate all caches containing this document"""
    # Find all cached queries containing this document
    affected_keys = await cache.find_keys_by_pattern(f"*:{document_id}:*")

    # Delete all tiers
    for key in affected_keys:
        await cache.delete(key)
        await cache.delete(key.replace("metadata:", "preview:"))
        await cache.delete(key.replace("metadata:", "full:"))
```

### 9.4 Cache Performance Estimates

**Metadata Cache**:
- Hit rate: 70-80% (metadata queries are common)
- Latency: <5ms (in-memory)
- Size: ~2KB per query × 1000 queries = 2MB

**Preview Cache**:
- Hit rate: 40-50% (less common, but valuable)
- Latency: <10ms
- Size: ~3KB per query × 500 queries = 1.5MB

**Full Cache**:
- Hit rate: 30-40% (lowest, due to variations in filter_ids)
- Latency: <50ms (larger payloads)
- Size: ~150KB per query × 200 queries = 30MB

**Total Cache Memory**: ~35MB for 1000 metadata + 500 preview + 200 full

---

## 10. Decision Matrix: MVP vs Full Integration

### 10.1 Option A: MVP (Task 10 Only, No Progressive Disclosure)

**Scope**:
- FastMCP server with `semantic_search` and `find_vendor_info` tools
- Single response mode: `full` (always return complete content)
- No tiered caching or metadata extraction
- No code execution integration

**Pros**:
- Simplest implementation (2-3 weeks)
- Minimal risk
- Fully functional for traditional use cases
- No architectural complexity

**Cons**:
- Higher token costs for Claude Desktop usage
- Requires complete refactoring for code execution integration
- Misses 90-95% token reduction opportunity
- Cache layer is inefficient (full content only)

**Effort Estimate**: 2-3 weeks (1.5 FTE)

**Risk Level**: LOW

---

### 10.2 Option B: Progressive Disclosure Foundation (Task 10 + Tiered Responses)

**Scope**:
- FastMCP server with `semantic_search` and `find_vendor_info` tools
- **Add**: `response_mode` parameter (ids_only, metadata, preview, full)
- **Add**: `ResponseFormatter` class for tiered output
- **Add**: Tiered caching (metadata/preview/full separation)
- No code execution (yet), but designed for future integration

**Pros**:
- **Token savings NOW**: Claude Desktop can use metadata mode (47-77% reduction)
- **Future-proof**: Code execution integration requires ZERO refactoring
- **Better UX**: Agents can progressively refine searches
- **Cache efficiency**: Smaller cache entries, higher hit rates

**Cons**:
- Additional 1 week development time
- Slightly more complex API (4 response modes vs. 1)
- Requires client education on when to use each mode

**Effort Estimate**: 3-4 weeks (1.5 FTE)

**Risk Level**: MEDIUM (new API pattern, requires testing)

---

### 10.3 Option C: Full Integration (Task 10 + Code Execution)

**Scope**:
- Everything in Option B PLUS:
- Code execution sandbox (subprocess isolation)
- Input validation (AST analysis)
- Security hardening (module whitelist, resource limits)
- In-memory MCP client for sandbox
- Comprehensive security testing

**Pros**:
- Complete solution with maximum token efficiency (90-95% reduction)
- No future architectural work needed
- Industry-leading performance (300-500ms latency)
- Enables complex search workflows

**Cons**:
- Significantly longer timeline (6-8 weeks)
- Higher risk (security testing, sandbox reliability)
- Requires security audit
- Code execution adds operational complexity

**Effort Estimate**: 6-8 weeks (2.0 FTE)

**Risk Level**: HIGH (security, complexity)

---

### 10.4 Recommendation: **Option B** (Progressive Disclosure Foundation)

**Rationale**:

1. **Best ROI**: 1 week additional effort → 47-77% token savings immediately
2. **Future-proof**: Code execution integration is a simple add-on, not a refactor
3. **Low risk**: No security concerns (just response formatting)
4. **Immediate value**: Claude Desktop users benefit NOW
5. **Incremental approach**: Can add code execution in Phase 2 if needed

**Implementation Path**:

**Week 1-2**: Core FastMCP server (Task 10.1-10.2)
- Server setup, tool definitions
- Request handlers with authentication
- Error handling

**Week 3**: Response formatting layer (Task 10.4 enhanced)
- `ResponseFormatter` class
- Tiered output (metadata/preview/full)
- Token counting

**Week 4**: Caching + testing (Task 10.5)
- Tiered cache implementation
- End-to-end testing
- Performance validation

**Week 5** (Optional): Code execution preparation
- Design sandbox architecture
- Security specification
- Integration plan

**Deliverables**:
- ✅ FastMCP server operational
- ✅ Progressive disclosure API working
- ✅ 47-77% token reduction available
- ✅ Architecture ready for code execution (Phase 2)

---

## 11. Risk Assessment

### 11.1 Technical Risks

**Risk 1: Response Mode Complexity**
- **Impact**: MEDIUM
- **Probability**: LOW
- **Description**: Clients confused by 4 response modes
- **Mitigation**:
  - Clear documentation with examples
  - Default to `full` mode (backward compatible)
  - Claude Desktop prompt templates guide usage
  - Gradual rollout with education

**Risk 2: Cache Invalidation Bugs**
- **Impact**: HIGH (stale data)
- **Probability**: MEDIUM
- **Description**: Tiered caches get out of sync
- **Mitigation**:
  - Atomic cache updates (all tiers together)
  - Cache versioning (invalidate on schema changes)
  - Monitoring for cache hit rates
  - Manual purge endpoint for emergencies

**Risk 3: Performance Degradation**
- **Impact**: HIGH
- **Probability**: LOW
- **Description**: Response formatting adds latency
- **Mitigation**:
  - Metadata extraction is O(n) simple field access
  - Preview truncation is string slicing (microseconds)
  - Benchmark all formatter operations (<20ms target)
  - Cache metadata aggressively (70-80% hit rate)

**Risk 4: Token Counting Accuracy**
- **Impact**: MEDIUM
- **Probability**: MEDIUM
- **Description**: Estimated tokens differ from actual usage
- **Mitigation**:
  - Use tiktoken library for exact counts
  - Include token counts in responses for verification
  - Log actual vs. estimated for tuning
  - Calibration testing across diverse queries

### 11.2 Integration Risks

**Risk 5: Backward Compatibility Break**
- **Impact**: HIGH
- **Probability**: LOW (if default="full")
- **Description**: Existing clients break when API changes
- **Mitigation**:
  - Default `response_mode="full"` (no change for existing clients)
  - Versioned API endpoints (v1 vs. v2)
  - Deprecation warnings for old patterns
  - Feature flag for progressive disclosure

**Risk 6: Code Execution Integration Complexity**
- **Impact**: MEDIUM
- **Probability**: MEDIUM
- **Description**: Sandbox integration harder than expected
- **Mitigation**:
  - Progressive disclosure API is independent of sandbox
  - Can deploy progressive disclosure WITHOUT code execution
  - Sandbox is Phase 2 (optional)
  - Detailed PRD already exists with architecture

### 11.3 Security Risks

**Risk 7: API Key Leakage**
- **Impact**: HIGH
- **Probability**: LOW
- **Description**: API keys exposed in logs or errors
- **Mitigation**:
  - Redact API keys in logs (mask all but last 4 chars)
  - Use environment variables (not config files)
  - Rotate keys regularly
  - Audit logging excludes sensitive data

**Risk 8: Rate Limit Bypass**
- **Impact**: MEDIUM
- **Probability**: MEDIUM
- **Description**: Attackers use multiple IPs/keys
- **Mitigation**:
  - Rate limit by IP AND API key
  - Exponential backoff on limit hits
  - IP-based anomaly detection
  - Manual IP blacklist capability

**Risk 9: Data Exfiltration via Progressive Disclosure**
- **Impact**: MEDIUM
- **Probability**: LOW
- **Description**: Attackers harvest metadata for all docs
- **Mitigation**:
  - Rate limiting prevents bulk extraction
  - Audit logging tracks unusual patterns (e.g., 1000 metadata queries)
  - Max results per query (100 docs)
  - Anomaly detection alerts on excessive metadata queries

### 11.4 Operational Risks

**Risk 10: Cache Memory Exhaustion**
- **Impact**: HIGH
- **Probability**: LOW
- **Description**: Cache grows unbounded, crashes server
- **Mitigation**:
  - LRU eviction with max size limits
  - Monitor cache memory usage (alert at 80%)
  - Separate cache limits per tier
  - Manual purge endpoint

**Risk 11: Database Connection Pool Exhaustion**
- **Impact**: HIGH
- **Probability**: MEDIUM (under load)
- **Description**: Too many concurrent queries
- **Mitigation**:
  - Connection pool with max size (default: 20)
  - Queue requests when pool exhausted
  - Query timeout enforcement (30s)
  - Circuit breaker on DB failures

---

## 12. Implementation Roadmap

### 12.1 Phase 1: Core FastMCP Server (Task 10.1-10.2)

**Duration**: 2 weeks

**Tasks**:
1. FastMCP server initialization
   - Install FastMCP dependencies
   - Create server entry point (`src/mcp_server/server.py`)
   - Configure logging and error handling

2. Tool registration
   - Define `semantic_search` schema
   - Define `find_vendor_info` schema
   - Implement tool discovery endpoint

3. Request handlers (basic)
   - Route requests to `handle_semantic_search`
   - Route requests to `handle_find_vendor_info`
   - Input validation (query length, top_k bounds)

4. Authentication
   - API key validation middleware
   - Environment variable configuration
   - Error responses for auth failures

**Deliverables**:
- ✅ FastMCP server running on localhost:3000
- ✅ Tools discoverable via MCP list_tools
- ✅ Basic search working (full mode only)

---

### 12.2 Phase 2: Progressive Disclosure Layer (Task 10.4 Enhanced)

**Duration**: 1 week

**Tasks**:
1. `ResponseFormatter` class
   - Implement `ResponseMode` enum
   - Implement metadata extraction
   - Implement preview generation
   - Implement full content formatting

2. Update request handlers
   - Add `response_mode` parameter to `semantic_search`
   - Add `include_content` parameter to `find_vendor_info`
   - Call `ResponseFormatter.format_results()`

3. Token counting
   - Integrate tiktoken library
   - Add token estimates to all responses
   - Log actual vs. estimated for calibration

4. Update tool schemas
   - Add `response_mode` to inputSchema
   - Add `filter_ids` parameter for selective fetching
   - Document progressive disclosure workflow

**Deliverables**:
- ✅ All 4 response modes functional (ids_only, metadata, preview, full)
- ✅ Token estimates in responses
- ✅ Updated documentation with examples

---

### 12.3 Phase 3: Tiered Caching (Task 10.3 Enhanced)

**Duration**: 1 week

**Tasks**:
1. Cache key design
   - Implement hierarchical keys (metadata:*, preview:*, full:*)
   - Hash function for query+params
   - TTL configuration per tier

2. Cache population logic
   - Populate all 3 tiers on search execution
   - Handle cache misses gracefully
   - Implement cache hit logging

3. Cache invalidation
   - Document update triggers
   - Manual purge endpoint
   - Cache version management

4. Monitoring
   - Cache hit rate metrics
   - Memory usage tracking
   - Performance impact analysis

**Deliverables**:
- ✅ Tiered cache operational
- ✅ 70%+ cache hit rate for metadata
- ✅ Monitoring dashboard

---

### 12.4 Phase 4: Testing & Performance Validation (Task 10.5)

**Duration**: 1 week

**Tasks**:
1. Unit tests
   - ResponseFormatter tests (all modes)
   - Cache logic tests
   - Input validation tests

2. Integration tests
   - End-to-end search workflows
   - Progressive disclosure scenarios
   - Cache hit/miss paths

3. Performance testing
   - Latency benchmarks (P50, P95, P99)
   - Throughput testing (concurrent requests)
   - Token reduction validation

4. Security testing
   - API key validation
   - Rate limiting enforcement
   - Input sanitization
   - SQL injection prevention

**Deliverables**:
- ✅ Test coverage >90%
- ✅ P95 latency <500ms
- ✅ 47-77% token reduction validated
- ✅ Security audit passed

---

### 12.5 Phase 5 (Optional): Code Execution Preparation

**Duration**: 1 week (design only, no implementation)

**Tasks**:
1. Sandbox architecture design
   - Subprocess isolation design
   - Resource limit specifications
   - Module whitelist definition

2. Security specification
   - AST analysis requirements
   - Timeout enforcement strategy
   - Network isolation approach

3. Integration design
   - In-memory MCP client design
   - Code validation pipeline
   - Error handling strategy

4. Documentation
   - PRD review and updates
   - Implementation guide
   - Security review checklist

**Deliverables**:
- ✅ Sandbox architecture document
- ✅ Security specification
- ✅ Implementation plan for Phase 6 (future)

---

## 13. Recommendations

### 13.1 Architecture Decisions

**Decision 1: Implement Progressive Disclosure in Task 10** ✅ RECOMMENDED

**Rationale**:
- 1 week additional effort → 47-77% token savings
- Future-proofs for code execution integration
- No refactoring needed for Phase 2
- Immediate value for Claude Desktop users

**Impact**: HIGH positive (token efficiency + future extensibility)

---

**Decision 2: Use Tiered Caching with Separate Metadata/Preview/Full** ✅ RECOMMENDED

**Rationale**:
- Metadata cache: 2KB vs. 150KB (75x smaller)
- Higher cache hit rates (70% vs. 40%)
- Better memory efficiency
- Enables fast metadata-only responses (<50ms)

**Impact**: HIGH positive (performance + cost)

---

**Decision 3: Default `response_mode="full"` for Backward Compatibility** ✅ RECOMMENDED

**Rationale**:
- Existing clients continue working without changes
- Opt-in progressive disclosure
- Reduces migration risk
- Gradual adoption curve

**Impact**: LOW risk, HIGH compatibility

---

**Decision 4: Defer Code Execution to Phase 2** ✅ RECOMMENDED

**Rationale**:
- Task 10 provides immediate value without code execution
- Progressive disclosure API is future-proof
- Code execution adds 4-5 weeks + security testing
- Can be added later without refactoring

**Impact**: MEDIUM timeline reduction, LOW future risk

---

### 13.2 Implementation Priorities

**Priority 1 (Must Have)**: Core FastMCP server with `semantic_search` and `find_vendor_info`
- Deliverable: Task 10.1-10.2
- Timeline: Week 1-2
- Risk: LOW

**Priority 2 (Should Have)**: Progressive disclosure API
- Deliverable: Task 10.4 enhanced with `ResponseFormatter`
- Timeline: Week 3
- Risk: MEDIUM
- Value: 47-77% token reduction

**Priority 3 (Should Have)**: Tiered caching
- Deliverable: Task 10.3 enhanced with metadata/preview/full separation
- Timeline: Week 4
- Risk: MEDIUM
- Value: 3-4x cache efficiency

**Priority 4 (Nice to Have)**: Code execution preparation
- Deliverable: Architecture design (no implementation)
- Timeline: Week 5
- Risk: LOW (design only)
- Value: Enables Phase 2 planning

---

### 13.3 Success Criteria

**Task 10 MVP** (Minimum Viable Product):
- ✅ FastMCP server operational on Claude Desktop
- ✅ `semantic_search` tool functional with full responses
- ✅ `find_vendor_info` tool functional
- ✅ Authentication working (API key validation)
- ✅ P95 latency <500ms
- ✅ Test coverage >80%

**Task 10 Enhanced** (With Progressive Disclosure):
- ✅ All MVP criteria PLUS:
- ✅ 4 response modes working (ids_only, metadata, preview, full)
- ✅ Token estimates accurate within 10%
- ✅ Tiered caching operational
- ✅ 47-77% token reduction validated
- ✅ Cache hit rate >70% for metadata
- ✅ Documentation complete with examples

**Future Code Execution Integration** (Phase 2):
- ✅ Sandbox operational with subprocess isolation
- ✅ Security audit passed
- ✅ 90-95% token reduction achieved
- ✅ 300-500ms latency for code execution path
- ✅ Zero architectural refactoring from Task 10

---

### 13.4 Next Steps

**Immediate Actions**:

1. **Review & Approve Architecture** (1 day)
   - Share this document with stakeholders
   - Confirm Option B (Progressive Disclosure Foundation)
   - Sign off on tiered caching approach

2. **Create Implementation Tasks** (1 day)
   - Expand Task 10 subtasks with progressive disclosure
   - Assign effort estimates
   - Identify dependencies

3. **Spike: Response Formatter Prototype** (2 days)
   - Implement basic `ResponseFormatter` class
   - Validate metadata extraction performance
   - Test token counting accuracy

4. **Begin Implementation** (Week 1)
   - Start with FastMCP server core (Task 10.1)
   - Parallel: Design tiered cache schema
   - Prepare test data and fixtures

**Risk Mitigation**:

1. **Token Counting Accuracy**: Run calibration tests early (Week 2)
2. **Cache Complexity**: Prototype cache layer before full integration (Week 3)
3. **Client Education**: Create usage guide with examples (Week 4)
4. **Performance**: Benchmark each component in isolation (ongoing)

---

## Conclusion

This architecture analysis demonstrates a clear path for implementing **Task 10 (FastMCP Server Integration)** with forward compatibility for **Code Execution MCP**. The recommended approach (Option B: Progressive Disclosure Foundation) balances immediate value delivery (47-77% token reduction) with future extensibility (code execution integration requires zero refactoring).

**Key Takeaways**:

1. **Progressive disclosure is a game-changer**: 4 response modes enable 90-95% token efficiency
2. **Tiered caching is essential**: Metadata cache provides 75x space savings and faster responses
3. **Architecture is future-proof**: Code execution integration is a simple add-on, not a refactor
4. **Risk is manageable**: Incremental implementation with clear milestones and validation
5. **Timeline is realistic**: 4-5 weeks for Task 10 Enhanced vs. 2-3 weeks for MVP (33% longer, 10x value)

**Final Recommendation**: Proceed with **Option B (Progressive Disclosure Foundation)** as the architectural foundation for Task 10, enabling both immediate token efficiency gains and seamless future code execution integration.

---

**Document Status**: READY FOR REVIEW
**Next Review**: Architecture approval meeting
**Implementation Target**: Start Week 1 upon approval
