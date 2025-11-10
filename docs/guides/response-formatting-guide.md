# Response Formatting Guide for MCP Tools

## Table of Contents

1. [Overview](#overview)
2. [Progressive Disclosure Pattern](#progressive-disclosure-pattern)
3. [Response Modes Explained](#response-modes-explained)
4. [Response Envelope Architecture](#response-envelope-architecture)
5. [Claude Desktop Integration](#claude-desktop-integration)
6. [Token Budget Management](#token-budget-management)
7. [Confidence Scores and Ranking](#confidence-scores-and-ranking)
8. [Implementation Patterns](#implementation-patterns)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

The BMCIS Knowledge MCP implements a sophisticated response formatting system designed to optimize token usage, improve Claude Desktop integration, and provide progressive disclosure of information. This guide explains the response formatting architecture, its benefits, and how to effectively use different response modes.

### Key Benefits

- **83% Token Reduction**: Metadata mode reduces tokens from ~15,000 to ~2,500 for 10 results
- **Progressive Disclosure**: Four levels of detail (ids_only → metadata → preview → full)
- **Desktop Optimization**: Structured responses designed for Claude Desktop's context limitations
- **Intelligent Caching**: Response-level caching with mode-specific keys
- **Field Filtering**: Whitelist-based field selection for minimal token usage
- **Confidence Scoring**: Multi-dimensional confidence indicators for result quality

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Tool Request                         │
│  (query, response_mode, page_size, cursor, fields)          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Request Validation                          │
│         (Pydantic models with field whitelisting)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cache Layer Check                         │
│          (Mode-specific caching with TTL)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   Response Formatting                        │
│    ┌──────────┬───────────┬────────────┬────────────┐      │
│    │ IDs Only │ Metadata  │  Preview   │    Full    │      │
│    │ ~10 tok  │ ~200 tok  │ ~500 tok   │ ~1500 tok  │      │
│    └──────────┴───────────┴────────────┴────────────┘      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   Response Envelope                          │
│         (MCPResponseEnvelope with metadata)                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop                            │
│            (Optimized for context window)                    │
└──────────────────────────────────────────────────────────────┘
```

## Progressive Disclosure Pattern

Progressive disclosure allows clients to request exactly the level of detail needed, minimizing token usage while maintaining information accessibility. This pattern is crucial for Claude Desktop integration where context windows are limited.

### Token Budget Per Result

| Mode | Token Budget | Use Case | Description |
|------|--------------|----------|-------------|
| `ids_only` | ~10 tokens | Quick scan | Chunk IDs and scores only |
| `metadata` | ~200 tokens | File browsing | Source file, category, position |
| `preview` | ~500 tokens | Content preview | Metadata + 200-char snippet |
| `full` | ~1500+ tokens | Deep analysis | Complete chunk content |

### Cumulative Token Usage (10 Results)

```python
# Example: Token usage comparison for 10 search results
{
    "ids_only": 100,      # Minimal overhead
    "metadata": 2500,     # 83% reduction vs full
    "preview": 7500,      # 50% reduction vs full
    "full": 15000         # Complete information
}
```

## Response Modes Explained

### 1. IDs Only Mode (`ids_only`)

Minimal response containing only identifiers and scores. Ideal for existence checks, counting, or when the client already has cached metadata.

**Response Structure:**
```json
{
    "results": [
        {
            "chunk_id": 123,
            "hybrid_score": 0.85,
            "rank": 1
        },
        {
            "chunk_id": 456,
            "hybrid_score": 0.72,
            "rank": 2
        }
    ],
    "total_found": 42,
    "strategy_used": "hybrid",
    "execution_time_ms": 124.5
}
```

**Use Cases:**
- Checking if results exist for a query
- Getting result counts
- Building custom UI result lists
- Pre-filtering before fetching details

### 2. Metadata Mode (`metadata`) - DEFAULT

Includes essential metadata for result identification and browsing without the actual content. This is the default mode as it provides the best balance between information and token usage.

**Response Structure:**
```json
{
    "results": [
        {
            "chunk_id": 123,
            "source_file": "docs/authentication/jwt-guide.md",
            "source_category": "security",
            "hybrid_score": 0.85,
            "rank": 1,
            "chunk_index": 3,
            "total_chunks": 15
        }
    ],
    "total_found": 42,
    "strategy_used": "hybrid",
    "execution_time_ms": 156.3
}
```

**Use Cases:**
- File discovery and browsing
- Source identification
- Result organization by category
- Quick relevance assessment

### 3. Preview Mode (`preview`)

Extends metadata with a 200-character snippet for content preview. Useful for quick content assessment without full retrieval.

**Response Structure:**
```json
{
    "results": [
        {
            "chunk_id": 123,
            "source_file": "docs/authentication/jwt-guide.md",
            "source_category": "security",
            "hybrid_score": 0.85,
            "rank": 1,
            "chunk_index": 3,
            "total_chunks": 15,
            "chunk_snippet": "JWT authentication provides a stateless mechanism for API security. By encoding user claims in a signed token, servers can verify identity without database lookups...",
            "context_header": "JWT Guide > Implementation > Best Practices"
        }
    ],
    "total_found": 42,
    "strategy_used": "hybrid",
    "execution_time_ms": 189.7
}
```

**Use Cases:**
- Content preview without full retrieval
- Quick relevance validation
- Building search result previews
- Identifying most relevant chunks

### 4. Full Mode (`full`)

Complete chunk content with all metadata and scoring details. Use sparingly due to high token cost.

**Response Structure:**
```json
{
    "results": [
        {
            "chunk_id": 123,
            "chunk_text": "JWT authentication provides a stateless mechanism for API security. By encoding user claims in a signed token, servers can verify identity without database lookups. This approach scales well in distributed systems where session state synchronization would be complex. Best practices include: using strong signing algorithms (RS256 preferred over HS256), implementing token rotation, setting appropriate expiration times, and never storing sensitive data in the payload...",
            "similarity_score": 0.82,
            "bm25_score": 0.88,
            "hybrid_score": 0.85,
            "rank": 1,
            "score_type": "hybrid",
            "source_file": "docs/authentication/jwt-guide.md",
            "source_category": "security",
            "context_header": "JWT Guide > Implementation > Best Practices",
            "chunk_index": 3,
            "total_chunks": 15,
            "chunk_token_count": 512
        }
    ],
    "total_found": 42,
    "strategy_used": "hybrid",
    "execution_time_ms": 234.5
}
```

**Use Cases:**
- Deep content analysis
- Implementation details extraction
- Complete context understanding
- Detailed scoring analysis

## Response Envelope Architecture

All MCP tool responses are wrapped in a standardized envelope (`MCPResponseEnvelope`) that provides consistent metadata, execution context, and error handling.

### Envelope Structure

```python
class MCPResponseEnvelope:
    """
    Standard wrapper for all MCP responses
    Token overhead: ~150-300 tokens per response
    """

    _metadata: ResponseMetadata       # Operation metadata
    results: T                        # Tool-specific results (generic)
    pagination: PaginationMetadata    # Optional pagination info
    execution_context: ExecutionContext  # Performance metrics
    warnings: List[ResponseWarning]   # Actionable warnings
```

### Response Metadata

Every response includes metadata describing the operation:

```json
{
    "_metadata": {
        "operation": "semantic_search",
        "version": "1.0.0",
        "timestamp": "2025-11-09T10:30:00Z",
        "request_id": "req_abc123def456",
        "status": "success",
        "message": null
    }
}
```

### Execution Context

Performance and token accounting information:

```json
{
    "execution_context": {
        "tokens_estimated": 2450,
        "tokens_used": 2523,
        "cache_hit": true,
        "execution_time_ms": 156.3,
        "request_id": "req_abc123def456"
    }
}
```

### Response Warnings

Actionable warnings for suboptimal configurations or approaching limits:

```json
{
    "warnings": [
        {
            "level": "warning",
            "code": "TOKEN_LIMIT_WARNING",
            "message": "Response approaching context window limit",
            "suggestion": "Consider using 'metadata' mode or reducing page_size"
        }
    ]
}
```

## Claude Desktop Integration

The response formatting system is specifically optimized for Claude Desktop's unique constraints and capabilities.

### Desktop Mode Optimization

When integrated with Claude Desktop, the system automatically:

1. **Enforces Token Budgets**: Prevents responses exceeding Desktop's context limits
2. **Provides Progressive Loading**: Starts with metadata, fetches full content on demand
3. **Includes Visual Hints**: Confidence scores and ranking for UI visualization
4. **Enables Smart Filtering**: Field-level filtering for minimal token transfer

### Desktop-Specific Features

#### Enhanced Metadata for Desktop UI

```json
{
    "confidence": {
        "score_reliability": 0.92,
        "source_quality": 0.88,
        "recency": 0.75
    },
    "ranking": {
        "percentile": 95,
        "explanation": "Strong semantic and keyword match",
        "score_method": "hybrid"
    },
    "deduplication": {
        "is_duplicate": false,
        "similar_chunk_ids": [124, 567],
        "confidence": 0.85
    }
}
```

#### Desktop Response Mode Selection

```python
# Automatic mode selection based on context budget
def select_desktop_mode(available_tokens: int, result_count: int) -> str:
    """
    Select optimal response mode for Desktop context

    Args:
        available_tokens: Remaining context tokens
        result_count: Number of results requested

    Returns:
        Optimal response_mode
    """
    tokens_per_result = {
        "ids_only": 10,
        "metadata": 200,
        "preview": 500,
        "full": 1500
    }

    for mode in ["full", "preview", "metadata", "ids_only"]:
        required = tokens_per_result[mode] * result_count + 300  # envelope
        if required <= available_tokens:
            return mode

    return "ids_only"  # Fallback to minimal
```

## Token Budget Management

Effective token management is crucial for optimal Desktop performance. The system provides multiple mechanisms for controlling token usage.

### Token Estimation Formula

```python
def estimate_tokens(mode: str, count: int) -> int:
    """
    Estimate total response tokens

    Formula: (base_per_result * count) + envelope_overhead
    """
    base_tokens = {
        "ids_only": 10,
        "metadata": 200,
        "preview": 500,
        "full": 1500
    }

    envelope_overhead = 300  # Metadata, pagination, warnings
    return (base_tokens[mode] * count) + envelope_overhead
```

### Budget Allocation Strategy

```python
# Example: Intelligent budget allocation
class TokenBudgetManager:
    def __init__(self, total_budget: int = 50000):
        self.total_budget = total_budget
        self.used = 0

    def allocate_for_search(self, priority_results: int = 3) -> dict:
        """
        Allocate tokens optimally:
        - Full details for top N results
        - Metadata for remaining results
        - Reserve budget for follow-ups
        """
        available = self.total_budget - self.used

        # Reserve 20% for follow-up queries
        search_budget = int(available * 0.8)

        # Full content for priority results
        priority_tokens = priority_results * 1500

        # Metadata for additional context
        remaining_budget = search_budget - priority_tokens
        metadata_results = remaining_budget // 200

        return {
            "full_results": priority_results,
            "metadata_results": metadata_results,
            "total_results": priority_results + metadata_results,
            "estimated_tokens": priority_tokens + (metadata_results * 200) + 300
        }
```

### Token Accounting Example

```json
{
    "execution_context": {
        "tokens_estimated": 2450,
        "tokens_used": 2523,
        "token_breakdown": {
            "results": 2200,
            "metadata": 150,
            "pagination": 73,
            "warnings": 100
        },
        "token_efficiency": 0.97  // used/estimated ratio
    }
}
```

## Confidence Scores and Ranking

The system provides multi-dimensional confidence scoring to help Claude Desktop make intelligent decisions about result quality and relevance.

### Confidence Score Components

```python
class ConfidenceScore:
    """
    Multi-dimensional confidence assessment
    All scores are 0.0-1.0
    """

    score_reliability: float  # Statistical confidence in the score
    source_quality: float     # Quality rating of the source document
    recency: float           # How recent/up-to-date the content is
```

### Confidence Calculation Example

```python
def calculate_confidence(result: SearchResult) -> ConfidenceScore:
    """
    Calculate multi-dimensional confidence scores
    """
    # Score reliability based on score distribution
    score_reliability = min(result.hybrid_score * 1.2, 1.0)

    # Source quality based on document metadata
    source_quality = 0.8  # Base quality
    if result.source_category in ["official", "documentation"]:
        source_quality = 0.95
    elif result.source_category in ["community", "forum"]:
        source_quality = 0.65

    # Recency based on document age (mock calculation)
    days_old = 30  # Would be calculated from metadata
    recency = max(0.0, 1.0 - (days_old / 365))

    return ConfidenceScore(
        score_reliability=score_reliability,
        source_quality=source_quality,
        recency=recency
    )
```

### Ranking Context

Provides human-readable ranking explanations and percentile positioning:

```python
class RankingContext:
    """
    Contextual ranking information
    """

    percentile: int          # 0-100, position in result set
    explanation: str         # Human-readable explanation
    score_method: str        # vector/bm25/hybrid
```

### Ranking Generation Example

```python
def generate_ranking_context(
    result: SearchResult,
    all_scores: List[float]
) -> RankingContext:
    """
    Generate ranking context for a result
    """
    # Calculate percentile
    below = sum(1 for s in all_scores if s < result.hybrid_score)
    percentile = int((below / len(all_scores)) * 100)

    # Generate explanation
    if result.similarity_score > result.bm25_score:
        explanation = "Strong semantic match"
    elif result.bm25_score > result.similarity_score:
        explanation = "Strong keyword match"
    else:
        explanation = "Balanced semantic and keyword match"

    if percentile >= 95:
        explanation = f"Top-tier result: {explanation}"
    elif percentile >= 80:
        explanation = f"High relevance: {explanation}"

    return RankingContext(
        percentile=percentile,
        explanation=explanation,
        score_method=result.score_type
    )
```

## Implementation Patterns

### Pattern 1: Progressive Fetching

Start with minimal data, fetch more as needed:

```python
async def progressive_search(query: str, client: MCPClient):
    """
    Progressive fetching pattern for optimal token usage
    """
    # Step 1: Get metadata for initial assessment
    metadata_response = await client.semantic_search(
        query=query,
        response_mode="metadata",
        page_size=20
    )

    # Step 2: Identify high-value results
    high_value_ids = [
        r.chunk_id for r in metadata_response.results
        if r.hybrid_score > 0.8
    ][:3]

    # Step 3: Fetch full content for high-value results only
    if high_value_ids:
        full_response = await client.semantic_search(
            query=query,
            response_mode="full",
            page_size=3,
            fields=["chunk_id", "chunk_text", "source_file"]
        )

    return {
        "overview": metadata_response,
        "details": full_response if high_value_ids else None
    }
```

### Pattern 2: Hybrid Mode Strategy

Combine different modes in a single workflow:

```python
class HybridSearchStrategy:
    """
    Combines multiple response modes for optimal results
    """

    async def execute(self, query: str, token_budget: int):
        # Quick overview with IDs
        ids = await self.search(query, mode="ids_only", size=50)

        if ids.total_found == 0:
            return None

        # Get metadata for top results
        top_ids = [r.chunk_id for r in ids.results[:10]]
        metadata = await self.search_by_ids(top_ids, mode="metadata")

        # Full content for best matches
        best_ids = [
            r.chunk_id for r in metadata.results
            if r.hybrid_score > 0.85
        ][:2]

        if best_ids:
            full = await self.search_by_ids(best_ids, mode="full")
            return self.combine_results(metadata, full)

        return metadata
```

### Pattern 3: Field-Filtered Responses

Use field filtering for minimal token transfer:

```python
# Example: Getting just filenames and scores
response = semantic_search(
    query="authentication",
    response_mode="metadata",
    fields=["chunk_id", "source_file", "hybrid_score"],
    page_size=30
)

# Result with only requested fields
{
    "results": [
        {
            "chunk_id": 123,
            "source_file": "docs/auth.md",
            "hybrid_score": 0.85
        }
        # ... more results
    ]
}
```

### Pattern 4: Cursor-Based Pagination

Efficiently navigate large result sets:

```python
async def paginate_all_results(query: str, client: MCPClient):
    """
    Fetch all results using cursor pagination
    """
    all_results = []
    cursor = None

    while True:
        response = await client.semantic_search(
            query=query,
            response_mode="metadata",
            page_size=20,
            cursor=cursor
        )

        all_results.extend(response.results)

        # Check if more pages available
        if not response.pagination or not response.pagination.has_more:
            break

        cursor = response.pagination.cursor

    return all_results
```

## Performance Optimization

### Caching Strategy

The system implements intelligent caching at multiple levels:

```python
class ResponseCache:
    """
    Multi-level caching for response optimization
    """

    def get_cache_key(self, query: str, mode: str, page: int) -> str:
        """Generate cache key including response mode"""
        params = {
            "query": query,
            "mode": mode,
            "page": page
        }
        hash_value = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"search:{mode}:{hash_value}"

    def cache_duration(self, mode: str) -> int:
        """Mode-specific cache durations"""
        durations = {
            "ids_only": 60,      # 1 minute - lightweight
            "metadata": 30,      # 30 seconds - moderate
            "preview": 20,       # 20 seconds - heavier
            "full": 10          # 10 seconds - heavyweight
        }
        return durations.get(mode, 30)
```

### Response Time Benchmarks

Expected response times under different conditions:

| Mode | Cached (P50/P95) | Fresh (P50/P95) |
|------|------------------|-----------------|
| `ids_only` | 15ms / 25ms | 80ms / 150ms |
| `metadata` | 20ms / 35ms | 120ms / 250ms |
| `preview` | 25ms / 45ms | 180ms / 350ms |
| `full` | 30ms / 55ms | 250ms / 500ms |

### Optimization Techniques

1. **Batch Processing**: Group multiple ID lookups
2. **Predictive Caching**: Pre-cache likely follow-up queries
3. **Compression**: Use gzip for large responses
4. **Connection Pooling**: Reuse database connections
5. **Query Optimization**: Use indexed fields for filtering

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Response Exceeds Token Limit

**Symptoms:**
```json
{
    "warnings": [{
        "level": "error",
        "code": "TOKEN_LIMIT_EXCEEDED",
        "message": "Response size (18500 tokens) exceeds limit (15000)",
        "suggestion": "Use 'metadata' mode or reduce page_size to 5"
    }]
}
```

**Solution:**
```python
# Reduce response size
response = semantic_search(
    query="same query",
    response_mode="metadata",  # Switch from "full"
    page_size=5                # Reduce from 10
)
```

#### Issue: Field Not Available in Response Mode

**Symptoms:**
```
ValueError: Invalid fields for response_mode 'metadata': ['chunk_text'].
Allowed fields: ['chunk_id', 'source_file', 'source_category', 'hybrid_score', 'rank', 'chunk_index', 'total_chunks']
```

**Solution:**
```python
# Use appropriate mode for desired fields
response = semantic_search(
    query="query",
    response_mode="full",  # Switch to mode that includes chunk_text
    fields=["chunk_id", "chunk_text"]
)
```

#### Issue: Cursor Mismatch

**Symptoms:**
```
Warning: Cursor query_hash mismatch - treating as new query
```

**Solution:**
```python
# Ensure consistent query parameters when paginating
page1 = semantic_search(query="auth", response_mode="metadata")
# Use same query and mode for next page
page2 = semantic_search(
    query="auth",  # Same query
    response_mode="metadata",  # Same mode
    cursor=page1.pagination.cursor
)
```

#### Issue: Slow Response Times

**Symptoms:**
- Response times > 500ms for metadata mode
- Cache hit rate < 20%

**Solution:**
```python
# Optimize query patterns
class QueryOptimizer:
    def optimize(self, query: str) -> str:
        """Normalize queries for better cache hits"""
        # Remove extra whitespace
        query = " ".join(query.split())
        # Lowercase for consistency
        query = query.lower()
        # Remove common words that don't affect results
        stopwords = {"the", "a", "an", "and", "or", "but"}
        words = [w for w in query.split() if w not in stopwords]
        return " ".join(words)
```

### Debugging Response Formatting

Enable debug logging to trace response formatting:

```python
import logging
from src.core.logging import StructuredLogger

# Enable debug logging
logger = StructuredLogger.get_logger(__name__)
logger.setLevel(logging.DEBUG)

# Log response formatting details
logger.debug("Response formatting", extra={
    "mode": response_mode,
    "results_count": len(results),
    "estimated_tokens": estimated_tokens,
    "actual_tokens": actual_tokens,
    "cache_hit": cache_hit,
    "execution_time_ms": execution_time
})
```

### Performance Monitoring

Track key metrics for response formatting:

```python
class ResponseMetrics:
    """Track response formatting performance"""

    def __init__(self):
        self.metrics = {
            "token_efficiency": [],  # actual/estimated ratio
            "cache_hit_rate": [],
            "response_times": {},
            "mode_distribution": {}
        }

    def record_response(self, response: MCPResponseEnvelope):
        """Record metrics from response"""
        # Token efficiency
        if response.execution_context.tokens_used:
            efficiency = (
                response.execution_context.tokens_used /
                response.execution_context.tokens_estimated
            )
            self.metrics["token_efficiency"].append(efficiency)

        # Cache hit rate
        self.metrics["cache_hit_rate"].append(
            response.execution_context.cache_hit
        )

        # Response time by mode
        mode = response._metadata.operation_details.get("mode", "unknown")
        if mode not in self.metrics["response_times"]:
            self.metrics["response_times"][mode] = []
        self.metrics["response_times"][mode].append(
            response.execution_context.execution_time_ms
        )

    def get_summary(self) -> dict:
        """Get performance summary"""
        return {
            "avg_token_efficiency": np.mean(self.metrics["token_efficiency"]),
            "cache_hit_rate": np.mean(self.metrics["cache_hit_rate"]),
            "avg_response_time_by_mode": {
                mode: np.mean(times)
                for mode, times in self.metrics["response_times"].items()
            }
        }
```

## Best Practices Summary

1. **Start with Metadata Mode**: Default to `metadata` for initial searches
2. **Use Progressive Fetching**: Get overview first, then fetch details
3. **Leverage Field Filtering**: Request only needed fields
4. **Monitor Token Usage**: Track efficiency with execution context
5. **Cache Strategically**: Use mode-appropriate cache durations
6. **Handle Warnings**: Respond to warnings proactively
7. **Optimize for Desktop**: Consider context limits in mode selection
8. **Document Mode Selection**: Explain why specific modes are chosen
9. **Test Edge Cases**: Verify behavior with empty results, large sets
10. **Profile Performance**: Monitor response times and token usage

## Conclusion

The response formatting system provides a powerful, flexible foundation for efficient MCP tool integration with Claude Desktop. By understanding and properly utilizing the progressive disclosure pattern, response modes, and optimization techniques described in this guide, you can build highly efficient, scalable applications that maximize the value of every token in the context window.

For implementation examples and API references, see:
- [API Reference: Response Formats](../api-reference/response-formats.md)
- [Claude Desktop Optimization Guide](./claude-desktop-optimization.md)
- [MCP Tools API Reference](../api-reference/mcp-tools.md)