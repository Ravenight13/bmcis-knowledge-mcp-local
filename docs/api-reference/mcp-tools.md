# BMCIS Knowledge MCP Tools API Reference

## Overview

The BMCIS Knowledge MCP (Model Context Protocol) server provides powerful semantic search and vendor information tools for Claude Desktop and other MCP-compatible clients. The server exposes knowledge graph data through a token-efficient progressive disclosure pattern, enabling Claude to access comprehensive business information while managing context window usage.

### Key Capabilities

- **Hybrid Semantic Search**: Combines vector similarity and BM25 keyword matching with Reciprocal Rank Fusion for superior search quality
- **Vendor Knowledge Graph**: Comprehensive vendor information with entity relationships and statistics
- **Progressive Disclosure**: 4-tier response system optimizes token usage (83-94% reduction vs full responses)
- **High Performance**: Sub-200ms median response times with intelligent caching
- **Security First**: API key authentication with constant-time comparison and multi-tier rate limiting

### Architecture Overview

```
┌─────────────────┐     MCP Protocol      ┌──────────────────┐
│ Claude Desktop  │◄─────────────────────►│  FastMCP Server  │
│   (MCP Client)  │                        │  (bmcis-knowledge)│
└─────────────────┘                        └──────────────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │   Authentication   │
                                          │  & Rate Limiting   │
                                          └─────────┬──────────┘
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              │                     │                     │
                     ┌────────▼────────┐  ┌────────▼────────┐  ┌─────────▼────────┐
                     │ semantic_search │  │ find_vendor_info│  │  Knowledge Graph │
                     │      Tool        │  │      Tool       │  │    PostgreSQL    │
                     └─────────────────┘  └─────────────────┘  └──────────────────┘
```

### Quick Start (5 Minutes to First Query)

1. **Set your API key**:
   ```bash
   export BMCIS_API_KEY="your-api-key-here"
   ```

2. **Configure Claude Desktop** (`.mcp.json`):
   ```json
   {
     "mcpServers": {
       "bmcis-knowledge": {
         "command": "python",
         "args": ["-m", "src.mcp.server"],
         "env": {
           "BMCIS_API_KEY": "your-api-key-here"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

4. **Test your first search**:
   ```
   Use the semantic_search tool to find information about "cloud providers"
   ```

## Authentication

### Getting Your API Key

API keys are managed by your BMCIS administrator. Contact your system administrator to obtain your unique API key.

### Setting the BMCIS_API_KEY Environment Variable

#### Option 1: System Environment Variable (Recommended)
```bash
# Linux/Mac
echo 'export BMCIS_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Windows (PowerShell)
[System.Environment]::SetEnvironmentVariable('BMCIS_API_KEY', 'your-api-key-here', 'User')
```

#### Option 2: MCP Configuration
Include the API key in your `.mcp.json` configuration file (shown in Quick Start above).

### API Key Security Best Practices

1. **Never commit API keys to version control** - Add `.mcp.json` to `.gitignore`
2. **Use environment variables** instead of hardcoding keys
3. **Rotate keys regularly** - Request new keys every 90 days
4. **Use unique keys per application** - Don't share keys between services
5. **Store keys securely** - Use password managers or secret management tools

### Authentication Error Responses

| Error Message | Cause | Solution |
|--------------|-------|----------|
| "Authentication required. API key must be provided" | No API key configured | Set BMCIS_API_KEY environment variable |
| "Invalid API key. Please check your credentials" | Wrong or expired API key | Verify your API key with administrator |
| "BMCIS_API_KEY environment variable not set" | Server misconfiguration | Contact system administrator |

## semantic_search Tool

### Purpose

Find entities in the knowledge graph using natural language queries with hybrid semantic search. Combines vector similarity (semantic understanding) with BM25 keyword matching for optimal results.

### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (natural language or keywords). Max 500 characters. |
| `top_k` | integer | No | 10 | Number of results to return (1-50). |
| `response_mode` | string | No | "metadata" | Response detail level: "ids_only", "metadata", "preview", or "full". |

### Output Schema

The response structure depends on the `response_mode` parameter:

#### Common Response Fields (All Modes)

```json
{
  "results": [...],           // Array of search results (type varies by mode)
  "total_found": 10,          // Number of results returned
  "strategy_used": "hybrid",  // Search strategy (always "hybrid" for MCP)
  "execution_time_ms": 145.3  // Query execution time in milliseconds
}
```

#### Response Mode: ids_only

Minimal response with chunk IDs and scores only (~100 tokens for 10 results).

```json
{
  "results": [
    {
      "chunk_id": 12345,
      "hybrid_score": 0.875,
      "rank": 1
    }
  ]
}
```

#### Response Mode: metadata (DEFAULT)

Balanced response with file information and scores (~2-4K tokens for 10 results).

```json
{
  "results": [
    {
      "chunk_id": 12345,
      "source_file": "docs/cloud-providers.md",
      "source_category": "documentation",
      "hybrid_score": 0.875,
      "rank": 1,
      "chunk_index": 3,
      "total_chunks": 15
    }
  ]
}
```

#### Response Mode: preview

Includes metadata plus 200-character text snippet (~5-10K tokens for 10 results).

```json
{
  "results": [
    {
      "chunk_id": 12345,
      "source_file": "docs/cloud-providers.md",
      "source_category": "documentation",
      "hybrid_score": 0.875,
      "rank": 1,
      "chunk_index": 3,
      "total_chunks": 15,
      "chunk_snippet": "AWS, Azure, and Google Cloud Platform are the three leading cloud providers. Each offers comprehensive compute, storage, and networking services with distinct pricing models and specializations...",
      "context_header": "## Major Cloud Providers"
    }
  ]
}
```

#### Response Mode: full

Complete chunk content with all scoring details (~15K+ tokens for 10 results).

```json
{
  "results": [
    {
      "chunk_id": 12345,
      "chunk_text": "AWS, Azure, and Google Cloud Platform are the three leading cloud providers. Each offers comprehensive compute, storage, and networking services with distinct pricing models and specializations. AWS leads in market share with the broadest service portfolio, Azure excels in enterprise integration, and GCP offers superior data analytics and ML capabilities...",
      "similarity_score": 0.892,
      "bm25_score": 0.743,
      "hybrid_score": 0.875,
      "rank": 1,
      "score_type": "hybrid",
      "source_file": "docs/cloud-providers.md",
      "source_category": "documentation",
      "context_header": "## Major Cloud Providers",
      "chunk_index": 3,
      "total_chunks": 15,
      "chunk_token_count": 487
    }
  ]
}
```

### Examples

#### Example 1: Basic Search (Default Metadata Mode)

```python
# Claude's request:
semantic_search("cloud security best practices")

# Response:
{
  "results": [
    {
      "chunk_id": 23456,
      "source_file": "guides/cloud-security.md",
      "source_category": "guides",
      "hybrid_score": 0.912,
      "rank": 1,
      "chunk_index": 7,
      "total_chunks": 42
    },
    // ... 9 more results
  ],
  "total_found": 10,
  "strategy_used": "hybrid",
  "execution_time_ms": 127.4
}
```

#### Example 2: Token-Efficient Research Workflow

```python
# Step 1: Quick relevance check with IDs only
results = semantic_search("authentication protocols", top_k=20, response_mode="ids_only")
# Cost: ~200 tokens

# Step 2: Get metadata for promising results
results = semantic_search("authentication protocols", top_k=10, response_mode="metadata")
# Cost: ~3,000 tokens

# Step 3: Full content for top 3 most relevant
results = semantic_search("authentication protocols", top_k=3, response_mode="full")
# Cost: ~5,000 tokens

# Total: ~8,200 tokens (vs ~30,000 for full mode on all 20)
```

#### Example 3: Preview Mode for Quick Context

```python
# Claude's request:
semantic_search("vendor management systems", response_mode="preview")

# Response includes 200-char snippets:
{
  "results": [
    {
      "chunk_id": 34567,
      "source_file": "systems/vendor-mgmt.md",
      "chunk_snippet": "Modern vendor management systems integrate procurement, compliance, and performance tracking. Key features include automated onboarding, risk assessment, contract management, and spend analytics...",
      "context_header": "## Vendor Management Platform Components",
      // ... other fields
    }
  ]
}
```

### Error Cases and Recovery

| Error | Cause | Recovery Strategy |
|-------|-------|-------------------|
| "Invalid request parameters: query exceeds 500 characters" | Query too long | Shorten query to key terms |
| "Invalid request parameters: top_k must be between 1 and 50" | Invalid top_k value | Use value between 1-50 |
| "Search failed: Database connection error" | Database unavailable | Retry after 5 seconds |
| "Rate limit exceeded" | Too many requests | Wait for rate limit reset |

### Token Budget by Mode

| Mode | Tokens (10 results) | Use Case |
|------|---------------------|----------|
| ids_only | ~100 | Quick relevance check, ID collection |
| metadata | ~2,500 | Default - balanced information |
| preview | ~5,000 | Content sampling with snippets |
| full | ~15,000+ | Deep analysis, complete context |

### Caching Behavior

The `semantic_search` tool implements automatic in-memory caching for improved performance:

- **Cache TTL**: 30 seconds (queries expire after 30s)
- **Cache Key**: Computed from `query + top_k + response_mode + fields` (exact match required)
- **Hit Rate**: Typically 70-90% in multi-turn conversations
- **Performance Impact**: Cached results return in <100ms vs 200-500ms for database queries
- **Manual Invalidation**: Not exposed through MCP (cache expires automatically via TTL)

**Example**:
```python
# First request (cache miss) - executes database query (~245ms)
result1 = semantic_search("authentication", top_k=10, response_mode="metadata")

# Same query 5 seconds later (cache hit) - returns instantly (~85ms)
result2 = semantic_search("authentication", top_k=10, response_mode="metadata")

# Different parameter (cache miss) - new cache entry
result3 = semantic_search("authentication", top_k=20, response_mode="metadata")
```

**Note**: Cache entries are automatically evicted after 30 seconds or when the cache reaches 1,000 entries (LRU eviction).

For detailed caching configuration, see [Caching Configuration Guide](../guides/caching-configuration.md).

### Pagination Support

The `semantic_search` tool supports cursor-based pagination for efficient browsing of large result sets:

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page_size` | integer | No | 10 | Number of results per page (1-50). Replaces `top_k` if both provided. |
| `cursor` | string | No | null | Pagination cursor from previous response. Omit for first page. |

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `cursor` | string \| null | Cursor for next page. Null on last page. |
| `has_more` | boolean | True if more results available. |
| `total_available` | integer | Total results matching query. |
| `page_size` | integer | Results per page. |
| `returned_count` | integer | Actual results in this page. |

#### Pagination Example

```python
# Fetch first page
page1 = semantic_search(
    query="database optimization",
    page_size=10,
    response_mode="metadata"
)

# Response:
# {
#   "results": [...],  # 10 results
#   "cursor": "eyJxdWVyeV9oYXNoIjogImFiYzEyMyIsICJvZmZzZXQiOiAxMH0=",
#   "has_more": True,
#   "total_available": 47,
#   "page_size": 10,
#   "returned_count": 10
# }

# Fetch second page using cursor
page2 = semantic_search(
    query="database optimization",
    page_size=10,
    cursor=page1.cursor,
    response_mode="metadata"
)

# Continue until has_more == False
```

**Cursor Stability**: Cursors are tied to the exact query and remain valid for the cache TTL period (30s). If the cached query expires, the cursor becomes invalid.

For comprehensive pagination patterns, see [Pagination & Filtering Guide](../guides/pagination-filtering-guide.md).

### Field Filtering

Select specific fields to minimize token consumption beyond progressive disclosure:

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `fields` | list[string] | No | null | Whitelist of fields to include. Omit for all fields. |

#### Available Fields by Response Mode

**ids_only mode**:
- `chunk_id`, `hybrid_score`, `rank`

**metadata mode** (includes ids_only fields):
- `source_file`, `source_category`, `chunk_index`, `total_chunks`

**preview mode** (includes metadata fields):
- `chunk_snippet`, `context_header`

**full mode** (includes preview fields):
- `chunk_text`, `similarity_score`, `bm25_score`, `score_type`, `chunk_token_count`

#### Field Filtering Example

```python
# Request only chunk_id and score (minimal token usage)
results = semantic_search(
    query="security vulnerabilities",
    page_size=20,
    response_mode="metadata",
    fields=["chunk_id", "hybrid_score"]
)

# Response (per result): ~10 tokens
# {
#   "chunk_id": 142,
#   "hybrid_score": 0.92
# }

# vs. unfiltered metadata response: ~200 tokens per result
```

**Token Savings**:
- **Unfiltered metadata** (10 results): ~2,000 tokens
- **Filtered (chunk_id + score)** (10 results): ~100 tokens
- **Savings**: 95% reduction

**Validation**: Requesting a field not available in the selected `response_mode` will raise a validation error.

For field combination strategies, see [Pagination & Filtering Guide](../guides/pagination-filtering-guide.md).

### Best Practices

1. **Start with metadata mode** (default) for initial exploration
2. **Use progressive disclosure**: ids_only → metadata → preview → full
3. **Leverage caching**: Identical queries within 30s return instantly
4. **Paginate large result sets**: Use `page_size` + `cursor` for browsing 50+ results
5. **Filter fields**: Request only needed fields for 85-95% token reduction
6. **Keep queries focused** - Specific terms yield better results
7. **Use natural language** - The semantic search understands context

## find_vendor_info Tool

### Purpose

Retrieve comprehensive information about a vendor from the knowledge graph, including related entities, relationships, and statistics. Supports progressive disclosure for efficient token usage.

### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vendor_name` | string | Yes | - | Exact vendor name (1-200 characters). Case-insensitive. |
| `response_mode` | string | No | "metadata" | Response detail level: "ids_only", "metadata", "preview", or "full". |
| `include_relationships` | boolean | No | true | Include relationship data in response. |

### Output Schema

The response structure varies by `response_mode`:

#### Response Mode: ids_only

Minimal vendor information (~100-500 tokens).

```json
{
  "vendor_name": "Acme Corp",
  "entity_ids": ["uuid-12345"],
  "relationship_ids": []
}
```

#### Response Mode: metadata (DEFAULT)

Vendor statistics and type distributions (~2-4K tokens).

```json
{
  "vendor_name": "Acme Corp",
  "statistics": {
    "entity_count": 47,
    "relationship_count": 89,
    "entity_type_distribution": {
      "PRODUCT": 15,
      "PERSON": 12,
      "ORG": 8,
      "LOCATION": 7,
      "EVENT": 5
    },
    "relationship_type_distribution": {
      "PRODUCES": 15,
      "WORKS_FOR": 12,
      "PARTNERS_WITH": 10,
      "LOCATED_IN": 7,
      "COMPETES_WITH": 5
    }
  },
  "top_entities": null,
  "last_updated": "2024-11-09T15:30:00Z"
}
```

#### Response Mode: preview

Top 5 entities and relationships (~5-10K tokens).

```json
{
  "vendor_name": "Acme Corp",
  "entities": [
    {
      "entity_id": "uuid-23456",
      "name": "Acme Cloud Platform",
      "entity_type": "PRODUCT",
      "confidence": 0.95,
      "snippet": null
    },
    // ... up to 5 entities
  ],
  "relationships": [
    {
      "source_id": "uuid-12345",
      "target_id": "uuid-23456",
      "relationship_type": "PRODUCES",
      "metadata": null
    },
    // ... up to 5 relationships
  ],
  "statistics": { /* same as metadata mode */ }
}
```

#### Response Mode: full

Complete vendor graph (max 100 entities, 500 relationships, ~10-50K+ tokens).

```json
{
  "vendor_name": "Acme Corp",
  "entities": [
    {
      "entity_id": "uuid-23456",
      "name": "Acme Cloud Platform",
      "entity_type": "PRODUCT",
      "confidence": 0.95,
      "snippet": "Enterprise cloud computing platform offering IaaS, PaaS, and SaaS solutions..."
    },
    // ... up to 100 entities
  ],
  "relationships": [
    {
      "source_id": "uuid-12345",
      "target_id": "uuid-23456",
      "relationship_type": "PRODUCES",
      "metadata": {
        "confidence": 0.92,
        "relationship_metadata": {"since": "2019", "primary": true}
      }
    },
    // ... up to 500 relationships
  ],
  "statistics": { /* same as metadata mode */ }
}
```

### Examples

#### Example 1: Quick Vendor Overview

```python
# Claude's request:
find_vendor_info("Microsoft Corporation")

# Response (metadata mode):
{
  "vendor_name": "Microsoft Corporation",
  "statistics": {
    "entity_count": 234,
    "relationship_count": 567,
    "entity_type_distribution": {
      "PRODUCT": 89,
      "PERSON": 45,
      "ORG": 34,
      "LOCATION": 23,
      "EVENT": 43
    }
  }
}
```

#### Example 2: Vendor Deep Dive Workflow

```python
# Step 1: Check if vendor exists and get overview
overview = find_vendor_info("Acme Corp", response_mode="metadata")
# Confirms vendor exists, shows 47 entities

# Step 2: Preview top entities and relationships
preview = find_vendor_info("Acme Corp", response_mode="preview")
# Shows top 5 products, partners, locations

# Step 3: Get complete vendor graph for analysis
full_data = find_vendor_info("Acme Corp", response_mode="full")
# Returns all 47 entities and 89 relationships for comprehensive analysis
```

#### Example 3: Handling Ambiguous Names

```python
# Claude's request:
find_vendor_info("Acme")

# Error response:
"Ambiguous vendor name. Multiple matches found: ['Acme Corp', 'ACME Inc', 'Acme Ltd'].
Please use full vendor name or a more specific term."

# Recovery - try first match:
find_vendor_info("Acme Corp")
# Success
```

### Error Cases and Recovery

| Error | Cause | Recovery Strategy |
|-------|-------|-------------------|
| "Vendor 'XYZ' not found" | No exact match | Use semantic_search first to find exact vendor names |
| "Ambiguous vendor name" | Multiple matches | Use full vendor name from error message |
| "Request validation failed" | Invalid parameters | Check vendor_name length (1-200 chars) |
| "Vendor info retrieval failed" | Database error | Retry request after 5 seconds |

### Caching Behavior

The `find_vendor_info` tool implements automatic in-memory caching optimized for vendor graph queries:

- **Cache TTL**: 300 seconds (5 minutes) - longer than semantic_search due to vendor data stability
- **Cache Key**: Computed from `vendor_name + response_mode + include_relationships`
- **Hit Rate**: Typically 85-95% for vendor-focused workflows
- **Performance Impact**: Cached results return in <50ms vs 200-1,500ms for database queries
- **Manual Invalidation**: Not exposed through MCP (cache expires automatically via TTL)

**Example**:
```python
# First request (cache miss) - executes database query (~500ms for full mode)
vendor1 = find_vendor_info("Acme Corp", response_mode="full")

# Same query 2 minutes later (cache hit) - returns instantly (~45ms)
vendor2 = find_vendor_info("Acme Corp", response_mode="full")

# Different response_mode (cache miss) - separate cache entry
vendor3 = find_vendor_info("Acme Corp", response_mode="metadata")
```

**Note**: Vendor data is relatively static, so the 5-minute TTL provides excellent cache efficiency while ensuring reasonable freshness.

For detailed caching configuration, see [Caching Configuration Guide](../guides/caching-configuration.md).

### When to Use vs semantic_search

| Use find_vendor_info when: | Use semantic_search when: |
|---------------------------|--------------------------|
| You know the exact vendor name | You're searching for vendors by description |
| You need entity relationships | You need text content about topics |
| You want vendor statistics | You're doing keyword searches |
| You need the vendor's knowledge graph | You want to find vendor names first |

### Real-World Workflow Examples

#### Vendor Competitive Analysis

```python
# 1. Find main vendor
vendor_info = find_vendor_info("Acme Corp", response_mode="preview")

# 2. Identify competitors from relationships
competitors = [e for e in vendor_info.entities if e.entity_type == "ORG"]

# 3. Deep dive on each competitor
for competitor in competitors[:3]:
    comp_info = find_vendor_info(competitor.name, response_mode="metadata")
    # Compare statistics, product counts, etc.
```

#### Supply Chain Mapping

```python
# 1. Start with vendor
main_vendor = find_vendor_info("GlobalTech Inc", response_mode="full")

# 2. Extract all partner organizations
partners = [e for e in main_vendor.entities
           if e.entity_type == "ORG" and
           any(r.relationship_type == "PARTNERS_WITH"
               for r in main_vendor.relationships)]

# 3. Build supply chain graph
for partner in partners:
    partner_info = find_vendor_info(partner.name, response_mode="preview")
    # Map relationships, identify critical suppliers
```

## Response Modes Explained

The progressive disclosure pattern allows you to optimize token usage by requesting only the level of detail needed for each query stage.

### ids_only Mode

**When to use:**
- Initial filtering of large result sets
- Collecting IDs for batch processing
- Quick relevance checks
- Token budget is extremely limited

**Example response:**
```json
{
  "chunk_id": 12345,
  "hybrid_score": 0.875,
  "rank": 1
}
```

**Token cost:** ~10 tokens per result

### metadata Mode (DEFAULT)

**When to use:**
- Most search operations (balanced approach)
- Understanding result distribution
- File and category identification
- Initial research phases

**Example response:**
```json
{
  "chunk_id": 12345,
  "source_file": "docs/security.md",
  "source_category": "documentation",
  "hybrid_score": 0.875,
  "rank": 1,
  "chunk_index": 3,
  "total_chunks": 15
}
```

**Token cost:** ~250 tokens per result

### preview Mode

**When to use:**
- Content sampling before full retrieval
- Quick context understanding
- Snippet-based summaries
- Validating relevance with content preview

**Example response:**
```json
{
  "chunk_id": 12345,
  "source_file": "docs/security.md",
  "chunk_snippet": "OAuth 2.0 provides authorization flows for web applications, desktop applications, mobile phones, and smart devices. The framework defines four grant types...",
  "context_header": "## OAuth 2.0 Framework",
  // ... other metadata fields
}
```

**Token cost:** ~500 tokens per result

### full Mode

**When to use:**
- Deep content analysis
- Complete context required
- Final detailed review
- Token budget allows full content

**Example response:**
```json
{
  "chunk_id": 12345,
  "chunk_text": "[Complete 1500+ token chunk content]",
  "similarity_score": 0.892,
  "bm25_score": 0.743,
  "hybrid_score": 0.875,
  // ... all fields including token count
}
```

**Token cost:** ~1500+ tokens per result

### Token Efficiency Comparison

| Scenario | Traditional (Full) | Progressive | Savings |
|----------|-------------------|-------------|---------|
| 10 results exploration | 15,000 tokens | 2,500 tokens (metadata) | 83% |
| 20 results filtering | 30,000 tokens | 200 tokens (ids_only) | 99% |
| Deep dive on 3 of 10 | 15,000 tokens | 7,500 tokens (metadata + 3 full) | 50% |
| Vendor analysis | 50,000 tokens | 3,000 tokens (metadata) | 94% |

## Rate Limiting

The MCP server implements multi-tier rate limiting to ensure fair usage and system stability.

### Rate Limits

| Tier | Limit | Window | Reset |
|------|-------|--------|-------|
| Per-minute | 100 requests | 60 seconds | Rolling window |
| Per-hour | 1,000 requests | 3,600 seconds | Rolling window |
| Per-day | 10,000 requests | 86,400 seconds | Rolling window |

### Token Bucket Algorithm

The server uses a token bucket algorithm for smooth rate limiting:
- Each tier has its own token bucket
- Tokens refill at tier boundaries (minute/hour/day)
- Requests consume 1 token from each bucket
- Request allowed only if all buckets have tokens

### Checking Remaining Quota

Rate limit information is included in error responses when limits are exceeded:

```json
{
  "error": "Rate limit exceeded",
  "remaining": {
    "minute": 0,
    "hour": 450,
    "day": 8500
  },
  "reset_in_seconds": 35
}
```

### Handling Rate Limit Errors

```python
try:
    result = semantic_search("query")
except RateLimitError as e:
    # Extract wait time from error message
    wait_seconds = e.reset_in_seconds
    print(f"Rate limited. Waiting {wait_seconds} seconds...")
    time.sleep(wait_seconds)
    # Retry request
    result = semantic_search("query")
```

### Best Practices for Rate Limit Management

1. **Implement exponential backoff** for retries
2. **Cache results** to avoid redundant queries
3. **Batch related queries** when possible
4. **Use progressive disclosure** to minimize requests
5. **Monitor usage patterns** to optimize query distribution

## Error Handling

### Complete Error Code Reference

| Error Code | Description | Recovery Action |
|------------|-------------|-----------------|
| AUTH_REQUIRED | No API key provided | Set BMCIS_API_KEY environment variable |
| AUTH_INVALID | Invalid API key | Verify API key with administrator |
| RATE_LIMIT | Rate limit exceeded | Wait for reset time specified in error |
| VENDOR_NOT_FOUND | Vendor doesn't exist | Use semantic_search to find correct name |
| VENDOR_AMBIGUOUS | Multiple vendor matches | Use full vendor name from suggestions |
| QUERY_TOO_LONG | Query exceeds 500 chars | Shorten query to essential terms |
| INVALID_PARAMS | Parameter validation failed | Check parameter types and ranges |
| DB_CONNECTION | Database unavailable | Retry with exponential backoff |
| SEARCH_FAILED | Search execution error | Check logs and retry |

### Authentication Errors

```python
# Error: No API key
{
  "error": "Authentication required. API key must be provided.",
  "action": "Set BMCIS_API_KEY environment variable"
}

# Error: Invalid API key
{
  "error": "Invalid API key. Please check your credentials.",
  "action": "Contact administrator for valid API key"
}
```

### Vendor Not Found Errors

```python
# Error: Vendor not found
{
  "error": "Vendor 'Acme' not found. Try using semantic_search first to find exact vendor names.",
  "suggestion": "semantic_search('Acme vendors')"
}

# Recovery workflow:
# 1. Search for vendor
results = semantic_search("Acme vendors", response_mode="metadata")
# 2. Extract vendor names from results
# 3. Use exact name
vendor_info = find_vendor_info("Acme Corporation")
```

### Ambiguous Name Errors

```python
# Error: Multiple matches
{
  "error": "Ambiguous vendor name. Multiple matches found: ['Acme Corp', 'ACME Inc', 'Acme Ltd']",
  "suggestions": ["Acme Corp", "ACME Inc", "Acme Ltd"]
}

# Recovery: Use suggested exact name
vendor_info = find_vendor_info("Acme Corp")
```

### Database Connection Errors

```python
# Implement retry with exponential backoff
import time

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except DatabaseError:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            time.sleep(wait_time)
```

## Integration Guide

### Claude Desktop Setup

1. **Locate Claude configuration directory:**
   - macOS: `~/Library/Application Support/Claude/`
   - Windows: `%APPDATA%\Claude\`
   - Linux: `~/.config/Claude/`

2. **Create or edit `.mcp.json`:**
   ```json
   {
     "mcpServers": {
       "bmcis-knowledge": {
         "command": "python",
         "args": ["-m", "src.mcp.server"],
         "cwd": "/path/to/bmcis-knowledge-mcp",
         "env": {
           "BMCIS_API_KEY": "your-api-key-here",
           "PYTHONPATH": "/path/to/bmcis-knowledge-mcp"
         }
       }
     }
   }
   ```

3. **Verify Python environment:**
   ```bash
   python --version  # Should be 3.11+
   pip install fastmcp pydantic psycopg2-binary
   ```

4. **Restart Claude Desktop**

### MCP Configuration Options

```json
{
  "mcpServers": {
    "bmcis-knowledge": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "env": {
        // Required
        "BMCIS_API_KEY": "your-api-key",

        // Optional rate limit overrides
        "BMCIS_RATE_LIMIT_MINUTE": "100",
        "BMCIS_RATE_LIMIT_HOUR": "1000",
        "BMCIS_RATE_LIMIT_DAY": "10000",

        // Optional database configuration
        "DATABASE_URL": "postgresql://user:pass@localhost/bmcis",

        // Optional logging
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Testing Tools in Claude

After setup, test the tools in Claude:

1. **Test authentication:**
   ```
   Use the semantic_search tool to search for "test"
   ```

2. **Verify response modes:**
   ```
   Use semantic_search with query "cloud providers" and response_mode "metadata"
   ```

3. **Test vendor lookup:**
   ```
   Use find_vendor_info to get information about "Microsoft Corporation"
   ```

### Debugging Tips

1. **Enable debug logging:**
   ```json
   "env": {
     "LOG_LEVEL": "DEBUG"
   }
   ```

2. **Check MCP server logs:**
   - macOS/Linux: `~/.claude/logs/mcp-server.log`
   - Windows: `%APPDATA%\Claude\logs\mcp-server.log`

3. **Common issues and solutions:**

   | Issue | Solution |
   |-------|----------|
   | Tools not appearing | Restart Claude after config changes |
   | "Module not found" | Check PYTHONPATH in configuration |
   | Authentication errors | Verify BMCIS_API_KEY is set correctly |
   | Connection refused | Ensure database is running |
   | Slow responses | Check database indexes and connection pool |

4. **Test direct server connection:**
   ```bash
   # Test server startup
   BMCIS_API_KEY=test python -m src.mcp.server

   # Check for errors in output
   ```

5. **Validate configuration:**
   ```bash
   # Check JSON syntax
   python -m json.tool < .mcp.json
   ```

## Performance Considerations

### Response Time Targets

#### Uncached (Database Queries)

| Operation | P50 (Median) | P95 | P99 |
|-----------|--------------|-----|-----|
| semantic_search (metadata) | <200ms | <500ms | <800ms |
| semantic_search (full) | <300ms | <800ms | <1200ms |
| find_vendor_info (metadata) | <200ms | <500ms | <800ms |
| find_vendor_info (full) | <500ms | <1500ms | <2500ms |

#### Cached (In-Memory)

| Operation | P50 (Median) | P95 | P99 | Cache Hit Rate |
|-----------|--------------|-----|-----|----------------|
| semantic_search (any mode) | <50ms | <95ms | <100ms | 70-90% |
| find_vendor_info (any mode) | <30ms | <50ms | <75ms | 85-95% |

**Performance Impact**: Caching provides 65-87% latency reduction for repeated queries with negligible memory overhead (~10-50MB).

### Optimization Strategies

1. **Use appropriate response modes** - Don't request full content unless needed
2. **Leverage automatic caching** - Identical queries within TTL window return instantly
3. **Use pagination for large result sets** - Browse incrementally instead of fetching all upfront
4. **Apply field filtering** - Request only needed fields for 85-95% token reduction
5. **Batch related queries** - Reduces overhead and improves cache hit rate
6. **Implement client-side caching** - Cache vendor info and search results beyond MCP server TTL
7. **Use connection pooling** - Server maintains PostgreSQL connection pool automatically

### Token Usage Optimization

#### Inefficient Approach (No Progressive Disclosure, No Filtering)

```python
# Fetch 50 results in full mode upfront
results = semantic_search("security", top_k=50, response_mode="full")
# Cost: ~75,000 tokens
```

#### Efficient Approach (Progressive Disclosure Only)

```python
# Step 1: Quick scan
ids = semantic_search("security", top_k=50, response_mode="ids_only")
# Cost: ~500 tokens

# Step 2: Review metadata for top 20
metadata = semantic_search("security", top_k=20, response_mode="metadata")
# Cost: ~5,000 tokens

# Step 3: Full content for top 5
full = semantic_search("security", top_k=5, response_mode="full")
# Cost: ~7,500 tokens

# Total: ~13,000 tokens (83% reduction)
```

#### Highly Optimized Approach (Progressive Disclosure + Pagination + Filtering)

```python
# Step 1: Browse first page with minimal fields
page1 = semantic_search(
    query="security",
    page_size=10,
    response_mode="metadata",
    fields=["chunk_id", "hybrid_score", "source_file"]
)
# Cost: ~300 tokens (10 results × 30 tokens)

# Step 2: If promising, fetch next page
if page1.has_more and max(r.hybrid_score for r in page1.results) > 0.8:
    page2 = semantic_search(
        query="security",
        page_size=10,
        cursor=page1.cursor,
        response_mode="metadata",
        fields=["chunk_id", "hybrid_score", "source_file"]
    )
    # Cost: ~300 tokens

# Step 3: Fetch full details for only top 3 highest-scoring chunks
top_3_ids = sorted(
    page1.results + page2.results,
    key=lambda r: r.hybrid_score,
    reverse=True
)[:3]

for chunk_id in top_3_ids:
    full_result = fetch_chunk_by_id(chunk_id, response_mode="full")
    # Cost: ~1,500 tokens each = 4,500 tokens total

# Total: ~5,100 tokens (93% reduction vs inefficient approach)
```

**Token Efficiency Summary**:
- **Inefficient**: 75,000 tokens
- **Progressive disclosure**: 13,000 tokens (83% reduction)
- **Progressive + pagination + filtering**: 5,100 tokens (93% reduction)

**Combined with caching**: If you repeat any queries within the TTL window, subsequent fetches are instant with zero additional token cost.

## Response Formatting System

### Overview

All MCP tool responses use a standardized envelope format (`MCPResponseEnvelope`) that provides consistent structure, metadata, and progressive disclosure capabilities. This system is specifically optimized for Claude Desktop integration.

### Response Envelope Structure

Every response follows this structure:

```json
{
    "_metadata": {
        "operation": "semantic_search",
        "version": "1.0.0",
        "timestamp": "2025-11-09T10:30:00Z",
        "request_id": "req_abc123",
        "status": "success",
        "message": null
    },
    "results": [...],  // Tool-specific results
    "pagination": {
        "cursor": "eyJxdWVyeV9oYXNoIjoi...",
        "page_size": 10,
        "has_more": true,
        "total_available": 42
    },
    "execution_context": {
        "tokens_estimated": 2450,
        "tokens_used": 2523,
        "cache_hit": true,
        "execution_time_ms": 156.3,
        "request_id": "req_abc123"
    },
    "warnings": []
}
```

### Response Modes and Token Budgets

The system supports four progressive disclosure levels:

| Mode | Per-Result Tokens | 10 Results Total | Use Case |
|------|------------------|------------------|----------|
| `ids_only` | ~10 | ~100 | Quick scan, ID collection |
| `metadata` | ~200 | ~2,500 | **Default** - balanced info |
| `preview` | ~500 | ~5,000 | Content sampling |
| `full` | ~1500+ | ~15,000+ | Deep analysis |

### Desktop Mode Optimization

When Claude Desktop is detected, responses automatically include:

#### Enhanced Metadata
```json
{
    "_metadata": {
        "visual_hints": {
            "highlight_top": 3,
            "group_by": "source_category",
            "show_confidence": true,
            "collapse_duplicates": true,
            "expandable_previews": true
        }
    }
}
```

#### Confidence Scoring
```json
{
    "confidence": {
        "score_reliability": 0.92,
        "source_quality": 0.88,
        "recency": 0.75
    },
    "ranking": {
        "percentile": 99,
        "explanation": "Top-tier result: Strong semantic and keyword match",
        "score_method": "hybrid"
    }
}
```

### Field Filtering

Request only specific fields to minimize token usage:

```python
# Request only essential fields
response = semantic_search(
    query="authentication",
    response_mode="metadata",
    fields=["chunk_id", "source_file", "hybrid_score"]
)

# Response contains only requested fields
{
    "results": [
        {
            "chunk_id": 123,
            "source_file": "docs/auth.md",
            "hybrid_score": 0.85
        }
    ]
}
```

### Progressive Fetching Pattern

Optimize token usage with progressive detail fetching:

```python
# 1. Start with IDs for overview
ids = semantic_search(query, response_mode="ids_only", page_size=50)
# Cost: ~500 tokens

# 2. Get metadata for promising results
if any(r.hybrid_score > 0.7 for r in ids.results):
    metadata = semantic_search(query, response_mode="metadata", page_size=20)
    # Cost: ~4,000 tokens

# 3. Full content for top matches only
top_ids = [r.chunk_id for r in metadata.results if r.hybrid_score > 0.85][:3]
for chunk_id in top_ids:
    full = get_chunk_by_id(chunk_id, response_mode="full")
    # Cost: ~1,500 per result

# Total: ~9,000 tokens (vs ~75,000 for full mode on all 50)
```

### Warning System

Responses include actionable warnings for optimization:

```json
{
    "warnings": [
        {
            "level": "warning",
            "code": "TOKEN_LIMIT_WARNING",
            "message": "Response approaching context window limit (18,500/20,000 tokens)",
            "suggestion": "Use 'metadata' mode or reduce page_size to 5"
        },
        {
            "level": "info",
            "code": "CACHE_MISS_SLOW",
            "message": "Cache miss resulted in slower response (456ms)",
            "suggestion": "Query will be cached for next 30 seconds"
        }
    ]
}
```

### Standard Warning Codes

| Code | Level | Description | Typical Action |
|------|-------|-------------|----------------|
| `TOKEN_LIMIT_WARNING` | warning | Near token limit | Reduce response size |
| `TOKEN_LIMIT_EXCEEDED` | error | Exceeded limit | Use smaller mode |
| `LOW_QUALITY_RESULTS` | info | No high-confidence results | Refine query |
| `PARTIAL_RESULTS` | warning | Some results omitted | Reduce page_size |
| `DEPRECATED_PARAMETER` | warning | Using deprecated param | Update to new param |

### Execution Context Details

Track performance and token usage:

```json
{
    "execution_context": {
        "tokens_estimated": 2450,      // Pre-calculated estimate
        "tokens_used": 2523,           // Actual tokens (when measured)
        "cache_hit": true,             // Result from cache?
        "execution_time_ms": 85.3,    // Total execution time
        "request_id": "req_abc123",   // Matches _metadata.request_id

        // Extended metrics (when available)
        "token_breakdown": {
            "results": 2200,
            "metadata": 150,
            "pagination": 73,
            "warnings": 100
        },
        "cache_key": "search:metadata:abc123def",
        "database_time_ms": 0,  // 0 if cached
        "formatting_time_ms": 12.5
    }
}
```

### Error Response Format

Errors follow the same envelope structure:

```json
{
    "_metadata": {
        "operation": "semantic_search",
        "version": "1.0.0",
        "timestamp": "2025-11-09T10:30:00Z",
        "request_id": "req_error_123",
        "status": "error",
        "message": "Token limit exceeded for requested response mode"
    },
    "results": [],
    "execution_context": {
        "tokens_estimated": 25000,
        "tokens_used": null,
        "cache_hit": false,
        "execution_time_ms": 125.4,
        "request_id": "req_error_123"
    },
    "warnings": [
        {
            "level": "error",
            "code": "TOKEN_LIMIT_EXCEEDED",
            "message": "Response would use 25,000 tokens, Desktop limit is 15,000",
            "suggestion": "Use 'metadata' mode or reduce page_size to 5"
        }
    ]
}
```

### Semantic Search Response Examples

#### Example: Metadata Mode with Pagination
```python
response = semantic_search(
    query="cloud security best practices",
    response_mode="metadata",
    page_size=10,
    cursor=None  # First page
)

# Response structure
{
    "_metadata": {...},
    "results": [
        {
            "chunk_id": 123,
            "source_file": "guides/cloud-security.md",
            "source_category": "security",
            "hybrid_score": 0.92,
            "rank": 1,
            "chunk_index": 5,
            "total_chunks": 20
        }
        // ... 9 more results
    ],
    "pagination": {
        "cursor": "eyJxdWVyeV9oYXNoIjoiYWJjMTIzIiwib2Zmc2V0IjoxMH0=",
        "page_size": 10,
        "has_more": true,
        "total_available": 42
    },
    "execution_context": {
        "tokens_estimated": 2000,
        "tokens_used": 2050,
        "cache_hit": false,
        "execution_time_ms": 245.3
    }
}
```

#### Example: Desktop-Enhanced Response
```python
# With desktop mode, additional metadata is included
response = semantic_search(
    query="authentication",
    response_mode="metadata",
    desktop_enhanced=True  # Automatic in Claude Desktop
)

# Enhanced response includes confidence and ranking
{
    "results": [
        {
            "chunk_id": 456,
            "source_file": "auth/jwt.md",
            "hybrid_score": 0.95,
            "rank": 1,
            // Standard metadata fields...

            // Desktop enhancements
            "confidence": {
                "score_reliability": 0.98,
                "source_quality": 0.90,
                "recency": 0.85
            },
            "ranking": {
                "percentile": 100,
                "explanation": "Perfect match: JWT authentication guide",
                "score_method": "hybrid"
            },
            "deduplication": {
                "is_duplicate": false,
                "similar_chunk_ids": [457, 458],
                "confidence": 0.92
            }
        }
    ]
}
```

### Vendor Info Response Examples

#### Example: Progressive Vendor Discovery
```python
# Step 1: Get vendor IDs only
vendor_ids = find_vendor_info(
    vendor_name="Acme",
    response_mode="ids_only"
)
# Response: ~500 tokens
{
    "vendor_name": "Acme Corporation",
    "entity_ids": ["e_001", "e_002", "e_003", ...],
    "relationship_ids": ["r_001", "r_002", ...]
}

# Step 2: Get metadata with statistics
vendor_meta = find_vendor_info(
    vendor_name="Acme Corporation",
    response_mode="metadata"
)
# Response: ~3,000 tokens
{
    "vendor_name": "Acme Corporation",
    "statistics": {
        "entity_count": 85,
        "relationship_count": 25,
        "entity_type_distribution": {
            "COMPANY": 50,
            "PERSON": 25,
            "PRODUCT": 10
        }
    },
    "top_entities": [/* Top 5 by relevance */]
}

# Step 3: Get full details if needed
vendor_full = find_vendor_info(
    vendor_name="Acme Corporation",
    response_mode="full",
    include_relationships=True
)
# Response: ~15,000 tokens
{
    "vendor_name": "Acme Corporation",
    "entities": [/* Up to 100 entities */],
    "relationships": [/* Up to 500 relationships */],
    "statistics": {...}
}
```

### Best Practices for Response Formatting

1. **Always start with metadata mode** - It's the default for a reason
2. **Use field filtering aggressively** - Request only what you need
3. **Implement progressive fetching** - Start broad, then narrow
4. **Monitor token usage** - Check `execution_context.tokens_used`
5. **Handle warnings proactively** - Adjust before hitting limits
6. **Cache strategically** - Leverage the 30-second cache window
7. **Use pagination for exploration** - Don't fetch everything upfront
8. **Check confidence scores** - Focus on high-confidence results
9. **Respect Desktop limits** - Stay under 15,000 tokens per response
10. **Document mode choices** - Explain why specific modes are used

### Response Format Configuration

Configure response formatting preferences:

```python
# In .mcp.json environment variables
{
    "env": {
        // Response format preferences
        "DEFAULT_RESPONSE_MODE": "metadata",
        "MAX_TOKENS_PER_RESPONSE": "15000",
        "ENABLE_DESKTOP_ENHANCEMENTS": "true",
        "INCLUDE_CONFIDENCE_SCORES": "true",

        // Warning thresholds
        "TOKEN_WARNING_THRESHOLD": "0.8",  // Warn at 80% of limit
        "LOW_QUALITY_THRESHOLD": "0.5",    // Warn if all scores < 0.5

        // Performance settings
        "ENABLE_RESPONSE_COMPRESSION": "true",
        "RESPONSE_STREAMING": "false"  // Not yet supported
    }
}
```

### Migration from Legacy Formats

If migrating from older versions without envelope format:

```python
# Legacy format (pre-1.0.0)
old_response = {
    "results": [...],
    "total_found": 42
}

# New envelope format (1.0.0+)
new_response = {
    "_metadata": {...},
    "results": [...],
    "pagination": {...},
    "execution_context": {...},
    "warnings": []
}

# Client adaptation layer
def adapt_legacy_response(response):
    if "_metadata" not in response:
        # Wrap legacy response in envelope
        return {
            "_metadata": {
                "operation": "unknown",
                "version": "0.9.0",
                "status": "success"
            },
            "results": response.get("results", []),
            "execution_context": {
                "tokens_estimated": 0,
                "cache_hit": False
            },
            "warnings": [{
                "level": "warning",
                "code": "LEGACY_FORMAT",
                "message": "Response in legacy format",
                "suggestion": "Update server to 1.0.0+"
            }]
        }
    return response
```

## Additional Resources

For detailed information about response formatting:

- [Response Formatting Guide](../guides/response-formatting-guide.md) - Comprehensive formatting patterns and examples
- [Claude Desktop Optimization](../guides/claude-desktop-optimization.md) - Desktop-specific optimizations
- [API Reference: Response Formats](./response-formats.md) - Complete schema documentation
- [Pagination and Filtering Guide](../guides/pagination-filtering-guide.md) - Advanced pagination patterns

---

*Last updated: November 2024 | Version: 1.0.0*