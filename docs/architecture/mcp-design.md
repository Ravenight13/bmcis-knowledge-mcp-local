# MCP Architecture & Design

This document provides a comprehensive overview of the BMCIS Knowledge MCP server architecture, design principles, and technical implementation details.

## Design Principles

### 1. Progressive Disclosure for Token Efficiency

The core design principle is **progressive disclosure** - providing information in layers to optimize token usage while maintaining utility.

**Why Progressive Disclosure?**

Traditional approaches dump complete information regardless of need, consuming valuable context window tokens. Our 4-tier system reduces token usage by 83-94% while preserving the ability to drill down when needed.

**Implementation:**
- **Level 0 (ids_only)**: Minimal identifiers for filtering (~10 tokens/result)
- **Level 1 (metadata)**: Balanced information for most use cases (~250 tokens/result)
- **Level 2 (preview)**: Sampling with snippets (~500 tokens/result)
- **Level 3 (full)**: Complete content when needed (~1500+ tokens/result)

### 2. MCP Best Practices

We follow Model Context Protocol best practices for optimal integration with Claude:

**Tool Design:**
- **Single responsibility**: Each tool has one clear purpose
- **Predictable schemas**: Consistent request/response structures
- **Graceful degradation**: Tools handle errors without crashing
- **Self-documenting**: Rich descriptions and examples in tool definitions

**Protocol Compliance:**
- **Standard JSON-RPC 2.0**: Full protocol compliance
- **Proper error codes**: Semantic error responses
- **Streaming support**: Ready for future streaming capabilities
- **Version compatibility**: Backward compatible changes only

### 3. Security-First Approach

Security is built into every layer, not added as an afterthought:

**Authentication:**
- **Constant-time comparison**: Prevents timing attacks on API keys
- **No key logging**: API keys never appear in logs
- **Environment isolation**: Keys stored in environment variables
- **Fail-secure defaults**: Authentication required unless explicitly disabled

**Rate Limiting:**
- **Multi-tier limits**: Per-minute, per-hour, and per-day
- **Token bucket algorithm**: Smooth rate limiting without spikes
- **Graceful degradation**: Clear error messages with retry information
- **Per-key tracking**: Individual limits per API key

### 4. Error Handling Philosophy

Errors should be **actionable**, **specific**, and **recoverable**:

**Actionable Errors:**
```python
# Bad: Generic error
"Search failed"

# Good: Actionable error with recovery path
"Vendor 'Acme' not found. Try using semantic_search first to find exact vendor names and IDs in the knowledge graph."
```

**Error Categories:**
- **User errors**: Clear guidance on fixing input
- **System errors**: Retry strategy and fallback options
- **Configuration errors**: Step-by-step resolution
- **Rate limits**: Exact wait times and quotas

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Clients                          │
│          (Claude Desktop, CLI tools, Custom apps)           │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol (JSON-RPC)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastMCP Server Layer                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Request Router                      │   │
│  │    (Tool registration, request dispatch, response)   │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                                │
│  ┌──────────────┬──────────▼──────────┬────────────────┐   │
│  │   Auth      │    Rate Limiter      │   Validators   │   │
│  │  Middleware │   (Token Bucket)      │   (Pydantic)   │   │
│  └──────────────┴─────────────────────┴────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tool Layer                             │
│  ┌─────────────────────────────┬────────────────────────┐  │
│  │    semantic_search Tool     │  find_vendor_info Tool │  │
│  │  ┌─────────────────────┐    │  ┌──────────────────┐ │  │
│  │  │ Request Validation  │    │  │ Vendor Lookup    │ │  │
│  │  │ Search Execution    │    │  │ Graph Traversal  │ │  │
│  │  │ Response Formatting │    │  │ Statistics Calc  │ │  │
│  │  │ Progressive Modes   │    │  │ Response Format  │ │  │
│  │  └─────────────────────┘    │  └──────────────────┘ │  │
│  └─────────────────────────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌──────────────────┬──────────────────┬────────────────┐  │
│  │  HybridSearch    │  QueryRepository  │  QueryCache    │  │
│  │  ┌────────────┐  │  ┌────────────┐  │  ┌──────────┐  │  │
│  │  │Vector      │  │  │Graph       │  │  │LRU Cache │  │  │
│  │  │Similarity  │  │  │Traversal   │  │  │TTL Mgmt  │  │  │
│  │  │BM25 Search │  │  │Aggregation │  │  │Hit Stats │  │  │
│  │  │RRF Merger  │  │  │Filtering   │  │  └──────────┘  │  │
│  │  └────────────┘  │  └────────────┘  │                │  │
│  └──────────────────┴──────────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               PostgreSQL Database                    │   │
│  │  ┌──────────────┬───────────────┬────────────────┐  │   │
│  │  │ Knowledge    │  Knowledge     │   Embeddings   │  │   │
│  │  │  Entities    │  Relationships │    Vectors     │  │   │
│  │  └──────────────┴───────────────┴────────────────┘  │   │
│  │          Connection Pool (2-10 connections)          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### FastMCP Server (`src/mcp/server.py`)

**Purpose**: Entry point and orchestrator for MCP protocol handling.

**Responsibilities:**
- Tool registration with MCP protocol
- Request routing to appropriate tools
- Server initialization and dependency injection
- Global state management (database pool, search instance)

**Key Features:**
- Automatic tool discovery via decorators
- Graceful startup with health checks
- Singleton pattern for shared resources

#### Authentication Layer (`src/mcp/auth.py`)

**Purpose**: Secure API key validation and rate limiting.

**Components:**
- **API Key Validator**: Constant-time comparison using `hmac.compare_digest()`
- **Rate Limiter**: Multi-tier token bucket implementation
- **Auth Decorator**: Transparent authentication for tools

**Security Features:**
- No timing attacks possible
- Keys never logged or exposed
- Clear error messages for debugging
- Configurable rate limits

#### Tool Implementations

##### semantic_search Tool (`src/mcp/tools/semantic_search.py`)

**Purpose**: Hybrid semantic search with progressive disclosure.

**Processing Pipeline:**
1. Request validation (Pydantic)
2. Search execution (HybridSearch service)
3. Response formatting (4 modes)
4. Performance logging

**Response Formatters:**
- `format_ids_only()`: Minimal response
- `format_metadata()`: Balanced response
- `format_preview()`: With snippets
- `format_full()`: Complete content

##### find_vendor_info Tool (`src/mcp/tools/find_vendor_info.py`)

**Purpose**: Knowledge graph vendor information retrieval.

**Processing Pipeline:**
1. Vendor name normalization
2. Database lookup with exact match
3. Graph traversal for relationships
4. Statistics aggregation
5. Progressive response formatting

**Key Functions:**
- `normalize_vendor_name()`: Case-insensitive matching
- `find_vendor_by_name()`: Database query with disambiguation
- `get_vendor_statistics()`: Aggregate entity/relationship counts
- Format functions for each response mode

#### Service Layer

##### HybridSearch Service

**Purpose**: Combines vector similarity and BM25 keyword search.

**Algorithm:**
```python
# Simplified RRF (Reciprocal Rank Fusion)
def merge_results(vector_results, bm25_results, k=60):
    scores = {}
    for rank, result in enumerate(vector_results):
        scores[result.id] = scores.get(result.id, 0) + 1/(k + rank)
    for rank, result in enumerate(bm25_results):
        scores[result.id] = scores.get(result.id, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Performance:**
- Vector search: ~50ms for 1M vectors
- BM25 search: ~30ms with GIN indexes
- RRF merge: ~5ms for 100 results
- Cache hit: <1ms

##### QueryRepository

**Purpose**: Knowledge graph traversal and aggregation.

**Key Methods:**
- `traverse_1hop()`: Get entities related to a vendor
- `aggregate_statistics()`: Count entities and relationships
- `filter_by_type()`: Type-specific entity queries

##### QueryCache

**Purpose**: LRU cache with TTL for expensive queries.

**Features:**
- Configurable size (default 1000 entries)
- TTL-based expiration (default 1 hour)
- Thread-safe operations
- Hit rate tracking

## Data Flow Diagrams

### Request Flow

```
Client Request
    │
    ▼
[MCP Protocol Layer]
    │ Parse JSON-RPC
    │
    ▼
[Authentication]
    │ Validate API Key
    │
    ▼
[Rate Limiting]
    │ Check Token Bucket
    │
    ▼
[Tool Router]
    │ Match tool name
    │
    ▼
[Tool Implementation]
    │ Validate parameters
    │ Execute operation
    │
    ▼
[Response Formatter]
    │ Apply response mode
    │
    ▼
[MCP Response]
    │ JSON-RPC result
    │
    ▼
Client Response
```

### Search Flow

```
semantic_search("query")
    │
    ▼
[Cache Check]
    │ Hash query + params
    ├─► Cache Hit ──► Return cached result
    │
    ▼ Cache Miss
[Vector Search]
    │ Encode query to embedding
    │ Cosine similarity search
    │
    ▼
[BM25 Search]
    │ Tokenize query
    │ Full-text search
    │
    ▼
[RRF Merge]
    │ Combine rankings
    │ Calculate hybrid scores
    │
    ▼
[Response Mode]
    ├─► ids_only: Return IDs + scores
    ├─► metadata: Add file info
    ├─► preview: Add snippets
    └─► full: Complete content
```

### Error Flow

```
Error Occurs
    │
    ▼
[Error Classification]
    ├─► User Error
    │     │ Validation failed
    │     │ Invalid parameters
    │     ▼
    │   [Actionable Message]
    │     "Query too long (max 500)"
    │
    ├─► Auth Error
    │     │ Invalid API key
    │     │ Rate limit exceeded
    │     ▼
    │   [Security Message]
    │     "Rate limit: 0/min, retry in 45s"
    │
    └─► System Error
          │ Database connection
          │ Service failure
          ▼
        [Retry Guidance]
          "Database unavailable, retry in 5s"
```

## Token Efficiency Strategy

### The Token Problem

Large Language Models have limited context windows. Every token counts:
- Claude 3: 200K token context
- Typical session: 50-100K tokens used
- Knowledge retrieval: Can consume 50K+ tokens if not optimized

### 4-Tier Response System

Our progressive disclosure system provides exponential token savings:

#### Tier Comparison

| Tier | Tokens/Result | Information | Use Case |
|------|---------------|-------------|----------|
| 0 (ids) | ~10 | IDs + scores | Filtering |
| 1 (metadata) | ~250 | + File info | Default search |
| 2 (preview) | ~500 | + 200-char snippet | Sampling |
| 3 (full) | ~1500+ | Complete content | Deep analysis |

#### Real-World Savings

**Scenario 1: Exploratory Search**
```
Traditional: 50 results × 1500 tokens = 75,000 tokens
Progressive: 50 ids (500) + 10 metadata (2,500) + 3 full (4,500) = 7,500 tokens
Savings: 90%
```

**Scenario 2: Vendor Analysis**
```
Traditional: Full vendor graph = 50,000 tokens
Progressive: Metadata overview = 3,000 tokens
Savings: 94%
```

### Implementation Details

#### Response Mode Selection

```python
def select_response_mode(request_context):
    """Smart response mode selection based on context."""

    if request_context.is_filtering:
        return "ids_only"  # Just need to filter

    if request_context.is_exploration:
        return "metadata"  # Need overview

    if request_context.needs_content_sample:
        return "preview"  # Need to see some content

    if request_context.is_final_analysis:
        return "full"  # Need complete information

    return "metadata"  # Safe default
```

#### Token Budget Management

```python
class TokenBudget:
    """Manage token allocation across operations."""

    def __init__(self, total_budget=100000):
        self.total = total_budget
        self.used = 0
        self.reserved = 0

    def can_afford(self, operation, mode="metadata"):
        """Check if operation fits in budget."""
        cost = self.estimate_cost(operation, mode)
        return (self.used + self.reserved + cost) <= self.total

    def estimate_cost(self, operation, mode):
        """Estimate token cost for operation."""
        costs = {
            "ids_only": 10,
            "metadata": 250,
            "preview": 500,
            "full": 1500
        }
        base_cost = costs.get(mode, 250)
        return base_cost * operation.expected_results
```

## Security Architecture

### API Key Management

```
┌──────────────────┐
│   Client App     │
│  ┌────────────┐  │
│  │ API Key    │  │──────► HTTPS/TLS ──────►
│  └────────────┘  │
└──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   MCP Server      │
                    │  ┌─────────────┐  │
                    │  │ Auth Module │  │
                    │  │             │  │
                    │  │ hmac.compare│  │
                    │  │ _digest()   │  │
                    │  └─────────────┘  │
                    └───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Environment Vars  │
                    │  BMCIS_API_KEY    │
                    │  (never logged)   │
                    └───────────────────┘
```

### Constant-Time Comparison

```python
def validate_api_key(provided_key: str) -> bool:
    """Timing-attack-safe API key validation."""
    valid_key = os.environ.get('BMCIS_API_KEY')

    if not valid_key:
        raise ValueError("API key not configured")

    # This comparison always takes the same time
    # regardless of how many characters match
    return hmac.compare_digest(provided_key, valid_key)
```

**Why Constant-Time?**

Regular string comparison (`==`) returns as soon as characters differ. Attackers can measure response times to guess correct characters. Constant-time comparison prevents this attack.

### Rate Limiting Strategy

#### Token Bucket Algorithm

```
Time →
Bucket fills with tokens at constant rate
    │
    ▼
┌─────────────────┐
│ Minute Bucket   │ 100 tokens/minute
│ ████████░░      │ 80/100 available
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Hour Bucket     │ 1000 tokens/hour
│ ██████░░░░      │ 600/1000 available
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Day Bucket      │ 10000 tokens/day
│ ███░░░░░░░      │ 3000/10000 available
└─────────────────┘
    │
    ▼
Request allowed if ALL buckets have tokens
```

**Benefits:**
- Allows bursts up to bucket size
- Smooth refill prevents edge spikes
- Multi-tier prevents abuse at any scale
- Clear feedback on limits and reset times

### Error Message Safety

```python
# Unsafe: Exposes internal details
except DatabaseError as e:
    return {"error": f"Database error: {e}"}  # BAD: Exposes internals

# Safe: Generic message with actionable guidance
except DatabaseError as e:
    logger.error(f"Database error: {e}")  # Log full error internally
    return {
        "error": "Service temporarily unavailable",
        "action": "Please retry in 5 seconds",
        "code": "SERVICE_UNAVAILABLE"
    }
```

## Performance Optimizations

### Database Connection Pooling

```python
class DatabasePool:
    """PostgreSQL connection pool with health checks."""

    def __init__(self, min_conn=2, max_conn=10):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            **connection_params
        )

    def get_connection(self):
        """Get connection with automatic retry."""
        try:
            conn = self.pool.getconn()
            # Test connection is alive
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception:
            # Return connection and get fresh one
            self.pool.putconn(conn, close=True)
            return self.pool.getconn()
```

### Query Caching Strategy

```python
class QueryCache:
    """LRU cache with TTL for search queries."""

    def __init__(self, max_size=1000, ttl=3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Get with LRU update and TTL check."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return value
            else:
                # Expired
                del self.cache[key]

        self.misses += 1
        return None

    def put(self, key, value):
        """Put with size limit enforcement."""
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)

        # Evict oldest if over size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
```

### Index Optimization

```sql
-- Vector similarity index (using pgvector)
CREATE INDEX idx_embeddings_vector ON embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search index
CREATE INDEX idx_chunks_text ON knowledge_chunks
USING gin(to_tsvector('english', chunk_text));

-- Vendor lookup index
CREATE INDEX idx_entities_text_lower ON knowledge_entities
(LOWER(text)) WHERE entity_type = 'ORG';

-- Relationship traversal index
CREATE INDEX idx_relationships_source ON knowledge_relationships
(source_id, relationship_type, confidence);
```

## Future Enhancements

### Planned Improvements

1. **Streaming Responses**: Support for streaming large result sets
2. **Batch Operations**: Multi-query execution in single request
3. **Custom Response Modes**: User-defined progressive disclosure levels
4. **Semantic Caching**: Cache based on query similarity, not exact match
5. **Adaptive Rate Limiting**: Dynamic limits based on usage patterns

### Scalability Considerations

1. **Horizontal Scaling**: Multiple MCP server instances behind load balancer
2. **Read Replicas**: PostgreSQL read replicas for search queries
3. **Distributed Cache**: Redis for shared cache across instances
4. **Query Optimization**: Materialized views for common aggregations
5. **CDN Integration**: Edge caching for static vendor information

---

*This architecture enables efficient, secure, and scalable knowledge graph access through the MCP protocol while maintaining exceptional token efficiency for LLM interactions.*