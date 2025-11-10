# Pagination and Filtering Guide

## Pagination Overview

### Why Pagination Matters

Pagination is essential for efficiently browsing large result sets without overwhelming token budgets or memory constraints. Without pagination, users face two problematic choices:

1. **Fetch everything upfront**: Risk exceeding token limits and wasting resources on unneeded data
2. **Arbitrary limits**: Miss potentially relevant results beyond the cutoff

Pagination solves this by enabling incremental result retrieval: fetch small batches, evaluate relevance, and fetch more only when needed.

**Token Efficiency Example**:
- **Without pagination**: Fetch 50 results (full mode) = ~75,000 tokens
- **With pagination**: Fetch 10 results per page (metadata mode) = ~2,500 tokens per page
  - View 2 pages total = ~5,000 tokens (**93% reduction**)

### Cursor-Based vs Offset Pagination

The MCP server uses **cursor-based pagination** instead of offset-based for reliability and performance:

| Feature | Cursor-Based (MCP) | Offset-Based (Traditional) |
|---------|-------------------|---------------------------|
| **Stability** | ✅ Results remain consistent even if data changes | ❌ Results shift if data is added/removed |
| **Performance** | ✅ O(1) lookup (direct index) | ❌ O(n) lookup (skip first N results) |
| **Caching** | ✅ Each page cached independently | ⚠️ Offset changes invalidate cache |
| **Use Case** | General-purpose pagination | Simple datasets with no concurrent updates |

**Example Problem with Offset Pagination**:
```python
# User fetches page 1 (offset=0, limit=10)
page1 = search(offset=0, limit=10)  # Results: IDs 1-10

# Meanwhile, new document is inserted at position 5

# User fetches page 2 (offset=10, limit=10)
page2 = search(offset=10, limit=10)  # Results: IDs 11-20
# User MISSED ID 10 (shifted to page 2) and SEES ID 11 TWICE (was ID 10)
```

**Cursor-Based Solution**:
```python
# User fetches page 1
page1 = search(page_size=10)
# cursor encodes: "query_hash + last_seen_id"

# New document inserted (cursor remains valid)

# User fetches page 2 with cursor
page2 = search(page_size=10, cursor=page1.cursor)
# Cursor points to last_seen_id=10, returns results starting from ID 11
# No duplicates, no gaps
```

### Use Cases

**1. Browsing Search Results**
```python
# User searches for "authentication"
# View first 10 results to gauge relevance
page1 = await semantic_search(
    query="authentication",
    page_size=10,
    response_mode="metadata"
)

# If promising, continue to next page
if page1.has_more:
    page2 = await semantic_search(
        query="authentication",
        page_size=10,
        cursor=page1.cursor,
        response_mode="metadata"
    )
```

**2. Incremental Loading (Progressive Disclosure)**
```python
# Fetch metadata for quick preview
preview = await semantic_search(
    query="JWT best practices",
    page_size=5,
    response_mode="metadata"
)

# User selects 2 interesting results by chunk_id
# Fetch full details for ONLY those 2 (filter by chunk_id)
for chunk_id in [142, 167]:
    full_result = fetch_chunk_by_id(chunk_id, response_mode="full")
```

**3. Deduplication and Aggregation**
```python
# Fetch all result IDs (for deduplication)
all_ids = []
cursor = None
while True:
    page = await semantic_search(
        query="vendor partnerships",
        page_size=50,
        response_mode="ids_only",
        cursor=cursor
    )
    all_ids.extend([r.chunk_id for r in page.results])
    if not page.has_more:
        break
    cursor = page.cursor

# Deduplicate and analyze
unique_ids = set(all_ids)  # Remove duplicates across pages
```

---

## Pagination Implementation

### Cursor-Based Pagination Mechanism

The cursor encodes the state needed to resume pagination:

```python
# Cursor format (base64-encoded JSON):
{
    "query_hash": "sha256(query + response_mode + top_k)",
    "offset": 10,  # Start position for next page
    "response_mode": "metadata"
}
```

**Security**: The `query_hash` prevents cursor tampering. If a user modifies the cursor, the hash validation fails and the request is rejected.

**Expiration**: Cursors are tied to cache TTL. If the cached query expires (30s for semantic_search, 300s for vendor_info), the cursor becomes invalid.

### Using the `cursor` Parameter

```python
# First request (no cursor)
page1 = await semantic_search(
    query="database indexing",
    page_size=10
)

# Response includes:
# - results: [SearchResultMetadata, ...]
# - cursor: "eyJxdWVyeV9oYXNoIjogImFiYzEyMyIsICJvZmZzZXQiOiAxMH0="
# - has_more: True
# - total_available: 47

# Second request (with cursor from page1)
page2 = await semantic_search(
    query="database indexing",  # Must match original query
    page_size=10,
    cursor=page1.cursor  # Resume from last position
)

# Response:
# - results: [Next 10 results]
# - cursor: "eyJxdWVyeV9oYXNoIjogImFiYzEyMyIsICJvZmZzZXQiOiAyMH0="
# - has_more: True
# - total_available: 47
```

### Handling `has_more` Flag

The `has_more` boolean indicates whether additional results are available:

```python
def fetch_all_pages(query: str) -> list:
    """Fetch all pages of results until exhausted."""
    all_results = []
    cursor = None

    while True:
        page = await semantic_search(
            query=query,
            page_size=10,
            cursor=cursor
        )

        all_results.extend(page.results)

        if not page.has_more:
            break  # No more results

        cursor = page.cursor  # Continue to next page

    return all_results
```

### Detecting Last Page

Two equivalent methods to detect the last page:

**Method 1: Check `has_more` flag**
```python
if not page.has_more:
    print("This is the last page")
```

**Method 2: Check `cursor` field**
```python
if page.cursor is None:
    print("This is the last page")
```

Both methods are reliable. Use `has_more` for clarity and explicitness.

### Code Examples for Common Patterns

**Pattern 1: Fetch-Until-Condition**
```python
# Fetch pages until we find a result with score > 0.9
def find_high_confidence_result(query: str):
    cursor = None
    while True:
        page = await semantic_search(query, page_size=10, cursor=cursor)

        for result in page.results:
            if result.hybrid_score > 0.9:
                return result  # Found it!

        if not page.has_more:
            return None  # Exhausted all results, none found

        cursor = page.cursor
```

**Pattern 2: Parallel Page Fetching** (Advanced)
```python
# WARNING: Only works if cursors are stable across multiple requests
# Not recommended for production due to cache expiration risks

async def fetch_pages_parallel(query: str, num_pages: int):
    # Fetch first page to get cursors
    page1 = await semantic_search(query, page_size=10)

    # Fetch subsequent pages in parallel
    tasks = [
        semantic_search(query, page_size=10, cursor=page1.cursor),
        semantic_search(query, page_size=10, cursor=page2.cursor),
        # ... (requires pre-fetching all cursors sequentially first)
    ]
    return await asyncio.gather(*tasks)

# LIMITATION: Cursors must be fetched sequentially (each cursor depends on previous page)
# Parallel fetching provides no real benefit; use sequential pagination instead
```

**Pattern 3: Page Size Adaptation**
```python
# Start with small page size, increase if results are relevant
def adaptive_pagination(query: str):
    # First page: small batch for quick preview
    page1 = await semantic_search(query, page_size=5, response_mode="metadata")

    if page1.results[0].hybrid_score > 0.8:
        # High relevance: fetch larger batches
        page2 = await semantic_search(
            query, page_size=20, cursor=page1.cursor, response_mode="metadata"
        )
    else:
        # Low relevance: stick with small batches
        page2 = await semantic_search(
            query, page_size=5, cursor=page1.cursor, response_mode="metadata"
        )
```

---

## Response Filtering

### Reducing Token Consumption with Filters

Field-level filtering allows you to select only the fields you need, reducing token consumption beyond progressive disclosure:

```python
# Without filtering (metadata mode): ~200 tokens per result
result = SearchResultMetadata(
    chunk_id=142,
    source_file="docs/auth.md",
    source_category="security",
    hybrid_score=0.85,
    rank=1,
    chunk_index=5,
    total_chunks=12
)  # Total: ~200 tokens

# With filtering (select only chunk_id + score): ~20 tokens per result
result_filtered = {
    "chunk_id": 142,
    "hybrid_score": 0.85
}  # Total: ~20 tokens (90% reduction)
```

### Whitelist of Available Fields

The `fields` parameter accepts a list of field names to include in the response. Only whitelisted fields are allowed:

**SearchResult Fields** (all modes):
```python
[
    "chunk_id",          # Always available
    "hybrid_score",      # Always available
    "rank",              # Always available
    "source_file",       # Available in metadata, preview, full modes
    "source_category",   # Available in metadata, preview, full modes
    "chunk_index",       # Available in metadata, preview, full modes
    "total_chunks",      # Available in metadata, preview, full modes
    "chunk_snippet",     # Available in preview, full modes
    "context_header",    # Available in preview, full modes
    "chunk_text",        # Available in full mode only
    "similarity_score",  # Available in full mode only
    "bm25_score",        # Available in full mode only
    "score_type",        # Available in full mode only
    "chunk_token_count"  # Available in full mode only
]
```

**Validation**: If you request a field that doesn't exist in the selected `response_mode`, the request is rejected:

```python
# ERROR: chunk_text not available in metadata mode
await semantic_search(
    query="test",
    response_mode="metadata",
    fields=["chunk_id", "chunk_text"]  # Invalid!
)
# Raises: ValueError("Field 'chunk_text' not available in metadata mode")
```

### Field Combinations and Their Token Costs

| Field Combination | Token Cost (per result) | Use Case |
|-------------------|------------------------|----------|
| `["chunk_id"]` | ~5 tokens | Deduplication, ID collection |
| `["chunk_id", "hybrid_score"]` | ~10 tokens | Relevance ranking |
| `["chunk_id", "source_file", "hybrid_score"]` | ~30 tokens | Source attribution |
| `["chunk_id", "chunk_snippet"]` | ~60 tokens | Quick content preview |
| All fields (no filter) | ~200 tokens (metadata)<br>~1,500 tokens (full) | Comprehensive analysis |

**Token Savings Example** (10 results):
- **Unfiltered metadata**: 10 × 200 = 2,000 tokens
- **Filtered (chunk_id + score)**: 10 × 10 = 100 tokens
- **Savings**: 95% reduction

### Use Cases

**Use Case 1: Deduplication**
```python
# Fetch all chunk IDs across multiple queries to deduplicate
query1_ids = await semantic_search(
    query="authentication",
    response_mode="ids_only",
    fields=["chunk_id"]
)

query2_ids = await semantic_search(
    query="security best practices",
    response_mode="ids_only",
    fields=["chunk_id"]
)

# Find intersection (chunks relevant to both queries)
common_chunks = set(query1_ids) & set(query2_ids)
```

**Use Case 2: Quick Summary**
```python
# Fetch minimal summary: source file + score
summary = await semantic_search(
    query="API design patterns",
    page_size=20,
    response_mode="metadata",
    fields=["source_file", "hybrid_score", "rank"]
)

# Display: "Top 20 results: docs/api.md (0.92), docs/rest.md (0.88), ..."
```

**Use Case 3: Selective Detail Fetching**
```python
# Step 1: Fetch IDs + scores for all results (lightweight)
all_results = await semantic_search(
    query="database optimization",
    page_size=50,
    response_mode="ids_only",
    fields=["chunk_id", "hybrid_score"]
)

# Step 2: Filter to top 5 by score
top_5_ids = sorted(all_results, key=lambda r: r.hybrid_score, reverse=True)[:5]

# Step 3: Fetch full details for ONLY top 5 (cache hit for chunk_id)
top_5_full = [
    fetch_chunk_by_id(chunk_id, response_mode="full")
    for chunk_id in top_5_ids
]

# Total tokens: 50 × 10 (ids_only) + 5 × 1,500 (full) = 8,000 tokens
# vs. 50 × 1,500 (all full) = 75,000 tokens (89% reduction)
```

### Code Examples

**Example 1: Minimal Fields**
```python
result = await semantic_search(
    query="JWT authentication",
    page_size=10,
    response_mode="metadata",
    fields=["chunk_id", "hybrid_score"]
)

# Response:
# {
#   "results": [
#     {"chunk_id": 142, "hybrid_score": 0.92},
#     {"chunk_id": 167, "hybrid_score": 0.88},
#     ...
#   ],
#   "total_found": 47,
#   "has_more": True,
#   "cursor": "..."
# }
```

**Example 2: Source Attribution**
```python
result = await semantic_search(
    query="microservices architecture",
    page_size=15,
    response_mode="metadata",
    fields=["chunk_id", "source_file", "source_category", "hybrid_score"]
)

# Response includes source attribution for citation/provenance tracking
```

---

## Combining Pagination + Filtering

### Best Practices for Combining Features

1. **Filter before pagination**: Apply field filtering at the query level (not after fetching)
2. **Use lightweight modes**: Combine `ids_only` or `metadata` with filtering for maximum efficiency
3. **Cache-friendly**: Filtered queries are cached separately from unfiltered queries

### Filter Before Pagination (Efficiency)

**Efficient Approach** (filter + paginate):
```python
# Fetch page 1 with filtering
page1 = await semantic_search(
    query="authentication",
    page_size=10,
    response_mode="metadata",
    fields=["chunk_id", "hybrid_score"]  # Filter applied at query time
)
# Token cost: 10 × 10 = 100 tokens

# Fetch page 2 with same filter
page2 = await semantic_search(
    query="authentication",
    page_size=10,
    cursor=page1.cursor,
    response_mode="metadata",
    fields=["chunk_id", "hybrid_score"]
)
# Token cost: 10 × 10 = 100 tokens
# Total: 200 tokens for 20 results
```

**Inefficient Approach** (fetch all, then filter client-side):
```python
# Fetch unfiltered results
page1 = await semantic_search(
    query="authentication",
    page_size=10,
    response_mode="metadata"
)
# Token cost: 10 × 200 = 2,000 tokens

# Manually filter client-side (wasteful)
filtered = [
    {"chunk_id": r.chunk_id, "hybrid_score": r.hybrid_score}
    for r in page1.results
]
# Total: 2,000 tokens (wasted 1,800 tokens on unused fields)
```

**Efficiency Gain**: 90% token reduction by filtering at query time.

### Common Patterns and Code Examples

**Pattern 1: Lightweight Pagination + Selective Full Fetch**
```python
# Browse metadata pages until you find interesting results
cursor = None
interesting_ids = []

for _ in range(5):  # Browse up to 5 pages
    page = await semantic_search(
        query="security vulnerabilities",
        page_size=10,
        cursor=cursor,
        response_mode="metadata",
        fields=["chunk_id", "hybrid_score", "source_file"]
    )

    # Identify interesting chunks (e.g., score > 0.85)
    interesting_ids.extend([
        r.chunk_id for r in page.results if r.hybrid_score > 0.85
    ])

    if not page.has_more:
        break
    cursor = page.cursor

# Fetch full details for only interesting chunks
full_results = [
    fetch_chunk_by_id(chunk_id, response_mode="full")
    for chunk_id in interesting_ids
]

# Token efficiency: 5 pages × 10 results × 30 tokens (filtered metadata)
#                 + 15 interesting × 1,500 tokens (full)
#                 = 1,500 + 22,500 = 24,000 tokens
# vs. 50 results × 1,500 tokens (all full) = 75,000 tokens (68% reduction)
```

**Pattern 2: Incremental Aggregation**
```python
# Aggregate statistics across all pages
total_score = 0.0
count = 0
cursor = None

while True:
    page = await semantic_search(
        query="performance optimization",
        page_size=50,  # Large pages for aggregation
        cursor=cursor,
        response_mode="ids_only",
        fields=["hybrid_score"]  # Only need scores
    )

    total_score += sum(r.hybrid_score for r in page.results)
    count += len(page.results)

    if not page.has_more:
        break
    cursor = page.cursor

average_score = total_score / count
print(f"Average relevance across {count} results: {average_score:.2f}")
```

### Performance Considerations

**Cache Behavior with Filtering**:
- Filtered and unfiltered queries are cached separately
- Cache key includes: `hash(query, response_mode, fields, page_size)`

**Example**:
```python
# Query 1: Cached with fields=["chunk_id", "score"]
q1 = await semantic_search(query="test", fields=["chunk_id", "hybrid_score"])

# Query 2: Different cache entry (no fields filter)
q2 = await semantic_search(query="test")  # Cache miss!

# Query 3: Cache hit (matches Query 1)
q3 = await semantic_search(query="test", fields=["chunk_id", "hybrid_score"])  # Cache hit!
```

**Implication**: Use consistent field filters across requests to maximize cache hit rate.

---

## API Examples

### Metadata Page Browsing

```python
# Browse search results page-by-page in metadata mode
query = "distributed systems architecture"
cursor = None

for page_num in range(1, 6):  # Browse up to 5 pages
    page = await semantic_search(
        query=query,
        page_size=10,
        cursor=cursor,
        response_mode="metadata"
    )

    print(f"Page {page_num}: {len(page.results)} results")
    for r in page.results:
        print(f"  [{r.rank}] {r.source_file} (score: {r.hybrid_score:.2f})")

    if not page.has_more:
        print("No more results")
        break

    cursor = page.cursor
```

### Filtered IDs for Deduplication

```python
# Fetch all chunk IDs for deduplication across multiple queries
queries = ["authentication", "authorization", "access control"]
all_chunk_ids = set()

for query in queries:
    cursor = None
    while True:
        page = await semantic_search(
            query=query,
            page_size=50,
            cursor=cursor,
            response_mode="ids_only",
            fields=["chunk_id"]
        )

        all_chunk_ids.update(r.chunk_id for r in page.results)

        if not page.has_more:
            break
        cursor = page.cursor

print(f"Total unique chunks across all queries: {len(all_chunk_ids)}")
```

### Preview with Pagination

```python
# Fetch preview snippets page-by-page
query = "API rate limiting strategies"
cursor = None

for page_num in range(1, 4):  # First 3 pages
    page = await semantic_search(
        query=query,
        page_size=5,
        cursor=cursor,
        response_mode="preview",
        fields=["chunk_id", "chunk_snippet", "hybrid_score"]
    )

    print(f"\n=== Page {page_num} ===")
    for r in page.results:
        print(f"[Score: {r.hybrid_score:.2f}] {r.chunk_snippet}")

    if not page.has_more:
        break
    cursor = page.cursor
```

### Full Content with Filtering

```python
# Fetch full content but filter to essential fields only
page = await semantic_search(
    query="database indexing best practices",
    page_size=5,
    response_mode="full",
    fields=[
        "chunk_id",
        "chunk_text",
        "hybrid_score",
        "source_file",
        "context_header"
    ]
)

for r in page.results:
    print(f"\n=== {r.source_file} (score: {r.hybrid_score:.2f}) ===")
    print(f"Context: {r.context_header}")
    print(f"Content:\n{r.chunk_text}")
```

### Multi-Page Workflows

```python
# Complete workflow: Browse → Select → Fetch Full Details
async def intelligent_search(query: str, max_pages: int = 10):
    # Phase 1: Browse metadata to find high-relevance results
    high_value_ids = []
    cursor = None

    for _ in range(max_pages):
        page = await semantic_search(
            query=query,
            page_size=10,
            cursor=cursor,
            response_mode="metadata",
            fields=["chunk_id", "hybrid_score", "source_file"]
        )

        # Select results with score > 0.8
        high_value_ids.extend([
            r.chunk_id for r in page.results if r.hybrid_score > 0.8
        ])

        if not page.has_more or len(high_value_ids) >= 20:
            break  # Found enough high-value results
        cursor = page.cursor

    # Phase 2: Fetch full details for selected high-value results
    full_results = []
    for chunk_id in high_value_ids[:10]:  # Top 10 only
        full_result = await fetch_chunk_by_id(chunk_id, response_mode="full")
        full_results.append(full_result)

    return full_results

# Usage
results = await intelligent_search("machine learning deployment patterns")
# Token cost: ~10 pages × 10 results × 30 tokens (metadata)
#           + 10 results × 1,500 tokens (full)
#           = 3,000 + 15,000 = 18,000 tokens
# vs. 100 results × 1,500 tokens (all full) = 150,000 tokens (88% reduction)
```

---

## Summary

Pagination and filtering provide 85-95% token reduction when combined with progressive disclosure. Use cursor-based pagination for stable, efficient result browsing, and apply field filtering to request only the data you need. Always filter at query time (not client-side) to maximize cache efficiency.

**Quick Best Practices**:
- ✅ Use cursor-based pagination (never offset-based)
- ✅ Set `page_size` based on use case (5-10 for browsing, 50+ for aggregation)
- ✅ Check `has_more` flag to detect last page
- ✅ Apply field filtering at query time (not after fetching)
- ✅ Combine metadata mode + filtering for 95%+ token reduction
- ✅ Use consistent filters across requests to maximize cache hit rate

For caching configuration and performance tuning, see the [Caching Configuration Guide](./caching-configuration.md).
