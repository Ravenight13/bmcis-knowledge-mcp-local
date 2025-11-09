# Knowledge Graph Traversal Queries

**Version**: 1.0.0
**Performance Target**: 1-hop <10ms, 2-hop <50ms (P95)
**Architecture**: Raw SQL CTEs with parameterized patterns

---

## Overview

This document describes the 5 core SQL graph traversal patterns implemented in `query_repository.py`.

All queries:
- Use parameterized SQL to prevent injection
- Return structured dataclasses for type safety
- Leverage PostgreSQL indexes for optimal performance
- Support connection pooling for scalability

---

## Query Patterns

### 1. One-Hop Outbound Traversal

**Purpose**: Get all entities directly related to a source entity.

**Performance**: P50 <5ms, P95 <10ms

**Usage**:
```python
related = repo.traverse_1hop(
    entity_id=123,
    min_confidence=0.7,
    relationship_types=['hierarchical', 'mentions-in-document'],
    max_results=50
)
```

**Returns**: `List[RelatedEntity]`

**SQL Pattern**:
```sql
WITH related_entities AS (
    SELECT target_entity_id, relationship_type, confidence
    FROM entity_relationships
    WHERE source_entity_id = $entity_id
      AND confidence >= $min_confidence
      AND ($relationship_types IS NULL OR relationship_type = ANY($relationship_types))
)
SELECT e.id, e.entity_name AS text, e.entity_type, e.confidence, ...
FROM related_entities re
JOIN knowledge_entities e ON e.id = re.target_entity_id
ORDER BY re.confidence DESC
LIMIT $max_results
```

**Required Index**:
```sql
CREATE INDEX idx_relationships_source ON entity_relationships(source_entity_id);
```

**Example Use Case**: Reranking boost - find all entities related to query entities for context boosting.

---

### 2. Two-Hop Traversal (Extended Network)

**Purpose**: Get entities reachable in 2 relationship steps from source entity.

**Performance**: P50 <20ms, P95 <50ms (depends on fanout)

**Usage**:
```python
two_hop = repo.traverse_2hop(
    entity_id=123,
    min_confidence=0.7,
    relationship_types=None,  # All types
    max_results=100
)
```

**Returns**: `List[TwoHopEntity]` (includes `intermediate_entity_id`, `path_confidence`)

**SQL Pattern**:
```sql
WITH hop1 AS (
    SELECT DISTINCT target_entity_id, confidence
    FROM entity_relationships
    WHERE source_entity_id = $entity_id AND confidence >= $min_confidence
),
hop2 AS (
    SELECT r2.target_entity_id, SQRT(h1.confidence * r2.confidence) AS path_confidence, ...
    FROM hop1 h1
    JOIN entity_relationships r2 ON r2.source_entity_id = h1.target_entity_id
    WHERE r2.confidence >= $min_confidence
      AND r2.target_entity_id != $entity_id  -- Prevent cycles
)
SELECT * FROM hop2
ORDER BY path_confidence DESC
LIMIT $max_results
```

**Path Confidence**: Geometric mean of hop1 × hop2 confidences (penalizes long, weak paths).

**Example Use Case**: Semantic expansion - find "colleagues of colleagues" for broader context.

---

### 3. Bidirectional Traversal

**Purpose**: Get all entities connected to source (both incoming and outgoing relationships).

**Performance**: P50 <15ms (1-hop), P95 <30ms

**Usage**:
```python
bidirectional = repo.traverse_bidirectional(
    entity_id=123,
    min_confidence=0.7,
    max_depth=1,  # 1 or 2
    max_results=50
)
```

**Returns**: `List[BidirectionalEntity]` (includes `outbound_rel_types`, `inbound_rel_types`, `relationship_count`)

**SQL Pattern**:
```sql
WITH outbound AS (
    SELECT target_entity_id AS related_entity_id, relationship_type, confidence
    FROM entity_relationships
    WHERE source_entity_id = $entity_id AND confidence >= $min_confidence
),
inbound AS (
    SELECT source_entity_id AS related_entity_id, relationship_type, confidence
    FROM entity_relationships
    WHERE target_entity_id = $entity_id AND confidence >= $min_confidence
),
combined AS (
    SELECT
        COALESCE(o.related_entity_id, i.related_entity_id) AS entity_id,
        ARRAY_AGG(DISTINCT o.relationship_type) FILTER (...) AS outbound_rel_types,
        ARRAY_AGG(DISTINCT i.relationship_type) FILTER (...) AS inbound_rel_types,
        GREATEST(MAX(o.confidence), MAX(i.confidence)) AS max_confidence,
        COUNT(*) AS relationship_count
    FROM outbound o FULL OUTER JOIN inbound i ON o.related_entity_id = i.related_entity_id
    GROUP BY entity_id
)
SELECT * FROM combined JOIN knowledge_entities ...
ORDER BY relationship_count DESC, max_confidence DESC
```

**Example Use Case**: Relationship visualization - show full entity network for graph UI.

---

### 4. Entity Type Filtering

**Purpose**: Get related entities of specific type(s) (e.g., only PRODUCT entities related to a VENDOR).

**Performance**: P50 <8ms, P95 <15ms

**Usage**:
```python
products = repo.traverse_with_type_filter(
    entity_id=123,
    relationship_type='hierarchical',
    target_entity_types=['PRODUCT', 'TECHNOLOGY'],
    min_confidence=0.7,
    max_results=50
)
```

**Returns**: `List[RelatedEntity]` (filtered by `entity_type`)

**SQL Pattern**:
```sql
SELECT e.id, e.entity_name AS text, e.entity_type, r.relationship_type, r.confidence, ...
FROM entity_relationships r
JOIN knowledge_entities e ON e.id = r.target_entity_id
WHERE r.source_entity_id = $entity_id
  AND r.relationship_type = $relationship_type
  AND e.entity_type = ANY($target_entity_types)
  AND r.confidence >= $min_confidence
ORDER BY r.confidence DESC
LIMIT $max_results
```

**Required Indexes**:
```sql
CREATE INDEX idx_entity_type ON knowledge_entities(entity_type);
CREATE INDEX idx_relationships_source ON entity_relationships(source_entity_id);
```

**Example Use Case**: Constrain reranking boosting to specific entity types (e.g., only boost PRODUCT mentions).

---

### 5. Entity Mentions Lookup

**Purpose**: Get documents and chunks where an entity is mentioned (for provenance).

**Performance**: P50 <10ms, P95 <20ms

**Usage**:
```python
mentions = repo.get_entity_mentions(
    entity_id=123,
    max_results=100
)
```

**Returns**: `List[EntityMention]` (includes `chunk_id`, `document_id`, `chunk_text`, `mention_confidence`)

**SQL Pattern**:
```sql
SELECT
    em.chunk_id,
    kb.source_file AS document_id,
    kb.chunk_text,
    kb.source_category,
    kb.chunk_index,
    em.confidence AS mention_confidence,
    kb.created_at AS indexed_at
FROM entity_mentions em
JOIN knowledge_base kb ON kb.id = em.chunk_id
WHERE em.entity_id = $entity_id
ORDER BY em.confidence DESC, kb.created_at DESC
LIMIT $max_results
```

**Required Index**:
```sql
CREATE INDEX idx_entity_mentions_entity ON entity_mentions(entity_id);
```

**Example Use Case**: Surface-level context retrieval - find all chunks mentioning an entity for highlighting.

---

## Index Strategy

### Core Indexes (Required for All Queries)

```sql
-- Entity relationships (source → target traversal)
CREATE INDEX idx_relationships_source ON entity_relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON entity_relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON entity_relationships(relationship_type);

-- Entity lookups
CREATE INDEX idx_entity_type ON knowledge_entities(entity_type);
CREATE INDEX idx_entity_name ON knowledge_entities(LOWER(text));

-- Entity mentions (provenance)
CREATE INDEX idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX idx_entity_mentions_chunk ON entity_mentions(chunk_id);
```

### Composite Indexes (Performance Optimization)

```sql
-- For type-filtered queries
CREATE INDEX idx_relationships_graph ON entity_relationships(source_entity_id, relationship_type, target_entity_id);
CREATE INDEX idx_entities_type_id ON knowledge_entities(entity_type, id);

-- For confidence-filtered queries
CREATE INDEX idx_relationships_source_conf ON entity_relationships(source_entity_id, confidence DESC);
CREATE INDEX idx_relationships_target_conf ON entity_relationships(target_entity_id, confidence DESC);

-- For document/chunk lookups
CREATE INDEX idx_entity_mentions_composite ON entity_mentions(entity_id, document_id);
CREATE INDEX idx_entity_mentions_chunk_doc ON entity_mentions(document_id, chunk_id);
```

---

## Performance Characteristics

### Query Latency (Indexed, 10k Entities, 30k Relationships)

| Query Type | P50 Latency | P95 Latency | Typical Fanout |
|-----------|-------------|-------------|----------------|
| 1-hop outbound | 5-8ms | 10-12ms | 10-20 entities |
| 2-hop | 20-30ms | 50-70ms | 50-100 entities (10 × 10) |
| Bidirectional | 10-15ms | 25-35ms | 20-30 entities (15 out + 15 in) |
| Type-filtered | 6-10ms | 12-18ms | 5-15 entities (filtered) |
| Mentions lookup | 8-12ms | 18-25ms | 20-50 mentions |

### Scalability

**Expected performance with different entity counts**:

| Entities | Relationships | 1-hop P95 | 2-hop P95 | Notes |
|----------|--------------|-----------|-----------|-------|
| 10k | 30k | <10ms | <50ms | Current target |
| 50k | 150k | <15ms | <80ms | Requires index tuning |
| 100k | 300k | <25ms | <120ms | May need partitioning |
| 500k | 1.5M | <50ms | <250ms | Requires query optimization (LIMIT, recursive CTEs) |

**Scaling strategies**:
1. **10k → 50k entities**: No changes needed, indexes handle this
2. **50k → 100k entities**: Tune `work_mem`, add composite indexes
3. **100k+ entities**: Consider table partitioning by `entity_type`, add LIMIT clauses to 2-hop queries

---

## Query Plan Analysis

### Example: 1-Hop Outbound Traversal

```sql
EXPLAIN (ANALYZE, BUFFERS) <query>;
```

**Expected plan (indexed)**:
```
Limit  (cost=0.29..25.45 rows=50 width=120) (actual time=0.034..2.456 rows=12 loops=1)
  ->  Sort  (cost=0.29..0.32 rows=12 width=120) (actual time=2.454..2.455 rows=12 loops=1)
        Sort Key: r.confidence DESC
        Sort Method: quicksort  Memory: 27kB
        ->  Hash Join  (cost=0.29..0.45 rows=12 width=120) (actual time=0.045..2.432 rows=12 loops=1)
              Hash Cond: (e.id = r.target_entity_id)
              ->  Seq Scan on knowledge_entities e  (cost=0.00..145.00 rows=10000 width=80)
              ->  Hash  (cost=0.28..0.28 rows=1 width=40) (actual time=0.021..0.021 rows=12 loops=1)
                    Buckets: 1024  Batches: 1  Memory Usage: 9kB
                    ->  Index Scan using idx_relationships_source on entity_relationships r
                          (cost=0.29..0.28 rows=1 width=40) (actual time=0.012..0.018 rows=12 loops=1)
                          Index Cond: (source_entity_id = '...'::uuid)
                          Filter: (confidence >= 0.7)
Buffers: shared hit=15 read=0
Planning Time: 0.254 ms
Execution Time: 2.512 ms
```

**Key observations**:
- **Index Scan** on `idx_relationships_source` (fast)
- **Hash Join** instead of Nested Loop (efficient for 10-20 results)
- **Buffers: shared hit=15** (all data in cache)
- **Execution Time: 2.5ms** (well under P50 <5ms target)

### Example: 2-Hop Traversal

**Expected plan (indexed)**:
```
Limit  (cost=25.45..80.12 rows=100 width=160) (actual time=5.234..28.456 rows=87 loops=1)
  ->  Sort  (cost=25.45..26.67 rows=487 width=160) (actual time=28.434..28.441 rows=87 loops=1)
        Sort Key: hop2.path_confidence DESC
        Sort Method: quicksort  Memory: 45kB
        ->  Hash Join  (cost=12.34..23.45 rows=487 width=160) (actual time=3.234..27.123 rows=87 loops=1)
              [... CTE scans with Index Scans on relationships ...]
Buffers: shared hit=124 read=8
Planning Time: 0.512 ms
Execution Time: 28.523 ms
```

**Optimization notes**:
- 2-hop queries are 5-10x slower than 1-hop (expected)
- CTE materialization can add overhead (consider `MATERIALIZED` hint if needed)
- Limit CTE size with confidence thresholds to reduce fanout

---

## Integration with Cache & Reranking

### Cache Integration

The `KnowledgeGraphService` wraps these queries with an LRU cache:

```python
# graph_service.py stub implementation
def traverse_1hop_cached(self, entity_id: UUID, rel_type: str) -> List[Entity]:
    # 1. Check cache
    cached = self._cache.get_relationships(entity_id, rel_type)
    if cached:
        return cached  # <1ms cache hit

    # 2. Query repository (5-10ms cache miss)
    entities = self.query_repo.traverse_1hop(entity_id, relationship_types=[rel_type])

    # 3. Cache result
    self._cache.set_relationships(entity_id, rel_type, entities)

    return entities
```

**Expected performance** (with >80% cache hit rate):
- P50: <2ms (mostly cache hits)
- P95: <15ms (some cache misses)
- P99: <30ms (cache misses + cold queries)

### Reranking Integration

Graph queries boost search results based on entity relationships:

```python
# Pseudocode for reranking pipeline
def rerank_with_graph(search_results: List[Chunk], query_entities: List[UUID]) -> List[Chunk]:
    # 1. Get 1-hop related entities for each query entity
    related_entities = set()
    for entity_id in query_entities:
        related = repo.traverse_1hop(entity_id, min_confidence=0.7)
        related_entities.update(e.id for e in related)

    # 2. Boost chunks mentioning related entities
    for chunk in search_results:
        chunk_entities = extract_entities_from_chunk(chunk.id)
        overlap = chunk_entities & related_entities

        if overlap:
            # Boost score by 40% for entity relationship overlap
            chunk.score += 0.4 * (len(overlap) / len(query_entities))

    # 3. Re-sort by boosted scores
    return sorted(search_results, key=lambda c: c.score, reverse=True)
```

**Performance target**: <50ms for full reranking pipeline (search + graph + rerank).

---

## Testing Strategy

### Unit Tests

**File**: `tests/knowledge_graph/test_query_repository.py`

Test cases:
1. **1-hop traversal**: Verify correctness with sample data
2. **2-hop traversal**: Test path confidence calculation
3. **Bidirectional traversal**: Verify inbound/outbound aggregation
4. **Type filtering**: Test entity type constraints
5. **Mentions lookup**: Verify chunk/document linkage
6. **Edge cases**: Empty results, missing entities, high fanout

### Performance Tests

Use `EXPLAIN ANALYZE` to verify query plans:

```python
def test_1hop_performance():
    # 1. Insert 10k entities + 30k relationships
    # 2. Run 1-hop query with EXPLAIN ANALYZE
    # 3. Assert execution time <10ms P95
    # 4. Assert index usage (Index Scan, not Seq Scan)
```

### Integration Tests

Test with real schema migration:

1. Create test database with schema
2. Run migrations to create indexes
3. Insert sample entities and relationships
4. Verify query results and performance
5. Test cache integration

---

## Query Optimization Checklist

Before deploying to production:

- [ ] All indexes created (8 core indexes, 4 composite)
- [ ] Query plans verified with `EXPLAIN ANALYZE`
- [ ] P95 latencies meet targets (<10ms 1-hop, <50ms 2-hop)
- [ ] Tested with production-scale data (10k entities, 30k relationships)
- [ ] Cache integration tested (>80% hit rate)
- [ ] Reranking pipeline integrated and tested
- [ ] Monitoring instrumented (query latency, index usage, cache hit rate)
- [ ] PostgreSQL tuning applied (`work_mem`, `random_page_cost`, `effective_cache_size`)

---

## Future Optimizations

### Phase 1 (Current): Raw SQL CTEs
- 1-hop: P95 <10ms
- 2-hop: P95 <50ms
- Scope: 10-20k entities

### Phase 2 (If needed): Recursive CTEs
- Variable-depth traversal (N-hop)
- Shortest path queries
- Scope: 50-100k entities

### Phase 3 (If needed): Materialized Views
- Pre-compute 2-hop paths for hot entities
- Trade storage for query speed
- Scope: 100k+ entities with high query load

### Phase 4 (If needed): Graph Extension (Apache AGE)
- Native graph traversal
- Cypher query language
- Scope: 500k+ entities with complex graph analytics

---

## Documentation Updates

This document should be updated when:
1. New query patterns are added
2. Index strategy changes
3. Performance characteristics change (e.g., after scaling to 50k entities)
4. Integration patterns change (e.g., new reranking strategy)

**Last Updated**: 2025-11-09
**Version**: 1.0.0
**Author**: Task 7.5 Phase 1 Implementation
