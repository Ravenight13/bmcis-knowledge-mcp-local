# Knowledge Graph Database Schema

**Task**: 7.3 Phase 1: Normalized PostgreSQL Schema
**Date**: 2025-11-09
**Architecture Pattern**: Hybrid Normalized + Cache
**Target Scale**: 10-20k entities, 500-750 documents, ~4MB total storage

---

## Overview

This document describes the normalized PostgreSQL schema for the BMCIS Knowledge Graph. The schema uses a relational design optimized for:

1. **Incremental updates**: Single INSERT/UPDATE per entity or relationship
2. **Query flexibility**: Support for 1-hop, 2-hop, and co-mention queries
3. **Provenance tracking**: Full history of entity mentions in documents
4. **Performance**: Sub-10ms P95 latency for common query patterns (with cache layer)
5. **Data integrity**: Constraints prevent invalid data (no self-loops, valid confidence scores)

---

## Entity-Relationship Diagram

```
┌──────────────────────────┐
│   knowledge_entities     │
├──────────────────────────┤
│ id (UUID, PK)            │
│ text (TEXT, UNIQUE)      │
│ entity_type (VARCHAR)    │ ◄───────────┐
│ confidence (FLOAT)       │             │
│ canonical_form (TEXT)    │             │
│ mention_count (INT)      │             │
│ created_at (TIMESTAMP)   │             │
│ updated_at (TIMESTAMP)   │             │
└──────────────────────────┘             │
         ▲         ▲                     │
         │         │                     │
    (FK) │         │ (FK)                │
         │         │                     │
    ┌────┴─────────┴────┐                │
    │                   │                │
┌───┴──────────────────┴───┐             │
│ entity_relationships      │             │
├───────────────────────────┤             │
│ id (UUID, PK)             │             │
│ source_entity_id (UUID)   ├─────────────┤
│ target_entity_id (UUID)   ├─────────────┘
│ relationship_type (VARCHAR)
│ confidence (FLOAT)        │
│ relationship_weight (FLOAT)
│ is_bidirectional (BOOL)   │
│ created_at (TIMESTAMP)    │
│ updated_at (TIMESTAMP)    │
└───────────────────────────┘

┌──────────────────────────┐
│   entity_mentions        │
├──────────────────────────┤
│ id (UUID, PK)            │
│ entity_id (UUID, FK)     │ ────► knowledge_entities.id
│ document_id (VARCHAR)    │
│ chunk_id (INT)           │
│ mention_text (TEXT)      │
│ offset_start (INT)       │
│ offset_end (INT)         │
│ created_at (TIMESTAMP)   │
└──────────────────────────┘
```

---

## Table Descriptions

### 1. `knowledge_entities`

**Purpose**: Core entity storage for the knowledge graph

**Columns**:

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Globally unique entity identifier (sharding-friendly) |
| `text` | TEXT | NOT NULL, UNIQUE with entity_type | Canonical entity text (e.g., "Lutron", "Anthropic") |
| `entity_type` | VARCHAR(50) | NOT NULL, INDEX | Entity classification (PERSON, ORG, PRODUCT, LOCATION, etc.) |
| `confidence` | FLOAT | NOT NULL, DEFAULT 1.0, CHECK [0.0-1.0] | Extraction confidence score |
| `canonical_form` | TEXT | NULLABLE, INDEX | Normalized form for deduplication (lowercase, stripped whitespace) |
| `mention_count` | INT | NOT NULL, DEFAULT 0, INDEX | Frequency of mentions in corpus (for popularity sorting) |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Record creation time (audit trail) |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last modification time (trigger-maintained) |

**Constraints**:
- `confidence` must be in [0.0, 1.0]
- `(text, entity_type)` must be unique (prevents duplicate entities)
- `text` must be non-empty (checked at application level)

**Indexes**:
- `pk`: id (primary key, automatic)
- `idx_knowledge_entities_text`: text (entity name lookups)
- `idx_knowledge_entities_type`: entity_type (filtering by type)
- `idx_knowledge_entities_canonical`: canonical_form (deduplication)
- `idx_knowledge_entities_mention_count`: mention_count DESC (popularity queries)

**Triggers**:
- `trigger_update_knowledge_entity_timestamp`: Updates `updated_at` on every BEFORE UPDATE

**Example Queries**:

```sql
-- Find entity by name
SELECT * FROM knowledge_entities WHERE text = 'Anthropic' AND entity_type = 'ORG';

-- Find all entities of a type
SELECT * FROM knowledge_entities WHERE entity_type = 'PRODUCT' ORDER BY mention_count DESC;

-- Deduplication check
SELECT * FROM knowledge_entities WHERE canonical_form = LOWER('anthropic') AND entity_type = 'ORG';
```

---

### 2. `entity_relationships`

**Purpose**: Typed directed relationships between entities (property graph edges)

**Columns**:

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Relationship identifier |
| `source_entity_id` | UUID | NOT NULL, FK, INDEX | Source entity of relationship |
| `target_entity_id` | UUID | NOT NULL, FK, INDEX | Target entity of relationship |
| `relationship_type` | VARCHAR(50) | NOT NULL, INDEX | Type of relationship (hierarchical, mentions-in-document, similar-to) |
| `confidence` | FLOAT | NOT NULL, DEFAULT 1.0, CHECK [0.0-1.0] | Relationship confidence (based on extraction method) |
| `relationship_weight` | FLOAT | NOT NULL, DEFAULT 1.0 | Frequency-based weight (higher = stronger) |
| `is_bidirectional` | BOOL | NOT NULL, DEFAULT FALSE, INDEX | Whether relationship is symmetric |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Record creation time |
| `updated_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Last modification time (trigger-maintained) |

**Constraints**:
- `source_entity_id != target_entity_id` (no self-loops)
- `confidence` must be in [0.0, 1.0]
- `(source_entity_id, target_entity_id, relationship_type)` must be unique
- Foreign keys ensure referential integrity (ON DELETE CASCADE)

**Relationship Types**:
- `hierarchical`: Source is parent/creator/owner of target (directed, non-symmetric)
- `mentions-in-document`: Source and target mentioned together in documents (typically bidirectional)
- `similar-to`: Source is semantically similar to target (typically bidirectional)

**Indexes**:
- `pk`: id (primary key, automatic)
- `idx_entity_relationships_source`: source_entity_id (outbound query optimization)
- `idx_entity_relationships_target`: target_entity_id (inbound query optimization)
- `idx_entity_relationships_type`: relationship_type (filtering by type)
- `idx_entity_relationships_graph`: (source_entity_id, relationship_type, target_entity_id) (graph traversal queries)
- `idx_entity_relationships_bidirectional`: is_bidirectional (symmetric relationship queries)

**Triggers**:
- `trigger_update_entity_relationship_timestamp`: Updates `updated_at` on every BEFORE UPDATE

**Example Queries**:

```sql
-- Get all outbound relationships from entity
SELECT r.*, e.text AS target_text
FROM entity_relationships r
JOIN knowledge_entities e ON r.target_entity_id = e.id
WHERE r.source_entity_id = ?
ORDER BY r.confidence DESC;

-- Find hierarchical relationships
SELECT r.*, e_source.text AS source_text, e_target.text AS target_text
FROM entity_relationships r
JOIN knowledge_entities e_source ON r.source_entity_id = e_source.id
JOIN knowledge_entities e_target ON r.target_entity_id = e_target.id
WHERE r.relationship_type = 'hierarchical'
ORDER BY r.confidence DESC;

-- Get entities related by any type
SELECT e.text, r.relationship_type, r.confidence
FROM entity_relationships r
JOIN knowledge_entities e ON r.target_entity_id = e.id
WHERE r.source_entity_id = ?
ORDER BY r.confidence DESC;
```

---

### 3. `entity_mentions`

**Purpose**: Provenance tracking - records where and how entities appear in documents

**Columns**:

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `id` | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Mention identifier |
| `entity_id` | UUID | NOT NULL, FK, INDEX | Reference to knowledge_entities |
| `document_id` | VARCHAR(255) | NOT NULL, INDEX | Source document (e.g., "docs/README.md", "kb_article_123") |
| `chunk_id` | INT | NOT NULL | Chunk/passage number within document (0-indexed) |
| `mention_text` | TEXT | NOT NULL | Actual text as it appears in the source |
| `offset_start` | INT | NULLABLE | Character offset start in chunk (for highlighting) |
| `offset_end` | INT | NULLABLE | Character offset end in chunk (exclusive, for highlighting) |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Record creation time |

**Constraints**:
- `entity_id` must reference valid entity (FK ensures this, ON DELETE CASCADE cleanup)
- `document_id` must be non-empty
- `chunk_id` must be non-negative (checked at application level)

**Indexes**:
- `pk`: id (primary key, automatic)
- `idx_entity_mentions_entity`: entity_id (entity-to-mention lookup)
- `idx_entity_mentions_document`: document_id (document-to-entity lookup)
- `idx_entity_mentions_chunk`: (document_id, chunk_id) (chunk-based queries)
- `idx_entity_mentions_composite`: (entity_id, document_id) (deduplication and co-mention analysis)

**Purpose of Each Index**:

| Index | Primary Use Case | Why Composite Index? |
|-------|-----------------|----------------------|
| `idx_entity_mentions_entity` | Find all mentions of a specific entity | Fast filtering by entity_id |
| `idx_entity_mentions_document` | Find all entities mentioned in a document | Fast filtering by document_id |
| `idx_entity_mentions_chunk` | Find entities in specific chunk | Composite allows covering index (document_id, chunk_id) |
| `idx_entity_mentions_composite` | Find co-mentioned entities, deduplication check | Both dimensions needed for efficient join |

**Example Queries**:

```sql
-- Find all mentions of an entity
SELECT * FROM entity_mentions WHERE entity_id = ? ORDER BY document_id, chunk_id;

-- Find all entities mentioned in a document
SELECT DISTINCT ke.id, ke.text, ke.entity_type
FROM entity_mentions em
JOIN knowledge_entities ke ON em.entity_id = ke.id
WHERE em.document_id = ?
ORDER BY ke.mention_count DESC;

-- Find co-mentioned entities (mentioned in same document)
SELECT e2.id, e2.text, COUNT(DISTINCT em1.chunk_id) AS chunk_count
FROM entity_mentions em1
JOIN entity_mentions em2 ON em1.document_id = em2.document_id
JOIN knowledge_entities e2 ON em2.entity_id = e2.id
WHERE em1.entity_id = ? AND em2.entity_id != em1.entity_id
GROUP BY e2.id, e2.text
ORDER BY chunk_count DESC;

-- Find mentions in specific chunk
SELECT ke.text, em.mention_text, em.offset_start, em.offset_end
FROM entity_mentions em
JOIN knowledge_entities ke ON em.entity_id = ke.id
WHERE em.document_id = ? AND em.chunk_id = ?
ORDER BY em.offset_start;
```

---

## Index Strategy and Performance Implications

### Why These Indexes?

**1. Graph Traversal Queries**:
- `idx_entity_relationships_graph` (source, type, target) enables fast 1-hop traversals
- Composite index allows single index scan instead of multiple lookups
- Estimated performance: <5ms for 10k entities

**2. Entity Lookups**:
- `idx_knowledge_entities_text` enables fast entity name lookups
- Combined with `idx_knowledge_entities_type`, supports (text, type) queries

**3. Co-mention Analysis**:
- `idx_entity_mentions_composite` (entity_id, document_id) enables fast deduplication checks
- Used for detecting multiple mentions of same entity in same document

**4. Popularity Sorting**:
- `idx_knowledge_entities_mention_count` DESC enables fast sorting by frequency
- Used for "top N entities" queries without full table scan

### Performance Expectations

| Query Pattern | Estimated Latency | Index Used |
|--------------|------------------|------------|
| Entity lookup by name | <1ms | idx_knowledge_entities_text |
| 1-hop relationships | 5-10ms | idx_entity_relationships_graph |
| 2-hop relationships | 20-50ms | idx_entity_relationships_source/target |
| Co-mention analysis | 10-20ms | idx_entity_mentions_composite |
| Most recent entities | <5ms | created_at index |
| Top entities by mention | <5ms | idx_knowledge_entities_mention_count |

**Notes**:
- All estimates assume ~10k entities, ~30k relationships, ~50k mentions
- Cache layer (Python in-memory LRU) provides <1ms for hot entities
- Full table scans should be avoided; always use indexed columns in WHERE clause

---

## Data Constraints and Validation

### Confidence Score Constraints

**Column**: `confidence` (in both `knowledge_entities` and `entity_relationships`)

**Constraint**: `CHECK (confidence >= 0.0 AND confidence <= 1.0)`

**Semantics**:
- `1.0`: High confidence (extracted with high certainty or manual entry)
- `0.5`: Medium confidence (multiple sources or weak signals)
- `0.0`: Low confidence (weak extraction signal, single mention)

**Application Level**:
- Always validate before INSERT/UPDATE
- Use sensible defaults based on extraction method
- Syntactic parsing: 0.8
- Co-occurrence: 0.5
- Manual entry: 1.0

### Entity Relationship Uniqueness

**Constraint**: `UNIQUE(source_entity_id, target_entity_id, relationship_type)`

**Semantics**:
- Only one relationship of each type between any two entities
- On duplicate insert, use `ON CONFLICT ... DO UPDATE` to update confidence/weight

**Example**:
```sql
INSERT INTO entity_relationships (source_entity_id, target_entity_id, relationship_type, confidence)
VALUES (?, ?, 'hierarchical', 0.8)
ON CONFLICT (source_entity_id, target_entity_id, relationship_type)
DO UPDATE SET confidence = EXCLUDED.confidence, updated_at = NOW();
```

### No Self-Loop Constraint

**Constraint**: `CHECK (source_entity_id != target_entity_id)`

**Semantics**:
- Relationships must connect different entities
- Prevents circular dependencies and simplifies traversal logic
- Self-reference relationships (e.g., entity tags) should be stored differently

---

## Incremental Update Patterns

### Adding a New Entity

```sql
INSERT INTO knowledge_entities (text, entity_type, confidence, canonical_form)
VALUES (?, ?, ?, LOWER(?))
ON CONFLICT (text, entity_type)
DO UPDATE SET
    confidence = EXCLUDED.confidence,
    updated_at = NOW(),
    mention_count = mention_count + 1;
```

### Adding a Mention

```sql
INSERT INTO entity_mentions (entity_id, document_id, chunk_id, mention_text, offset_start, offset_end)
VALUES (?, ?, ?, ?, ?, ?);

-- Update mention count
UPDATE knowledge_entities
SET mention_count = mention_count + 1
WHERE id = ?;
```

### Adding a Relationship

```sql
INSERT INTO entity_relationships (source_entity_id, target_entity_id, relationship_type, confidence)
VALUES (?, ?, ?, ?)
ON CONFLICT (source_entity_id, target_entity_id, relationship_type)
DO UPDATE SET
    confidence = EXCLUDED.confidence,
    relationship_weight = relationship_weight + 1.0,
    updated_at = NOW();
```

---

## Example Queries for Common Operations

### 1. Get All Relationships for an Entity (1-hop)

```sql
-- Get all entities that SOURCE connects to
SELECT
    e.id, e.text, e.entity_type, e.mention_count,
    r.relationship_type, r.confidence
FROM entity_relationships r
JOIN knowledge_entities e ON r.target_entity_id = e.id
WHERE r.source_entity_id = $1
ORDER BY r.confidence DESC;

-- Get all entities that connect TO source (inbound)
SELECT
    e.id, e.text, e.entity_type,
    r.relationship_type, r.confidence
FROM entity_relationships r
JOIN knowledge_entities e ON r.source_entity_id = e.id
WHERE r.target_entity_id = $1
ORDER BY r.confidence DESC;
```

### 2. Get 2-Hop Relationships

```sql
WITH first_hop AS (
    SELECT
        r.target_entity_id AS entity_id,
        r.relationship_type AS rel1_type,
        r.confidence AS conf1
    FROM entity_relationships r
    WHERE r.source_entity_id = $1
    ORDER BY r.confidence DESC
    LIMIT 10
)
SELECT
    e.id, e.text, e.entity_type,
    fh.rel1_type, fh.conf1,
    r.relationship_type AS rel2_type, r.confidence AS conf2,
    (fh.conf1 * r.confidence) AS combined_confidence
FROM first_hop fh
JOIN entity_relationships r ON r.source_entity_id = fh.entity_id
JOIN knowledge_entities e ON e.id = r.target_entity_id
ORDER BY combined_confidence DESC
LIMIT 20;
```

### 3. Find Co-Mentioned Entities

```sql
SELECT
    e2.id, e2.text, e2.entity_type,
    COUNT(DISTINCT em1.document_id) AS doc_count,
    COUNT(DISTINCT em1.chunk_id) AS chunk_count
FROM entity_mentions em1
JOIN entity_mentions em2
    ON em1.document_id = em2.document_id
    AND em1.entity_id != em2.entity_id
JOIN knowledge_entities e2 ON em2.entity_id = e2.id
WHERE em1.entity_id = $1
GROUP BY e2.id, e2.text, e2.entity_type
HAVING COUNT(DISTINCT em1.document_id) >= 2  -- Co-mentioned in at least 2 docs
ORDER BY COUNT(DISTINCT em1.chunk_id) DESC
LIMIT 20;
```

### 4. Get Entities in Document with Mention Locations

```sql
SELECT
    ke.id, ke.text, ke.entity_type, ke.mention_count,
    em.mention_text, em.chunk_id, em.offset_start, em.offset_end,
    em.created_at
FROM entity_mentions em
JOIN knowledge_entities ke ON em.entity_id = ke.id
WHERE em.document_id = $1
ORDER BY em.chunk_id, em.offset_start;
```

---

## Storage Estimates

**Assumptions**:
- 10-20k entities (use ~15k average)
- 30-50k relationships (use ~40k average)
- 50k mentions (1-5 mentions per entity)
- Average text length: 50 bytes
- Average document_id: 50 bytes

| Table | Rows | Avg Row Size | Total Size |
|-------|------|--------------|-----------|
| knowledge_entities | 15,000 | 40 bytes | 600 KB |
| entity_relationships | 40,000 | 50 bytes | 2 MB |
| entity_mentions | 50,000 | 40 bytes | 2 MB |
| **Indexes** | - | - | **1 MB** |
| **Total** | - | - | **~5.6 MB** |

**Actual estimates may vary**:
- Text fields can be longer (product names, descriptions)
- Document_id can be longer (full paths)
- Conservative estimate: ~8-10 MB for production data

---

## Migration and Schema Evolution

### Creating the Schema

Use the migration script in `migrations/001_create_knowledge_graph.py`:

```python
from src.knowledge_graph.migrations.migration_001 import upgrade
from src.core.database import DatabasePool

with DatabasePool.get_connection() as conn:
    upgrade(conn)
    conn.commit()
```

### Idempotent Design

All CREATE statements use `IF NOT EXISTS`:
- `CREATE TABLE IF NOT EXISTS`
- `CREATE INDEX IF NOT EXISTS`
- `CREATE TRIGGER IF NOT EXISTS`

This allows safe re-execution without errors.

### Future Schema Changes

When adding new columns/indexes:
1. Create new migration file: `002_add_xyz_column.py`
2. Use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
3. Use `CREATE INDEX IF NOT EXISTS`
4. Test on staging database first
5. Deploy with zero downtime

---

## Monitoring and Maintenance

### Index Monitoring

```sql
-- Check index sizes
SELECT
    schemaname, tablename, indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
AND tablename IN ('knowledge_entities', 'entity_relationships', 'entity_mentions')
ORDER BY pg_relation_size(indexrelid) DESC;

-- Identify unused indexes
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
AND (idx_scan = 0 OR idx_tup_read = 0)
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Query Performance

```sql
-- Analyze query plans
EXPLAIN ANALYZE
SELECT * FROM knowledge_entities WHERE text = ? AND entity_type = ?;

-- Show index effectiveness
SELECT
    indexrelname, idx_blks_read, idx_blks_hit,
    ROUND(100.0 * idx_blks_hit / (idx_blks_hit + idx_blks_read), 2) AS hit_ratio
FROM pg_statio_idx_blks
WHERE schemaname = 'public';
```

### Table Statistics

```sql
-- Update statistics for query planner
ANALYZE knowledge_entities;
ANALYZE entity_relationships;
ANALYZE entity_mentions;

-- Check row counts
SELECT
    tablename,
    n_live_tup AS live_rows,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) AS total_size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
AND tablename IN ('knowledge_entities', 'entity_relationships', 'entity_mentions');
```

---

## Security Considerations

### No Direct SQL Injection Risk

- All example queries use parameterized queries (`$1`, `$2`, etc.)
- Use prepared statements in application code
- Never concatenate user input into SQL strings

### Data Access Control

- Consider adding `created_by` and `updated_by` columns for audit trail
- Implement row-level security (RLS) if multi-tenant access needed
- Document permission model for knowledge graph access

### Backups

- Regular backups should include all three tables
- Test backup restoration regularly
- Consider point-in-time recovery needs

---

## Related Documentation

- **Architecture Review**: See `docs/subagent-reports/architecture-review/2025-11-09-task7-schema-relationships.md`
- **ORM Models**: See `src/knowledge_graph/models.py`
- **Migration Scripts**: See `src/knowledge_graph/migrations/`
- **SQL Schema**: See `src/knowledge_graph/schema.sql`

---

**End of Schema Documentation**
