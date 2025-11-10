-- ============================================================================
-- Knowledge Graph SQL Traversal Queries
-- ============================================================================
-- Performance targets: 1-hop <10ms, 2-hop <50ms (P95)
-- Index requirements: See QUERIES.md for full index strategy
-- Version: 1.0.0
-- ============================================================================

-- ============================================================================
-- Query 1: 1-HOP OUTBOUND TRAVERSAL
-- ============================================================================
-- Purpose: Get all entities directly related to source entity
-- Performance: P50 <5ms, P95 <10ms (with index on source_entity_id + relationship_type)
-- Latency profile: ~8ms for 20 related entities, ~5ms for <10 entities
-- Index: idx_relationships_source ON entity_relationships(source_entity_id)
-- ============================================================================

-- Query Name: 1_hop_outbound_traversal
-- Parameters:
--   $1: source_entity_id (INTEGER) - Source entity ID
--   $2: min_confidence (FLOAT) - Minimum relationship confidence (default: 0.7)
--   $3: relationship_types (VARCHAR[]) - Optional filter by relationship types
--   $4: max_results (INTEGER) - Limit results (default: 50)

WITH related_entities AS (
    SELECT
        r.target_entity_id,
        r.relationship_type,
        r.confidence AS relationship_confidence,
        r.metadata AS relationship_metadata
    FROM entity_relationships r
    WHERE r.source_entity_id = $1
      AND r.confidence >= $2
      AND ($3 IS NULL OR r.relationship_type = ANY($3))
)
SELECT
    e.id,
    e.entity_name AS text,
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,
    re.relationship_type,
    re.relationship_confidence,
    re.relationship_metadata
FROM related_entities re
JOIN knowledge_entities e ON e.id = re.target_entity_id
ORDER BY re.relationship_confidence DESC
LIMIT $4;

-- Expected plan:
-- -> Limit
--   -> Sort (on relationship_confidence DESC)
--     -> Hash Join (entity_relationships.target_entity_id = knowledge_entities.id)
--       -> Index Scan using idx_relationship_source on entity_relationships
--       -> Seq Scan on knowledge_entities
-- Estimated cost: 10-20 for 50 entities


-- ============================================================================
-- Query 2: 2-HOP TRAVERSAL (EXTENDED NETWORK)
-- ============================================================================
-- Purpose: Get entities reachable in 2 relationship steps from source entity
-- Performance: P50 <20ms, P95 <50ms (depends on fanout, indexed)
-- Latency profile: ~30ms for 10 intermediate entities with 5 targets each
-- Index: idx_relationships_source, idx_relationships_target
-- ============================================================================

-- Query Name: 2_hop_traversal
-- Parameters:
--   $1: source_entity_id (INTEGER) - Source entity ID
--   $2: min_confidence (FLOAT) - Minimum relationship confidence (default: 0.7)
--   $3: relationship_types (VARCHAR[]) - Optional filter by relationship types
--   $4: max_results (INTEGER) - Limit results (default: 100)

WITH hop1 AS (
    SELECT DISTINCT
        r1.target_entity_id AS entity_id,
        r1.confidence AS hop1_confidence,
        r1.relationship_type AS hop1_rel_type
    FROM entity_relationships r1
    WHERE r1.source_entity_id = $1
      AND r1.confidence >= $2
      AND ($3 IS NULL OR r1.relationship_type = ANY($3))
),
hop2 AS (
    SELECT
        r2.target_entity_id AS entity_id,
        e2.entity_name AS text,
        e2.entity_type,
        e2.metadata->>'confidence' AS entity_confidence,
        r2.relationship_type AS hop2_rel_type,
        r2.confidence AS hop2_confidence,
        h1.entity_id AS intermediate_entity_id,
        ei.entity_name AS intermediate_entity_name,
        h1.hop1_confidence,
        h1.hop1_rel_type,
        -- Distance metric: geometric mean of confidences
        SQRT(h1.hop1_confidence * r2.confidence) AS path_confidence
    FROM hop1 h1
    JOIN entity_relationships r2 ON r2.source_entity_id = h1.entity_id
    JOIN knowledge_entities e2 ON e2.id = r2.target_entity_id
    JOIN knowledge_entities ei ON ei.id = h1.entity_id
    WHERE r2.confidence >= $2
      AND r2.target_entity_id != $1  -- Prevent cycles back to source
      AND ($3 IS NULL OR r2.relationship_type = ANY($3))
)
SELECT
    h2.entity_id,
    h2.text,
    h2.entity_type,
    h2.entity_confidence,
    h2.hop2_rel_type AS relationship_type,
    h2.hop2_confidence AS relationship_confidence,
    h2.intermediate_entity_id,
    h2.intermediate_entity_name,
    h2.path_confidence,
    2 AS path_depth
FROM hop2 h2
ORDER BY h2.path_confidence DESC
LIMIT $4;

-- Expected plan:
-- -> Limit
--   -> Sort (on path_confidence DESC)
--     -> Hash Join (hop2 relationships)
--       -> CTE Scan on hop1
--       -> Index Scan using idx_relationships_source
--       -> Hash Join with knowledge_entities
-- Estimated cost: 50-100 for 100 entities with moderate fanout


-- ============================================================================
-- Query 3: BIDIRECTIONAL TRAVERSAL (FULL RELATIONSHIP NETWORK)
-- ============================================================================
-- Purpose: Get all entities connected to source (both incoming and outgoing)
-- Performance: P50 <15ms (1-hop), P95 <30ms
-- Latency profile: ~20ms for 30 total relationships (15 inbound + 15 outbound)
-- Index: idx_relationships_source, idx_relationships_target
-- ============================================================================

-- Query Name: bidirectional_traversal
-- Parameters:
--   $1: source_entity_id (INTEGER) - Source entity ID
--   $2: min_confidence (FLOAT) - Minimum relationship confidence (default: 0.7)
--   $3: max_depth (INTEGER) - Maximum traversal depth (1 or 2, default: 1)
--   $4: max_results (INTEGER) - Limit results (default: 50)

WITH outbound AS (
    SELECT
        r.target_entity_id AS related_entity_id,
        r.relationship_type,
        r.confidence,
        'outbound' AS direction,
        1 AS distance
    FROM entity_relationships r
    WHERE r.source_entity_id = $1
      AND r.confidence >= $2
),
inbound AS (
    SELECT
        r.source_entity_id AS related_entity_id,
        r.relationship_type,
        r.confidence,
        'inbound' AS direction,
        1 AS distance
    FROM entity_relationships r
    WHERE r.target_entity_id = $1
      AND r.confidence >= $2
),
combined AS (
    SELECT
        COALESCE(o.related_entity_id, i.related_entity_id) AS entity_id,
        ARRAY_AGG(DISTINCT o.relationship_type) FILTER (WHERE o.relationship_type IS NOT NULL) AS outbound_rel_types,
        ARRAY_AGG(DISTINCT i.relationship_type) FILTER (WHERE i.relationship_type IS NOT NULL) AS inbound_rel_types,
        GREATEST(COALESCE(MAX(o.confidence), 0), COALESCE(MAX(i.confidence), 0)) AS max_confidence,
        COUNT(*) AS relationship_count,
        MIN(COALESCE(o.distance, i.distance)) AS min_distance
    FROM outbound o
    FULL OUTER JOIN inbound i ON o.related_entity_id = i.related_entity_id
    GROUP BY COALESCE(o.related_entity_id, i.related_entity_id)
)
SELECT
    c.entity_id,
    e.entity_name AS text,
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,
    c.outbound_rel_types,
    c.inbound_rel_types,
    c.max_confidence,
    c.relationship_count,
    c.min_distance
FROM combined c
JOIN knowledge_entities e ON e.id = c.entity_id
ORDER BY c.relationship_count DESC, c.max_confidence DESC
LIMIT $4;

-- Expected plan:
-- -> Limit
--   -> Sort (on relationship_count DESC, max_confidence DESC)
--     -> Hash Join (combined.entity_id = knowledge_entities.id)
--       -> Hash Full Join (outbound ⋈ inbound)
--         -> CTE Scan on outbound (Index Scan idx_relationships_source)
--         -> CTE Scan on inbound (Index Scan idx_relationships_target)
-- Estimated cost: 20-40 for 50 entities


-- ============================================================================
-- Query 4: ENTITY TYPE FILTERING (CONTEXT-AWARE RELATIONSHIPS)
-- ============================================================================
-- Purpose: Get related entities of specific type(s)
-- Performance: P50 <8ms, P95 <15ms (with additional entity_type filter)
-- Latency profile: ~10ms for 20 entities filtered by 2 types
-- Index: idx_relationships_source, idx_entity_type
-- ============================================================================

-- Query Name: type_filtered_traversal
-- Parameters:
--   $1: source_entity_id (INTEGER) - Source entity ID
--   $2: relationship_type (VARCHAR) - Specific relationship type to filter
--   $3: target_entity_types (VARCHAR[]) - Entity types to include (e.g., ['VENDOR', 'PRODUCT'])
--   $4: min_confidence (FLOAT) - Minimum relationship confidence (default: 0.7)
--   $5: max_results (INTEGER) - Limit results (default: 50)

SELECT
    e.id,
    e.entity_name AS text,
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,
    r.relationship_type,
    r.confidence AS relationship_confidence,
    r.metadata AS relationship_metadata
FROM entity_relationships r
JOIN knowledge_entities e ON e.id = r.target_entity_id
WHERE r.source_entity_id = $1
  AND r.relationship_type = $2
  AND e.entity_type = ANY($3)
  AND r.confidence >= $4
ORDER BY r.confidence DESC
LIMIT $5;

-- Expected plan:
-- -> Limit
--   -> Sort (on confidence DESC)
--     -> Nested Loop
--       -> Index Scan using idx_relationships_source on entity_relationships
--         Filter: (relationship_type = $2 AND confidence >= $4)
--       -> Index Scan using idx_entity_type on knowledge_entities
--         Filter: (entity_type = ANY($3))
-- Estimated cost: 15-25 for 50 entities


-- ============================================================================
-- Query 5: ENTITY MENTIONS LOOKUP (DOCUMENT CONTEXT RETRIEVAL)
-- ============================================================================
-- Purpose: Get documents and chunks where entity is mentioned
-- Performance: P50 <10ms, P95 <20ms (with index on entity_id)
-- Latency profile: ~12ms for 50 mentions across documents
-- Index: idx_chunk_entities_entity ON chunk_entities(entity_id)
-- ============================================================================

-- Query Name: entity_mentions_lookup
-- Parameters:
--   $1: entity_id (INTEGER) - Entity ID to find mentions for
--   $2: max_results (INTEGER) - Limit results (default: 100)

SELECT
    ce.chunk_id AS chunk_id,
    kb.source_file AS document_id,
    kb.chunk_text AS chunk_text,
    kb.source_category AS document_category,
    kb.chunk_index,
    ce.confidence AS mention_confidence,
    kb.created_at AS indexed_at
FROM chunk_entities ce
JOIN knowledge_base kb ON kb.id = ce.chunk_id
WHERE ce.entity_id = $1
ORDER BY ce.confidence DESC, kb.created_at DESC
LIMIT $2;

-- Expected plan:
-- -> Limit
--   -> Sort (on confidence DESC, created_at DESC)
--     -> Hash Join (chunk_entities.chunk_id = knowledge_base.id)
--       -> Index Scan using idx_chunk_entities_entity on chunk_entities
--       -> Seq Scan on knowledge_base
-- Estimated cost: 10-20 for 100 mentions


-- ============================================================================
-- Query 6: CONFIDENCE-WEIGHTED PATHS (HIGH-CONFIDENCE RELATIONSHIPS ONLY)
-- ============================================================================
-- Purpose: Find paths where ALL relationships exceed a high confidence threshold
-- Performance: P50 <25ms, P95 <60ms (fewer relationships to traverse)
-- Latency profile: ~40ms for 2-hop with strict threshold (0.8+)
-- Index: idx_relationships_source with confidence filter
-- ============================================================================

-- Query Name: high_confidence_paths
-- Parameters:
--   $1: source_entity_id (INTEGER) - Source entity ID
--   $2: high_confidence_threshold (FLOAT) - Minimum confidence (default: 0.8)
--   $3: max_results (INTEGER) - Limit results (default: 50)

WITH high_conf_1hop AS (
    SELECT
        r1.target_entity_id AS entity_id,
        r1.confidence,
        r1.relationship_type
    FROM entity_relationships r1
    WHERE r1.source_entity_id = $1
      AND r1.confidence >= $2
),
high_conf_2hop AS (
    SELECT
        r2.target_entity_id AS entity_id,
        e2.entity_name AS text,
        e2.entity_type,
        r2.relationship_type,
        -- Geometric mean of path confidences
        SQRT(h1.confidence * r2.confidence) AS path_confidence,
        h1.entity_id AS intermediate_entity_id,
        ei.entity_name AS intermediate_entity_name
    FROM high_conf_1hop h1
    JOIN entity_relationships r2 ON r2.source_entity_id = h1.entity_id
    JOIN knowledge_entities e2 ON e2.id = r2.target_entity_id
    JOIN knowledge_entities ei ON ei.id = h1.entity_id
    WHERE r2.confidence >= $2
      AND r2.target_entity_id != $1
)
SELECT * FROM high_conf_2hop
ORDER BY path_confidence DESC
LIMIT $3;

-- Expected plan:
-- -> Limit
--   -> Sort (on path_confidence DESC)
--     -> Hash Join (2-hop relationships)
--       -> CTE Scan on high_conf_1hop
--       -> Index Scan using idx_relationships_source
--         Filter: (confidence >= $2)
-- Estimated cost: 30-50 for 50 entities (fewer relationships due to threshold)


-- ============================================================================
-- QUERY PERFORMANCE NOTES
-- ============================================================================

-- Required Indexes for Optimal Performance:
--
-- Core indexes (already in schema):
-- 1. idx_relationship_source ON entity_relationships(source_entity_id)
-- 2. idx_relationship_target ON entity_relationships(target_entity_id)
-- 3. idx_relationship_type ON entity_relationships(relationship_type)
-- 4. idx_entity_type ON knowledge_entities(entity_type)
-- 5. idx_chunk_entities_entity ON chunk_entities(entity_id)
--
-- Composite indexes for advanced queries:
-- 6. idx_relationships_source_conf ON entity_relationships(source_entity_id, confidence DESC)
-- 7. idx_relationships_target_conf ON entity_relationships(target_entity_id, confidence DESC)
-- 8. idx_entities_type_id ON knowledge_entities(entity_type, id)
--
-- Query plan analysis commands:
-- EXPLAIN (ANALYZE, BUFFERS) <query>;
-- EXPLAIN (FORMAT JSON) <query>;
--
-- Performance tuning parameters:
-- SET work_mem = '256MB';  -- For large 2-hop queries
-- SET random_page_cost = 1.1;  -- SSD optimization
-- SET effective_cache_size = '4GB';  -- Available cache

-- ============================================================================
-- END OF QUERIES
-- ============================================================================
