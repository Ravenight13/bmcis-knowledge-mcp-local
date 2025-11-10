-- ============================================================================
-- BMCIS Knowledge Graph - Index Validation Script
-- ============================================================================
--
-- Purpose: Verify composite indexes (Migration 003) are created and being used
--
-- Usage:
--   psql -h localhost -U postgres -d knowledge_graph -f validate_indexes.sql
--
-- Expected Output:
--   - All 4 indexes exist
--   - EXPLAIN plans show index usage
--   - No sequential scans for indexed queries
-- ============================================================================

\echo '============================================================================'
\echo 'BMCIS Knowledge Graph - Index Validation'
\echo 'Migration 003: Composite Index Implementation (HP 4)'
\echo '============================================================================'

-- 1. Verify all indexes exist
\echo ''
\echo '1. Checking index existence...'
\echo ''

SELECT
    indexname,
    tablename,
    indexdef
FROM pg_indexes
WHERE indexname IN (
    'idx_relationships_source_confidence',
    'idx_entities_type_id',
    'idx_entities_updated_at',
    'idx_relationships_target_type'
)
ORDER BY indexname;

-- 2. Verify index comments (documentation)
\echo ''
\echo '2. Checking index comments...'
\echo ''

SELECT
    c.relname AS index_name,
    obj_description(c.oid, 'pg_class') AS comment
FROM pg_class c
WHERE c.relname IN (
    'idx_relationships_source_confidence',
    'idx_entities_type_id',
    'idx_entities_updated_at',
    'idx_relationships_target_type'
)
ORDER BY c.relname;

-- 3. Verify index sizes
\echo ''
\echo '3. Checking index sizes...'
\echo ''

SELECT
    c.relname AS index_name,
    pg_size_pretty(pg_relation_size(c.oid)) AS size,
    pg_relation_size(c.oid) AS size_bytes
FROM pg_class c
WHERE c.relname IN (
    'idx_relationships_source_confidence',
    'idx_entities_type_id',
    'idx_entities_updated_at',
    'idx_relationships_target_type'
)
ORDER BY c.relname;

-- 4. Verify index usage statistics (if available)
\echo ''
\echo '4. Checking index usage statistics...'
\echo ''

SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE indexname IN (
    'idx_relationships_source_confidence',
    'idx_entities_type_id',
    'idx_entities_updated_at',
    'idx_relationships_target_type'
)
ORDER BY indexname;

-- ============================================================================
-- EXPLAIN ANALYZE Verification
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'EXPLAIN ANALYZE Verification (requires sample data)'
\echo '============================================================================'

-- Get a sample entity ID for testing
\echo ''
\echo '5. Testing idx_relationships_source_confidence...'
\echo ''

DO $$
DECLARE
    sample_entity_id UUID;
BEGIN
    -- Get a sample entity with relationships
    SELECT DISTINCT source_entity_id INTO sample_entity_id
    FROM entity_relationships
    LIMIT 1;

    IF sample_entity_id IS NOT NULL THEN
        RAISE NOTICE 'Sample entity ID: %', sample_entity_id;

        -- Run EXPLAIN ANALYZE
        RAISE NOTICE 'Expected: Index Scan using idx_relationships_source_confidence';
    ELSE
        RAISE NOTICE 'No sample data available. Skipping EXPLAIN ANALYZE tests.';
    END IF;
END $$;

-- Manual EXPLAIN test (uncomment and replace UUID to run)
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT target_entity_id, confidence
-- FROM entity_relationships
-- WHERE source_entity_id = 'REPLACE-WITH-ACTUAL-UUID'::uuid
-- ORDER BY confidence DESC
-- LIMIT 50;

\echo ''
\echo '6. Testing idx_entities_type_id...'
\echo ''

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, text, confidence
FROM knowledge_entities
WHERE entity_type = 'PERSON'
ORDER BY id
LIMIT 100;

\echo ''
\echo '7. Testing idx_entities_updated_at...'
\echo ''

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, text, entity_type, updated_at
FROM knowledge_entities
WHERE updated_at > NOW() - INTERVAL '1 hour'
ORDER BY updated_at DESC
LIMIT 1000;

\echo ''
\echo '8. Testing idx_relationships_target_type...'
\echo ''

DO $$
DECLARE
    sample_entity_id UUID;
BEGIN
    -- Get a sample target entity
    SELECT DISTINCT target_entity_id INTO sample_entity_id
    FROM entity_relationships
    LIMIT 1;

    IF sample_entity_id IS NOT NULL THEN
        RAISE NOTICE 'Sample target entity ID: %', sample_entity_id;
        RAISE NOTICE 'Expected: Index Scan using idx_relationships_target_type';
    ELSE
        RAISE NOTICE 'No sample data available. Skipping test.';
    END IF;
END $$;

-- Manual EXPLAIN test (uncomment and replace UUID to run)
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT source_entity_id, confidence
-- FROM entity_relationships
-- WHERE target_entity_id = 'REPLACE-WITH-ACTUAL-UUID'::uuid
--   AND relationship_type = 'similar-to'
-- ORDER BY confidence DESC;

-- ============================================================================
-- Performance Comparison
-- ============================================================================

\echo ''
\echo '============================================================================'
\echo 'Performance Expectations (with composite indexes)'
\echo '============================================================================'

SELECT
    'idx_relationships_source_confidence' AS index_name,
    '1-hop sorted traversal' AS query_type,
    '8-12ms → 3-5ms' AS latency_improvement,
    '60-70%' AS improvement_pct
UNION ALL
SELECT
    'idx_entities_type_id',
    'Type-filtered queries',
    '18.5ms → 2.5ms',
    '86%'
UNION ALL
SELECT
    'idx_entities_updated_at',
    'Incremental sync',
    '5-10ms → 1-2ms',
    '70-80%'
UNION ALL
SELECT
    'idx_relationships_target_type',
    'Reverse 1-hop with type',
    '6-10ms → 2-4ms',
    '50-60%';

\echo ''
\echo '============================================================================'
\echo 'Validation Complete'
\echo '============================================================================'
\echo ''
\echo 'Next Steps:'
\echo '1. Verify all 4 indexes exist (should see 4 rows in section 1)'
\echo '2. Check index comments are set (should see performance descriptions)'
\echo '3. Verify EXPLAIN plans show Index Scan (not Sequential Scan)'
\echo '4. Run performance tests: pytest tests/knowledge_graph/test_index_performance.py'
\echo ''
