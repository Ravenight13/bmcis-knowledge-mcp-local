-- ============================================================================
-- BMCIS Knowledge Graph - Enum Type Validation
-- Migration: 004_add_enum_types.sql
-- Date: 2025-11-09
-- Author: Security Team
-- ============================================================================
--
-- Purpose: Add enum validation for entity_type and relationship_type
--
-- Security Impact:
-- - Prevents invalid entity types (data corruption, injection risk)
-- - Ensures consistent type values across entire database
-- - Enables reliable type-based queries and analytics
--
-- Schema Changes:
-- 1. Create entity_type_enum (PERSON, ORG, GPE, PRODUCT, EVENT, FACILITY, LAW, LANGUAGE, DATE, TIME, MONEY, PERCENT)
-- 2. Create relationship_type_enum (hierarchical, mentions-in-document, similar-to)
-- 3. Alter knowledge_entities.entity_type to use enum
-- 4. Alter entity_relationships.relationship_type to use enum
-- ============================================================================

-- Step 1: Create entity type enum
CREATE TYPE entity_type_enum AS ENUM (
    'PERSON',      -- People, including fictional characters
    'ORG',         -- Organizations, companies, agencies, institutions
    'GPE',         -- Geopolitical entities (countries, cities, states)
    'PRODUCT',     -- Products, technologies, services
    'EVENT',       -- Named events (conferences, wars, etc.)
    'FACILITY',    -- Buildings, airports, highways, bridges
    'LAW',         -- Named laws, regulations, legal documents
    'LANGUAGE',    -- Named languages
    'DATE',        -- Absolute or relative dates
    'TIME',        -- Times smaller than a day
    'MONEY',       -- Monetary values with currency
    'PERCENT'      -- Percentage values
);

COMMENT ON TYPE entity_type_enum IS
'Valid entity types from spaCy en_core_web_md NER model';

-- Step 2: Create relationship type enum
CREATE TYPE relationship_type_enum AS ENUM (
    'hierarchical',          -- Parent/child, creator/creation relationships
    'mentions-in-document',  -- Co-occurrence in same document/chunk
    'similar-to'             -- Semantic similarity (embedding-based)
);

COMMENT ON TYPE relationship_type_enum IS
'Valid relationship types for knowledge graph edges';

-- Step 3: Backfill check - validate any non-conforming entity types (if needed)
DO $$
DECLARE
    invalid_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO invalid_count
    FROM knowledge_entities
    WHERE entity_type NOT IN ('PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'FACILITY', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'MONEY', 'PERCENT');

    IF invalid_count > 0 THEN
        RAISE NOTICE 'Found % invalid entity types. Review before proceeding:', invalid_count;

        -- Log invalid types
        RAISE NOTICE 'Invalid entity types: %', (
            SELECT STRING_AGG(DISTINCT entity_type, ', ')
            FROM knowledge_entities
            WHERE entity_type NOT IN ('PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'FACILITY', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'MONEY', 'PERCENT')
        );

        RAISE EXCEPTION 'Invalid entity types detected. Fix manually before migration.';
    END IF;
END $$;

-- Step 4: Backfill check - validate any non-conforming relationship types (if needed)
DO $$
DECLARE
    invalid_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO invalid_count
    FROM entity_relationships
    WHERE relationship_type NOT IN ('hierarchical', 'mentions-in-document', 'similar-to');

    IF invalid_count > 0 THEN
        RAISE NOTICE 'Found % invalid relationship types:', invalid_count;
        RAISE NOTICE 'Invalid types: %', (
            SELECT STRING_AGG(DISTINCT relationship_type, ', ')
            FROM entity_relationships
            WHERE relationship_type NOT IN ('hierarchical', 'mentions-in-document', 'similar-to')
        );

        RAISE EXCEPTION 'Invalid relationship types detected. Fix manually before migration.';
    END IF;
END $$;

-- Step 5: Alter knowledge_entities.entity_type to use enum
-- This will fail if any values don't match enum
ALTER TABLE knowledge_entities
    ALTER COLUMN entity_type TYPE entity_type_enum
    USING entity_type::entity_type_enum;

-- Step 6: Alter entity_relationships.relationship_type to use enum
ALTER TABLE entity_relationships
    ALTER COLUMN relationship_type TYPE relationship_type_enum
    USING relationship_type::relationship_type_enum;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify enum types exist
SELECT typname, typcategory
FROM pg_type
WHERE typname IN ('entity_type_enum', 'relationship_type_enum');

-- Verify columns use enum types
SELECT
    table_name,
    column_name,
    udt_name
FROM information_schema.columns
WHERE table_name IN ('knowledge_entities', 'entity_relationships')
  AND column_name IN ('entity_type', 'relationship_type');

-- Test insert with valid enum value (should succeed)
-- INSERT INTO knowledge_entities (text, entity_type, confidence)
-- VALUES ('Test Entity', 'PERSON', 0.9);

-- Test insert with invalid enum value (should fail)
-- INSERT INTO knowledge_entities (text, entity_type, confidence)
-- VALUES ('Test Entity', 'INVALID_TYPE', 0.9);
-- Expected: ERROR: invalid input value for enum entity_type_enum: "INVALID_TYPE"

-- ============================================================================
-- FUTURE EXTENSION PATTERN
-- ============================================================================
--
-- To add new entity types in future migrations:
-- ALTER TYPE entity_type_enum ADD VALUE 'NEW_TYPE';
--
-- Note: Adding enum values is non-blocking (no table lock)
-- ============================================================================
