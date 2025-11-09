-- Migration: Add metadata column and optimize BM25 full-text search
-- Date: 2025-11-08
-- Description: Adds metadata JSONB column and chunk_token_count for BM25 search optimization

-- 1. Add metadata column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base'
        AND column_name = 'metadata'
    ) THEN
        ALTER TABLE knowledge_base ADD COLUMN metadata JSONB DEFAULT '{}';
        RAISE NOTICE 'Added metadata column to knowledge_base table';
    ELSE
        RAISE NOTICE 'metadata column already exists in knowledge_base table';
    END IF;
END $$;

-- 2. Add chunk_token_count column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base'
        AND column_name = 'chunk_token_count'
    ) THEN
        ALTER TABLE knowledge_base ADD COLUMN chunk_token_count INTEGER;
        RAISE NOTICE 'Added chunk_token_count column to knowledge_base table';
    ELSE
        RAISE NOTICE 'chunk_token_count column already exists in knowledge_base table';
    END IF;
END $$;

-- 3. Create GIN index for JSONB metadata searches if not exists
CREATE INDEX IF NOT EXISTS idx_knowledge_metadata ON knowledge_base USING GIN(metadata);

-- 4. Verify ts_vector column and GIN index exist (should already exist)
-- The ts_vector column and idx_knowledge_fts index are already present in the schema
-- This migration ensures they exist and are properly configured

-- 5. Create optimized BM25 search function
-- This function uses ts_rank_cd with normalization for better relevance ranking
CREATE OR REPLACE FUNCTION search_bm25(
    query_text TEXT,
    top_k INTEGER DEFAULT 10,
    category_filter TEXT DEFAULT NULL,
    min_score FLOAT DEFAULT 0.0
) RETURNS TABLE (
    id INTEGER,
    chunk_text TEXT,
    context_header TEXT,
    source_file VARCHAR(512),
    source_category VARCHAR(128),
    document_date DATE,
    chunk_index INTEGER,
    total_chunks INTEGER,
    chunk_token_count INTEGER,
    metadata JSONB,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.chunk_text,
        kb.context_header,
        kb.source_file,
        kb.source_category,
        kb.document_date,
        kb.chunk_index,
        kb.total_chunks,
        kb.chunk_token_count,
        kb.metadata,
        -- Normalize ts_rank_cd score to 0-1 range
        -- ts_rank_cd uses cover density ranking (more sophisticated than ts_rank)
        -- Normalization: 1 = divide by document length, 2 = divide by log(length), 4 = harmonic distance
        ts_rank_cd(kb.ts_vector, plainto_tsquery('english', query_text), 1 | 2) AS similarity
    FROM knowledge_base kb
    WHERE
        kb.ts_vector @@ plainto_tsquery('english', query_text)
        AND (category_filter IS NULL OR kb.source_category = category_filter)
        AND ts_rank_cd(kb.ts_vector, plainto_tsquery('english', query_text), 1 | 2) >= min_score
    ORDER BY similarity DESC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- 6. Create phrase search function for exact matches
CREATE OR REPLACE FUNCTION search_bm25_phrase(
    phrase TEXT,
    top_k INTEGER DEFAULT 10,
    category_filter TEXT DEFAULT NULL
) RETURNS TABLE (
    id INTEGER,
    chunk_text TEXT,
    context_header TEXT,
    source_file VARCHAR(512),
    source_category VARCHAR(128),
    document_date DATE,
    chunk_index INTEGER,
    total_chunks INTEGER,
    chunk_token_count INTEGER,
    metadata JSONB,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.chunk_text,
        kb.context_header,
        kb.source_file,
        kb.source_category,
        kb.document_date,
        kb.chunk_index,
        kb.total_chunks,
        kb.chunk_token_count,
        kb.metadata,
        ts_rank_cd(kb.ts_vector, phraseto_tsquery('english', phrase), 1 | 2) AS similarity
    FROM knowledge_base kb
    WHERE
        kb.ts_vector @@ phraseto_tsquery('english', phrase)
        AND (category_filter IS NULL OR kb.source_category = category_filter)
    ORDER BY similarity DESC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- 7. Verify migration success
DO $$
DECLARE
    metadata_exists BOOLEAN;
    token_count_exists BOOLEAN;
    ts_vector_exists BOOLEAN;
    gin_index_exists BOOLEAN;
BEGIN
    -- Check metadata column
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base' AND column_name = 'metadata'
    ) INTO metadata_exists;

    -- Check chunk_token_count column
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base' AND column_name = 'chunk_token_count'
    ) INTO token_count_exists;

    -- Check ts_vector column
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base' AND column_name = 'ts_vector'
    ) INTO ts_vector_exists;

    -- Check GIN indexes
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'knowledge_base' AND indexname = 'idx_knowledge_fts'
    ) INTO gin_index_exists;

    -- Report status
    IF metadata_exists AND token_count_exists AND ts_vector_exists AND gin_index_exists THEN
        RAISE NOTICE 'Migration successful: All required columns and indexes exist';
        RAISE NOTICE '  ✓ metadata column: %', metadata_exists;
        RAISE NOTICE '  ✓ chunk_token_count column: %', token_count_exists;
        RAISE NOTICE '  ✓ ts_vector column: %', ts_vector_exists;
        RAISE NOTICE '  ✓ idx_knowledge_fts index: %', gin_index_exists;
    ELSE
        RAISE EXCEPTION 'Migration failed: Missing required columns or indexes';
    END IF;
END $$;

-- 8. Performance analysis query (for manual testing)
-- EXPLAIN (ANALYZE, BUFFERS)
-- SELECT * FROM search_bm25('authentication', 10);
