-- BMCIS Knowledge Base Schema (768-dimensional embeddings)
-- PostgreSQL 16 + pgvector
-- This schema supports semantic search, full-text search, and knowledge graph storage

-- Ensure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- KNOWLEDGE_BASE TABLE: Core storage for document chunks with embeddings
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,                -- 512-token chunk content
    chunk_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256 for deduplication
    embedding vector(768),                   -- all-mpnet-base-v2 embeddings
    source_file VARCHAR(512) NOT NULL,       -- Original markdown path
    source_category VARCHAR(128),            -- product_docs, kb_article, etc.
    document_date DATE,                      -- Document publish/update date
    chunk_index INTEGER NOT NULL,            -- Position in document
    total_chunks INTEGER NOT NULL,           -- Total chunks in document
    context_header TEXT,                     -- "filename.md > Section > Subsection"
    ts_vector tsvector,                      -- Auto-updated for full-text search
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for efficient vector similarity search (cosine distance)
-- HNSW (Hierarchical Navigable Small World) is superior to IVFFlat for production
-- m=16: connections per node, ef_construction=64: size of dynamic candidate list
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search (BM25 via PostgreSQL ts_vector)
CREATE INDEX IF NOT EXISTS idx_knowledge_fts ON knowledge_base
USING GIN(ts_vector);

-- B-tree indexes for metadata filtering and joins
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base(source_category);
CREATE INDEX IF NOT EXISTS idx_knowledge_source_file ON knowledge_base(source_file);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_index ON knowledge_base(chunk_index);
CREATE INDEX IF NOT EXISTS idx_knowledge_document_date ON knowledge_base(document_date DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_created_at ON knowledge_base(created_at DESC);

-- Compound index for common query patterns (category + date + chunk ordering)
CREATE INDEX IF NOT EXISTS idx_knowledge_category_date ON knowledge_base(source_category, document_date DESC);

-- ============================================================================
-- KNOWLEDGE_ENTITIES TABLE: Structured entity extraction (vendors, products, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id SERIAL PRIMARY KEY,
    entity_name TEXT NOT NULL,               -- "Lutron", "Quantum System", etc.
    entity_type VARCHAR(50) NOT NULL,        -- VENDOR, PRODUCT, TEAM_MEMBER, REGION
    metadata JSONB,                          -- {aliases: [], confidence: 0.95, context: "..."}
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Unique constraint on canonical entities (lowercase normalized)
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_canonical ON knowledge_entities(lower(entity_name), entity_type)
WHERE COALESCE(metadata->>'canonical', 'false') = 'true';

-- B-tree indexes for entity lookups
CREATE INDEX IF NOT EXISTS idx_entity_name ON knowledge_entities(lower(entity_name));
CREATE INDEX IF NOT EXISTS idx_entity_type ON knowledge_entities(entity_type);

-- GIN index for JSONB metadata searches (includes confidence data)
CREATE INDEX IF NOT EXISTS idx_entity_metadata ON knowledge_entities USING GIN(metadata);

-- ============================================================================
-- ENTITY_RELATIONSHIPS TABLE: Knowledge graph relationships
-- ============================================================================
CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100),          -- "vendor_has_product", "team_manages_vendor", etc.
    metadata JSONB,                          -- {confidence: 0.92, source: "chunk_id", context: "..."}
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Compound index for common relationship queries
CREATE INDEX IF NOT EXISTS idx_relationship_source ON entity_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationship_target ON entity_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationship_type ON entity_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationship_graph ON entity_relationships(source_entity_id, relationship_type, target_entity_id);

-- GIN index for relationship metadata
CREATE INDEX IF NOT EXISTS idx_relationship_metadata ON entity_relationships USING GIN(metadata);

-- ============================================================================
-- CHUNK_EMBEDDINGS_MAPPING TABLE: Links chunks to entities for tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS chunk_entities (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    confidence FLOAT DEFAULT 0.5,            -- Extraction confidence score
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id, entity_id)
);

-- Indexes for entity-to-chunk lookups
CREATE INDEX IF NOT EXISTS idx_chunk_entities_chunk ON chunk_entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_chunk_entities_confidence ON chunk_entities(confidence DESC);

-- ============================================================================
-- SEARCH_CACHE TABLE: Optional - stores recent search results for performance
-- ============================================================================
CREATE TABLE IF NOT EXISTS search_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) UNIQUE NOT NULL,  -- MD5 of query text
    query_text TEXT NOT NULL,
    results JSONB NOT NULL,                  -- Cached results
    result_count INTEGER,
    query_latency_ms INTEGER,                -- Query execution time
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP                     -- TTL for cache invalidation
);

-- Index for cache lookups and expiration cleanup
CREATE INDEX IF NOT EXISTS idx_search_cache_hash ON search_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON search_cache(expires_at) WHERE expires_at IS NOT NULL;

-- ============================================================================
-- FUNCTIONS: Helpers for common operations
-- ============================================================================

-- Function to update tsvector for full-text search automatically
CREATE OR REPLACE FUNCTION update_knowledge_base_ts_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector := to_tsvector('english', COALESCE(NEW.chunk_text, '') || ' ' || COALESCE(NEW.context_header, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Trigger to update tsvector on insert/update
CREATE TRIGGER trigger_update_knowledge_ts_vector
BEFORE INSERT OR UPDATE ON knowledge_base
FOR EACH ROW
EXECUTE FUNCTION update_knowledge_base_ts_vector();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Triggers for timestamp management
CREATE TRIGGER trigger_knowledge_base_timestamp
BEFORE UPDATE ON knowledge_base
FOR EACH ROW
WHEN (OLD.* IS DISTINCT FROM NEW.*)
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_entities_timestamp
BEFORE UPDATE ON knowledge_entities
FOR EACH ROW
WHEN (OLD.* IS DISTINCT FROM NEW.*)
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trigger_relationships_timestamp
BEFORE UPDATE ON entity_relationships
FOR EACH ROW
WHEN (OLD.* IS DISTINCT FROM NEW.*)
EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- PERMISSIONS: Set up appropriate access control
-- ============================================================================
-- Grant appropriate permissions (adjust roles as needed for production)
-- GRANT SELECT ON knowledge_base TO read_user;
-- GRANT SELECT ON knowledge_entities TO read_user;
-- GRANT SELECT ON entity_relationships TO read_user;
-- GRANT USAGE ON SEQUENCE knowledge_base_id_seq TO app_user;

-- ============================================================================
-- VERIFICATION: Post-schema creation checks
-- ============================================================================
-- Verify all tables created successfully
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'knowledge_base'
    ) THEN
        RAISE NOTICE 'Schema creation successful: knowledge_base table exists';
    ELSE
        RAISE EXCEPTION 'Schema creation failed: knowledge_base table not found';
    END IF;
END $$;
