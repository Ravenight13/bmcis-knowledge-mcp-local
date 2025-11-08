-- Migration: Add metadata JSONB column to knowledge_base table
-- Date: 2025-11-08
-- Description: Adds metadata column for storing additional chunk metadata as JSON

-- Add metadata column if it doesn't exist
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

-- Create GIN index for JSONB metadata searches if not exists
CREATE INDEX IF NOT EXISTS idx_knowledge_metadata ON knowledge_base USING GIN(metadata);

-- Verify column was added
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_base'
        AND column_name = 'metadata'
    ) THEN
        RAISE NOTICE 'Migration successful: metadata column exists';
    ELSE
        RAISE EXCEPTION 'Migration failed: metadata column not found';
    END IF;
END $$;
