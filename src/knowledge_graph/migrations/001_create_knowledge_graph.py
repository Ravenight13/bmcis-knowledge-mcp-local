"""Migration 001: Create knowledge graph schema (entities, relationships, mentions).

This migration creates the initial normalized PostgreSQL schema for the knowledge
graph, including:
1. knowledge_entities table with UUID primary keys
2. entity_relationships table with typed edges and constraints
3. entity_mentions table for provenance tracking
4. Indexes optimized for 1-hop and 2-hop queries
5. Triggers for automatic timestamp updates

This migration is idempotent: it uses CREATE TABLE IF NOT EXISTS and CREATE INDEX
IF NOT EXISTS, so it's safe to apply multiple times.

Revision: 001
Date: 2025-11-09
Task: 7.3 Phase 1: Normalized PostgreSQL Schema
"""

from typing import Any

# SQL statements for migration
UP_SQL: list[str] = [
    # Enable UUID extension
    """CREATE EXTENSION IF NOT EXISTS "uuid-ossp";""",

    # Create knowledge_entities table
    """
    CREATE TABLE IF NOT EXISTS knowledge_entities (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        text TEXT NOT NULL,
        entity_type VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
        canonical_form TEXT,
        mention_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(text, entity_type)
    );
    """,

    # Indexes for knowledge_entities
    """CREATE INDEX IF NOT EXISTS idx_knowledge_entities_text ON knowledge_entities(text);""",
    """CREATE INDEX IF NOT EXISTS idx_knowledge_entities_type ON knowledge_entities(entity_type);""",
    """CREATE INDEX IF NOT EXISTS idx_knowledge_entities_canonical ON knowledge_entities(canonical_form);""",
    """CREATE INDEX IF NOT EXISTS idx_knowledge_entities_mention_count ON knowledge_entities(mention_count DESC);""",

    # Create entity_relationships table
    """
    CREATE TABLE IF NOT EXISTS entity_relationships (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
        target_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
        relationship_type VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
        relationship_weight FLOAT NOT NULL DEFAULT 1.0,
        is_bidirectional BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT no_self_loops CHECK (source_entity_id != target_entity_id),
        UNIQUE(source_entity_id, target_entity_id, relationship_type)
    );
    """,

    # Indexes for entity_relationships
    """CREATE INDEX IF NOT EXISTS idx_entity_relationships_source ON entity_relationships(source_entity_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_relationships_target ON entity_relationships(target_entity_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_relationships_type ON entity_relationships(relationship_type);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_relationships_graph ON entity_relationships(source_entity_id, relationship_type, target_entity_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_relationships_bidirectional ON entity_relationships(is_bidirectional);""",

    # Create entity_mentions table
    """
    CREATE TABLE IF NOT EXISTS entity_mentions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
        document_id VARCHAR(255) NOT NULL,
        chunk_id INTEGER NOT NULL,
        mention_text TEXT NOT NULL,
        offset_start INTEGER,
        offset_end INTEGER,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,

    # Indexes for entity_mentions
    """CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_mentions_document ON entity_mentions(document_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk ON entity_mentions(document_id, chunk_id);""",
    """CREATE INDEX IF NOT EXISTS idx_entity_mentions_composite ON entity_mentions(entity_id, document_id);""",

    # Create trigger functions
    """
    CREATE OR REPLACE FUNCTION update_knowledge_entity_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at := CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """,

    """
    CREATE TRIGGER IF NOT EXISTS trigger_update_knowledge_entity_timestamp
    BEFORE UPDATE ON knowledge_entities
    FOR EACH ROW
    EXECUTE FUNCTION update_knowledge_entity_timestamp();
    """,

    """
    CREATE OR REPLACE FUNCTION update_entity_relationship_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at := CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """,

    """
    CREATE TRIGGER IF NOT EXISTS trigger_update_entity_relationship_timestamp
    BEFORE UPDATE ON entity_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_entity_relationship_timestamp();
    """,
]

DOWN_SQL: list[str] = [
    # Drop triggers
    """DROP TRIGGER IF EXISTS trigger_update_entity_relationship_timestamp ON entity_relationships;""",
    """DROP TRIGGER IF EXISTS trigger_update_knowledge_entity_timestamp ON knowledge_entities;""",

    # Drop functions
    """DROP FUNCTION IF EXISTS update_entity_relationship_timestamp();""",
    """DROP FUNCTION IF EXISTS update_knowledge_entity_timestamp();""",

    # Drop tables (in reverse dependency order)
    """DROP TABLE IF EXISTS entity_mentions;""",
    """DROP TABLE IF EXISTS entity_relationships;""",
    """DROP TABLE IF EXISTS knowledge_entities;""",

    # Drop extension
    """DROP EXTENSION IF EXISTS "uuid-ossp";""",
]


def upgrade(connection: Any) -> None:
    """Apply migration (create tables and indexes).

    Args:
        connection: Database connection object with execute() method.

    Raises:
        Exception: If any SQL statement fails.
    """
    for sql_statement in UP_SQL:
        # Skip empty statements
        if sql_statement.strip():
            connection.execute(sql_statement)


def downgrade(connection: Any) -> None:
    """Reverse migration (drop tables and functions).

    Args:
        connection: Database connection object with execute() method.

    Raises:
        Exception: If any SQL statement fails.
    """
    for sql_statement in DOWN_SQL:
        # Skip empty statements
        if sql_statement.strip():
            connection.execute(sql_statement)
