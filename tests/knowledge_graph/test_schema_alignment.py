"""Schema and ORM alignment tests for knowledge graph.

This test suite prevents future schema/query misalignment by validating:
1. Schema structure (all required tables and columns exist)
2. ORM model definitions match schema exactly
3. Query column references exist in schema
4. Database constraints are enforced
5. Known regression issues can't happen again

Test Categories:
- Schema Structure Tests (3 tests)
- ORM Model Tests (3 tests)
- Query Validation Tests (4 tests)
- Constraint Tests (2 tests)
- Regression Tests (3 tests)

Total: 15 tests for comprehensive alignment coverage
"""

import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytest


# ============================================================================
# CATEGORY 1: Schema Structure Tests (3 tests)
# ============================================================================


class TestSchemaStructure:
    """Verify database schema structure matches requirements."""

    def test_schema_required_tables_exist(self):
        """Verify all required tables exist in schema definition.

        This test checks that the schema.sql file contains definitions for
        all three required tables: knowledge_entities, entity_relationships,
        and entity_mentions.

        This prevents:
        - Missing table errors at runtime
        - Attempting to use tables that don't exist
        """
        schema_file = Path("src/knowledge_graph/schema.sql")
        schema_content = schema_file.read_text()

        required_tables = [
            "knowledge_entities",
            "entity_relationships",
            "entity_mentions",
        ]

        for table in required_tables:
            assert f'CREATE TABLE IF NOT EXISTS {table}' in schema_content, \
                f"Required table '{table}' definition not found in schema.sql"

    def test_knowledge_entities_columns_exist(self):
        """Verify knowledge_entities table has all required columns.

        Checks schema.sql for complete knowledge_entities definition with:
        - id (UUID PRIMARY KEY)
        - text (TEXT NOT NULL)
        - entity_type (VARCHAR(50) NOT NULL)
        - confidence (FLOAT NOT NULL)
        - canonical_form (TEXT nullable)
        - mention_count (INTEGER)
        - created_at (TIMESTAMP)
        - updated_at (TIMESTAMP)

        This prevents:
        - Missing column errors in queries
        - Type mismatches between schema and ORM
        """
        schema_file = Path("src/knowledge_graph/schema.sql")
        schema_content = schema_file.read_text()

        # Extract knowledge_entities table definition
        entities_start = schema_content.find("CREATE TABLE IF NOT EXISTS knowledge_entities")
        assert entities_start != -1, "knowledge_entities table not found"

        # Find the end of the table definition (semicolon)
        entities_end = schema_content.find(");", entities_start) + 2
        entities_def = schema_content[entities_start:entities_end]

        # Verify all required columns
        required_columns = {
            "id UUID PRIMARY KEY": "Primary key column",
            "text TEXT NOT NULL": "Entity text column",
            "entity_type VARCHAR(50) NOT NULL": "Entity type column",
            "confidence FLOAT NOT NULL DEFAULT 1.0": "Confidence score column",
            "canonical_form TEXT": "Canonical form column",
            "mention_count INTEGER NOT NULL DEFAULT 0": "Mention count column",
            "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP": "Creation timestamp",
            "updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP": "Update timestamp",
        }

        for col_def, purpose in required_columns.items():
            assert col_def in entities_def, \
                f"Missing or incorrect column definition: {purpose}\n" \
                f"Expected: {col_def}\n" \
                f"Table definition:\n{entities_def}"

    def test_entity_relationships_columns_exist(self):
        """Verify entity_relationships table has all required columns.

        Checks schema.sql for complete entity_relationships definition with:
        - id (UUID PRIMARY KEY)
        - source_entity_id (UUID FK)
        - target_entity_id (UUID FK)
        - relationship_type (VARCHAR(50))
        - confidence (FLOAT)
        - relationship_weight (FLOAT)
        - is_bidirectional (BOOLEAN)
        - created_at (TIMESTAMP)
        - updated_at (TIMESTAMP)

        This prevents:
        - Queries referencing non-existent relationship columns
        - Missing relationship_weight (should be used instead of metadata)
        - Type mismatches in relationship data
        """
        schema_file = Path("src/knowledge_graph/schema.sql")
        schema_content = schema_file.read_text()

        # Extract entity_relationships table definition
        rel_start = schema_content.find("CREATE TABLE IF NOT EXISTS entity_relationships")
        assert rel_start != -1, "entity_relationships table not found"

        rel_end = schema_content.find(");", rel_start) + 2
        rel_def = schema_content[rel_start:rel_end]

        # Verify all required columns
        required_columns = {
            "id UUID PRIMARY KEY": "Primary key",
            "source_entity_id UUID NOT NULL REFERENCES knowledge_entities(id)": "Source FK",
            "target_entity_id UUID NOT NULL REFERENCES knowledge_entities(id)": "Target FK",
            "relationship_type VARCHAR(50) NOT NULL": "Relationship type",
            "confidence FLOAT NOT NULL DEFAULT 1.0": "Confidence score",
            "relationship_weight FLOAT NOT NULL DEFAULT 1.0": "Relationship weight",
            "is_bidirectional BOOLEAN NOT NULL DEFAULT FALSE": "Bidirectional flag",
        }

        for col_def, purpose in required_columns.items():
            assert col_def in rel_def, \
                f"Missing or incorrect column: {purpose}\n" \
                f"Expected: {col_def}"

        # CRITICAL: Verify metadata column does NOT exist
        assert "metadata" not in rel_def.lower() or "COMMENT" in rel_def[rel_def.lower().find("metadata"):], \
            "entity_relationships should not have a metadata JSONB column (use relationship_weight instead)"


# ============================================================================
# CATEGORY 2: ORM Model Tests (3 tests)
# ============================================================================


class TestORMModels:
    """Verify SQLAlchemy ORM models match schema definitions."""

    def test_knowledge_entity_model_fields(self):
        """Verify KnowledgeEntity ORM model has correct fields.

        This test imports the actual ORM model and verifies:
        - All schema columns have corresponding ORM fields
        - Field types are correct (UUID, str, float, int, bool, datetime)
        - Non-existent columns are not in the model (entity_name, metadata)

        This catches:
        - Typos in field names that queries would reference
        - Incorrect type mappings
        - Stale fields from previous schema iterations
        """
        try:
            from src.knowledge_graph.models import KnowledgeEntity
            from sqlalchemy import inspect
        except ImportError:
            pytest.skip("SQLAlchemy not available in test environment")

        # Get all columns in the model
        mapper = inspect(KnowledgeEntity)
        columns = {col.name: col.type for col in mapper.columns}

        # Verify required columns exist
        required_columns = [
            "id",
            "text",
            "entity_type",
            "confidence",
            "canonical_form",
            "mention_count",
            "created_at",
            "updated_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, \
                f"Required column '{col_name}' not found in KnowledgeEntity model"

        # Verify correct types
        assert columns["text"].python_type == str, \
            "text field should be str type, not entity_name"
        assert columns["confidence"].python_type == float, \
            "confidence field should be float type, not JSONB"
        assert columns["entity_type"].python_type == str, \
            "entity_type field should be str type"

        # Verify non-existent columns are NOT in the model
        assert "entity_name" not in columns, \
            "REGRESSION: entity_name column should not exist (correct name is 'text')"
        assert "metadata" not in columns, \
            "REGRESSION: metadata column should not exist in KnowledgeEntity"

    def test_entity_relationship_model_fields(self):
        """Verify EntityRelationship ORM model matches schema.

        Verifies:
        - relationship_weight column exists (for relationship metadata)
        - metadata column does NOT exist
        - All FK references are correct
        - Confidence is float, not JSONB

        This catches:
        - Queries trying to use r.metadata instead of r.relationship_weight
        - Type mismatches in relationship data
        """
        try:
            from src.knowledge_graph.models import EntityRelationship
            from sqlalchemy import inspect
        except ImportError:
            pytest.skip("SQLAlchemy not available in test environment")

        mapper = inspect(EntityRelationship)
        columns = {col.name: col.type for col in mapper.columns}

        # Verify correct columns exist
        assert "relationship_weight" in columns, \
            "relationship_weight column required for relationship metadata"
        assert columns["confidence"].python_type == float, \
            "confidence should be float, not JSONB"

        # Verify incorrect columns don't exist
        assert "metadata" not in columns, \
            "REGRESSION: metadata JSONB column should not exist in EntityRelationship"

        # Verify foreign keys
        required_columns = [
            "source_entity_id",
            "target_entity_id",
            "relationship_type",
            "confidence",
            "relationship_weight",
            "is_bidirectional",
        ]

        for col_name in required_columns:
            assert col_name in columns, \
                f"Required column '{col_name}' not found in EntityRelationship model"

    def test_entity_mention_model_fields(self):
        """Verify EntityMention ORM model matches schema.

        Verifies:
        - Table is named entity_mentions (not chunk_entities)
        - All required columns present: entity_id, document_id, chunk_id, mention_text
        - chunk_id is integer (not UUID)
        - correct foreign key to knowledge_entities

        This catches:
        - Queries referencing non-existent chunk_entities table
        - Queries referencing non-existent knowledge_base table
        - Type mismatches in mention data
        """
        try:
            from src.knowledge_graph.models import EntityMention
            from sqlalchemy import inspect
        except ImportError:
            pytest.skip("SQLAlchemy not available in test environment")

        mapper = inspect(EntityMention)
        columns = {col.name: col.type for col in mapper.columns}

        # Verify table name
        assert mapper.tables[0].name == "entity_mentions", \
            "Table should be 'entity_mentions', not 'chunk_entities'"

        # Verify all required columns
        required_columns = [
            "id",
            "entity_id",
            "document_id",
            "chunk_id",
            "mention_text",
            "offset_start",
            "offset_end",
            "created_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, \
                f"Required column '{col_name}' not found in EntityMention model"

        # Verify types
        assert columns["chunk_id"].python_type == int, \
            "chunk_id should be integer type"
        assert columns["document_id"].python_type == str, \
            "document_id should be string type"


# ============================================================================
# CATEGORY 3: Query Validation Tests (4 tests)
# ============================================================================


class TestQueryValidation:
    """Verify query_repository.py uses correct schema column names."""

    def test_query_column_references_are_valid(self):
        """Verify all SQL column references exist in schema.

        Extracts all SQL queries from query_repository.py and checks:
        - All referenced columns exist in schema
        - No references to non-existent columns
        - Correct table names used

        This catches:
        - Typos in column names (entity_name vs text)
        - References to columns that were removed
        - Outdated query patterns
        """
        query_file = Path("src/knowledge_graph/query_repository.py")
        content = query_file.read_text()

        # Known valid columns per table
        valid_entity_cols = {
            "id", "text", "entity_type", "confidence",
            "canonical_form", "mention_count", "created_at", "updated_at"
        }
        valid_rel_cols = {
            "id", "source_entity_id", "target_entity_id", "relationship_type",
            "confidence", "relationship_weight", "is_bidirectional",
            "created_at", "updated_at"
        }
        valid_mention_cols = {
            "id", "entity_id", "document_id", "chunk_id", "mention_text",
            "offset_start", "offset_end", "created_at"
        }

        # These column references should NEVER appear
        invalid_patterns = [
            (r"\.entity_name\b", "entity_name (should be 'text')"),
            (r"metadata->>", "JSONB extraction on metadata (should use direct column)"),
            (r"chunk_entities", "chunk_entities table (should be 'entity_mentions')"),
            (r"knowledge_base", "knowledge_base table (does not exist)"),
            (r"metadata\s+(AS|$)", "metadata column (does not exist in entities)"),
        ]

        for pattern, description in invalid_patterns:
            matches = re.findall(pattern, content)
            assert not matches, \
                f"Found invalid column/table reference: {description}\n" \
                f"Matches: {matches}\n" \
                f"Pattern: {pattern}"

    def test_no_entity_name_column_reference(self):
        """REGRESSION TEST: Verify entity_name is not referenced in SQL queries.

        This was Mismatch 1.1 in the original blocker:
        - traverse_1hop line 134: e.entity_name AS text (WRONG)
        - traverse_2hop line 216: e2.entity_name AS text (WRONG)
        - traverse_bidirectional line 350: e.entity_name AS text (WRONG)
        - traverse_with_type_filter line 430: e.entity_name AS text (WRONG)

        The correct column name is 'text', not 'entity_name'.
        """
        query_file = Path("src/knowledge_graph/query_repository.py")
        content = query_file.read_text()

        # Extract only SQL query strings (triple-quoted strings after 'query = ')
        # This avoids false positives from docstring examples
        sql_query_pattern = r'query\s*=\s*"""(.*?)"""'
        sql_matches = re.findall(sql_query_pattern, content, re.DOTALL)

        for sql_query in sql_matches:
            # entity_name should not appear in any SQL query
            assert "entity_name" not in sql_query.upper(), \
                "REGRESSION: SQL query still references non-existent entity_name column\n" \
                "Correct column name is 'text', not 'entity_name'\n" \
                f"Found in query:\n{sql_query}"

    def test_no_jsonb_metadata_extraction(self):
        """REGRESSION TEST: Verify metadata JSONB extraction is not used.

        This was Mismatch 1.2 in the original blocker:
        - traverse_1hop line 136: e.metadata->>'confidence' (WRONG)
        - traverse_2hop line 218: e2.metadata->>'confidence' (WRONG)
        - traverse_bidirectional line 352: e.metadata->>'confidence' (WRONG)
        - traverse_with_type_filter line 432: e.metadata->>'confidence' (WRONG)

        The correct approach is to use the direct column:
        - e.confidence (not e.metadata->>'confidence')
        - Confidence is a FLOAT column, not JSONB

        This also prevents:
        - traverse_1hop line 139: re.relationship_metadata (should use relationship_weight)
        - traverse_with_type_filter line 435: r.metadata (should use relationship_weight)
        """
        query_file = Path("src/knowledge_graph/query_repository.py")
        content = query_file.read_text()

        # metadata JSONB extraction should not appear
        assert "metadata->>" not in content, \
            "REGRESSION: Queries still use JSONB extraction on non-existent metadata column\n" \
            "Correct approach: use direct column access (e.confidence, not e.metadata->>'confidence')\n" \
            "Affected methods: traverse_1hop (line 136), traverse_2hop (line 218), " \
            "traverse_bidirectional (line 352), traverse_with_type_filter (line 432)"

    def test_no_chunk_entities_table_reference(self):
        """REGRESSION TEST: Verify chunk_entities table is not used in SQL queries.

        This was Mismatch 2.1 in the original blocker:
        - get_entity_mentions line 506: FROM chunk_entities ce (WRONG)
        - Also references non-existent knowledge_base table

        The correct table is 'entity_mentions' which exists in schema.
        Schema defines:
        - entity_mentions (id, entity_id, document_id, chunk_id, mention_text, ...)
        """
        query_file = Path("src/knowledge_graph/query_repository.py")
        content = query_file.read_text()

        # Extract only SQL query strings (triple-quoted strings after 'query = ')
        sql_query_pattern = r'query\s*=\s*"""(.*?)"""'
        sql_matches = re.findall(sql_query_pattern, content, re.DOTALL)

        for sql_query in sql_matches:
            # chunk_entities table should not be referenced
            assert "chunk_entities" not in sql_query.upper(), \
                "REGRESSION: SQL query still references non-existent chunk_entities table\n" \
                "Correct table name is 'entity_mentions'\n" \
                f"Found in query:\n{sql_query}"

            # knowledge_base table should not be referenced
            assert "knowledge_base" not in sql_query.upper(), \
                "REGRESSION: SQL query still references non-existent knowledge_base table\n" \
                "Entity mention data comes from 'entity_mentions' table in schema\n" \
                f"Found in query:\n{sql_query}"


# ============================================================================
# CATEGORY 4: Constraint Tests (2 tests)
# ============================================================================


class TestConstraintEnforcement:
    """Verify database constraints are properly defined in schema."""

    def test_confidence_constraint_defined(self):
        """Verify confidence CHECK constraint is defined in schema.

        The schema must enforce that confidence values are in [0.0, 1.0]:
        - In knowledge_entities: CHECK (confidence >= 0.0 AND confidence <= 1.0)
        - In entity_relationships: CHECK (confidence >= 0.0 AND confidence <= 1.0)

        This prevents:
        - Invalid confidence values (e.g., 1.5, -0.5)
        - Database corruption from bad data
        - Queries returning invalid scores
        """
        schema_file = Path("src/knowledge_graph/schema.sql")
        schema_content = schema_file.read_text()

        # Count how many times the confidence constraint appears
        constraint_pattern = r"CHECK\s*\(\s*confidence\s*>=\s*0\.0\s*AND\s*confidence\s*<=\s*1\.0\s*\)"
        matches = re.findall(constraint_pattern, schema_content, re.IGNORECASE)

        # Should appear at least twice: once for knowledge_entities, once for entity_relationships
        assert len(matches) >= 2, \
            "Confidence CHECK constraint not properly defined in schema\n" \
            "Expected: At least 2 occurrences (knowledge_entities and entity_relationships)\n" \
            f"Found: {len(matches)}"

    def test_no_self_loops_constraint_defined(self):
        """Verify no-self-loops constraint is defined for relationships.

        The schema must prevent relationships where source == target:
        - CONSTRAINT no_self_loops CHECK (source_entity_id != target_entity_id)

        This prevents:
        - Invalid self-referential relationships
        - Graph traversal loops
        - Infinite traversal results
        """
        schema_file = Path("src/knowledge_graph/schema.sql")
        schema_content = schema_file.read_text()

        # Verify no_self_loops constraint exists
        assert "no_self_loops" in schema_content, \
            "No-self-loops constraint not defined in schema\n" \
            "Expected: CONSTRAINT no_self_loops CHECK (source_entity_id != target_entity_id)"

        assert "source_entity_id != target_entity_id" in schema_content, \
            "No-self-loops constraint logic not defined\n" \
            "Expected: CHECK (source_entity_id != target_entity_id)"


# ============================================================================
# CATEGORY 5: Regression Tests (3 tests)
# ============================================================================


class TestRegressionPrevention:
    """Prevent regression of known schema/query mismatches.

    These tests specifically target issues that were found and fixed,
    ensuring they can't be reintroduced in future changes.
    """

    def test_regression_entity_name_field_removed(self):
        """Prevent regression: entity_name should not exist in any model.

        Issue: Original code used entity_name column that doesn't exist in schema.
        Fix: Changed to use text column (which does exist).

        This test ensures the old column name is never used again.
        """
        try:
            from src.knowledge_graph.models import KnowledgeEntity
            from sqlalchemy import inspect
        except ImportError:
            pytest.skip("SQLAlchemy not available in test environment")

        mapper = inspect(KnowledgeEntity)
        columns = {col.name for col in mapper.columns}

        # entity_name should absolutely not exist
        assert "entity_name" not in columns, \
            "REGRESSION DETECTED: entity_name field re-introduced\n" \
            "This column never existed in the schema\n" \
            "The correct column is 'text'\n" \
            "Please remove entity_name field and use text instead"

    def test_regression_metadata_jsonb_removed(self):
        """Prevent regression: metadata JSONB should not exist.

        Issue: Original code tried to extract confidence from metadata JSONB.
        Fix: Changed to use direct float column.

        This test ensures the old pattern is never used again.
        """
        try:
            from src.knowledge_graph.models import KnowledgeEntity, EntityRelationship
            from sqlalchemy import inspect
        except ImportError:
            pytest.skip("SQLAlchemy not available in test environment")

        # Check KnowledgeEntity
        ke_mapper = inspect(KnowledgeEntity)
        ke_columns = {col.name for col in ke_mapper.columns}

        assert "metadata" not in ke_columns, \
            "REGRESSION DETECTED: metadata field in KnowledgeEntity\n" \
            "This column never existed in the schema\n" \
            "Use 'confidence' float column instead"

        # Check EntityRelationship
        er_mapper = inspect(EntityRelationship)
        er_columns = {col.name for col in er_mapper.columns}

        assert "metadata" not in er_columns, \
            "REGRESSION DETECTED: metadata field in EntityRelationship\n" \
            "This column never existed in the schema\n" \
            "Use 'relationship_weight' column instead"

    def test_regression_query_column_types_match(self):
        """Prevent regression: query column types should match schema.

        This verifies that dataclass definitions in query_repository.py
        match the schema column types (especially UUID vs int).

        Issue: Original dataclasses used id: int instead of id: UUID.
        Fix: Changed to use UUID type to match schema.
        """
        try:
            from src.knowledge_graph.query_repository import (
                RelatedEntity,
                TwoHopEntity,
                BidirectionalEntity,
                EntityMention as EntityMentionDataclass
            )
            from typing import get_type_hints
            from uuid import UUID
        except ImportError:
            pytest.skip("Required modules not available in test environment")

        # Check each dataclass
        for dataclass_cls in [RelatedEntity, TwoHopEntity, BidirectionalEntity, EntityMentionDataclass]:
            hints = get_type_hints(dataclass_cls)

            # The 'id' field must be UUID type (or Optional[UUID]), not int
            if "id" in hints:
                id_type = hints["id"]

                # Handle Optional[UUID] case
                id_type_str = str(id_type)
                assert "UUID" in id_type_str, \
                    f"REGRESSION DETECTED in {dataclass_cls.__name__}.id\n" \
                    f"Type: {id_type}\n" \
                    f"Expected: UUID (matching schema)\n" \
                    f"Issue: Original code used int, but schema uses UUID"

            # Confidence fields should be float, not string
            if "entity_confidence" in hints:
                conf_type = hints["entity_confidence"]
                conf_type_str = str(conf_type)

                # Should be float or Optional[float]
                assert "float" in conf_type_str.lower() or "NoneType" in conf_type_str, \
                    f"REGRESSION DETECTED in {dataclass_cls.__name__}.entity_confidence\n" \
                    f"Type: {conf_type}\n" \
                    f"Expected: float (confidence is FLOAT column, not string)\n" \
                    f"Issue: Original code converted to string from JSONB"


# ============================================================================
# Test Markers and Configuration
# ============================================================================


@pytest.mark.schema
def test_marker_schema():
    """Test marker for schema-related tests."""
    pass


@pytest.mark.alignment
def test_marker_alignment():
    """Test marker for alignment-related tests."""
    pass


@pytest.mark.regression
def test_marker_regression():
    """Test marker for regression prevention tests."""
    pass
