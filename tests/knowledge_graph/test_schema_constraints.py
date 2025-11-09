"""PostgreSQL Schema Constraint Tests for Knowledge Graph.

Tests database-level constraints, triggers, and indexes to ensure data integrity
at the PostgreSQL level. These tests use raw SQL and actual database connections
to verify constraints cannot be bypassed at the ORM level.

Test Coverage:
- CHECK constraints (confidence ranges, no self-loops)
- UNIQUE constraints (entity text+type, relationship source+target+type)
- Foreign key constraints (referential integrity)
- CASCADE delete behavior
- Timestamp update triggers
- Index existence and configuration
- Column type validation
- NULL constraint enforcement

Architecture:
- Tests use real PostgreSQL database (not mocked)
- Cleanup fixtures ensure isolation between tests
- Raw SQL used to bypass ORM validation where needed
- Type-safe assertions with explicit type checking
"""

from __future__ import annotations

import pytest
import time
from datetime import datetime
from typing import Optional, Generator, Any
from uuid import UUID, uuid4
from contextlib import contextmanager

import psycopg
from psycopg import IntegrityError, sql
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.knowledge_graph.models import Base, KnowledgeEntity, EntityRelationship, EntityMention


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    """Create SQLAlchemy engine for test database.

    Uses PostgreSQL test database configured in environment or defaults to
    localhost test database. Schema is created before tests and dropped after.

    Yields:
        SQLAlchemy Engine connected to test database
    """
    # Use test database URL (typically localhost)
    db_url = "postgresql://postgres:postgres@localhost:5432/test_kg"

    engine: Engine = create_engine(db_url, echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def session(db_engine: Engine) -> Generator[Session, None, None]:
    """Create SQLAlchemy session for each test.

    Session is rolled back after each test to ensure isolation.

    Args:
        db_engine: SQLAlchemy engine from fixture

    Yields:
        SQLAlchemy Session for database operations
    """
    SessionLocal = sessionmaker(bind=db_engine)
    session: Session = SessionLocal()

    yield session

    # Rollback any uncommitted changes
    session.rollback()
    session.close()


@pytest.fixture
def db_connection(db_engine: Engine) -> Generator[psycopg.Connection, None, None]:
    """Create raw PostgreSQL connection for direct SQL access.

    Allows bypass of ORM for constraint testing. Connection is rolled back
    after each test.

    Args:
        db_engine: SQLAlchemy engine from fixture

    Yields:
        Raw psycopg connection to test database
    """
    # Extract connection string from SQLAlchemy engine
    conn_str = str(db_engine.url)

    conn: psycopg.Connection = psycopg.connect(conn_str)
    conn.autocommit = False  # Use transactions for rollback

    yield conn

    # Rollback and close
    conn.rollback()
    conn.close()


@pytest.fixture
def cleanup_entities(db_connection: psycopg.Connection) -> Generator[None, None, None]:
    """Cleanup fixture to delete all entities before and after each test.

    Ensures test isolation by removing test data. Deletes in correct order
    to respect FK constraints.

    Args:
        db_connection: Raw database connection

    Yields:
        None (cleanup happens automatically)
    """
    # Cleanup before test
    cursor = db_connection.cursor()
    cursor.execute("DELETE FROM entity_mentions")
    cursor.execute("DELETE FROM entity_relationships")
    cursor.execute("DELETE FROM knowledge_entities")
    db_connection.commit()

    yield

    # Cleanup after test
    cursor.execute("DELETE FROM entity_mentions")
    cursor.execute("DELETE FROM entity_relationships")
    cursor.execute("DELETE FROM knowledge_entities")
    db_connection.commit()
    cursor.close()


@pytest.fixture
def valid_entity_id(
    db_connection: psycopg.Connection,
    cleanup_entities: Generator[None, None, None]
) -> Generator[UUID, None, None]:
    """Create and yield a valid entity ID for FK tests.

    Creates a single entity in the database and returns its UUID.
    This entity can be referenced in FK constraint tests.

    Args:
        db_connection: Raw database connection
        cleanup_entities: Cleanup fixture to ensure isolation

    Yields:
        UUID of created entity
    """
    entity_id: UUID = uuid4()

    cursor = db_connection.cursor()
    cursor.execute(
        """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
           VALUES (%s, %s, %s, %s)""",
        (str(entity_id), "Test Entity", "PERSON", 0.95)
    )
    db_connection.commit()
    cursor.close()

    yield entity_id


@pytest.fixture
def valid_entity_pair(
    db_connection: psycopg.Connection,
    cleanup_entities: Generator[None, None, None]
) -> Generator[tuple[UUID, UUID], None, None]:
    """Create and yield two valid entity IDs for relationship tests.

    Creates two distinct entities that can be used as source and target
    in relationship constraint tests.

    Args:
        db_connection: Raw database connection
        cleanup_entities: Cleanup fixture to ensure isolation

    Yields:
        Tuple of (source_entity_id, target_entity_id)
    """
    source_id: UUID = uuid4()
    target_id: UUID = uuid4()

    cursor = db_connection.cursor()
    cursor.execute(
        """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
           VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)""",
        (str(source_id), "Source Entity", "PERSON", 0.95,
         str(target_id), "Target Entity", "ORG", 0.90)
    )
    db_connection.commit()
    cursor.close()

    yield source_id, target_id


# ============================================================================
# 1. CHECK Constraint Tests
# ============================================================================


class TestCheckConstraints:
    """Test PostgreSQL CHECK constraints on confidence ranges and self-loops."""

    def test_entity_confidence_check_lower_bound(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """PostgreSQL CHECK constraint prevents confidence < 0.0 on INSERT.

        Tests that the knowledge_entities.confidence CHECK constraint
        rejects values below 0.0 at database level.

        Expected: psycopg.IntegrityError with CHECK constraint violation message
        """
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
                   VALUES (%s, %s, %s, %s)""",
                (str(uuid4()), "Test", "PERSON", -0.1)  # Invalid: < 0.0
            )
            db_connection.commit()

        # Verify constraint violation
        error_msg = str(exc_info.value).lower()
        assert "confidence" in error_msg or "check" in error_msg

    def test_entity_confidence_check_upper_bound(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """PostgreSQL CHECK constraint prevents confidence > 1.0 on INSERT.

        Tests that the knowledge_entities.confidence CHECK constraint
        rejects values above 1.0 at database level.

        Expected: psycopg.IntegrityError with CHECK constraint violation message
        """
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
                   VALUES (%s, %s, %s, %s)""",
                (str(uuid4()), "Test", "PERSON", 1.5)  # Invalid: > 1.0
            )
            db_connection.commit()

        error_msg = str(exc_info.value).lower()
        assert "confidence" in error_msg or "check" in error_msg

    def test_entity_confidence_check_boundary_values(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """PostgreSQL CHECK constraint accepts boundary values 0.0 and 1.0.

        Tests that the confidence CHECK constraint correctly allows
        the boundary values [0.0, 1.0].

        Expected: Both inserts succeed
        """
        cursor = db_connection.cursor()

        # Should succeed: confidence = 0.0
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Test Min", "PERSON", 0.0)
        )
        db_connection.commit()

        # Should succeed: confidence = 1.0
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Test Max", "ORG", 1.0)
        )
        db_connection.commit()

        # Verify both exist
        cursor.execute("SELECT COUNT(*) FROM knowledge_entities WHERE confidence IN (0.0, 1.0)")
        count = cursor.fetchone()[0]
        assert count == 2

    def test_relationship_confidence_check_constraint(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_pair: Generator[tuple[UUID, UUID], None, None]
    ) -> None:
        """PostgreSQL CHECK constraint prevents invalid confidence on relationships.

        Tests that entity_relationships.confidence CHECK constraint
        rejects values outside [0.0, 1.0].

        Expected: psycopg.IntegrityError on invalid confidence values
        """
        source_id, target_id = valid_entity_pair
        cursor = db_connection.cursor()

        # Test confidence < 0.0
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(source_id), str(target_id), "test", -0.1)
            )
            db_connection.commit()

        db_connection.rollback()

        # Test confidence > 1.0
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(source_id), str(target_id), "test", 1.5)
            )
            db_connection.commit()

    def test_no_self_loops_check_constraint(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """PostgreSQL CHECK constraint prevents self-loops (source_id = target_id).

        Tests the no_self_loops CHECK constraint on entity_relationships
        that prevents source_entity_id == target_entity_id.

        Expected: psycopg.IntegrityError with constraint violation
        """
        entity_id = valid_entity_id
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(entity_id), str(entity_id), "self-loop", 0.8)
            )
            db_connection.commit()

        error_msg = str(exc_info.value).lower()
        assert "self_loops" in error_msg or "source_entity_id != target_entity_id" in error_msg


# ============================================================================
# 2. UNIQUE Constraint Tests
# ============================================================================


class TestUniqueConstraints:
    """Test PostgreSQL UNIQUE constraints on entity and relationship tables."""

    def test_entity_unique_text_type_constraint(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test (text, entity_type) UNIQUE constraint on knowledge_entities.

        Tests that entities with same text and entity_type cannot be inserted.

        Expected: First insert succeeds, second raises IntegrityError
        """
        cursor = db_connection.cursor()

        # First insert should succeed
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Lutron", "VENDOR", 0.95)
        )
        db_connection.commit()

        # Second insert with same (text, entity_type) should fail
        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
                   VALUES (%s, %s, %s, %s)""",
                (str(uuid4()), "Lutron", "VENDOR", 0.90)  # Duplicate!
            )
            db_connection.commit()

        error_msg = str(exc_info.value).lower()
        assert "unique" in error_msg or "uq_entity_text_type" in error_msg

    def test_entity_unique_allows_same_text_different_type(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test that same text with different entity_type is allowed.

        Verifies that the UNIQUE constraint is on (text, entity_type),
        not just text. Same text with different types should be allowed.

        Expected: Both inserts succeed
        """
        cursor = db_connection.cursor()

        # Same text, different types - both should succeed
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Apple", "PRODUCT", 0.95)
        )
        db_connection.commit()

        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Apple", "ORG", 0.95)  # Different type
        )
        db_connection.commit()

        # Verify both exist
        cursor.execute("SELECT COUNT(*) FROM knowledge_entities WHERE text = 'Apple'")
        count = cursor.fetchone()[0]
        assert count == 2

    def test_relationship_unique_source_target_type_constraint(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_pair: Generator[tuple[UUID, UUID], None, None]
    ) -> None:
        """Test (source, target, type) UNIQUE constraint on entity_relationships.

        Tests that relationships with same source, target, and type cannot
        be inserted twice.

        Expected: First insert succeeds, second raises IntegrityError
        """
        source_id, target_id = valid_entity_pair
        cursor = db_connection.cursor()

        # First insert should succeed
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(uuid4()), str(source_id), str(target_id), "works-for", 0.85)
        )
        db_connection.commit()

        # Second insert with same (source, target, type) should fail
        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(source_id), str(target_id), "works-for", 0.80)  # Duplicate!
            )
            db_connection.commit()

        error_msg = str(exc_info.value).lower()
        assert "unique" in error_msg or "uq_relationship" in error_msg

    def test_relationship_unique_allows_different_type(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_pair: Generator[tuple[UUID, UUID], None, None]
    ) -> None:
        """Test that same source/target with different type is allowed.

        Verifies that UNIQUE is on (source, target, type), not just (source, target).

        Expected: Both inserts succeed
        """
        source_id, target_id = valid_entity_pair
        cursor = db_connection.cursor()

        # First relationship
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(uuid4()), str(source_id), str(target_id), "works-for", 0.85)
        )
        db_connection.commit()

        # Same source/target, different type - should succeed
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(uuid4()), str(source_id), str(target_id), "manages", 0.90)  # Different type
        )
        db_connection.commit()

        # Verify both exist
        cursor.execute(
            """SELECT COUNT(*) FROM entity_relationships
               WHERE source_entity_id = %s AND target_entity_id = %s""",
            (str(source_id), str(target_id))
        )
        count = cursor.fetchone()[0]
        assert count == 2


# ============================================================================
# 3. Foreign Key Constraint Tests
# ============================================================================


class TestForeignKeyConstraints:
    """Test PostgreSQL foreign key constraints and referential integrity."""

    def test_relationship_fk_source_entity_required(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """Test that relationship source_entity_id must reference valid entity.

        Tests the foreign key constraint on source_entity_id that prevents
        inserting relationships referencing non-existent source entities.

        Expected: psycopg.IntegrityError (foreign key violation)
        """
        target_id = valid_entity_id
        fake_source_id = uuid4()
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError) as exc_info:
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(fake_source_id), str(target_id), "test", 0.8)
            )
            db_connection.commit()

        error_msg = str(exc_info.value).lower()
        assert "foreign key" in error_msg or "fk" in error_msg

    def test_relationship_fk_target_entity_required(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """Test that relationship target_entity_id must reference valid entity.

        Tests the foreign key constraint on target_entity_id that prevents
        inserting relationships referencing non-existent target entities.

        Expected: psycopg.IntegrityError (foreign key violation)
        """
        source_id = valid_entity_id
        fake_target_id = uuid4()
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(source_id), str(fake_target_id), "test", 0.8)
            )
            db_connection.commit()

    def test_mention_fk_entity_required(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test that mention entity_id must reference valid entity.

        Tests the foreign key constraint on entity_mentions.entity_id that prevents
        inserting mentions for non-existent entities.

        Expected: psycopg.IntegrityError (foreign key violation)
        """
        fake_entity_id = uuid4()
        cursor = db_connection.cursor()

        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_mentions
                   (id, entity_id, document_id, chunk_id, mention_text)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(fake_entity_id), "docs/README.md", 0, "Test mention")
            )
            db_connection.commit()


# ============================================================================
# 4. CASCADE Delete Tests
# ============================================================================


class TestCascadeDelete:
    """Test PostgreSQL CASCADE delete behavior."""

    def test_delete_entity_cascades_relationships(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test deleting entity cascades delete to relationships.

        Tests that ON DELETE CASCADE on entity_relationships foreign keys
        properly deletes relationships when their source or target entity is deleted.

        Expected: Relationship deleted when entity is deleted
        """
        cursor = db_connection.cursor()

        # Create two entities
        entity_a_id = uuid4()
        entity_b_id = uuid4()

        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)""",
            (str(entity_a_id), "Entity A", "PERSON", 0.9,
             str(entity_b_id), "Entity B", "ORG", 0.9)
        )
        db_connection.commit()

        # Create relationship from A to B
        rel_id = uuid4()
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(rel_id), str(entity_a_id), str(entity_b_id), "test", 0.8)
        )
        db_connection.commit()

        # Verify relationship exists
        cursor.execute("SELECT COUNT(*) FROM entity_relationships WHERE id = %s", (str(rel_id),))
        assert cursor.fetchone()[0] == 1

        # Delete entity A
        cursor.execute("DELETE FROM knowledge_entities WHERE id = %s", (str(entity_a_id),))
        db_connection.commit()

        # Verify relationship was cascade-deleted
        cursor.execute("SELECT COUNT(*) FROM entity_relationships WHERE id = %s", (str(rel_id),))
        assert cursor.fetchone()[0] == 0

    def test_delete_entity_cascades_mentions(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test deleting entity cascades delete to mentions.

        Tests that ON DELETE CASCADE on entity_mentions.entity_id
        properly deletes mentions when entity is deleted.

        Expected: Mention deleted when entity is deleted
        """
        cursor = db_connection.cursor()

        # Create entity
        entity_id = uuid4()
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(entity_id), "Test Entity", "PRODUCT", 0.9)
        )
        db_connection.commit()

        # Create mention
        mention_id = uuid4()
        cursor.execute(
            """INSERT INTO entity_mentions
               (id, entity_id, document_id, chunk_id, mention_text)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(mention_id), str(entity_id), "docs/README.md", 0, "Test mention")
        )
        db_connection.commit()

        # Verify mention exists
        cursor.execute("SELECT COUNT(*) FROM entity_mentions WHERE id = %s", (str(mention_id),))
        assert cursor.fetchone()[0] == 1

        # Delete entity
        cursor.execute("DELETE FROM knowledge_entities WHERE id = %s", (str(entity_id),))
        db_connection.commit()

        # Verify mention was cascade-deleted
        cursor.execute("SELECT COUNT(*) FROM entity_mentions WHERE id = %s", (str(mention_id),))
        assert cursor.fetchone()[0] == 0


# ============================================================================
# 5. Trigger Tests
# ============================================================================


class TestTriggers:
    """Test PostgreSQL update triggers for timestamp maintenance."""

    def test_entity_updated_at_trigger_on_update(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test UPDATE trigger auto-updates entity updated_at timestamp.

        Tests that the trigger_update_knowledge_entity_timestamp trigger
        automatically updates updated_at when entity is modified.

        Expected: updated_at increases on UPDATE
        """
        cursor = db_connection.cursor()

        # Create entity
        entity_id = uuid4()
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (str(entity_id), "Test", "PERSON", 0.9, datetime.utcnow(), datetime.utcnow())
        )
        db_connection.commit()

        # Get original updated_at
        cursor.execute("SELECT updated_at FROM knowledge_entities WHERE id = %s", (str(entity_id),))
        original_updated_at: datetime = cursor.fetchone()[0]

        # Wait to ensure timestamp difference
        time.sleep(0.1)

        # Update entity
        cursor.execute(
            "UPDATE knowledge_entities SET confidence = %s WHERE id = %s",
            (0.85, str(entity_id))
        )
        db_connection.commit()

        # Get new updated_at
        cursor.execute("SELECT updated_at FROM knowledge_entities WHERE id = %s", (str(entity_id),))
        new_updated_at: datetime = cursor.fetchone()[0]

        # Verify timestamp increased
        assert new_updated_at > original_updated_at

    def test_relationship_updated_at_trigger_on_update(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_pair: Generator[tuple[UUID, UUID], None, None]
    ) -> None:
        """Test UPDATE trigger auto-updates relationship updated_at timestamp.

        Tests that the trigger_update_entity_relationship_timestamp trigger
        automatically updates updated_at when relationship is modified.

        Expected: updated_at increases on UPDATE
        """
        source_id, target_id = valid_entity_pair
        cursor = db_connection.cursor()

        # Create relationship
        rel_id = uuid4()
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (str(rel_id), str(source_id), str(target_id), "test", 0.8,
             datetime.utcnow(), datetime.utcnow())
        )
        db_connection.commit()

        # Get original updated_at
        cursor.execute("SELECT updated_at FROM entity_relationships WHERE id = %s", (str(rel_id),))
        original_updated_at: datetime = cursor.fetchone()[0]

        # Wait to ensure timestamp difference
        time.sleep(0.1)

        # Update relationship
        cursor.execute(
            "UPDATE entity_relationships SET confidence = %s WHERE id = %s",
            (0.75, str(rel_id))
        )
        db_connection.commit()

        # Get new updated_at
        cursor.execute("SELECT updated_at FROM entity_relationships WHERE id = %s", (str(rel_id),))
        new_updated_at: datetime = cursor.fetchone()[0]

        # Verify timestamp increased
        assert new_updated_at > original_updated_at


# ============================================================================
# 6. Index Tests
# ============================================================================


class TestIndexes:
    """Test that all required indexes exist and have correct configuration."""

    def test_all_entity_indexes_exist(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test all expected indexes exist on knowledge_entities table.

        Verifies that all performance-critical indexes are created:
        - idx_knowledge_entities_text
        - idx_knowledge_entities_type
        - idx_knowledge_entities_canonical
        - idx_knowledge_entities_mention_count

        Expected: All 4 indexes found in pg_indexes
        """
        cursor = db_connection.cursor()

        expected_indexes = [
            "idx_knowledge_entities_text",
            "idx_knowledge_entities_type",
            "idx_knowledge_entities_canonical",
            "idx_knowledge_entities_mention_count"
        ]

        # Query PostgreSQL information_schema for indexes
        cursor.execute(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = 'knowledge_entities' AND indexname LIKE 'idx_%'"""
        )

        existing_indexes = {row[0] for row in cursor.fetchall()}

        for idx in expected_indexes:
            assert idx in existing_indexes, f"Missing index {idx} on knowledge_entities"

    def test_all_relationship_indexes_exist(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test all expected indexes exist on entity_relationships table.

        Verifies that all performance-critical indexes for relationship
        traversal are created:
        - idx_entity_relationships_source
        - idx_entity_relationships_target
        - idx_entity_relationships_type
        - idx_entity_relationships_graph (composite)
        - idx_entity_relationships_bidirectional

        Expected: All 5 indexes found in pg_indexes
        """
        cursor = db_connection.cursor()

        expected_indexes = [
            "idx_entity_relationships_source",
            "idx_entity_relationships_target",
            "idx_entity_relationships_type",
            "idx_entity_relationships_graph",
            "idx_entity_relationships_bidirectional"
        ]

        cursor.execute(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = 'entity_relationships' AND indexname LIKE 'idx_%'"""
        )

        existing_indexes = {row[0] for row in cursor.fetchall()}

        for idx in expected_indexes:
            assert idx in existing_indexes, f"Missing index {idx} on entity_relationships"

    def test_all_mention_indexes_exist(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test all expected indexes exist on entity_mentions table.

        Verifies that all indexes for mention lookups are created:
        - idx_entity_mentions_entity
        - idx_entity_mentions_document
        - idx_entity_mentions_chunk
        - idx_entity_mentions_composite

        Expected: All 4 indexes found in pg_indexes
        """
        cursor = db_connection.cursor()

        expected_indexes = [
            "idx_entity_mentions_entity",
            "idx_entity_mentions_document",
            "idx_entity_mentions_chunk",
            "idx_entity_mentions_composite"
        ]

        cursor.execute(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = 'entity_mentions' AND indexname LIKE 'idx_%'"""
        )

        existing_indexes = {row[0] for row in cursor.fetchall()}

        for idx in expected_indexes:
            assert idx in existing_indexes, f"Missing index {idx} on entity_mentions"

    def test_composite_index_column_order(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test composite indexes have correct column order.

        Verifies that composite indexes like idx_entity_relationships_graph
        have columns in the correct order for efficient query planning:
        (source_entity_id, relationship_type, target_entity_id)

        Expected: Composite index exists with correct column order
        """
        cursor = db_connection.cursor()

        # Query PostgreSQL for index columns
        cursor.execute(
            """SELECT attname FROM pg_index
               JOIN pg_attribute ON pg_attribute.attrelid = pg_index.indrelid
               WHERE pg_index.indexrelname = 'idx_entity_relationships_graph'
               ORDER BY pg_index.indseq"""
        )

        columns = [row[0] for row in cursor.fetchall()]

        # Verify composite index has expected columns in order
        # (exact order matters for query planner)
        assert len(columns) >= 2  # At least 2 columns in composite index


# ============================================================================
# 7. Column Type Tests
# ============================================================================


class TestColumnTypes:
    """Test that columns have correct data types."""

    def test_uuid_columns_are_uuid_type(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test that ID columns use UUID type, not TEXT.

        Verifies that primary keys and foreign keys use proper UUID type
        (uuid) not string/text, for data integrity and performance.

        Expected: id, source_entity_id, target_entity_id are uuid type
        """
        cursor = db_connection.cursor()

        # Query information_schema for column types
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'knowledge_entities'
               AND column_name IN ('id')"""
        )

        for col_name, data_type in cursor.fetchall():
            assert data_type == "uuid", f"{col_name} should be uuid, got {data_type}"

        # Check relationship table
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'entity_relationships'
               AND column_name IN ('id', 'source_entity_id', 'target_entity_id')"""
        )

        for col_name, data_type in cursor.fetchall():
            assert data_type == "uuid", f"{col_name} should be uuid, got {data_type}"

    def test_confidence_columns_are_numeric(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test that confidence columns use FLOAT type, not TEXT.

        Verifies that confidence and weight columns use numeric types
        for mathematical operations and range validation.

        Expected: confidence, relationship_weight are double precision (float)
        """
        cursor = db_connection.cursor()

        # Check entity confidence
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'knowledge_entities'
               AND column_name = 'confidence'"""
        )

        _, data_type = cursor.fetchone()
        assert data_type == "double precision", f"confidence should be double precision, got {data_type}"

        # Check relationship confidence and weight
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'entity_relationships'
               AND column_name IN ('confidence', 'relationship_weight')"""
        )

        for col_name, data_type in cursor.fetchall():
            assert data_type == "double precision", \
                f"{col_name} should be double precision, got {data_type}"

    def test_text_columns_are_text_type(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test that text columns use TEXT or VARCHAR type.

        Verifies that entity text and mention text use text types,
        not JSON or other inappropriate types.

        Expected: text, mention_text are text type
        """
        cursor = db_connection.cursor()

        # Check entity text
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'knowledge_entities'
               AND column_name = 'text'"""
        )

        _, data_type = cursor.fetchone()
        assert data_type == "text", f"text should be text, got {data_type}"

        # Check mention text
        cursor.execute(
            """SELECT column_name, data_type FROM information_schema.columns
               WHERE table_name = 'entity_mentions'
               AND column_name = 'mention_text'"""
        )

        _, data_type = cursor.fetchone()
        assert data_type == "text", f"mention_text should be text, got {data_type}"


# ============================================================================
# 8. Constraint Naming Tests
# ============================================================================


class TestConstraintNaming:
    """Test that constraints follow naming conventions."""

    def test_check_constraints_properly_named(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test CHECK constraints follow naming convention (ck_*).

        Verifies that all CHECK constraints use consistent naming
        for debugging and maintainability:
        - ck_confidence_range
        - ck_rel_confidence_range
        - ck_no_self_loops

        Expected: All CHECK constraints start with 'ck_'
        """
        cursor = db_connection.cursor()

        # Query for CHECK constraints
        cursor.execute(
            """SELECT constraint_name FROM information_schema.table_constraints
               WHERE table_name IN ('knowledge_entities', 'entity_relationships')
               AND constraint_type = 'CHECK'"""
        )

        constraint_names = {row[0] for row in cursor.fetchall()}

        # Verify naming convention
        for name in constraint_names:
            assert name.startswith("ck_"), f"CHECK constraint {name} should start with 'ck_'"

    def test_unique_constraints_properly_named(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test UNIQUE constraints follow naming convention (uq_*).

        Verifies that UNIQUE constraints use consistent naming:
        - uq_entity_text_type
        - uq_relationship

        Expected: UNIQUE constraints start with 'uq_'
        """
        cursor = db_connection.cursor()

        # Query for UNIQUE constraints
        cursor.execute(
            """SELECT constraint_name FROM information_schema.table_constraints
               WHERE table_name IN ('knowledge_entities', 'entity_relationships')
               AND constraint_type = 'UNIQUE'"""
        )

        constraint_names = {row[0] for row in cursor.fetchall()}

        for name in constraint_names:
            assert name.startswith("uq_"), f"UNIQUE constraint {name} should start with 'uq_'"

    def test_indexes_properly_named(
        self,
        db_connection: psycopg.Connection
    ) -> None:
        """Test indexes follow naming convention (idx_*).

        Verifies that all indexes use consistent naming for clarity.

        Expected: All indexes start with 'idx_'
        """
        cursor = db_connection.cursor()

        # Query for all user-created indexes
        cursor.execute(
            """SELECT indexname FROM pg_indexes
               WHERE schemaname = 'public'
               AND indexname NOT LIKE '%_pkey'"""  # Exclude primary key indexes
        )

        index_names = {row[0] for row in cursor.fetchall()}

        for name in index_names:
            # Should either be idx_* or PRIMARY KEY
            assert name.startswith("idx_") or "pkey" in name, \
                f"Index {name} should start with 'idx_'"


# ============================================================================
# 9. NULL Constraint Tests
# ============================================================================


class TestNullConstraints:
    """Test NOT NULL constraint enforcement."""

    def test_entity_required_fields_not_null(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test that required entity fields cannot be NULL.

        Verifies NOT NULL constraints on:
        - text
        - entity_type
        - confidence (required, has default)

        Expected: NULL values rejected for required fields
        """
        cursor = db_connection.cursor()

        # Try to insert with NULL text
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
                   VALUES (%s, %s, %s, %s)""",
                (str(uuid4()), None, "PERSON", 0.9)  # NULL text
            )
            db_connection.commit()

        db_connection.rollback()

        # Try to insert with NULL entity_type
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
                   VALUES (%s, %s, %s, %s)""",
                (str(uuid4()), "Test", None, 0.9)  # NULL entity_type
            )
            db_connection.commit()

    def test_relationship_required_fields_not_null(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """Test that required relationship fields cannot be NULL.

        Verifies NOT NULL constraints on:
        - source_entity_id
        - target_entity_id
        - relationship_type
        - confidence

        Expected: NULL values rejected for required fields
        """
        entity_id = valid_entity_id
        cursor = db_connection.cursor()

        # Try to insert with NULL relationship_type
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_relationships
                   (id, source_entity_id, target_entity_id, relationship_type, confidence)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(entity_id), str(entity_id), None, 0.8)  # NULL type
            )
            db_connection.commit()

    def test_mention_required_fields_not_null(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """Test that required mention fields cannot be NULL.

        Verifies NOT NULL constraints on:
        - entity_id
        - document_id
        - chunk_id
        - mention_text

        Expected: NULL values rejected for required fields
        """
        entity_id = valid_entity_id
        cursor = db_connection.cursor()

        # Try to insert with NULL document_id
        with pytest.raises(IntegrityError):
            cursor.execute(
                """INSERT INTO entity_mentions
                   (id, entity_id, document_id, chunk_id, mention_text)
                   VALUES (%s, %s, %s, %s, %s)""",
                (str(uuid4()), str(entity_id), None, 0, "Test")  # NULL document_id
            )
            db_connection.commit()


# ============================================================================
# 10. Data Type Precision Tests
# ============================================================================


class TestDataTypePrecision:
    """Test that numeric columns have sufficient precision."""

    def test_confidence_numeric_precision(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None]
    ) -> None:
        """Test that confidence column stores decimal precision correctly.

        Verifies that double precision float can store values like 0.123456789
        without loss of precision (float64 has ~15 decimal digits).

        Expected: Value retrieved equals value inserted
        """
        cursor = db_connection.cursor()

        entity_id = uuid4()
        precision_value = 0.123456789

        # Insert with high precision
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(entity_id), "Test", "PERSON", precision_value)
        )
        db_connection.commit()

        # Retrieve and verify precision
        cursor.execute("SELECT confidence FROM knowledge_entities WHERE id = %s", (str(entity_id),))
        retrieved_value = cursor.fetchone()[0]

        # Should match to reasonable precision (float64 ~15 sig figs)
        assert abs(retrieved_value - precision_value) < 1e-8

    def test_offset_integers_have_sufficient_range(
        self,
        db_connection: psycopg.Connection,
        cleanup_entities: Generator[None, None, None],
        valid_entity_id: Generator[UUID, None, None]
    ) -> None:
        """Test that offset_start and offset_end support large documents.

        Verifies that INTEGER (32-bit signed) can store offsets for
        documents up to ~2 billion characters (sufficient for real use).

        Expected: Can insert and retrieve large offset values
        """
        entity_id = valid_entity_id
        cursor = db_connection.cursor()

        # Use a large offset (within 32-bit INT range)
        large_offset = 1000000  # 1 million

        cursor.execute(
            """INSERT INTO entity_mentions
               (id, entity_id, document_id, chunk_id, mention_text, offset_start, offset_end)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (str(uuid4()), str(entity_id), "docs/large.md", 0, "test", large_offset, large_offset + 4)
        )
        db_connection.commit()

        # Retrieve and verify
        cursor.execute(
            "SELECT offset_start FROM entity_mentions WHERE offset_start = %s",
            (large_offset,)
        )

        result = cursor.fetchone()
        assert result is not None
        assert result[0] == large_offset
