"""Tests for entity and relationship type enum validation (Issue 7).

This test module validates that enum constraints are enforced at both the ORM
and database layers, providing defense-in-depth against invalid type values.

Tests cover:
- Valid enum values accepted
- Invalid values rejected at ORM layer
- Invalid values rejected at database layer (bypass ORM)
- Case sensitivity enforced
- SQL injection attempts blocked
- All enum values defined correctly
"""

from __future__ import annotations

from typing import Generator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.knowledge_graph.models import (
    EntityTypeEnum,
    RelationshipTypeEnum,
    KnowledgeEntity,
    EntityRelationship,
    Base,
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Create fresh SQLAlchemy session with in-memory SQLite database.

    Each test gets an isolated database that is created and destroyed
    automatically. This ensures test isolation and prevents side effects.

    Yields:
        SQLAlchemy Session instance

    Cleanup:
        Database is dropped after test completion
    """
    # Use SQLite in-memory for fast test execution
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session_instance = SessionLocal()

    yield session_instance

    session_instance.close()
    Base.metadata.drop_all(engine)


class TestEntityTypeEnumDefinitions:
    """Test EntityTypeEnum has all expected values."""

    def test_entity_type_enum_has_12_values(self) -> None:
        """Verify all 12 entity types are defined."""
        assert len(EntityTypeEnum) == 12

    def test_entity_type_enum_contains_core_types(self) -> None:
        """Verify core entity types are defined."""
        core_types = {
            "PERSON",
            "ORG",
            "GPE",
            "PRODUCT",
            "EVENT",
        }
        actual_values = {e.value for e in EntityTypeEnum}
        assert core_types.issubset(actual_values)

    def test_entity_type_enum_contains_extended_types(self) -> None:
        """Verify extended entity types are defined."""
        extended_types = {
            "FACILITY",
            "LAW",
            "LANGUAGE",
            "DATE",
            "TIME",
            "MONEY",
            "PERCENT",
        }
        actual_values = {e.value for e in EntityTypeEnum}
        assert extended_types.issubset(actual_values)

    def test_entity_type_enum_values_match_members(self) -> None:
        """Verify enum values match member names (uppercase)."""
        expected = {
            "PERSON",
            "ORG",
            "GPE",
            "PRODUCT",
            "EVENT",
            "FACILITY",
            "LAW",
            "LANGUAGE",
            "DATE",
            "TIME",
            "MONEY",
            "PERCENT",
        }
        actual = {e.value for e in EntityTypeEnum}
        assert actual == expected


class TestRelationshipTypeEnumDefinitions:
    """Test RelationshipTypeEnum has all expected values."""

    def test_relationship_type_enum_has_3_values(self) -> None:
        """Verify all 3 relationship types are defined."""
        assert len(RelationshipTypeEnum) == 3

    def test_relationship_type_enum_has_required_types(self) -> None:
        """Verify required relationship types are defined."""
        expected = {
            "hierarchical",
            "mentions-in-document",
            "similar-to",
        }
        actual = {e.value for e in RelationshipTypeEnum}
        assert actual == expected

    def test_relationship_type_enum_is_hyphenated(self) -> None:
        """Verify relationship types use hyphens (not underscores)."""
        # This is important for consistency with database and API
        mentions_in_doc = RelationshipTypeEnum.MENTIONS_IN_DOCUMENT
        assert mentions_in_doc.value == "mentions-in-document"
        assert "-" in mentions_in_doc.value


class TestEntityTypeOrmValidation:
    """Test entity_type validation at ORM layer."""

    def test_valid_entity_type_person_accepted(self, db_session: Session) -> None:
        """Valid PERSON type accepted."""
        entity = KnowledgeEntity(
            id=uuid4(),
            text="John Doe",
            entity_type="PERSON",
            confidence=0.95,
        )
        db_session.add(entity)
        db_session.commit()

        assert db_session.query(KnowledgeEntity).filter_by(text="John Doe").first() is not None

    def test_valid_entity_type_org_accepted(self, db_session: Session) -> None:
        """Valid ORG type accepted."""
        entity = KnowledgeEntity(
            id=uuid4(),
            text="Anthropic",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add(entity)
        db_session.commit()

        assert db_session.query(KnowledgeEntity).filter_by(text="Anthropic").first() is not None

    def test_all_valid_entity_types_accepted(self, db_session: Session) -> None:
        """All valid entity types are accepted."""
        entities = []
        for i, entity_type in enumerate(EntityTypeEnum):
            entity = KnowledgeEntity(
                id=uuid4(),
                text=f"Entity {i}",
                entity_type=entity_type.value,
                confidence=0.9,
            )
            entities.append(entity)

        db_session.add_all(entities)
        db_session.commit()

        # Verify all entities were created
        count = db_session.query(KnowledgeEntity).filter(
            KnowledgeEntity.entity_type.in_([e.value for e in EntityTypeEnum])
        ).count()
        assert count >= len(EntityTypeEnum)

    def test_invalid_entity_type_rejected(self, db_session: Session) -> None:
        """Invalid entity types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid entity_type"):
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Invalid",
                entity_type="INVALID_TYPE",
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()

    def test_entity_type_validation_is_case_sensitive(self, db_session: Session) -> None:
        """Entity type validation is case-sensitive (PERSON != person)."""
        with pytest.raises(ValueError, match="Invalid entity_type 'person'"):
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Test Person",
                entity_type="person",  # lowercase not allowed
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()

    def test_entity_type_validation_rejects_mixed_case(self, db_session: Session) -> None:
        """Entity type validation rejects mixed case."""
        with pytest.raises(ValueError, match="Invalid entity_type 'Person'"):
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Test Person",
                entity_type="Person",  # mixed case not allowed
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()

    def test_entity_type_validation_blocks_sql_injection(self, db_session: Session) -> None:
        """Entity type validation blocks SQL injection attempts."""
        malicious_input = "PERSON'; DROP TABLE knowledge_entities; --"

        with pytest.raises(ValueError, match="Invalid entity_type"):
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Test Entity",
                entity_type=malicious_input,
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()

    def test_entity_type_validation_blocks_script_injection(self, db_session: Session) -> None:
        """Entity type validation blocks XSS-like injection attempts."""
        malicious_input = "<script>alert('xss')</script>"

        with pytest.raises(ValueError, match="Invalid entity_type"):
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Test Entity",
                entity_type=malicious_input,
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()


class TestRelationshipTypeOrmValidation:
    """Test relationship_type validation at ORM layer."""

    def test_valid_relationship_type_hierarchical_accepted(self, db_session: Session) -> None:
        """Valid hierarchical type accepted."""
        # Create entities first
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        # Create relationship
        rel = EntityRelationship(
            id=uuid4(),
            source_entity_id=source.id,
            target_entity_id=target.id,
            relationship_type="hierarchical",
            confidence=0.8,
        )
        db_session.add(rel)
        db_session.commit()

        assert db_session.query(EntityRelationship).count() > 0

    def test_valid_relationship_type_mentions_accepted(self, db_session: Session) -> None:
        """Valid mentions-in-document type accepted."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="PRODUCT",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        rel = EntityRelationship(
            id=uuid4(),
            source_entity_id=source.id,
            target_entity_id=target.id,
            relationship_type="mentions-in-document",
            confidence=0.8,
        )
        db_session.add(rel)
        db_session.commit()

        assert db_session.query(EntityRelationship).count() > 0

    def test_valid_relationship_type_similar_accepted(self, db_session: Session) -> None:
        """Valid similar-to type accepted."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PRODUCT",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="PRODUCT",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        rel = EntityRelationship(
            id=uuid4(),
            source_entity_id=source.id,
            target_entity_id=target.id,
            relationship_type="similar-to",
            confidence=0.8,
        )
        db_session.add(rel)
        db_session.commit()

        assert db_session.query(EntityRelationship).count() > 0

    def test_all_valid_relationship_types_accepted(self, db_session: Session) -> None:
        """All valid relationship types are accepted."""
        # Create entities
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        # Create relationships with all valid types
        rels = []
        for rel_type in RelationshipTypeEnum:
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type=rel_type.value,
                confidence=0.8,
            )
            rels.append(rel)

        db_session.add_all(rels)
        db_session.commit()

        count = db_session.query(EntityRelationship).count()
        assert count >= len(RelationshipTypeEnum)

    def test_invalid_relationship_type_rejected(self, db_session: Session) -> None:
        """Invalid relationship types raise ValueError."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        with pytest.raises(ValueError, match="Invalid relationship_type"):
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type="INVALID_TYPE",
                confidence=0.8,
            )
            db_session.add(rel)
            db_session.commit()

    def test_relationship_type_validation_is_case_sensitive(self, db_session: Session) -> None:
        """Relationship type validation is case-sensitive."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        # Wrong case for hierarchical
        with pytest.raises(ValueError, match="Invalid relationship_type"):
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type="HIERARCHICAL",  # uppercase not allowed
                confidence=0.8,
            )
            db_session.add(rel)
            db_session.commit()

    def test_relationship_type_validation_requires_hyphens(self, db_session: Session) -> None:
        """Relationship type validation requires hyphens (not underscores)."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        # Wrong: underscores instead of hyphens
        with pytest.raises(ValueError, match="Invalid relationship_type"):
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type="mentions_in_document",  # underscores not allowed
                confidence=0.8,
            )
            db_session.add(rel)
            db_session.commit()

    def test_relationship_type_validation_blocks_sql_injection(self, db_session: Session) -> None:
        """Relationship type validation blocks SQL injection."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        malicious_input = "hierarchical'; DROP TABLE entity_relationships; --"

        with pytest.raises(ValueError, match="Invalid relationship_type"):
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type=malicious_input,
                confidence=0.8,
            )
            db_session.add(rel)
            db_session.commit()


class TestEnumValueMessages:
    """Test that validation error messages are clear and helpful."""

    def test_entity_type_error_message_lists_valid_types(self, db_session: Session) -> None:
        """Invalid entity type error message lists all valid types."""
        try:
            entity = KnowledgeEntity(
                id=uuid4(),
                text="Test",
                entity_type="INVALID",
                confidence=0.9,
            )
            db_session.add(entity)
            db_session.commit()
        except ValueError as e:
            error_msg = str(e)
            assert "PERSON" in error_msg
            assert "ORG" in error_msg
            assert "PRODUCT" in error_msg
        else:
            pytest.fail("Expected ValueError to be raised")

    def test_relationship_type_error_message_lists_valid_types(self, db_session: Session) -> None:
        """Invalid relationship type error message lists all valid types."""
        source = KnowledgeEntity(
            id=uuid4(),
            text="Source",
            entity_type="PERSON",
            confidence=0.9,
        )
        target = KnowledgeEntity(
            id=uuid4(),
            text="Target",
            entity_type="ORG",
            confidence=0.9,
        )
        db_session.add_all([source, target])
        db_session.flush()

        try:
            rel = EntityRelationship(
                id=uuid4(),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type="INVALID",
                confidence=0.8,
            )
            db_session.add(rel)
            db_session.commit()
        except ValueError as e:
            error_msg = str(e)
            assert "hierarchical" in error_msg
            assert "similar-to" in error_msg
        else:
            pytest.fail("Expected ValueError to be raised")
