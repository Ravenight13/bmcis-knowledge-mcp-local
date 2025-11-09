"""Comprehensive test suite for ORM model-level constraints.

This module validates all SQLAlchemy model constraints including:
- Confidence range validation (0.0-1.0)
- No self-loop validation for relationships
- Unique constraints (entity text+type, relationship source+target+type)
- Required field validation
- Type validation (UUID, enum types)
- Relationship weight bounds

These tests focus on ORM-level validation before database insertion.
Type safety is enforced throughout with explicit type annotations.
"""

from __future__ import annotations

from typing import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.knowledge_graph.models import Base, EntityMention, EntityRelationship, KnowledgeEntity


@pytest.fixture
def session() -> Generator[Session, None, None]:
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


@pytest.fixture
def valid_entity(session: Session) -> KnowledgeEntity:
    """Create and persist a valid test entity.

    Args:
        session: SQLAlchemy session fixture

    Returns:
        Persisted KnowledgeEntity instance with standard test values
    """
    entity: KnowledgeEntity = KnowledgeEntity(
        id=uuid4(),
        text="Test Entity",
        entity_type="PERSON",
        confidence=0.95,
    )
    session.add(entity)
    session.commit()
    return entity


@pytest.fixture
def valid_entities(session: Session) -> tuple[KnowledgeEntity, KnowledgeEntity]:
    """Create and persist two distinct valid entities.

    Args:
        session: SQLAlchemy session fixture

    Returns:
        Tuple of two persisted KnowledgeEntity instances
    """
    entity1: KnowledgeEntity = KnowledgeEntity(
        id=uuid4(),
        text="Entity One",
        entity_type="PERSON",
        confidence=0.90,
    )
    entity2: KnowledgeEntity = KnowledgeEntity(
        id=uuid4(),
        text="Entity Two",
        entity_type="ORG",
        confidence=0.85,
    )
    session.add_all([entity1, entity2])
    session.commit()
    return entity1, entity2


# ============================================================================
# Confidence Range Validation Tests (3 tests)
# ============================================================================


class TestConfidenceRangeValidation:
    """Test entity and relationship confidence bounds [0.0, 1.0]."""

    def test_entity_confidence_valid_values(
        self, session: Session
    ) -> None:
        """Test entity accepts valid confidence values in [0.0, 1.0].

        Valid boundary values and midpoint should be accepted without error.
        Each value is persisted and verified in database.
        """
        entity_min: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="MinConfidence",
            entity_type="PERSON",
            confidence=0.0,
        )
        entity_mid: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="MidConfidence",
            entity_type="ORG",
            confidence=0.5,
        )
        entity_max: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="MaxConfidence",
            entity_type="PRODUCT",
            confidence=1.0,
        )

        session.add_all([entity_min, entity_mid, entity_max])
        session.commit()

        # Verify boundary values were persisted correctly
        assert entity_min.confidence == 0.0
        assert entity_mid.confidence == 0.5
        assert entity_max.confidence == 1.0

    def test_entity_confidence_too_high_rejected(
        self, session: Session
    ) -> None:
        """Test entity rejects confidence > 1.0.

        Database CHECK constraint should reject confidence exceeding 1.0.
        SQLAlchemy raises IntegrityError on constraint violation.
        """
        with pytest.raises(IntegrityError):
            entity: KnowledgeEntity = KnowledgeEntity(
                id=uuid4(),
                text="BadConfidence",
                entity_type="PERSON",
                confidence=1.5,
            )
            # Force validation by adding to session
            session.add(entity)
            session.commit()

    def test_entity_confidence_negative_rejected(
        self, session: Session
    ) -> None:
        """Test entity rejects confidence < 0.0.

        Database CHECK constraint should reject negative confidence.
        SQLAlchemy raises IntegrityError on constraint violation.
        """
        with pytest.raises(IntegrityError):
            entity: KnowledgeEntity = KnowledgeEntity(
                id=uuid4(),
                text="NegativeConfidence",
                entity_type="PERSON",
                confidence=-0.1,
            )
            # Force validation by adding to session
            session.add(entity)
            session.commit()

    def test_relationship_confidence_valid_values(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship accepts valid confidence values [0.0, 1.0].

        Relationships should support same confidence bounds as entities.
        """
        entity1, entity2 = valid_entities

        rel_min: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test-min",
            confidence=0.0,
        )
        rel_max: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test-max",
            confidence=1.0,
        )

        session.add_all([rel_min, rel_max])
        session.commit()

        assert rel_min.confidence == 0.0
        assert rel_max.confidence == 1.0

    def test_relationship_confidence_too_high_rejected(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship rejects confidence > 1.0."""
        entity1, entity2 = valid_entities

        with pytest.raises(IntegrityError):
            rel: EntityRelationship = EntityRelationship(
                id=uuid4(),
                source_entity_id=entity1.id,
                target_entity_id=entity2.id,
                relationship_type="bad-rel",
                confidence=1.5,
            )
            session.add(rel)
            session.commit()

    def test_relationship_confidence_negative_rejected(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship rejects confidence < 0.0."""
        entity1, entity2 = valid_entities

        with pytest.raises(IntegrityError):
            rel: EntityRelationship = EntityRelationship(
                id=uuid4(),
                source_entity_id=entity1.id,
                target_entity_id=entity2.id,
                relationship_type="bad-rel",
                confidence=-0.5,
            )
            session.add(rel)
            session.commit()


# ============================================================================
# No Self-Loop Validation Tests (2 tests)
# ============================================================================


class TestNoSelfLoopValidation:
    """Test that relationships cannot have self-loops (source == target)."""

    def test_relationship_prevents_self_loops(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test relationship rejects source_entity_id == target_entity_id.

        Self-loops violate the CHECK constraint source_entity_id != target_entity_id
        and should be rejected on database commit.
        """
        with pytest.raises(IntegrityError):
            # Attempt to create self-loop
            bad_rel: EntityRelationship = EntityRelationship(
                id=uuid4(),
                source_entity_id=valid_entity.id,
                target_entity_id=valid_entity.id,  # Self-loop!
                relationship_type="self-reference",
                confidence=0.8,
            )
            session.add(bad_rel)
            session.commit()

    def test_relationship_allows_different_entities(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship between different entities is allowed.

        Valid relationships with distinct source and target should persist
        without error.
        """
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="works-for",
            confidence=0.9,
        )
        session.add(rel)
        session.commit()

        # Verify relationship was persisted
        assert rel.source_entity_id == entity1.id
        assert rel.target_entity_id == entity2.id
        assert rel.relationship_type == "works-for"
        assert rel.confidence == 0.9


# ============================================================================
# Unique Constraint Tests (3 tests)
# ============================================================================


class TestUniqueConstraints:
    """Test UNIQUE constraints on entities, relationships, and mentions."""

    def test_entity_unique_constraint_text_type(
        self, session: Session
    ) -> None:
        """Test (text, entity_type) uniqueness constraint.

        Two entities with same text and type should be rejected.
        Different types with same text should be allowed.
        """
        entity1: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="Lutron",
            entity_type="VENDOR",
            confidence=0.95,
        )
        entity2: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="Lutron",
            entity_type="VENDOR",
            confidence=0.90,
        )

        session.add(entity1)
        session.commit()

        # Second entity with same text + type should fail
        session.add(entity2)
        with pytest.raises(Exception):  # IntegrityError on commit
            session.commit()

    def test_entity_allows_same_text_different_type(
        self, session: Session
    ) -> None:
        """Test entities with same text but different types are allowed.

        (text, entity_type) uniqueness should allow same text with different types.
        """
        entity1: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="Apple",
            entity_type="COMPANY",
            confidence=0.95,
        )
        entity2: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="Apple",
            entity_type="PRODUCT",
            confidence=0.90,
        )

        session.add_all([entity1, entity2])
        session.commit()

        # Both should exist in database
        results = session.query(KnowledgeEntity).filter_by(text="Apple").all()
        assert len(results) == 2

    def test_relationship_unique_constraint_source_target_type(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test (source_entity_id, target_entity_id, relationship_type) uniqueness.

        Two relationships with same source, target, and type should be rejected.
        Different types with same source/target should be allowed.
        """
        entity1, entity2 = valid_entities

        rel1: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="works-for",
            confidence=0.85,
        )
        rel2: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="works-for",  # Duplicate!
            confidence=0.80,
        )

        session.add(rel1)
        session.commit()

        session.add(rel2)
        with pytest.raises(Exception):  # IntegrityError on commit
            session.commit()

    def test_relationship_allows_same_entities_different_type(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationships with same source/target but different types allowed.

        (source, target, type) uniqueness should allow same source/target with different types.
        """
        entity1, entity2 = valid_entities

        rel1: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="works-for",
            confidence=0.9,
        )
        rel2: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="reports-to",  # Different type
            confidence=0.85,
        )

        session.add_all([rel1, rel2])
        session.commit()

        # Both should exist
        results = (
            session.query(EntityRelationship)
            .filter_by(source_entity_id=entity1.id, target_entity_id=entity2.id)
            .all()
        )
        assert len(results) == 2


# ============================================================================
# Required Field Validation Tests (2 tests)
# ============================================================================


class TestRequiredFieldValidation:
    """Test that required fields are enforced."""

    def test_entity_required_text_field(
        self, session: Session
    ) -> None:
        """Test entity requires non-null text field.

        Creating entity without text will trigger NOT NULL constraint violation.
        """
        with pytest.raises(IntegrityError):
            entity: KnowledgeEntity = KnowledgeEntity(  # type: ignore
                id=uuid4(),
                entity_type="PERSON",
                confidence=0.9,
            )
            session.add(entity)
            session.commit()

    def test_entity_required_entity_type_field(
        self, session: Session
    ) -> None:
        """Test entity requires non-null entity_type field."""
        with pytest.raises(IntegrityError):
            entity: KnowledgeEntity = KnowledgeEntity(  # type: ignore
                id=uuid4(),
                text="Test",
                confidence=0.9,
            )
            session.add(entity)
            session.commit()

    def test_entity_confidence_has_default_value(
        self, session: Session
    ) -> None:
        """Test entity confidence defaults to 1.0 when not specified.

        Confidence should default to 1.0 (maximum confidence).
        """
        entity: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="DefaultConfidence",
            entity_type="PERSON",
        )
        session.add(entity)
        session.commit()

        assert entity.confidence == 1.0

    def test_relationship_required_source_entity_id(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test relationship requires non-null source_entity_id."""
        with pytest.raises(IntegrityError):
            rel: EntityRelationship = EntityRelationship(  # type: ignore
                id=uuid4(),
                target_entity_id=valid_entity.id,
                relationship_type="test",
                confidence=0.8,
            )
            session.add(rel)
            session.commit()

    def test_relationship_required_target_entity_id(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test relationship requires non-null target_entity_id."""
        with pytest.raises(IntegrityError):
            rel: EntityRelationship = EntityRelationship(  # type: ignore
                id=uuid4(),
                source_entity_id=valid_entity.id,
                relationship_type="test",
                confidence=0.8,
            )
            session.add(rel)
            session.commit()

    def test_relationship_required_relationship_type(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship requires non-null relationship_type."""
        entity1, entity2 = valid_entities

        with pytest.raises(IntegrityError):
            rel: EntityRelationship = EntityRelationship(  # type: ignore
                id=uuid4(),
                source_entity_id=entity1.id,
                target_entity_id=entity2.id,
                confidence=0.8,
            )
            session.add(rel)
            session.commit()

    def test_relationship_confidence_has_default_value(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship confidence defaults to 1.0.

        When confidence is not specified, should default to 1.0.
        """
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test-default",
        )
        session.add(rel)
        session.commit()

        assert rel.confidence == 1.0


# ============================================================================
# Type Validation Tests (2 tests)
# ============================================================================


class TestTypeValidation:
    """Test UUID and enum type validation."""

    def test_uuid_primary_keys_properly_typed(
        self, session: Session
    ) -> None:
        """Test UUID primary keys are UUID type, not string.

        Entity IDs should be UUID type. This enables proper type checking
        and prevents string-based UUID lookups.
        """
        entity: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="UUIDTest",
            entity_type="PERSON",
            confidence=0.9,
        )
        session.add(entity)
        session.commit()

        # Verify id is UUID type
        assert isinstance(entity.id, UUID)

        # Verify can query by UUID
        retrieved: KnowledgeEntity | None = session.query(KnowledgeEntity).filter_by(
            id=entity.id
        ).first()
        assert retrieved is not None
        assert retrieved.id == entity.id

    def test_entity_type_string_accepted(
        self, session: Session
    ) -> None:
        """Test entity_type accepts string values.

        entity_type is stored as VARCHAR(50) and accepts various string values.
        """
        valid_types: list[str] = [
            "PERSON",
            "ORG",
            "PRODUCT",
            "LOCATION",
            "TECHNOLOGY",
            "VENDOR",
        ]

        for entity_type in valid_types:
            entity: KnowledgeEntity = KnowledgeEntity(
                id=uuid4(),
                text=f"Test{entity_type}",
                entity_type=entity_type,
                confidence=0.9,
            )
            session.add(entity)

        session.commit()

        # Verify all were persisted
        for entity_type in valid_types:
            result: KnowledgeEntity | None = session.query(KnowledgeEntity).filter_by(
                entity_type=entity_type
            ).first()
            assert result is not None
            assert result.entity_type == entity_type


# ============================================================================
# Relationship Weight Validation Tests (1 test)
# ============================================================================


class TestRelationshipWeightValidation:
    """Test relationship_weight bounds and defaults."""

    def test_relationship_weight_has_default_value(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship_weight defaults to 1.0.

        Relationship weight (frequency score) should default to 1.0.
        """
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test-weight",
            confidence=0.8,
        )
        session.add(rel)
        session.commit()

        assert rel.relationship_weight == 1.0

    def test_relationship_weight_can_be_set(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test relationship_weight can be set to custom value."""
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test-custom-weight",
            confidence=0.8,
            relationship_weight=2.5,
        )
        session.add(rel)
        session.commit()

        assert rel.relationship_weight == 2.5


# ============================================================================
# Entity Mention Field Validation Tests (1 test)
# ============================================================================


class TestEntityMentionFieldValidation:
    """Test EntityMention required fields."""

    def test_mention_required_entity_id(
        self, session: Session
    ) -> None:
        """Test EntityMention requires non-null entity_id."""
        with pytest.raises(IntegrityError):
            mention: EntityMention = EntityMention(  # type: ignore
                id=uuid4(),
                document_id="docs/README.md",
                chunk_id=1,
                mention_text="Test mention",
            )
            session.add(mention)
            session.commit()

    def test_mention_required_document_id(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test EntityMention requires non-null document_id."""
        with pytest.raises(IntegrityError):
            mention: EntityMention = EntityMention(  # type: ignore
                id=uuid4(),
                entity_id=valid_entity.id,
                chunk_id=1,
                mention_text="Test mention",
            )
            session.add(mention)
            session.commit()

    def test_mention_required_chunk_id(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test EntityMention requires non-null chunk_id."""
        with pytest.raises(IntegrityError):
            mention: EntityMention = EntityMention(  # type: ignore
                id=uuid4(),
                entity_id=valid_entity.id,
                document_id="docs/README.md",
                mention_text="Test mention",
            )
            session.add(mention)
            session.commit()

    def test_mention_required_mention_text(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test EntityMention requires non-null mention_text."""
        with pytest.raises(IntegrityError):
            mention: EntityMention = EntityMention(  # type: ignore
                id=uuid4(),
                entity_id=valid_entity.id,
                document_id="docs/README.md",
                chunk_id=1,
            )
            session.add(mention)
            session.commit()

    def test_mention_can_be_created_with_required_fields(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test EntityMention creation with all required fields.

        A valid mention with all required fields should persist successfully.
        """
        mention: EntityMention = EntityMention(
            id=uuid4(),
            entity_id=valid_entity.id,
            document_id="docs/README.md",
            chunk_id=1,
            mention_text="Test mention",
        )
        session.add(mention)
        session.commit()

        assert mention.entity_id == valid_entity.id
        assert mention.document_id == "docs/README.md"
        assert mention.chunk_id == 1
        assert mention.mention_text == "Test mention"


# ============================================================================
# Model Instantiation Tests (2 tests)
# ============================================================================


class TestModelInstantiation:
    """Test complete model instantiation with all fields."""

    def test_entity_can_be_instantiated_completely(
        self, session: Session
    ) -> None:
        """Test Entity creation with all fields and persistence.

        Complete entity instantiation should work with all fields specified.
        """
        entity: KnowledgeEntity = KnowledgeEntity(
            id=uuid4(),
            text="Complete Entity",
            entity_type="PERSON",
            confidence=0.95,
            canonical_form="complete entity",
        )
        session.add(entity)
        session.commit()

        assert entity.text == "Complete Entity"
        assert entity.entity_type == "PERSON"
        assert entity.confidence == 0.95
        assert entity.canonical_form == "complete entity"
        assert entity.mention_count == 0  # Default value

    def test_relationship_can_be_instantiated_completely(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test Relationship creation with all fields and persistence.

        Complete relationship instantiation with all fields should work.
        """
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="works-for",
            confidence=0.85,
            relationship_weight=1.5,
            is_bidirectional=False,
        )
        session.add(rel)
        session.commit()

        assert rel.source_entity_id == entity1.id
        assert rel.target_entity_id == entity2.id
        assert rel.relationship_type == "works-for"
        assert rel.confidence == 0.85
        assert rel.relationship_weight == 1.5
        assert rel.is_bidirectional is False

    def test_mention_can_be_instantiated_completely(
        self, valid_entity: KnowledgeEntity, session: Session
    ) -> None:
        """Test EntityMention creation with all fields.

        Complete mention instantiation with offset information should work.
        """
        mention: EntityMention = EntityMention(
            id=uuid4(),
            entity_id=valid_entity.id,
            document_id="docs/example.md",
            chunk_id=5,
            mention_text="test mention",
            offset_start=10,
            offset_end=23,
        )
        session.add(mention)
        session.commit()

        assert mention.entity_id == valid_entity.id
        assert mention.document_id == "docs/example.md"
        assert mention.chunk_id == 5
        assert mention.mention_text == "test mention"
        assert mention.offset_start == 10
        assert mention.offset_end == 23


# ============================================================================
# Bidirectional Relationship Tests (1 test)
# ============================================================================


class TestBidirectionalFlag:
    """Test is_bidirectional flag on relationships."""

    def test_relationship_is_bidirectional_defaults_to_false(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test is_bidirectional defaults to False.

        By default relationships should be directional (one-way).
        """
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="test",
            confidence=0.9,
        )
        session.add(rel)
        session.commit()

        assert rel.is_bidirectional is False

    def test_relationship_is_bidirectional_can_be_set_true(
        self, valid_entities: tuple[KnowledgeEntity, KnowledgeEntity], session: Session
    ) -> None:
        """Test is_bidirectional can be explicitly set to True."""
        entity1, entity2 = valid_entities

        rel: EntityRelationship = EntityRelationship(
            id=uuid4(),
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="similar-to",
            confidence=0.9,
            is_bidirectional=True,
        )
        session.add(rel)
        session.commit()

        assert rel.is_bidirectional is True
