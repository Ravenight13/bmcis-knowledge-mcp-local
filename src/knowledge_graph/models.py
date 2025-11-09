"""SQLAlchemy ORM models for knowledge graph entities and relationships.

This module defines SQLAlchemy models corresponding to the normalized PostgreSQL
schema for storing knowledge entities, relationships, and mention provenance data.

Models use UUID primary keys for sharding-friendliness and follow a type-safe
approach with comprehensive validation.

Architecture: Hybrid Normalized + Cache
- Models represent normalized relational data
- In-memory cache layer (separate module) handles hot-path performance
- Incremental updates via simple INSERT/UPDATE operations
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


class KnowledgeEntity(Base):
    """Entity in the knowledge graph (vendor, product, person, organization, etc.).

    Attributes:
        id: UUID primary key, globally unique
        text: Canonical entity text (e.g., "Lutron", "Claude AI")
        entity_type: Classification type (PERSON, ORG, PRODUCT, GPE, LOCATION, etc.)
        confidence: Extraction confidence score (0.0-1.0)
        canonical_form: Normalized form for deduplication (e.g., lowercase variant)
        mention_count: Total count of mentions in corpus
        created_at: Record creation timestamp
        updated_at: Last modification timestamp
        relationships_from: Outbound relationships (this entity as source)
        relationships_to: Inbound relationships (this entity as target)
        mentions: List of document mentions for this entity

    Constraints:
        - confidence must be in [0.0, 1.0]
        - text + entity_type must be unique
        - text must be non-empty
    """

    __tablename__ = "knowledge_entities"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=None)
    text: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    canonical_form: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    relationships_from: Mapped[list["EntityRelationship"]] = relationship(
        "EntityRelationship",
        foreign_keys="EntityRelationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    relationships_to: Mapped[list["EntityRelationship"]] = relationship(
        "EntityRelationship",
        foreign_keys="EntityRelationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )
    mentions: Mapped[list["EntityMention"]] = relationship(
        "EntityMention",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("text", "entity_type", name="uq_entity_text_type"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_confidence_range"),
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"KnowledgeEntity(id={self.id}, text={self.text!r}, type={self.entity_type})"


class EntityRelationship(Base):
    """Directed relationship between two entities (typed property graph edge).

    Attributes:
        id: UUID primary key
        source_entity_id: FK to source entity
        target_entity_id: FK to target entity
        relationship_type: Type of relationship
            - 'hierarchical': Source is parent/creator/owner of target
            - 'mentions-in-document': Source and target co-mentioned
            - 'similar-to': Source is semantically similar to target
        confidence: Relationship confidence (0.0-1.0), based on extraction method
        relationship_weight: Frequency-based weight (higher = stronger)
        is_bidirectional: Whether relationship is symmetric (e.g., similar-to is bidirectional)
        created_at: Record creation timestamp
        updated_at: Last modification timestamp
        source_entity: Relationship to source entity
        target_entity: Relationship to target entity

    Constraints:
        - confidence must be in [0.0, 1.0]
        - source_entity_id != target_entity_id (no self-loops)
        - (source_entity_id, target_entity_id, relationship_type) must be unique
        - Foreign keys ensure referential integrity
    """

    __tablename__ = "entity_relationships"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=None)
    source_entity_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True
    )
    target_entity_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True
    )
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    relationship_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_bidirectional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    source_entity: Mapped[KnowledgeEntity] = relationship(
        "KnowledgeEntity",
        foreign_keys=[source_entity_id],
        back_populates="relationships_from",
    )
    target_entity: Mapped[KnowledgeEntity] = relationship(
        "KnowledgeEntity",
        foreign_keys=[target_entity_id],
        back_populates="relationships_to",
    )

    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint(
            "source_entity_id", "target_entity_id", "relationship_type", name="uq_relationship"
        ),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_rel_confidence_range"),
        CheckConstraint(
            "source_entity_id != target_entity_id", name="ck_no_self_loops"
        ),
        Index(
            "idx_entity_relationships_graph",
            "source_entity_id",
            "relationship_type",
            "target_entity_id",
        ),
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EntityRelationship(source={self.source_entity_id}, "
            f"target={self.target_entity_id}, type={self.relationship_type!r})"
        )


class EntityMention(Base):
    """Provenance tracking: where and how an entity appears in a document.

    Attributes:
        id: UUID primary key
        entity_id: FK to knowledge entity
        document_id: Reference to source document (e.g., "docs/README.md")
        chunk_id: Chunk/passage number within document
        mention_text: Actual text as it appears in source
        offset_start: Character offset start in chunk (for highlighting)
        offset_end: Character offset end in chunk (exclusive)
        created_at: Record creation timestamp
        entity: Relationship to the knowledge entity

    Purpose:
        - Provides provenance: where was this entity extracted?
        - Enables chunk-based queries
        - Supports highlighting in UI
        - Enables frequency analysis
        - Enables co-mention analysis (entities in same chunk/document)

    Constraints:
        - entity_id must reference valid entity
        - document_id must be non-empty
        - chunk_id must be non-negative
    """

    __tablename__ = "entity_mentions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=None)
    entity_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    mention_text: Mapped[str] = mapped_column(Text, nullable=False)
    offset_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    offset_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    entity: Mapped[KnowledgeEntity] = relationship(
        "KnowledgeEntity",
        back_populates="mentions",
    )

    # Indexes
    __table_args__ = (
        Index("idx_entity_mentions_chunk", "document_id", "chunk_id"),
        Index("idx_entity_mentions_composite", "entity_id", "document_id"),
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EntityMention(entity={self.entity_id}, doc={self.document_id!r}, "
            f"chunk={self.chunk_id})"
        )
