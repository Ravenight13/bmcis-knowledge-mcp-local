"""Test dependency injection with mock cache and repository.

This test suite verifies that KnowledgeGraphService properly supports
dependency injection for cache and repository implementations.

Test Coverage:
1. Service accepts injected cache (MockCache)
2. Service accepts injected repository (MockRepository)
3. Service uses default implementations if none provided
4. MockCache tracks all operations (get, set, invalidate)
5. Service delegates operations to injected dependencies
6. Mock implementations enable testability

Benefits of DI Testing:
- Verify service behavior without database
- Test cache interaction patterns
- Validate proper dependency usage
- Enable fast unit tests (no I/O)
"""

from __future__ import annotations

import pytest
from uuid import uuid4, UUID
from typing import List, Optional, Any

from src.knowledge_graph.cache_protocol import Entity, CacheStats, CacheProtocol
from src.knowledge_graph.graph_service import KnowledgeGraphService
from src.knowledge_graph.cache import KnowledgeGraphCache
from src.knowledge_graph.query_repository import KnowledgeGraphQueryRepository


# Mock Implementations

class MockCache:
    """Mock cache implementation for testing dependency injection.

    Implements CacheProtocol and tracks all method calls for verification.
    Stores entities/relationships in memory dictionaries.

    Tracking:
        - get_calls: List of entity IDs requested via get_entity
        - set_calls: List of entity IDs stored via set_entity
        - invalidate_calls: List of entity IDs invalidated
        - get_rel_calls: List of (entity_id, rel_type) tuples
        - set_rel_calls: List of (entity_id, rel_type) tuples

    Usage:
        mock_cache = MockCache()
        service = KnowledgeGraphService(db_pool, cache=mock_cache)

        # Perform operations
        service.get_entity(entity_id)

        # Verify cache was called
        assert entity_id in mock_cache.get_calls
    """

    def __init__(self) -> None:
        """Initialize mock cache with tracking."""
        # Storage
        self.entities: dict[UUID, Entity] = {}
        self.relationships: dict[tuple[UUID, str], List[Entity]] = {}

        # Call tracking
        self.get_calls: List[UUID] = []
        self.set_calls: List[UUID] = []
        self.invalidate_calls: List[UUID] = []
        self.get_rel_calls: List[tuple[UUID, str]] = []
        self.set_rel_calls: List[tuple[UUID, str]] = []
        self.invalidate_rel_calls: List[tuple[UUID, str]] = []
        self.clear_calls: int = 0
        self.stats_calls: int = 0

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Track get calls and return entity if exists."""
        self.get_calls.append(entity_id)
        return self.entities.get(entity_id)

    def set_entity(self, entity: Entity) -> None:
        """Track set calls and store entity."""
        self.set_calls.append(entity.id)
        self.entities[entity.id] = entity

    def get_relationships(
        self,
        entity_id: UUID,
        rel_type: str
    ) -> Optional[List[Entity]]:
        """Track relationship get calls."""
        self.get_rel_calls.append((entity_id, rel_type))
        return self.relationships.get((entity_id, rel_type))

    def set_relationships(
        self,
        entity_id: UUID,
        rel_type: str,
        entities: List[Entity]
    ) -> None:
        """Track relationship set calls and store."""
        self.set_rel_calls.append((entity_id, rel_type))
        self.relationships[(entity_id, rel_type)] = entities

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Track invalidation calls and remove entity."""
        self.invalidate_calls.append(entity_id)
        self.entities.pop(entity_id, None)

    def invalidate_relationships(
        self,
        entity_id: UUID,
        rel_type: str
    ) -> None:
        """Track relationship invalidation."""
        self.invalidate_rel_calls.append((entity_id, rel_type))
        self.relationships.pop((entity_id, rel_type), None)

    def clear(self) -> None:
        """Track clear calls and clear storage."""
        self.clear_calls += 1
        self.entities.clear()
        self.relationships.clear()

    def stats(self) -> CacheStats:
        """Track stats calls and return mock statistics."""
        self.stats_calls += 1
        return CacheStats(
            hits=len(self.get_calls),
            misses=0,
            evictions=0,
            size=len(self.entities) + len(self.relationships),
            max_size=10000
        )


class MockDatabasePool:
    """Mock database pool for testing (minimal implementation)."""

    def __init__(self) -> None:
        """Initialize mock pool."""
        pass


class MockRepository:
    """Mock repository implementation for testing dependency injection.

    Provides minimal implementation of KnowledgeGraphQueryRepository interface
    for testing service behavior without database.
    """

    def __init__(self) -> None:
        """Initialize mock repository with tracking."""
        self.traverse_1hop_calls: List[tuple[UUID, float, Optional[List[str]]]] = []
        self.traverse_2hop_calls: List[tuple[UUID, float, Optional[List[str]]]] = []
        self.traverse_bidirectional_calls: List[tuple[UUID, float, int]] = []
        self.get_entity_mentions_calls: List[tuple[UUID, int]] = []
        self.traverse_with_type_filter_calls: List[tuple[UUID, str, List[str], float]] = []

    def traverse_1hop(
        self,
        entity_id: UUID,
        min_confidence: float = 0.7,
        relationship_types: Optional[List[str]] = None
    ) -> List[Any]:
        """Track 1-hop traversal calls."""
        self.traverse_1hop_calls.append((entity_id, min_confidence, relationship_types))
        return []

    def traverse_2hop(
        self,
        entity_id: UUID,
        min_confidence: float = 0.7,
        relationship_types: Optional[List[str]] = None
    ) -> List[Any]:
        """Track 2-hop traversal calls."""
        self.traverse_2hop_calls.append((entity_id, min_confidence, relationship_types))
        return []

    def traverse_bidirectional(
        self,
        entity_id: UUID,
        min_confidence: float = 0.7,
        max_depth: int = 1
    ) -> List[Any]:
        """Track bidirectional traversal calls."""
        self.traverse_bidirectional_calls.append((entity_id, min_confidence, max_depth))
        return []

    def get_entity_mentions(
        self,
        entity_id: UUID,
        max_results: int = 100
    ) -> List[Any]:
        """Track mention retrieval calls."""
        self.get_entity_mentions_calls.append((entity_id, max_results))
        return []

    def traverse_with_type_filter(
        self,
        entity_id: UUID,
        relationship_type: str,
        target_entity_types: List[str],
        min_confidence: float = 0.7
    ) -> List[Any]:
        """Track type-filtered traversal calls."""
        self.traverse_with_type_filter_calls.append(
            (entity_id, relationship_type, target_entity_types, min_confidence)
        )
        return []


# Test Cases

def test_service_accepts_injected_cache() -> None:
    """Service accepts cache dependency via constructor."""
    mock_cache = MockCache()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache
    )

    # Verify service uses injected cache
    assert service._cache is mock_cache


def test_service_accepts_injected_repository() -> None:
    """Service accepts repository dependency via constructor."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Verify service uses injected repository
    assert service._repo is mock_repo


def test_service_uses_default_cache_if_none() -> None:
    """Service creates KnowledgeGraphCache if none provided."""
    service = KnowledgeGraphService(db_pool=MockDatabasePool())

    # Verify default cache is KnowledgeGraphCache
    assert isinstance(service._cache, KnowledgeGraphCache)


def test_service_uses_default_repository_if_none() -> None:
    """Service creates KnowledgeGraphQueryRepository if none provided."""
    service = KnowledgeGraphService(db_pool=MockDatabasePool())

    # Verify default repository is KnowledgeGraphQueryRepository
    assert isinstance(service._repo, KnowledgeGraphQueryRepository)


def test_mock_cache_tracks_get_calls() -> None:
    """MockCache tracks entity get operations."""
    mock_cache = MockCache()
    entity_id = uuid4()

    # Call get_entity
    result = mock_cache.get_entity(entity_id)

    # Verify call was tracked
    assert entity_id in mock_cache.get_calls
    assert result is None  # Not yet in cache


def test_mock_cache_tracks_set_calls() -> None:
    """MockCache tracks entity set operations."""
    mock_cache = MockCache()
    entity = Entity(
        id=uuid4(),
        text="Test Entity",
        type="PERSON",
        confidence=0.95,
        mention_count=5
    )

    # Call set_entity
    mock_cache.set_entity(entity)

    # Verify call was tracked
    assert entity.id in mock_cache.set_calls
    assert mock_cache.entities[entity.id] == entity


def test_mock_cache_tracks_invalidation() -> None:
    """MockCache tracks invalidation operations."""
    mock_cache = MockCache()
    entity_id = uuid4()

    # Add entity first
    entity = Entity(
        id=entity_id,
        text="Test",
        type="PERSON",
        confidence=0.9,
        mention_count=3
    )
    mock_cache.set_entity(entity)

    # Invalidate entity
    mock_cache.invalidate_entity(entity_id)

    # Verify invalidation was tracked
    assert entity_id in mock_cache.invalidate_calls
    assert entity_id not in mock_cache.entities


def test_mock_cache_tracks_relationship_gets() -> None:
    """MockCache tracks relationship get operations."""
    mock_cache = MockCache()
    entity_id = uuid4()
    rel_type = "hierarchical"

    # Call get_relationships
    result = mock_cache.get_relationships(entity_id, rel_type)

    # Verify call was tracked
    assert (entity_id, rel_type) in mock_cache.get_rel_calls
    assert result is None  # Not yet cached


def test_mock_cache_tracks_relationship_sets() -> None:
    """MockCache tracks relationship set operations."""
    mock_cache = MockCache()
    entity_id = uuid4()
    rel_type = "hierarchical"
    entities = [
        Entity(id=uuid4(), text="Child 1", type="PRODUCT", confidence=0.9, mention_count=2),
        Entity(id=uuid4(), text="Child 2", type="PRODUCT", confidence=0.85, mention_count=1)
    ]

    # Call set_relationships
    mock_cache.set_relationships(entity_id, rel_type, entities)

    # Verify call was tracked
    assert (entity_id, rel_type) in mock_cache.set_rel_calls
    assert mock_cache.relationships[(entity_id, rel_type)] == entities


def test_mock_cache_tracks_relationship_invalidation() -> None:
    """MockCache tracks relationship invalidation."""
    mock_cache = MockCache()
    entity_id = uuid4()
    rel_type = "hierarchical"

    # Add relationships first
    entities = [Entity(id=uuid4(), text="Test", type="PRODUCT", confidence=0.9, mention_count=1)]
    mock_cache.set_relationships(entity_id, rel_type, entities)

    # Invalidate relationships
    mock_cache.invalidate_relationships(entity_id, rel_type)

    # Verify invalidation was tracked
    assert (entity_id, rel_type) in mock_cache.invalidate_rel_calls
    assert (entity_id, rel_type) not in mock_cache.relationships


def test_mock_cache_tracks_clear() -> None:
    """MockCache tracks clear operations."""
    mock_cache = MockCache()

    # Add some data
    entity = Entity(id=uuid4(), text="Test", type="PERSON", confidence=0.9, mention_count=1)
    mock_cache.set_entity(entity)

    # Clear cache
    mock_cache.clear()

    # Verify clear was tracked and data removed
    assert mock_cache.clear_calls == 1
    assert len(mock_cache.entities) == 0


def test_mock_cache_tracks_stats() -> None:
    """MockCache tracks stats calls."""
    mock_cache = MockCache()

    # Call stats
    stats = mock_cache.stats()

    # Verify stats call was tracked
    assert mock_cache.stats_calls == 1
    assert isinstance(stats, CacheStats)


def test_service_with_mock_cache_delegates_get_entity() -> None:
    """Service delegates get_entity to injected cache."""
    mock_cache = MockCache()
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache,
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.get_entity(entity_id)

    # Verify cache was called (not found, so returned None)
    assert entity_id in mock_cache.get_calls


def test_service_with_mock_cache_delegates_invalidate() -> None:
    """Service delegates invalidation to injected cache."""
    mock_cache = MockCache()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache
    )

    # Call invalidation
    entity_id = uuid4()
    service.invalidate_entity(entity_id)

    # Verify cache invalidation was called
    assert entity_id in mock_cache.invalidate_calls


def test_service_with_mock_cache_delegates_invalidate_relationships() -> None:
    """Service delegates relationship invalidation to injected cache."""
    mock_cache = MockCache()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache
    )

    # Call relationship invalidation
    entity_id = uuid4()
    rel_type = "hierarchical"
    service.invalidate_relationships(entity_id, rel_type)

    # Verify cache was called
    assert (entity_id, rel_type) in mock_cache.invalidate_rel_calls


def test_service_with_mock_cache_delegates_stats() -> None:
    """Service delegates stats to injected cache."""
    mock_cache = MockCache()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache
    )

    # Call stats
    stats = service.get_cache_stats()

    # Verify cache stats was called
    assert mock_cache.stats_calls == 1
    assert 'hits' in stats


def test_service_with_mock_repo_delegates_traverse_1hop() -> None:
    """Service delegates traverse_1hop to injected repository."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.traverse_1hop(entity_id, rel_type="hierarchical", min_confidence=0.8)

    # Verify repository was called
    assert len(mock_repo.traverse_1hop_calls) == 1
    call = mock_repo.traverse_1hop_calls[0]
    assert call[0] == entity_id
    assert call[1] == 0.8
    assert call[2] == ["hierarchical"]


def test_service_with_mock_repo_delegates_traverse_2hop() -> None:
    """Service delegates traverse_2hop to injected repository."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.traverse_2hop(entity_id, rel_type="hierarchical", min_confidence=0.75)

    # Verify repository was called
    assert len(mock_repo.traverse_2hop_calls) == 1
    call = mock_repo.traverse_2hop_calls[0]
    assert call[0] == entity_id
    assert call[1] == 0.75


def test_service_with_mock_repo_delegates_traverse_bidirectional() -> None:
    """Service delegates traverse_bidirectional to injected repository."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.traverse_bidirectional(entity_id, min_confidence=0.8, max_depth=2)

    # Verify repository was called
    assert len(mock_repo.traverse_bidirectional_calls) == 1
    call = mock_repo.traverse_bidirectional_calls[0]
    assert call[0] == entity_id
    assert call[1] == 0.8
    assert call[2] == 2


def test_service_with_mock_repo_delegates_get_mentions() -> None:
    """Service delegates get_mentions to injected repository."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.get_mentions(entity_id, max_results=50)

    # Verify repository was called
    assert len(mock_repo.get_entity_mentions_calls) == 1
    call = mock_repo.get_entity_mentions_calls[0]
    assert call[0] == entity_id
    assert call[1] == 50


def test_service_with_mock_repo_delegates_traverse_with_type_filter() -> None:
    """Service delegates traverse_with_type_filter to injected repository."""
    mock_repo = MockRepository()
    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        query_repo=mock_repo
    )

    # Call service method
    entity_id = uuid4()
    service.traverse_with_type_filter(
        entity_id,
        rel_type="hierarchical",
        target_entity_types=["PRODUCT", "VENDOR"],
        min_confidence=0.85
    )

    # Verify repository was called
    assert len(mock_repo.traverse_with_type_filter_calls) == 1
    call = mock_repo.traverse_with_type_filter_calls[0]
    assert call[0] == entity_id
    assert call[1] == "hierarchical"
    assert call[2] == ["PRODUCT", "VENDOR"]
    assert call[3] == 0.85


def test_both_cache_and_repo_can_be_injected_together() -> None:
    """Service accepts both cache and repository simultaneously."""
    mock_cache = MockCache()
    mock_repo = MockRepository()

    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache,
        query_repo=mock_repo
    )

    # Verify both dependencies are injected
    assert service._cache is mock_cache
    assert service._repo is mock_repo

    # Perform operation that uses both
    entity_id = uuid4()
    service.traverse_1hop(entity_id, rel_type="hierarchical")

    # Verify both were called
    assert (entity_id, "hierarchical") in mock_cache.get_rel_calls
    assert len(mock_repo.traverse_1hop_calls) == 1


def test_mock_cache_get_and_set_roundtrip() -> None:
    """MockCache stores and retrieves entities correctly."""
    mock_cache = MockCache()
    entity = Entity(
        id=uuid4(),
        text="Test Entity",
        type="ORGANIZATION",
        confidence=0.92,
        mention_count=10
    )

    # Set entity
    mock_cache.set_entity(entity)

    # Get entity back
    retrieved = mock_cache.get_entity(entity.id)

    # Verify roundtrip
    assert retrieved is not None
    assert retrieved.id == entity.id
    assert retrieved.text == entity.text
    assert retrieved.type == entity.type
    assert retrieved.confidence == entity.confidence


def test_mock_cache_relationship_get_and_set_roundtrip() -> None:
    """MockCache stores and retrieves relationships correctly."""
    mock_cache = MockCache()
    entity_id = uuid4()
    rel_type = "mentions-in-document"
    entities = [
        Entity(id=uuid4(), text="Entity 1", type="PERSON", confidence=0.88, mention_count=3),
        Entity(id=uuid4(), text="Entity 2", type="PERSON", confidence=0.91, mention_count=7)
    ]

    # Set relationships
    mock_cache.set_relationships(entity_id, rel_type, entities)

    # Get relationships back
    retrieved = mock_cache.get_relationships(entity_id, rel_type)

    # Verify roundtrip
    assert retrieved is not None
    assert len(retrieved) == 2
    assert retrieved[0].id == entities[0].id
    assert retrieved[1].id == entities[1].id
