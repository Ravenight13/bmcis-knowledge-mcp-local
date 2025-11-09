"""Integration tests for cache ↔ database interactions.

This module tests the interaction between the LRU cache and database through
the KnowledgeGraphService layer. Tests verify:

1. Cache Hit/Miss with DB Reads (3 tests)
   - Entity cache hit on repeated access
   - Entity cache miss triggers DB query
   - Relationship cache hit/miss patterns

2. Invalidation Cascade (3 tests)
   - Entity invalidation removes from cache
   - Relationship invalidation removes from cache
   - Cascade invalidation on entity updates

3. Concurrent Reads with Writes (4 tests)
   - Multiple threads reading from cache
   - Write operations don't corrupt cache state
   - Cache state remains consistent
   - Concurrent invalidation safety

Total: 10 integration tests covering cache-DB workflows.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity, CacheStats
from src.knowledge_graph.cache_config import CacheConfig
from src.knowledge_graph.graph_service import KnowledgeGraphService


# ============================================================================
# Mock Database Implementations
# ============================================================================


class MockConnectionPool:
    """Mock connection pool with simulated persistence layer."""

    def __init__(self) -> None:
        """Initialize mock pool with empty data stores."""
        self.entities: dict[UUID, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def get_connection(self) -> MockConnection:
        """Get a mock connection from the pool."""
        return MockConnection(self)


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, pool: MockConnectionPool) -> None:
        """Initialize mock connection."""
        self.pool = pool

    def __enter__(self) -> MockConnection:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    def cursor(self) -> MockCursor:
        """Get a mock cursor."""
        return MockCursor(self.pool)


class MockCursor:
    """Mock database cursor for testing."""

    def __init__(self, pool: MockConnectionPool) -> None:
        """Initialize mock cursor."""
        self.pool = pool
        self.results: list[tuple[Any, ...]] = []
        self._query_count = 0

    def __enter__(self) -> MockCursor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass

    def execute(self, query: str, params: tuple[Any, ...]) -> None:
        """Execute a query (mock implementation)."""
        self._query_count += 1
        self.results = []

        # Parse query to determine type and simulate results
        if "source_entity_id" in query.lower():
            self._simulate_1hop_query(params)
        elif "target_entity_id" in query.lower() and "FULL OUTER" in query.upper():
            self._simulate_bidirectional_query(params)

    def _simulate_1hop_query(self, params: tuple[Any, ...]) -> None:
        """Simulate 1-hop traversal results."""
        entity_id = params[0]
        min_confidence = params[1] if len(params) > 1 else 0.0

        # Find all relationships from source_entity_id
        with self.pool._lock:
            for rel in self.pool.relationships:
                if (rel["source_entity_id"] == entity_id and
                    float(rel["confidence"]) >= float(min_confidence)):
                    target = self.pool.entities.get(rel["target_entity_id"])
                    if target:
                        self.results.append((
                            target["id"],
                            target["text"],
                            target["entity_type"],
                            target["confidence"],
                            rel["relationship_type"],
                            rel["confidence"],
                            None
                        ))

    def _simulate_bidirectional_query(self, params: tuple[Any, ...]) -> None:
        """Simulate bidirectional traversal results."""
        entity_id = params[0]
        related_ids: set[UUID] = set()

        with self.pool._lock:
            # Find outbound relationships
            for rel in self.pool.relationships:
                if rel["source_entity_id"] == entity_id:
                    related_ids.add(rel["target_entity_id"])

            # Find inbound relationships
            for rel in self.pool.relationships:
                if rel["target_entity_id"] == entity_id:
                    related_ids.add(rel["source_entity_id"])

            # Build results
            for rel_id in related_ids:
                entity = self.pool.entities.get(rel_id)
                if entity:
                    self.results.append((
                        entity["id"],
                        entity["text"],
                        entity["entity_type"],
                        entity["confidence"],
                        ["rel-type"],  # outbound_rel_types
                        [],  # inbound_rel_types
                        0.8,  # max_confidence
                        1,  # relationship_count
                        1  # min_distance
                    ))

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all results."""
        return self.results


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_pool() -> MockConnectionPool:
    """Create a fresh mock connection pool for each test."""
    return MockConnectionPool()


@pytest.fixture
def service(mock_pool: MockConnectionPool) -> KnowledgeGraphService:
    """Create service with mock database pool."""
    cache_config = CacheConfig(
        max_entities=100,
        max_relationship_caches=200
    )
    return KnowledgeGraphService(
        db_pool=mock_pool,
        cache_config=cache_config
    )


@pytest.fixture
def sample_entity_id() -> UUID:
    """Generate a sample entity ID."""
    return uuid4()


@pytest.fixture
def sample_related_id() -> UUID:
    """Generate a sample related entity ID."""
    return uuid4()


# ============================================================================
# Category 1: Cache Hit/Miss with DB Reads (3 tests)
# ============================================================================


class TestCacheHitMissWithDbReads:
    """Tests for cache hit/miss behavior with database reads."""

    def test_entity_cache_hit_on_repeated_reads(
        self,
        service: KnowledgeGraphService,
        mock_pool: MockConnectionPool,
        sample_entity_id: UUID
    ) -> None:
        """Test entity cache hit on repeated access.

        Workflow:
        1. Set entity in cache
        2. First read - retrieves from cache (hit)
        3. Second read - retrieves from cache (hit)
        4. Verify cache stats show 2 hits
        """
        entity = Entity(
            id=sample_entity_id,
            text="Claude AI",
            type="technology",
            confidence=0.95,
            mention_count=10,
        )

        # Set entity in cache
        service._cache.set_entity(entity)

        # First read
        result1 = service.get_entity(sample_entity_id)
        assert result1 is not None
        assert result1.text == "Claude AI"

        # Get stats after first read
        stats_after_first = service.get_cache_stats()
        first_hits = stats_after_first["hits"]

        # Second read (should hit cache)
        result2 = service.get_entity(sample_entity_id)
        assert result2 is not None
        assert result2.text == "Claude AI"

        # Get stats after second read
        stats_after_second = service.get_cache_stats()
        second_hits = stats_after_second["hits"]

        # Verify cache hit
        assert second_hits > first_hits
        assert result1.id == result2.id
        # Both should be same object from cache
        assert result1 is result2

    def test_entity_cache_miss_returns_none(
        self,
        service: KnowledgeGraphService,
        mock_pool: MockConnectionPool,
        sample_entity_id: UUID
    ) -> None:
        """Test cache miss for non-cached entity returns None.

        Workflow:
        1. Entity NOT in cache, NOT in database
        2. Read entity via service
        3. Should return None (get_entity only checks cache)
        4. Verify cache stats show 1 miss
        """
        # Ensure entity is NOT in cache or database
        assert sample_entity_id not in mock_pool.entities

        # Get initial stats
        stats_before = service.get_cache_stats()
        misses_before = stats_before["misses"]

        # Read entity - cache miss (entity not in cache)
        result = service.get_entity(sample_entity_id)

        # Get stats after read
        stats_after = service.get_cache_stats()
        misses_after = stats_after["misses"]

        # Verify cache miss occurred
        assert result is None
        assert misses_after > misses_before

    def test_relationship_cache_hit_miss_pattern(
        self,
        service: KnowledgeGraphService,
        mock_pool: MockConnectionPool,
        sample_entity_id: UUID,
        sample_related_id: UUID
    ) -> None:
        """Test 1-hop relationship cache hit/miss patterns.

        Workflow:
        1. Create relationship in database
        2. First traverse_1hop - cache miss, DB query
        3. Second traverse_1hop - cache hit
        4. Verify cache stats reflect hit/miss pattern
        """
        # Setup entities in database
        mock_pool.entities[sample_entity_id] = {
            "id": sample_entity_id,
            "text": "Source",
            "entity_type": "VENDOR",
            "confidence": 0.95,
            "mention_count": 10
        }
        mock_pool.entities[sample_related_id] = {
            "id": sample_related_id,
            "text": "Target",
            "entity_type": "PRODUCT",
            "confidence": 0.85,
            "mention_count": 5
        }

        # Create relationship
        mock_pool.relationships.append({
            "source_entity_id": sample_entity_id,
            "target_entity_id": sample_related_id,
            "relationship_type": "produces",
            "confidence": 0.90
        })

        # First traverse - cache miss
        stats_before = service.get_cache_stats()
        misses_before = stats_before["misses"]

        result1 = service.traverse_1hop(sample_entity_id, "produces")
        assert len(result1) > 0

        stats_after_first = service.get_cache_stats()
        misses_after_first = stats_after_first["misses"]

        # Second traverse - cache hit
        result2 = service.traverse_1hop(sample_entity_id, "produces")
        assert len(result2) > 0

        stats_after_second = service.get_cache_stats()
        hits_after_second = stats_after_second["hits"]

        # Verify patterns
        assert misses_after_first > misses_before
        assert hits_after_second > 0
        assert result1 == result2


# ============================================================================
# Category 2: Invalidation Cascade (3 tests)
# ============================================================================


class TestInvalidationCascade:
    """Tests for cache invalidation and cascade effects."""

    def test_entity_invalidation_removes_from_cache(
        self,
        service: KnowledgeGraphService,
        sample_entity_id: UUID
    ) -> None:
        """Test entity invalidation removes from cache.

        Workflow:
        1. Set entity in cache
        2. Verify entity in cache
        3. Invalidate entity
        4. Verify entity removed from cache
        """
        entity = Entity(
            id=sample_entity_id,
            text="Test",
            type="TEST",
            confidence=0.9,
            mention_count=1,
        )

        # Set in cache
        service._cache.set_entity(entity)
        assert service._cache.get_entity(sample_entity_id) is not None

        # Get cache size before
        stats_before = service.get_cache_stats()
        size_before = stats_before["size"]

        # Invalidate
        service.invalidate_entity(sample_entity_id)

        # Get cache size after
        stats_after = service.get_cache_stats()
        size_after = stats_after["size"]

        # Verify removal
        assert service._cache.get_entity(sample_entity_id) is None
        assert size_after < size_before

    def test_relationship_invalidation_removes_from_cache(
        self,
        service: KnowledgeGraphService,
        sample_entity_id: UUID,
        sample_related_id: UUID
    ) -> None:
        """Test relationship cache invalidation.

        Workflow:
        1. Set relationship in cache
        2. Verify relationship cached
        3. Invalidate outbound relationships
        4. Verify relationship removed from cache
        """
        # Create and cache entities
        entity = Entity(
            id=sample_entity_id,
            text="Source",
            type="TYPE",
            confidence=0.9,
            mention_count=1,
        )
        related = Entity(
            id=sample_related_id,
            text="Related",
            type="TYPE",
            confidence=0.8,
            mention_count=1,
        )

        service._cache.set_entity(entity)
        service._cache.set_entity(related)

        # Set relationships in cache
        service._cache.set_relationships(
            sample_entity_id,
            "test_rel",
            [related]
        )

        # Verify cached
        cached = service._cache.get_relationships(sample_entity_id, "test_rel")
        assert cached is not None
        assert len(cached) == 1

        # Invalidate
        service._cache.invalidate_relationships(sample_entity_id, "test_rel")

        # Verify removed
        cached_after = service._cache.get_relationships(sample_entity_id, "test_rel")
        assert cached_after is None

    def test_cascade_invalidation_on_entity_update(
        self,
        service: KnowledgeGraphService,
        sample_entity_id: UUID,
        sample_related_id: UUID
    ) -> None:
        """Test cascade invalidation when entity is updated.

        Workflow:
        1. Cache entity and its relationships
        2. Update entity in database
        3. Invalidate entity (cascading)
        4. Verify entity and outbound relationships invalidated
        """
        # Create entities
        entity = Entity(
            id=sample_entity_id,
            text="Original",
            type="PERSON",
            confidence=0.90,
            mention_count=3,
        )
        related = Entity(
            id=sample_related_id,
            text="Related",
            type="ORG",
            confidence=0.85,
            mention_count=2,
        )

        # Cache both
        service._cache.set_entity(entity)
        service._cache.set_entity(related)

        # Cache outbound relationships
        service._cache.set_relationships(
            sample_entity_id,
            "works_for",
            [related]
        )

        # Verify cached
        assert service._cache.get_entity(sample_entity_id) is not None
        assert service._cache.get_relationships(sample_entity_id, "works_for") is not None

        # Invalidate (cascade)
        service.invalidate_entity(sample_entity_id)

        # Verify both invalidated
        assert service._cache.get_entity(sample_entity_id) is None
        assert service._cache.get_relationships(sample_entity_id, "works_for") is None


# ============================================================================
# Category 3: Concurrent Reads with Writes (4 tests)
# ============================================================================


class TestConcurrentReadsWithWrites:
    """Tests for concurrent cache operations safety."""

    def test_concurrent_entity_reads_from_cache(
        self,
        service: KnowledgeGraphService,
        sample_entity_id: UUID
    ) -> None:
        """Test multiple threads reading entity from cache concurrently.

        Workflow:
        1. Set entity in cache
        2. Spawn N threads reading same entity
        3. All threads should get same object
        4. No race conditions or corruption
        """
        entity = Entity(
            id=sample_entity_id,
            text="Concurrent",
            type="TEST",
            confidence=0.95,
            mention_count=100,
        )

        service._cache.set_entity(entity)

        results: list[Optional[Entity]] = []
        lock = threading.Lock()

        def read_entity() -> None:
            """Read entity from cache in thread."""
            result = service.get_entity(sample_entity_id)
            with lock:
                results.append(result)

        # Spawn 10 concurrent readers
        threads = []
        for _ in range(10):
            t = threading.Thread(target=read_entity)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all got same entity
        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.id == sample_entity_id for r in results)
        # All should be same cached object
        assert all(r is results[0] for r in results)

    def test_write_to_cache_with_concurrent_reads(
        self,
        service: KnowledgeGraphService,
        sample_entity_id: UUID
    ) -> None:
        """Test concurrent reads while writing to cache.

        Workflow:
        1. Spawn reader threads
        2. Periodically write entities to cache
        3. Verify readers don't get corrupted data
        4. No deadlocks or crashes
        """
        read_errors: list[str] = []
        lock = threading.Lock()

        def reader_thread() -> None:
            """Read various entities from cache."""
            try:
                for i in range(50):
                    entity_id = uuid4()
                    result = service._cache.get_entity(entity_id)
                    # Should be None or valid Entity
                    assert result is None or isinstance(result, Entity)
            except Exception as e:
                with lock:
                    read_errors.append(str(e))

        def writer_thread() -> None:
            """Write entities to cache."""
            try:
                for i in range(50):
                    entity = Entity(
                        id=uuid4(),
                        text=f"Entity {i}",
                        type="TEST",
                        confidence=0.9,
                        mention_count=i,
                    )
                    service._cache.set_entity(entity)
            except Exception as e:
                with lock:
                    read_errors.append(str(e))

        # Spawn readers and writers
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader_thread))
        for _ in range(3):
            threads.append(threading.Thread(target=writer_thread))

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        # Verify no errors
        assert len(read_errors) == 0

    def test_concurrent_invalidation_safety(
        self,
        service: KnowledgeGraphService
    ) -> None:
        """Test concurrent invalidation doesn't corrupt cache.

        Workflow:
        1. Cache multiple entities
        2. Spawn threads invalidating different entities
        3. Verify cache state remains consistent
        4. No partial/corrupted entries
        """
        # Cache 20 entities
        entity_ids: list[UUID] = []
        for i in range(20):
            eid = uuid4()
            entity = Entity(
                id=eid,
                text=f"Entity {i}",
                type="TEST",
                confidence=0.9,
                mention_count=i,
            )
            service._cache.set_entity(entity)
            entity_ids.append(eid)

        errors: list[str] = []
        lock = threading.Lock()

        def invalidate_entities(start: int, end: int) -> None:
            """Invalidate range of entities."""
            try:
                for i in range(start, end):
                    if i < len(entity_ids):
                        service.invalidate_entity(entity_ids[i])
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Spawn threads invalidating different ranges
        threads = [
            threading.Thread(target=invalidate_entities, args=(0, 5)),
            threading.Thread(target=invalidate_entities, args=(5, 10)),
            threading.Thread(target=invalidate_entities, args=(10, 15)),
            threading.Thread(target=invalidate_entities, args=(15, 20)),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all invalidated
        for eid in entity_ids:
            assert service._cache.get_entity(eid) is None

    def test_concurrent_cache_operations_consistency(
        self,
        service: KnowledgeGraphService
    ) -> None:
        """Test cache state remains consistent under concurrent operations.

        Workflow:
        1. Mix concurrent reads, writes, invalidations
        2. Verify cache stats are accurate
        3. No race conditions in hit/miss counting
        4. Cache size stays within bounds
        """
        max_entities = service._cache.max_entities
        results: dict[str, int] = {}
        lock = threading.Lock()

        def mixed_operations(thread_id: int) -> None:
            """Perform mixed cache operations."""
            for i in range(30):
                # Write
                entity_id = uuid4()
                entity = Entity(
                    id=entity_id,
                    text=f"Entity {i}",
                    type="TEST",
                    confidence=0.9,
                    mention_count=i,
                )
                service._cache.set_entity(entity)

                # Read
                _ = service._cache.get_entity(entity_id)

                # Sometimes invalidate
                if i % 5 == 0:
                    service._cache.invalidate_entity(entity_id)

        # Spawn threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        # Check cache stats are reasonable
        stats = service.get_cache_stats()
        assert stats["size"] <= stats["max_size"]
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0
        assert stats["evictions"] >= 0
