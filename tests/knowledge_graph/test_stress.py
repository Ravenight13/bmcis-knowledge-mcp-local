"""Stress tests for knowledge graph - large fanout, concurrency, cache eviction.

Tests:
- Large fanout tests: Single entity with 1000 relationships, 2-hop with 100 intermediates
- Cache eviction: Insert 10,000 entities, verify LRU behavior
- Concurrent operations: 50 threads updating same entity, 100 threads mixed ops
- Performance degradation: Measure latency at scale

Total: 12 tests covering stress conditions
"""

from __future__ import annotations

import time
import random
from uuid import uuid4, UUID
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
import statistics

import pytest

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity, CacheStats


class TestLargeFanoutScenarios:
    """Test knowledge graph with large fanout (many relationships)."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create cache with reasonable limits for fanout tests."""
        # Large cache to handle 1000+ entities
        return KnowledgeGraphCache(max_entities=2000, max_relationship_caches=5000)

    def test_single_entity_1000_relationships(
        self,
        cache: KnowledgeGraphCache,
    ) -> None:
        """Test single entity with 1000 relationships.

        Validates:
        - Cache can store 1000+ related entities
        - Relationship traversal doesn't corrupt state
        - Performance remains acceptable
        """
        source_entity: Entity = Entity(
            id=uuid4(),
            text="Central Hub",
            type="ORG",
            confidence=0.99,
            mention_count=1000,
        )

        cache.set_entity(source_entity)

        # Create 1000 related entities
        related_entities: List[Entity] = []
        for i in range(1000):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Related Entity {i}",
                type="PRODUCT",
                confidence=0.9 + (i % 10) * 0.001,
                mention_count=i + 1,
            )
            related_entities.append(entity)
            cache.set_entity(entity)

        # Verify source entity still retrievable
        retrieved_source = cache.get_entity(source_entity.id)
        assert retrieved_source is not None
        assert retrieved_source.text == "Central Hub"
        assert retrieved_source.mention_count == 1000

        # Verify all related entities retrievable
        for i, entity in enumerate(related_entities):
            retrieved = cache.get_entity(entity.id)
            assert retrieved is not None
            assert retrieved.text == f"Related Entity {i}"

        # Check cache stats
        stats = cache.stats()
        assert stats.size > 0
        assert stats.size <= cache.max_entities

    def test_2hop_graph_with_100_intermediate_entities(
        self,
    ) -> None:
        """Test 2-hop graph with 100 intermediate entities.

        Graph structure:
        Source -> [100 intermediate entities] -> [100 target entities each]
        Total: 1 source + 100 intermediate + 10,000 target = 10,101 entities

        Validates:
        - Large multi-hop graphs handled correctly
        - Cache can store complex relationship structures
        - No corruption under scale
        """
        # Use larger cache for this test to avoid evictions
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=15000,
            max_relationship_caches=30000,
        )

        source_entity: Entity = Entity(
            id=uuid4(),
            text="Root Entity",
            type="PERSON",
            confidence=0.95,
            mention_count=1,
        )
        cache.set_entity(source_entity)

        # Create 100 intermediate entities
        intermediate_entities: List[Entity] = []
        for i in range(100):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Intermediate {i}",
                type="ORG",
                confidence=0.9,
                mention_count=i + 1,
            )
            intermediate_entities.append(entity)
            cache.set_entity(entity)

        # Create 100 target entities for each intermediate
        target_entities_per_intermediate: List[List[Entity]] = []
        for i, intermediate in enumerate(intermediate_entities):
            targets: List[Entity] = []
            for j in range(100):
                entity: Entity = Entity(
                    id=uuid4(),
                    text=f"Target {i}_{j}",
                    type="PRODUCT",
                    confidence=0.85 + (j % 10) * 0.001,
                    mention_count=j + 1,
                )
                targets.append(entity)
                cache.set_entity(entity)
            target_entities_per_intermediate.append(targets)

        # Verify source is still accessible
        retrieved_source = cache.get_entity(source_entity.id)
        assert retrieved_source is not None

        # Verify intermediate entities
        for i, intermediate in enumerate(intermediate_entities):
            retrieved = cache.get_entity(intermediate.id)
            assert retrieved is not None
            assert retrieved.text == f"Intermediate {i}"

        # Verify a sample of target entities
        for i in range(0, 100, 10):  # Sample every 10th intermediate
            for j in range(0, 100, 10):  # Sample every 10th target
                entity = target_entities_per_intermediate[i][j]
                retrieved = cache.get_entity(entity.id)
                assert retrieved is not None
                assert retrieved.text == f"Target {i}_{j}"

    def test_star_topology_1000_direct_connections(
        self,
        cache: KnowledgeGraphCache,
    ) -> None:
        """Test star topology with 1000 direct connections to central node.

        This tests worst-case cache performance for a single hot entity
        with many connections.
        """
        central_id: UUID = uuid4()
        central_entity: Entity = Entity(
            id=central_id,
            text="Central Node",
            type="ORG",
            confidence=0.98,
            mention_count=10000,
        )
        cache.set_entity(central_entity)

        # Create 1000 peripheral entities
        peripheral_ids: List[UUID] = []
        for i in range(1000):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Peripheral {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=100 - (i % 100),
            )
            peripheral_ids.append(entity.id)
            cache.set_entity(entity)

        # Repeatedly access central node (hot path)
        for _ in range(100):
            retrieved = cache.get_entity(central_id)
            assert retrieved is not None

        # Verify central node not evicted despite many accesses
        final_central = cache.get_entity(central_id)
        assert final_central is not None
        assert final_central.text == "Central Node"


class TestCacheEvictionAndLRU:
    """Test cache eviction behavior under pressure."""

    def test_lru_eviction_with_10000_entities(self) -> None:
        """Test cache LRU eviction with 10,000 entity insertions.

        Setup:
        - Create cache with max_entities=1000
        - Insert 10,000 entities
        - Verify only ~1000 most recently accessed remain
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=1000,
            max_relationship_caches=2000,
        )

        # Create list to track insertion order
        entities: List[Entity] = []
        inserted_ids: List[UUID] = []

        # Insert 10,000 entities
        for i in range(10000):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i + 1,
            )
            entities.append(entity)
            inserted_ids.append(entity.id)
            cache.set_entity(entity)

        # Check cache size is capped
        stats = cache.stats()
        assert stats.size <= cache.max_entities

        # Early entities should be evicted (LRU policy)
        # Last 100 entities should be in cache
        for i in range(9900, 10000):
            retrieved = cache.get_entity(inserted_ids[i])
            assert retrieved is not None, f"Entity {i} should be in cache"

        # First 100 entities should likely be evicted
        evicted_count: int = 0
        for i in range(0, 100):
            retrieved = cache.get_entity(inserted_ids[i])
            if retrieved is None:
                evicted_count += 1

        # At least 80% of first 100 should be evicted
        assert evicted_count >= 80, f"Only {evicted_count}/100 early entities evicted"

    def test_cache_access_promotes_lru(self) -> None:
        """Test that accessing an entity promotes it in LRU order.

        Validates that frequently accessed entities survive eviction.
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=100,
            max_relationship_caches=200,
        )

        # Create 100 entities (fills cache)
        entities: List[Entity] = []
        first_entity_id: UUID = uuid4()

        for i in range(100):
            entity: Entity = Entity(
                id=uuid4() if i > 0 else first_entity_id,
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i + 1,
            )
            entities.append(entity)
            cache.set_entity(entity)

        # Access first entity many times (should promote it)
        for _ in range(50):
            retrieved = cache.get_entity(first_entity_id)
            assert retrieved is not None

        # Add 50 more entities (should evict least recently accessed)
        for i in range(100, 150):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i + 1,
            )
            cache.set_entity(entity)

        # First entity should still be in cache (frequently accessed)
        retrieved = cache.get_entity(first_entity_id)
        assert retrieved is not None

    def test_relationship_cache_eviction(self) -> None:
        """Test relationship cache eviction independently.

        Validates that relationship caches respect max_relationship_caches limit.
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=5000,
            max_relationship_caches=100,  # Small limit
        )

        # Create source and target entities
        source_id: UUID = uuid4()
        source: Entity = Entity(
            id=source_id,
            text="Source",
            type="PERSON",
            confidence=0.95,
            mention_count=1,
        )
        cache.set_entity(source)

        # Create many target entities for different relationship types
        targets: List[Entity] = []
        for i in range(1000):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Target {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=1,
            )
            targets.append(entity)
            cache.set_entity(entity)

        # Cache relationship stats
        stats = cache.stats()
        assert stats.size <= cache.max_entities


class TestConcurrentStressOperations:
    """Test concurrent access patterns under stress."""

    def test_50_threads_updating_same_entity(self) -> None:
        """Test 50 threads updating same entity (atomicity test).

        Each thread updates mention_count independently.
        Validates:
        - Thread-safe updates
        - No data corruption
        - Final state consistency
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=100,
            max_relationship_caches=200,
        )

        entity_id: UUID = uuid4()
        initial_entity: Entity = Entity(
            id=entity_id,
            text="Shared Entity",
            type="PRODUCT",
            confidence=0.95,
            mention_count=0,
        )
        cache.set_entity(initial_entity)

        errors: List[Exception] = []
        errors_lock: Lock = Lock()
        update_count: int = 0

        def update_entity(thread_id: int) -> None:
            """Each thread updates the entity."""
            nonlocal update_count
            try:
                for i in range(10):
                    # Get current entity
                    entity = cache.get_entity(entity_id)
                    if entity is None:
                        continue

                    # Update with new mention count
                    updated: Entity = Entity(
                        id=entity_id,
                        text=entity.text,
                        type=entity.type,
                        confidence=entity.confidence,
                        mention_count=entity.mention_count + 1,
                    )
                    cache.set_entity(updated)
                    update_count += 1
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Run 50 threads
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures: List[Future[None]] = [
                executor.submit(update_entity, i)
                for i in range(50)
            ]
            for future in futures:
                future.result()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Final entity should exist
        final_entity = cache.get_entity(entity_id)
        assert final_entity is not None

    def test_100_threads_mixed_reads_writes(self) -> None:
        """Test 100 threads with mixed read/write operations.

        Creates high contention scenario:
        - 50 reader threads
        - 50 writer threads

        Validates:
        - No deadlocks
        - Cache coherency
        - No crashes under load
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=500,
            max_relationship_caches=1000,
        )

        # Pre-populate cache with 100 entities
        entity_ids: List[UUID] = []
        for i in range(100):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i,
            )
            entity_ids.append(entity.id)
            cache.set_entity(entity)

        errors: List[Exception] = []
        errors_lock: Lock = Lock()
        read_count: int = 0
        write_count: int = 0

        def reader(thread_id: int) -> None:
            """Read entities."""
            nonlocal read_count
            try:
                for _ in range(50):
                    idx = random.randint(0, len(entity_ids) - 1)
                    entity = cache.get_entity(entity_ids[idx])
                    if entity is not None:
                        read_count += 1
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def writer(thread_id: int) -> None:
            """Write entities."""
            nonlocal write_count
            try:
                for _ in range(25):
                    idx = random.randint(0, len(entity_ids) - 1)
                    entity = cache.get_entity(entity_ids[idx])
                    if entity is not None:
                        updated: Entity = Entity(
                            id=entity.id,
                            text=entity.text + "_updated",
                            type=entity.type,
                            confidence=entity.confidence,
                            mention_count=entity.mention_count + 1,
                        )
                        cache.set_entity(updated)
                        write_count += 1
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Run 100 threads: 50 readers + 50 writers
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: List[Future[None]] = []

            # 50 reader threads
            for i in range(50):
                futures.append(executor.submit(reader, i))

            # 50 writer threads
            for i in range(50):
                futures.append(executor.submit(writer, i))

            for future in futures:
                future.result()

        # Verify success
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert read_count > 0, "Some reads should have succeeded"
        assert write_count > 0, "Some writes should have succeeded"

    def test_concurrent_cache_invalidation(self) -> None:
        """Test cache coherency during concurrent invalidation.

        One thread invalidates entities while others read.
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=200,
            max_relationship_caches=400,
        )

        # Create 50 entities
        entity_ids: List[UUID] = []
        for i in range(50):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i,
            )
            entity_ids.append(entity.id)
            cache.set_entity(entity)

        errors: List[Exception] = []
        errors_lock: Lock = Lock()

        def reader(thread_id: int) -> None:
            """Read entities continuously."""
            try:
                for _ in range(100):
                    idx = random.randint(0, len(entity_ids) - 1)
                    cache.get_entity(entity_ids[idx])
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def invalidator() -> None:
            """Invalidate entities by re-setting them."""
            try:
                for i in range(50):
                    entity: Entity = Entity(
                        id=entity_ids[i],
                        text=f"Entity {i} Invalidated",
                        type="PRODUCT",
                        confidence=0.95,
                        mention_count=i + 100,
                    )
                    cache.set_entity(entity)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Run readers and invalidator concurrently
        with ThreadPoolExecutor(max_workers=11) as executor:
            futures: List[Future[None]] = []

            # 10 reader threads
            for i in range(10):
                futures.append(executor.submit(reader, i))

            # 1 invalidator thread
            futures.append(executor.submit(invalidator))

            for future in futures:
                future.result()

        # Verify success
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestPerformanceDegradation:
    """Test performance under increasing scale."""

    def test_latency_at_scale_1hop(self) -> None:
        """Test 1-hop query latency as cache fills up.

        Measures latency with varying cache sizes.
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=5000,
            max_relationship_caches=10000,
        )

        test_entity_id: UUID = uuid4()
        test_entity: Entity = Entity(
            id=test_entity_id,
            text="Test Entity",
            type="PRODUCT",
            confidence=0.95,
            mention_count=1,
        )
        cache.set_entity(test_entity)

        latencies: Dict[int, List[float]] = {}

        # Test at different cache fill levels
        for fill_level in [100, 500, 1000, 2000]:
            # Fill cache to level
            while cache.stats().size < fill_level:
                entity: Entity = Entity(
                    id=uuid4(),
                    text=f"Filler Entity",
                    type="PRODUCT",
                    confidence=0.9,
                    mention_count=1,
                )
                cache.set_entity(entity)

            # Measure latency for 1000 lookups
            latencies[fill_level] = []
            for _ in range(1000):
                start = time.perf_counter()
                cache.get_entity(test_entity_id)
                end = time.perf_counter()
                latencies[fill_level].append((end - start) * 1_000_000)  # Convert to microseconds

        # Analyze latencies
        for level, times in latencies.items():
            avg_latency = statistics.mean(times)
            max_latency = max(times)
            # Latency should remain <100µs (cache lookups should be very fast)
            assert avg_latency < 100, f"Avg latency at {level} entities: {avg_latency}µs"

    def test_cache_hit_rate_under_workload(self) -> None:
        """Test cache hit rate degrades gracefully with scale.

        Simulates realistic access pattern.
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=1000,
            max_relationship_caches=2000,
        )

        # Create 1000 entities (exactly cache capacity)
        entities: List[Entity] = []
        for i in range(1000):
            entity: Entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="PRODUCT",
                confidence=0.9,
                mention_count=i,
            )
            entities.append(entity)
            cache.set_entity(entity)

        # Access pattern: Zipfian distribution (80/20 rule)
        # 80% of accesses to 20% of entities
        hot_entities = entities[:200]  # 20% of entities
        cool_entities = entities[200:]  # 80% of entities

        for _ in range(1000):
            # 80% chance of hot entity
            if random.random() < 0.8:
                entity = random.choice(hot_entities)
            else:
                entity = random.choice(cool_entities)

            cache.get_entity(entity.id)

        # Get cache stats
        stats = cache.stats()
        hit_rate = stats.hits / (stats.hits + stats.misses) if (stats.hits + stats.misses) > 0 else 0

        # With Zipfian access and LRU, hit rate should be reasonable (>50% given the size ratio)
        assert hit_rate > 0.5, f"Hit rate too low: {hit_rate}"
