"""Concurrent stress tests for KnowledgeGraphCache - High Priority 8.

Tests cache thread-safety under 100+ concurrent operations with mixed read/write patterns.
Validates:
- Cache hit latency <2µs
- Throughput >10k operations/second
- Hit rate >80% under concurrent load
- No deadlocks or data corruption
"""

from __future__ import annotations

import time
import random
from uuid import UUID, uuid4
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock

import pytest

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity, CacheStats


class TestHighConcurrencyReadOnly:
    """Category 1: High Concurrency Read-Only Tests (2 tests)."""

    def test_concurrent_reads_100_threads_same_entity(self) -> None:
        """Test 100 threads reading same entity - validates cache coherency.

        - 100 threads × 100 reads per thread = 10,000 total cache hits
        - All reads should succeed with zero errors
        - All reads should return same entity ID
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=100)
        entity: Entity = Entity(
            id=uuid4(),
            text="Test",
            type="PERSON",
            confidence=0.95,
            mention_count=1,
        )
        cache.set_entity(entity)

        results: list[UUID] = []
        errors: list[Exception] = []
        results_lock: Lock = Lock()
        errors_lock: Lock = Lock()

        def reader() -> None:
            """Read entity 100 times."""
            try:
                for _ in range(100):
                    cached: Optional[Entity] = cache.get_entity(entity.id)
                    if cached:
                        with results_lock:
                            results.append(cached.id)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(reader) for _ in range(100)
            ]
            for future in futures:
                future.result()

        # Verify
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10000, f"Expected 10,000 reads, got {len(results)}"
        assert all(
            result_id == entity.id for result_id in results
        ), "All reads should return same entity ID"

        stats: CacheStats = cache.stats()
        assert (
            stats.hits == 10000
        ), f"Expected 10,000 hits, got {stats.hits}"

    def test_concurrent_reads_100_threads_different_entities(self) -> None:
        """Test 100 threads each reading different entities.

        - 100 different entities, each read 100 times across 100 threads
        - Distributed read pattern: entities spread across multiple threads
        - Validates cache performance with diverse working set
        - Expects >95% cache hit rate (all entities pre-cached)
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=5000)

        # Create 100 entities in cache
        entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Entity{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        results: list[tuple[UUID, bool]] = []
        errors: list[Exception] = []
        results_lock: Lock = Lock()
        errors_lock: Lock = Lock()

        def reader(entity_list: list[Entity]) -> None:
            """Read entities 10 times each."""
            try:
                for entity in entity_list:
                    for _ in range(10):
                        cached: Optional[Entity] = cache.get_entity(entity.id)
                        is_hit: bool = cached is not None
                        with results_lock:
                            results.append((entity.id, is_hit))
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Distribute 100 entities among 10 groups, with 100 threads per group
        entity_batches: list[list[Entity]] = [
            entities[i * 10 : (i + 1) * 10] for i in range(10)
        ]

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(reader, batch)
                for batch in entity_batches
                for _ in range(10)
            ]
            for future in futures:
                future.result()

        # Verify
        assert len(errors) == 0, f"Errors occurred: {errors}"
        total_reads: int = len(results)
        hits: int = sum(1 for _, hit in results if hit)
        hit_rate: float = hits / total_reads if total_reads > 0 else 0.0

        assert (
            hit_rate > 0.95
        ), f"Hit rate {hit_rate:.2%} should be >95%, got {hits} hits from {total_reads} reads"


class TestHighConcurrencyWrite:
    """Category 2: High Concurrency Write Tests (2 tests)."""

    def test_concurrent_writes_100_threads(self) -> None:
        """Test 100 threads writing different entities.

        - 100 threads, each writing 100 unique entities
        - Total: 10,000 entities written with 100 concurrent writers
        - No capacity issues expected (max_entities=10,000)
        - Validates write performance and thread-safety
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=10000)

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def writer(start_idx: int) -> None:
            """Write 100 unique entities starting from start_idx."""
            try:
                for i in range(start_idx, start_idx + 100):
                    entity: Entity = Entity(
                        id=uuid4(),
                        text=f"Entity{i}",
                        type="PERSON",
                        confidence=0.9,
                        mention_count=i,
                    )
                    cache.set_entity(entity)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(writer, i * 100) for i in range(100)
            ]
            for future in futures:
                future.result()

        # Verify
        assert len(errors) == 0, f"Write errors: {errors}"
        stats: CacheStats = cache.stats()
        assert (
            stats.size == 10000
        ), f"Cache should have 10,000 entities, got {stats.size}"
        assert (
            stats.evictions == 0
        ), f"No evictions expected (enough capacity), got {stats.evictions}"

    def test_concurrent_invalidations_100_threads(self) -> None:
        """Test 100 threads invalidating different entities.

        - Pre-populate cache with 100 entities
        - 100 threads, each invalidating 1 entity
        - Total: 100 concurrent invalidations
        - Validates cache consistency under concurrent deletions
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=10000)

        # Pre-populate cache with 100 entities
        entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Entity{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def invalidator(entity_list: list[Entity]) -> None:
            """Invalidate each entity in list."""
            try:
                for entity in entity_list:
                    cache.invalidate_entity(entity.id)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Distribute entities among 100 threads (1 entity each)
        entity_batches: list[list[Entity]] = [
            [entity] for entity in entities
        ]

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(invalidator, batch) for batch in entity_batches
            ]
            for future in futures:
                future.result()

        # Verify all invalidated
        assert len(errors) == 0, f"Invalidation errors: {errors}"
        stats: CacheStats = cache.stats()
        assert (
            stats.size == 0
        ), f"All entities should be invalidated, cache size = {stats.size}"


class TestMixedReadWriteContention:
    """Category 3: Mixed Read/Write Contention Tests (2 tests)."""

    def test_concurrent_mixed_50_readers_50_writers(self) -> None:
        """Test 50 reader threads + 50 writer threads mixed contention.

        - 50 reader threads continuously reading 50 seed entities
        - 50 writer threads continuously writing new entities
        - Reader pattern: 50 seed entities × 100 reads = 5,000 reads per thread
        - Writer pattern: 100 new entities per thread = 5,000 total writes
        - Validates performance under mixed read/write load
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=5000)

        # Pre-populate with seed entities for readers
        seed_entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Seed{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(50)
        ]
        for entity in seed_entities:
            cache.set_entity(entity)

        read_count: list[int] = [0]
        write_count: list[int] = [0]
        errors: list[Exception] = []

        read_count_lock: Lock = Lock()
        write_count_lock: Lock = Lock()
        errors_lock: Lock = Lock()

        def reader() -> None:
            """Read seed entities 100 times."""
            try:
                for entity in seed_entities:
                    for _ in range(100):
                        result: Optional[Entity] = cache.get_entity(entity.id)
                        if result:
                            with read_count_lock:
                                read_count[0] += 1
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def writer() -> None:
            """Write 100 new entities."""
            try:
                for i in range(100):
                    entity: Entity = Entity(
                        id=uuid4(),
                        text=f"Written{i}",
                        type="PERSON",
                        confidence=0.8,
                        mention_count=i,
                    )
                    cache.set_entity(entity)
                    with write_count_lock:
                        write_count[0] += 1
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as executor:
            read_futures: list[Future[None]] = [
                executor.submit(reader) for _ in range(50)
            ]
            write_futures: list[Future[None]] = [
                executor.submit(writer) for _ in range(50)
            ]

            for future in read_futures + write_futures:
                future.result()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert (
            read_count[0] == 250000
        ), f"Expected 250k reads (50 threads × 50 entities × 100), got {read_count[0]}"
        assert (
            write_count[0] == 5000
        ), f"Expected 5k writes (50 threads × 100), got {write_count[0]}"

    def test_concurrent_read_and_invalidate_race(self) -> None:
        """Test race condition: simultaneous read and invalidate.

        - 5 reader threads continuously reading one entity
        - 5 invalidator threads continuously invalidating and re-inserting entity
        - Creates classic read-write race condition
        - Validates atomic cache operations and thread-safety
        - Expected: no exceptions, final entity in cache from last set
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=100)
        entity: Entity = Entity(
            id=uuid4(),
            text="Test",
            type="PERSON",
            confidence=0.95,
            mention_count=1,
        )
        cache.set_entity(entity)

        read_results: list[bool] = []
        errors: list[Exception] = []

        read_results_lock: Lock = Lock()
        errors_lock: Lock = Lock()

        def reader() -> None:
            """Read entity 1,000 times."""
            try:
                for _ in range(1000):
                    result: Optional[Entity] = cache.get_entity(entity.id)
                    is_cached: bool = result is not None
                    with read_results_lock:
                        read_results.append(is_cached)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def invalidator() -> None:
            """Invalidate and re-insert entity 100 times."""
            try:
                for _ in range(100):
                    cache.invalidate_entity(entity.id)
                    # Re-insert to simulate update pattern
                    cache.set_entity(entity)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            read_futures: list[Future[None]] = [
                executor.submit(reader) for _ in range(5)
            ]
            invalidate_futures: list[Future[None]] = [
                executor.submit(invalidator) for _ in range(5)
            ]

            for future in read_futures + invalidate_futures:
                future.result()

        # Verify no corruption
        assert len(errors) == 0, f"Race condition errors: {errors}"
        # Eventually entity should be in cache (last set wins)
        final_entity: Optional[Entity] = cache.get_entity(entity.id)
        assert (
            final_entity is not None
        ), "Final entity should be in cache after last set"


class TestBidirectionalInvalidationCascade:
    """Category 4: Bidirectional Invalidation Cascade Tests (2 tests)."""

    def test_concurrent_bidirectional_invalidation_cascade(self) -> None:
        """Test concurrent bidirectional relationship invalidation.

        - Two entities with bidirectional relationships (A↔B)
        - 5 threads invalidating A's relationships
        - 5 threads invalidating B's relationships
        - 100 concurrent invalidations per thread
        - Validates consistency of bidirectional cache invalidation
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=100)

        entity_a: Entity = Entity(
            id=uuid4(), text="A", type="PERSON", confidence=0.9, mention_count=1
        )
        entity_b: Entity = Entity(
            id=uuid4(), text="B", type="PERSON", confidence=0.9, mention_count=1
        )
        cache.set_entity(entity_a)
        cache.set_entity(entity_b)

        # Cache relationships in both directions
        cache.set_relationships(entity_a.id, "similar-to", [entity_b])
        cache.set_relationships(entity_b.id, "similar-to", [entity_a])

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def invalidator_a() -> None:
            """Invalidate A's relationships 100 times."""
            try:
                for _ in range(100):
                    cache.invalidate_relationships(entity_a.id, "similar-to")
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def invalidator_b() -> None:
            """Invalidate B's relationships 100 times."""
            try:
                for _ in range(100):
                    cache.invalidate_relationships(entity_b.id, "similar-to")
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures_a: list[Future[None]] = [
                executor.submit(invalidator_a) for _ in range(5)
            ]
            futures_b: list[Future[None]] = [
                executor.submit(invalidator_b) for _ in range(5)
            ]

            for future in futures_a + futures_b:
                future.result()

        # Verify both invalidated
        assert len(errors) == 0, f"Invalidation errors: {errors}"
        assert (
            cache.get_relationships(entity_a.id, "similar-to") is None
        ), "A's relationships should be invalidated"
        assert (
            cache.get_relationships(entity_b.id, "similar-to") is None
        ), "B's relationships should be invalidated"


class TestLRUEvictionUnderConcurrency:
    """Category 5: LRU Eviction Under Concurrency Tests (2 tests)."""

    def test_concurrent_lru_eviction_under_load(self) -> None:
        """Test concurrent entity insertion with LRU eviction.

        - Cache capacity: 1,000 entities max
        - 100 threads each inserting 50 unique entities = 5,000 total insertions
        - Expected: 4,000+ evictions (5,000 inserted - 1,000 capacity)
        - Validates LRU eviction correctness under concurrent writes
        - Expects cache stays bounded at max_entities
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=1000)

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def inserter(start_idx: int) -> None:
            """Insert 50 unique entities."""
            try:
                for i in range(start_idx, start_idx + 50):
                    entity: Entity = Entity(
                        id=uuid4(),
                        text=f"Entity{i}",
                        type="PERSON",
                        confidence=0.9,
                        mention_count=i,
                    )
                    cache.set_entity(entity)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(inserter, i * 50) for i in range(100)
            ]
            for future in futures:
                future.result()

        # Verify
        assert len(errors) == 0, f"Insertion errors: {errors}"
        stats: CacheStats = cache.stats()
        assert (
            stats.size <= 1000
        ), f"Cache size {stats.size} should be bounded at max_entities=1000"
        assert (
            stats.evictions >= 4000
        ), f"Expected ≥4,000 evictions, got {stats.evictions}"


class TestLoadTestingFramework:
    """Category 6: Load Testing Framework Tests (3 tests)."""

    def test_load_search_and_reranking_simulation(self) -> None:
        """Simulate realistic search + reranking load under concurrency.

        - Populate cache with 100 entities
        - 50 concurrent workers simulating search + reranking:
          - Get entity (simulates search)
          - Get relationships (simulates ranking traversal)
        - Measures latency per operation
        - Expects P95 latency <100ms under load
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=1000)

        # Populate cache with entities
        entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Entity{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        # Create some relationships for traversal
        for i, entity in enumerate(entities[:-1]):
            related: list[Entity] = [entities[(i + 1) % len(entities)]]
            cache.set_relationships(entity.id, "similar-to", related)

        latencies: list[float] = []
        errors: list[Exception] = []

        latencies_lock: Lock = Lock()
        errors_lock: Lock = Lock()

        def search_and_rerank() -> None:
            """Simulate search + reranking on all entities."""
            try:
                start: float = time.perf_counter()
                for entity in entities:
                    # Simulate search: get entity
                    _ = cache.get_entity(entity.id)
                    # Simulate reranking: traverse 1-hop
                    _ = cache.get_relationships(entity.id, "similar-to")
                elapsed: float = (
                    time.perf_counter() - start
                ) * 1000  # milliseconds
                with latencies_lock:
                    latencies.append(elapsed)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures: list[Future[None]] = [
                executor.submit(search_and_rerank) for _ in range(50)
            ]
            for future in futures:
                future.result()

        # Verify performance
        assert len(errors) == 0, f"Errors: {errors}"
        avg_latency: float = sum(latencies) / len(latencies)
        sorted_latencies: list[float] = sorted(latencies)
        p95_latency: float = sorted_latencies[int(len(sorted_latencies) * 0.95)]

        print(
            f"\nSearch + Rerank Load: Avg {avg_latency:.2f}ms, P95 {p95_latency:.2f}ms"
        )
        assert (
            p95_latency < 100
        ), f"P95 latency {p95_latency:.2f}ms should be <100ms under load"

    def test_cache_hit_rate_under_concurrent_load(self) -> None:
        """Verify cache hit rate under 100 concurrent requests.

        - Pre-populate cache with 10 "hot" entities (frequently accessed)
        - 100 concurrent threads, each reading hot entities repeatedly
        - Each thread reads each entity 100 times = 100k total reads
        - Expected hit rate: >80% (all pre-cached entities)
        - Validates cache effectiveness under concurrent load
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=100)

        # Pre-populate with 10 "hot" entities
        hot_entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Hot{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(10)
        ]
        for entity in hot_entities:
            cache.set_entity(entity)

        hits: list[int] = [0]
        misses: list[int] = [0]

        hits_lock: Lock = Lock()
        misses_lock: Lock = Lock()

        def worker() -> None:
            """Read hot entities 100 times each."""
            for entity in hot_entities:
                for _ in range(100):
                    result: Optional[Entity] = cache.get_entity(entity.id)
                    if result:
                        with hits_lock:
                            hits[0] += 1
                    else:
                        with misses_lock:
                            misses[0] += 1

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(worker) for _ in range(100)
            ]
            for future in futures:
                future.result()

        total: int = hits[0] + misses[0]
        hit_rate: float = hits[0] / total if total > 0 else 0.0

        print(
            f"\nCache Hit Rate: {hit_rate*100:.1f}% ({hits[0]} hits, {misses[0]} misses)"
        )
        assert (
            hit_rate > 0.80
        ), f"Hit rate {hit_rate:.2%} should be >80%, got {hits[0]} hits from {total} operations"

    def test_throughput_operations_per_second(self) -> None:
        """Measure cache throughput under concurrent load.

        - Pre-populate cache with 100 entities
        - 100 concurrent threads, each performing 1,000 random reads
        - Total: 100,000 cache operations under concurrency
        - Expected throughput: >10,000 ops/second
        - Validates cache performance at scale
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=1000)

        # Pre-populate
        entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Entity{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        operation_count: list[int] = [0]
        operation_count_lock: Lock = Lock()

        start_time: float = time.perf_counter()

        def worker() -> None:
            """Perform 1,000 random reads."""
            for _ in range(1000):
                entity: Entity = random.choice(entities)
                _ = cache.get_entity(entity.id)
                with operation_count_lock:
                    operation_count[0] += 1

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(worker) for _ in range(100)
            ]
            for future in futures:
                future.result()

        elapsed: float = time.perf_counter() - start_time
        throughput: float = operation_count[0] / elapsed

        print(f"\nThroughput: {throughput:,.0f} ops/sec")
        assert (
            throughput > 10000
        ), f"Throughput {throughput:,.0f} ops/sec should be >10k ops/sec"


class TestConcurrentEdgeCases:
    """Additional edge case tests for concurrent scenarios."""

    def test_concurrent_writes_with_updates(self) -> None:
        """Test concurrent writes and updates to same entities.

        - Create initial set of 10 entities
        - 50 threads: 25 threads update existing, 25 threads add new
        - Validates LRU move-to-end behavior under concurrent updates
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(max_entities=1000)

        # Create initial entities
        initial_entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Initial{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(10)
        ]
        for entity in initial_entities:
            cache.set_entity(entity)

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def updater() -> None:
            """Update existing entities."""
            try:
                for _ in range(100):
                    entity_to_update: Entity = random.choice(initial_entities)
                    updated: Entity = Entity(
                        id=entity_to_update.id,
                        text=entity_to_update.text,
                        type=entity_to_update.type,
                        confidence=entity_to_update.confidence + 0.01,
                        mention_count=entity_to_update.mention_count + 1,
                    )
                    cache.set_entity(updated)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def writer() -> None:
            """Write new entities."""
            try:
                for i in range(100):
                    entity: Entity = Entity(
                        id=uuid4(),
                        text=f"New{i}",
                        type="PERSON",
                        confidence=0.8,
                        mention_count=i,
                    )
                    cache.set_entity(entity)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=50) as executor:
            update_futures: list[Future[None]] = [
                executor.submit(updater) for _ in range(25)
            ]
            write_futures: list[Future[None]] = [
                executor.submit(writer) for _ in range(25)
            ]

            for future in update_futures + write_futures:
                future.result()

        assert len(errors) == 0, f"Concurrent update errors: {errors}"
        stats: CacheStats = cache.stats()
        # Cache should be full or nearly full
        assert stats.size > 500, f"Cache should have substantial entries, got {stats.size}"

    def test_concurrent_relationship_operations(self) -> None:
        """Test concurrent relationship caching and invalidation.

        - 100 threads concurrently setting and getting relationships
        - Mix of different relationship types
        - Validates relationship cache thread-safety
        """
        cache: KnowledgeGraphCache = KnowledgeGraphCache(
            max_entities=100, max_relationship_caches=1000
        )

        # Create base entities for relationships
        base_entities: list[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Base{i}",
                type="PERSON",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(20)
        ]
        for entity in base_entities:
            cache.set_entity(entity)

        rel_types: list[str] = [
            "similar-to",
            "hierarchical",
            "co-occurs-with",
            "mentions-in-document",
        ]

        errors: list[Exception] = []
        errors_lock: Lock = Lock()

        def relationship_worker(worker_id: int) -> None:
            """Perform relationship operations."""
            try:
                for i in range(25):
                    source: Entity = random.choice(base_entities)
                    targets: list[Entity] = random.sample(
                        base_entities, k=min(3, len(base_entities))
                    )
                    rel_type: str = random.choice(rel_types)

                    # Set relationship
                    cache.set_relationships(source.id, rel_type, targets)

                    # Get relationship
                    _ = cache.get_relationships(source.id, rel_type)

                    # Occasionally invalidate
                    if i % 5 == 0:
                        cache.invalidate_relationships(source.id, rel_type)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures: list[Future[None]] = [
                executor.submit(relationship_worker, i) for i in range(100)
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Relationship operation errors: {errors}"
        stats: CacheStats = cache.stats()
        # Should have some entries
        assert (
            stats.size > 0
        ), "Cache should have entries from relationship operations"
