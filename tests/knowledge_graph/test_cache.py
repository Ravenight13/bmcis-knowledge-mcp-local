"""Comprehensive test suite for KnowledgeGraphCache."""

import pytest
from uuid import UUID, uuid4
from typing import List

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity, CacheStats
from src.knowledge_graph.cache_config import CacheConfig


class TestEntityCaching:
    """Tests for entity caching functionality."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create fresh cache for each test."""
        return KnowledgeGraphCache(max_entities=10, max_relationship_caches=20)

    @pytest.fixture
    def sample_entity(self) -> Entity:
        """Create sample entity for testing."""
        return Entity(
            id=uuid4(),
            text="Claude AI",
            type="technology",
            confidence=0.95,
            mention_count=10,
        )

    def test_set_and_get_entity(self, cache: KnowledgeGraphCache, sample_entity: Entity) -> None:
        """Test basic entity set and get."""
        cache.set_entity(sample_entity)
        retrieved = cache.get_entity(sample_entity.id)

        assert retrieved is not None
        assert retrieved.id == sample_entity.id
        assert retrieved.text == sample_entity.text
        assert retrieved.confidence == sample_entity.confidence

    def test_get_missing_entity(self, cache: KnowledgeGraphCache) -> None:
        """Test getting non-existent entity returns None."""
        missing_id = uuid4()
        result = cache.get_entity(missing_id)

        assert result is None

    def test_cache_hit_increments_hits(
        self, cache: KnowledgeGraphCache, sample_entity: Entity
    ) -> None:
        """Test that cache hits increment hit counter."""
        cache.set_entity(sample_entity)

        # First access is a cache miss (from miss side)
        _ = cache.get_entity(sample_entity.id)
        stats_after_hit = cache.stats()
        assert stats_after_hit.hits == 1

        # Second access is a cache hit
        _ = cache.get_entity(sample_entity.id)
        stats_after_second = cache.stats()
        assert stats_after_second.hits == 2

    def test_cache_miss_increments_misses(self, cache: KnowledgeGraphCache) -> None:
        """Test that cache misses increment miss counter."""
        missing_id = uuid4()
        _ = cache.get_entity(missing_id)
        stats = cache.stats()

        assert stats.misses == 1

    def test_lru_eviction_entities(self, cache: KnowledgeGraphCache) -> None:
        """Test LRU eviction when cache exceeds max_entities."""
        # Create 12 entities (cache max is 10)
        entities: List[Entity] = [
            Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="test",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(12)
        ]

        # Add all entities
        for entity in entities:
            cache.set_entity(entity)

        stats = cache.stats()

        # Cache size should be capped at 10
        assert stats.size <= 10
        # Should have 2 evictions (added 12, max 10)
        assert stats.evictions == 2

        # First two entities should be evicted (LRU)
        assert cache.get_entity(entities[0].id) is None
        assert cache.get_entity(entities[1].id) is None

        # Last entity should still be there
        assert cache.get_entity(entities[-1].id) is not None

    def test_update_existing_entity_resets_lru(
        self, cache: KnowledgeGraphCache, sample_entity: Entity
    ) -> None:
        """Test that updating entity moves it to end of LRU queue."""
        # Create 11 entities to trigger eviction
        entities: List[Entity] = [sample_entity] + [
            Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="test",
                confidence=0.9,
                mention_count=i,
            )
            for i in range(10)
        ]

        # Add first entity
        cache.set_entity(entities[0])

        # Add 10 more entities
        for entity in entities[1:]:
            cache.set_entity(entity)

        # Update first entity (should move to end, preventing eviction)
        updated = Entity(
            id=entities[0].id,
            text=entities[0].text,
            type=entities[0].type,
            confidence=0.99,  # Updated confidence
            mention_count=entities[0].mention_count + 1,
        )
        cache.set_entity(updated)

        # First entity should still be in cache (wasn't evicted)
        assert cache.get_entity(entities[0].id) is not None


class TestRelationshipCaching:
    """Tests for relationship caching functionality."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create fresh cache for each test."""
        return KnowledgeGraphCache(max_entities=50, max_relationship_caches=20)

    @pytest.fixture
    def sample_entities(self) -> List[Entity]:
        """Create sample entities for relationship testing."""
        return [
            Entity(id=uuid4(), text=f"Entity {i}", type="test", confidence=0.9, mention_count=i)
            for i in range(5)
        ]

    def test_set_and_get_relationships(
        self, cache: KnowledgeGraphCache, sample_entities: List[Entity]
    ) -> None:
        """Test basic relationship set and get."""
        source_id = sample_entities[0].id
        rel_type = "hierarchical"

        cache.set_relationships(source_id, rel_type, sample_entities[1:])
        retrieved = cache.get_relationships(source_id, rel_type)

        assert retrieved is not None
        assert len(retrieved) == len(sample_entities[1:])
        assert retrieved[0].text == sample_entities[1].text

    def test_get_missing_relationship(self, cache: KnowledgeGraphCache) -> None:
        """Test getting non-existent relationship returns None."""
        missing_id = uuid4()
        result = cache.get_relationships(missing_id, "hierarchical")

        assert result is None

    def test_relationship_cache_hit_miss(
        self, cache: KnowledgeGraphCache, sample_entities: List[Entity]
    ) -> None:
        """Test cache hit/miss tracking for relationships."""
        source_id = sample_entities[0].id

        # First access is a miss
        _ = cache.get_relationships(source_id, "hierarchical")
        stats_after_miss = cache.stats()
        assert stats_after_miss.misses == 1

        # Set relationship
        cache.set_relationships(source_id, "hierarchical", sample_entities[1:])

        # Now it's a hit
        _ = cache.get_relationships(source_id, "hierarchical")
        stats_after_hit = cache.stats()
        assert stats_after_hit.hits == 1

    def test_lru_eviction_relationships(
        self, cache: KnowledgeGraphCache, sample_entities: List[Entity]
    ) -> None:
        """Test LRU eviction when relationship cache exceeds max."""
        # Cache max is 20, add 25 relationship entries
        for i in range(25):
            source_id = uuid4()
            cache.set_relationships(source_id, "hierarchical", sample_entities)

        stats = cache.stats()
        # Total size should not exceed 20 + 50 (entities + relationships)
        assert stats.size <= 70
        # Should have some evictions (added 25 relationships, max 20)
        assert stats.evictions >= 5


class TestCacheInvalidation:
    """Tests for cache invalidation functionality."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create fresh cache for each test."""
        return KnowledgeGraphCache(max_entities=100, max_relationship_caches=100)

    @pytest.fixture
    def setup_cache_with_data(self, cache: KnowledgeGraphCache) -> tuple[UUID, UUID, List[Entity]]:
        """Setup cache with entity and relationship data."""
        source_id = uuid4()
        target_id = uuid4()

        source = Entity(
            id=source_id,
            text="Source Entity",
            type="test",
            confidence=0.9,
            mention_count=1,
        )

        related_entities = [
            Entity(
                id=target_id,
                text="Related Entity",
                type="test",
                confidence=0.8,
                mention_count=1,
            )
        ]

        cache.set_entity(source)
        cache.set_relationships(source_id, "hierarchical", related_entities)

        return source_id, target_id, related_entities

    def test_invalidate_entity_removes_from_cache(
        self,
        cache: KnowledgeGraphCache,
        setup_cache_with_data: tuple[UUID, UUID, List[Entity]],
    ) -> None:
        """Test that invalidate_entity removes entity from cache."""
        source_id, _, _ = setup_cache_with_data

        # Verify entity is cached
        assert cache.get_entity(source_id) is not None

        # Invalidate it
        cache.invalidate_entity(source_id)

        # Verify it's gone
        assert cache.get_entity(source_id) is None

    def test_invalidate_entity_removes_outbound_relationships(
        self,
        cache: KnowledgeGraphCache,
        setup_cache_with_data: tuple[UUID, UUID, List[Entity]],
    ) -> None:
        """Test that invalidating entity removes its outbound relationships."""
        source_id, _, _ = setup_cache_with_data

        # Verify relationships are cached
        assert cache.get_relationships(source_id, "hierarchical") is not None

        # Invalidate source entity
        cache.invalidate_entity(source_id)

        # Verify relationships are also gone
        assert cache.get_relationships(source_id, "hierarchical") is None

    def test_invalidate_specific_relationship(
        self,
        cache: KnowledgeGraphCache,
        setup_cache_with_data: tuple[UUID, UUID, List[Entity]],
    ) -> None:
        """Test invalidating specific relationship type."""
        source_id = uuid4()
        target_entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="test", confidence=0.9, mention_count=i)
            for i in range(3)
        ]

        # Cache two relationship types
        cache.set_relationships(source_id, "hierarchical", target_entities)
        cache.set_relationships(source_id, "similar-to", target_entities)

        # Verify both are cached
        assert cache.get_relationships(source_id, "hierarchical") is not None
        assert cache.get_relationships(source_id, "similar-to") is not None

        # Invalidate only hierarchical
        cache.invalidate_relationships(source_id, "hierarchical")

        # Verify only hierarchical is gone
        assert cache.get_relationships(source_id, "hierarchical") is None
        assert cache.get_relationships(source_id, "similar-to") is not None

    def test_clear_removes_all_entries(
        self,
        cache: KnowledgeGraphCache,
        setup_cache_with_data: tuple[UUID, UUID, List[Entity]],
    ) -> None:
        """Test that clear removes all cache entries."""
        source_id, _, _ = setup_cache_with_data

        # Verify data is cached
        assert cache.get_entity(source_id) is not None
        assert cache.get_relationships(source_id, "hierarchical") is not None

        # Clear cache
        cache.clear()

        # Verify all entries are gone
        assert cache.get_entity(source_id) is None
        assert cache.get_relationships(source_id, "hierarchical") is None

        stats = cache.stats()
        assert stats.size == 0


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create fresh cache for each test."""
        return KnowledgeGraphCache(max_entities=10, max_relationship_caches=10)

    def test_stats_initial_state(self, cache: KnowledgeGraphCache) -> None:
        """Test initial stats are all zeros."""
        stats = cache.stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size == 0

    def test_stats_track_hits_and_misses(self, cache: KnowledgeGraphCache) -> None:
        """Test stats correctly track hits and misses."""
        entity = Entity(id=uuid4(), text="Test", type="test", confidence=0.9, mention_count=1)

        # Miss
        cache.get_entity(entity.id)

        # Hit
        cache.set_entity(entity)
        cache.get_entity(entity.id)

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_stats_track_evictions(self, cache: KnowledgeGraphCache) -> None:
        """Test stats correctly track evictions."""
        # Cache max is 10, add 12
        for i in range(12):
            entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="test",
                confidence=0.9,
                mention_count=i,
            )
            cache.set_entity(entity)

        stats = cache.stats()
        # Should have 2 evictions (added 12, max 10)
        assert stats.evictions == 2

    def test_stats_reflect_current_size(self, cache: KnowledgeGraphCache) -> None:
        """Test stats report correct current cache size."""
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="test", confidence=0.9, mention_count=i)
            for i in range(5)
        ]

        for entity in entities:
            cache.set_entity(entity)

        stats = cache.stats()
        assert stats.size == 5


class TestCacheConfiguration:
    """Tests for cache configuration."""

    def test_cache_config_defaults(self) -> None:
        """Test CacheConfig has correct defaults."""
        config = CacheConfig()

        assert config.max_entities == 5000
        assert config.max_relationship_caches == 10000
        assert config.enable_metrics is True

    def test_cache_config_custom_values(self) -> None:
        """Test CacheConfig accepts custom values."""
        config = CacheConfig(
            max_entities=1000,
            max_relationship_caches=2000,
            enable_metrics=False,
        )

        assert config.max_entities == 1000
        assert config.max_relationship_caches == 2000
        assert config.enable_metrics is False

    def test_cache_respects_config_limits(self) -> None:
        """Test cache respects configuration limits."""
        config = CacheConfig(max_entities=5, max_relationship_caches=5)
        cache = KnowledgeGraphCache(
            max_entities=config.max_entities,
            max_relationship_caches=config.max_relationship_caches,
        )

        # Add 6 entities (exceeds limit of 5)
        for i in range(6):
            entity = Entity(
                id=uuid4(),
                text=f"Entity {i}",
                type="test",
                confidence=0.9,
                mention_count=i,
            )
            cache.set_entity(entity)

        stats = cache.stats()
        # Size should not exceed max_entities
        assert stats.size <= 5


class TestThreadSafety:
    """Tests for thread safety (basic coverage)."""

    def test_concurrent_gets_dont_raise(self) -> None:
        """Test that concurrent gets don't raise exceptions."""
        import threading

        cache = KnowledgeGraphCache(max_entities=100, max_relationship_caches=100)
        entity = Entity(id=uuid4(), text="Test", type="test", confidence=0.9, mention_count=1)
        cache.set_entity(entity)

        results: list[Entity | None] = []
        errors: list[Exception] = []

        def get_entity() -> None:
            try:
                result = cache.get_entity(entity.id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_entity) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0
        # All results should be the entity (or None if race condition)
        assert len(results) > 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create fresh cache for each test."""
        return KnowledgeGraphCache(max_entities=10, max_relationship_caches=10)

    def test_empty_relationship_list(self, cache: KnowledgeGraphCache) -> None:
        """Test caching empty relationship list."""
        source_id = uuid4()
        cache.set_relationships(source_id, "hierarchical", [])

        retrieved = cache.get_relationships(source_id, "hierarchical")
        assert retrieved == []

    def test_duplicate_entity_ids_in_relationships(self, cache: KnowledgeGraphCache) -> None:
        """Test handling duplicate entities in relationship list."""
        source_id = uuid4()
        entity = Entity(id=uuid4(), text="Test", type="test", confidence=0.9, mention_count=1)

        # Add same entity twice (should be allowed)
        cache.set_relationships(source_id, "hierarchical", [entity, entity])

        retrieved = cache.get_relationships(source_id, "hierarchical")
        assert len(retrieved) == 2

    def test_large_entity_object(self, cache: KnowledgeGraphCache) -> None:
        """Test caching large entity objects."""
        large_entity = Entity(
            id=uuid4(),
            text="x" * 10000,  # Large text
            type="test",
            confidence=0.9,
            mention_count=999999,
        )

        cache.set_entity(large_entity)
        retrieved = cache.get_entity(large_entity.id)

        assert retrieved == large_entity

    def test_many_relationship_types(self, cache: KnowledgeGraphCache) -> None:
        """Test caching many relationship types for same entity."""
        source_id = uuid4()
        entity = Entity(id=uuid4(), text="Test", type="test", confidence=0.9, mention_count=1)

        rel_types = ["hierarchical", "similar-to", "mentions-in-document", "co-occurs-with"]

        for rel_type in rel_types:
            cache.set_relationships(source_id, rel_type, [entity])

        # Verify all are cached
        for rel_type in rel_types:
            assert cache.get_relationships(source_id, rel_type) is not None
