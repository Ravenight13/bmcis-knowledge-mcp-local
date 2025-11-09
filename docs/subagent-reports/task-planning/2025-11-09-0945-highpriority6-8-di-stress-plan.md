# High Priority Issues 6 & 8: Dependency Injection Refactor and Concurrent Cache Stress Tests

**Planning Document**
**Date**: 2025-11-09
**Issues**: #6 (Dependency Injection Missing), #8 (Concurrent Cache Stress Tests - Partial)
**Status**: Planning Complete - Ready for Implementation
**Dependencies**: Blocker 3 (Repository Integration) must complete first

---

## Executive Summary

This document provides detailed specifications for two critical architectural improvements to the knowledge graph caching system:

1. **Issue 6 - Dependency Injection Refactor**: Refactor `KnowledgeGraphService` to use constructor injection with Protocol-based interfaces, enabling cache implementation swapping (LRU → Redis) without code changes and improving testability.

2. **Issue 8 - Concurrent Cache Stress Tests**: Implement comprehensive concurrent stress tests (100+ threads) to verify cache thread-safety under realistic mixed read/write workloads, including cache invalidation race conditions.

**Key Benefits**:
- **DI Refactor**: Future-proof architecture for Redis migration, improved testability with mock injection, maintains backward compatibility
- **Stress Tests**: Production-grade thread-safety verification, performance validation under load (P95 <10ms maintained), detection of race conditions

**Timeline**: 8.5-9.5 hours implementation (can parallelize DI + stress tests)
**Success Criteria**: All stress tests pass, backward compatibility maintained, P95 latency targets met

---

## Issue 6: Dependency Injection Refactor

### Problem Statement

**Current Architecture Limitation**:
```python
class KnowledgeGraphService:
    def __init__(self, db_session, cache=None, cache_config=None):
        self._db_session = db_session

        # Hardcoded dependency - tightly couples to KnowledgeGraphCache implementation
        if cache is None:
            config = cache_config if cache_config is not None else CacheConfig()
            self._cache = KnowledgeGraphCache(
                max_entities=config.max_entities,
                max_relationship_caches=config.max_relationship_caches,
            )
        else:
            self._cache = cache  # Type not enforced via protocol
```

**Problems**:
1. **No Interface Contract**: `cache` parameter has no type hint/protocol, allowing any object to be passed
2. **Reduced Testability**: Can't easily inject test mocks with assertion tracking
3. **Violates Dependency Inversion Principle**: Service depends on concrete implementation, not abstraction
4. **Hard to Swap Implementations**: Switching to Redis cache requires code changes across codebase

**Impact**:
- **Future Redis Migration**: Cannot swap to Redis without modifying service code
- **Testing Friction**: Mock cache requires guessing method signatures, no IDE autocomplete
- **Maintenance Risk**: Changing cache interface breaks all consumers without compile-time errors

---

### Solution: Protocol-Based Constructor Injection

**Approach**: Define `CacheProtocol` interface using Python's `typing.Protocol`, update service to accept protocol-typed cache parameter.

**Architecture Decision**: Constructor Injection (over Factory Pattern) for:
- **Simplicity**: Direct, explicit dependencies
- **Testability**: Easy mock injection in tests
- **Flexibility**: Supports both default and custom implementations
- **IDE Support**: Full autocomplete and type checking

---

### Implementation Specification

#### 1. Define Cache Protocol Interface

**File**: `src/knowledge_graph/cache_protocol.py` (NEW)

**Purpose**: Establish interface contract that all cache implementations must satisfy.

```python
"""Protocol interface for knowledge graph cache implementations."""

from typing import Protocol, Optional, List
from uuid import UUID

from src.knowledge_graph.cache import Entity, CacheStats


class CacheProtocol(Protocol):
    """Protocol defining the interface for knowledge graph cache implementations.

    All cache implementations (in-memory LRU, Redis, etc.) must implement these methods.
    This enables dependency injection and allows swapping cache backends without
    changing service code.

    Thread Safety: All methods must be thread-safe (safe for concurrent access).
    """

    # Entity operations
    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get cached entity by ID.

        Args:
            entity_id: UUID of entity to retrieve

        Returns:
            Entity if found in cache, None otherwise

        Thread Safety: Must be safe to call concurrently
        """
        ...

    def set_entity(self, entity: Entity) -> None:
        """Cache entity object.

        Args:
            entity: Entity object to cache

        Thread Safety: Must be safe to call concurrently with get_entity
        """
        ...

    # Relationship operations
    def get_relationships(
        self, entity_id: UUID, rel_type: str
    ) -> Optional[List[Entity]]:
        """Get cached 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type (e.g., 'hierarchical', 'mentions-in-document')

        Returns:
            List of related entities if cached, None otherwise

        Thread Safety: Must be safe to call concurrently
        """
        ...

    def set_relationships(
        self, entity_id: UUID, rel_type: str, entities: List[Entity]
    ) -> None:
        """Cache 1-hop relationships for entity.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type
            entities: List of related entities

        Thread Safety: Must be safe to call concurrently with get_relationships
        """
        ...

    # Cache invalidation
    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity and all outbound 1-hop caches.

        Args:
            entity_id: Entity UUID to invalidate

        Thread Safety: Must be safe to call concurrently with all other methods
        """
        ...

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate specific relationship cache.

        Args:
            entity_id: Source entity UUID
            rel_type: Relationship type to invalidate

        Thread Safety: Must be safe to call concurrently
        """
        ...

    # Utility operations
    def clear(self) -> None:
        """Clear all cache entries (entity and relationship).

        Thread Safety: Must be safe to call concurrently (will acquire exclusive lock)
        """
        ...

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hits, misses, evictions, and current size

        Thread Safety: Must be safe to call concurrently
        """
        ...

    # Public properties (for configuration introspection)
    @property
    def max_entities(self) -> int:
        """Maximum entity entries in cache."""
        ...

    @property
    def max_relationship_caches(self) -> int:
        """Maximum relationship cache entries."""
        ...
```

**Design Notes**:
- **Protocol vs ABC**: Using `Protocol` for structural subtyping (duck typing with type checking)
- **Thread Safety**: All methods documented as thread-safe (implementation must honor this)
- **Optional Return Types**: `None` indicates cache miss (not error)
- **Properties**: Exposed for monitoring/debugging (e.g., cache size limits)

**Effort**: 45 minutes (protocol definition + documentation)

---

#### 2. Update Service Constructor

**File**: `src/knowledge_graph/graph_service.py` (MODIFY)

**Changes**:
```python
"""Graph query service with integrated LRU cache for knowledge graph."""

from __future__ import annotations

from typing import Optional, List, Any
from uuid import UUID
import logging

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity
from src.knowledge_graph.cache_config import CacheConfig
from src.knowledge_graph.cache_protocol import CacheProtocol  # NEW IMPORT

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Graph query service with integrated LRU cache.

    Provides high-level interface for entity and relationship queries:
    - Checks cache first for hot path optimization
    - Falls back to database for cache misses
    - Manages cache invalidation on writes

    Expected performance:
    - Cache hit: 1-2 microseconds (in-memory OrderedDict lookup)
    - Cache miss + DB query: 5-20ms (normalized schema with indexes)
    - Overall with >80% hit rate: P95 <10ms for 1-hop queries

    Dependency Injection:
    - Accepts optional cache implementation via CacheProtocol interface
    - Defaults to in-memory LRU cache if not provided
    - Enables testing with mock caches and future Redis migration
    """

    def __init__(
        self,
        db_session: Any,
        cache: Optional[CacheProtocol] = None,  # CHANGED: Type hint to CacheProtocol
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Initialize graph service with database session and optional cache.

        Args:
            db_session: SQLAlchemy session for database access
            cache: CacheProtocol implementation (optional, defaults to KnowledgeGraphCache)
            cache_config: CacheConfig instance (optional, used only if cache not provided)

        Examples:
            # Default in-memory cache
            service = KnowledgeGraphService(db_session)

            # Custom cache with config
            config = CacheConfig(max_entities=10000)
            service = KnowledgeGraphService(db_session, cache_config=config)

            # Inject specific cache implementation (e.g., for testing)
            mock_cache = MockCache()
            service = KnowledgeGraphService(db_session, cache=mock_cache)

            # Future Redis cache
            redis_cache = RedisCache(redis_client)
            service = KnowledgeGraphService(db_session, cache=redis_cache)
        """
        self._db_session: Any = db_session

        # Initialize cache if not provided (backward compatible default behavior)
        if cache is None:
            config = cache_config if cache_config is not None else CacheConfig()
            self._cache: CacheProtocol = KnowledgeGraphCache(
                max_entities=config.max_entities,
                max_relationship_caches=config.max_relationship_caches,
            )
        else:
            # Use provided cache (must satisfy CacheProtocol)
            self._cache: CacheProtocol = cache

        logger.info(
            f"Initialized KnowledgeGraphService with cache "
            f"(max_entities={self._cache.max_entities}, "
            f"max_relationships={self._cache.max_relationship_caches})"
        )

    # ... rest of methods unchanged (already use self._cache correctly)
```

**Changes Summary**:
1. Import `CacheProtocol` from new protocol module
2. Type hint `cache` parameter as `Optional[CacheProtocol]`
3. Type hint `self._cache` as `CacheProtocol`
4. Add docstring examples showing DI usage
5. No changes to method implementations (already generic)

**Backward Compatibility**: YES
- Existing code passes `None` for cache → gets default `KnowledgeGraphCache`
- Existing code passes `KnowledgeGraphCache` instance → works (satisfies protocol)
- No breaking changes to API

**Effort**: 30 minutes (type hints + docstrings + testing)

---

#### 3. Verify Cache Implementation Satisfies Protocol

**File**: `src/knowledge_graph/cache.py` (NO CHANGES REQUIRED)

**Verification**: Current `KnowledgeGraphCache` implementation already satisfies `CacheProtocol`:

```python
class KnowledgeGraphCache:
    # Already implements all protocol methods:
    def get_entity(self, entity_id: UUID) -> Optional[Entity]: ...  ✓
    def set_entity(self, entity: Entity) -> None: ...  ✓
    def get_relationships(self, entity_id: UUID, rel_type: str) -> Optional[List[Entity]]: ...  ✓
    def set_relationships(self, entity_id: UUID, rel_type: str, entities: List[Entity]) -> None: ...  ✓
    def invalidate_entity(self, entity_id: UUID) -> None: ...  ✓
    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None: ...  ✓
    def clear(self) -> None: ...  ✓
    def stats(self) -> CacheStats: ...  ✓

    # Properties
    @property
    def max_entities(self) -> int:
        return self._max_entities  ✓

    @property
    def max_relationship_caches(self) -> int:
        return self._max_relationship_caches  ✓
```

**Action Required**: Add `@property` decorators for `max_entities` and `max_relationship_caches` if not already present (currently accessed via `self.max_entities`, need to verify compatibility).

**Effort**: 15 minutes (add properties if needed + type check validation)

---

#### 4. Update Tests to Inject Mock Caches

**File**: `tests/knowledge_graph/test_service_di.py` (NEW)

**Purpose**: Demonstrate DI in action with mock caches that track method calls.

```python
"""Tests for KnowledgeGraphService dependency injection."""

import pytest
from uuid import UUID, uuid4
from typing import Optional, List

from src.knowledge_graph.graph_service import KnowledgeGraphService
from src.knowledge_graph.cache import Entity, CacheStats
from src.knowledge_graph.cache_protocol import CacheProtocol


class MockCache:
    """Mock cache for testing DI and method call tracking.

    Implements CacheProtocol with tracking of all method calls.
    Used to verify service correctly delegates to cache.
    """

    def __init__(self):
        """Initialize mock with empty tracking lists."""
        # Track method calls
        self.get_entity_calls: List[UUID] = []
        self.set_entity_calls: List[Entity] = []
        self.get_relationships_calls: List[tuple[UUID, str]] = []
        self.set_relationships_calls: List[tuple[UUID, str, List[Entity]]] = []
        self.invalidate_entity_calls: List[UUID] = []
        self.invalidate_relationships_calls: List[tuple[UUID, str]] = []
        self.clear_calls: int = 0
        self.stats_calls: int = 0

        # Mock data store
        self._entities: dict[UUID, Entity] = {}
        self._relationships: dict[tuple[UUID, str], List[Entity]] = {}

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Track get_entity calls and return mock data."""
        self.get_entity_calls.append(entity_id)
        return self._entities.get(entity_id)

    def set_entity(self, entity: Entity) -> None:
        """Track set_entity calls and store mock data."""
        self.set_entity_calls.append(entity)
        self._entities[entity.id] = entity

    def get_relationships(self, entity_id: UUID, rel_type: str) -> Optional[List[Entity]]:
        """Track get_relationships calls and return mock data."""
        self.get_relationships_calls.append((entity_id, rel_type))
        return self._relationships.get((entity_id, rel_type))

    def set_relationships(self, entity_id: UUID, rel_type: str, entities: List[Entity]) -> None:
        """Track set_relationships calls and store mock data."""
        self.set_relationships_calls.append((entity_id, rel_type, entities))
        self._relationships[(entity_id, rel_type)] = entities

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Track invalidate_entity calls."""
        self.invalidate_entity_calls.append(entity_id)
        self._entities.pop(entity_id, None)

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Track invalidate_relationships calls."""
        self.invalidate_relationships_calls.append((entity_id, rel_type))
        self._relationships.pop((entity_id, rel_type), None)

    def clear(self) -> None:
        """Track clear calls."""
        self.clear_calls += 1
        self._entities.clear()
        self._relationships.clear()

    def stats(self) -> CacheStats:
        """Track stats calls and return mock data."""
        self.stats_calls += 1
        return CacheStats(hits=10, misses=5, evictions=2, size=len(self._entities), max_size=100)

    @property
    def max_entities(self) -> int:
        """Return mock max_entities."""
        return 5000

    @property
    def max_relationship_caches(self) -> int:
        """Return mock max_relationship_caches."""
        return 10000


class TestServiceDependencyInjection:
    """Tests for service DI with mock caches."""

    @pytest.fixture
    def mock_cache(self) -> MockCache:
        """Create fresh mock cache for each test."""
        return MockCache()

    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return None  # Service doesn't use it in current stub implementation

    def test_service_accepts_injected_cache(self, db_session, mock_cache):
        """Test that service accepts cache via constructor."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        # Verify service uses the injected cache
        assert service._cache is mock_cache

    def test_service_delegates_get_entity_to_cache(self, db_session, mock_cache):
        """Test that service delegates get_entity to injected cache."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        entity_id = uuid4()
        entity = Entity(id=entity_id, text="Test", type="PERSON", confidence=0.9, mention_count=5)
        mock_cache.set_entity(entity)

        # Call service method
        result = service.get_entity(entity_id)

        # Verify cache method was called
        assert entity_id in mock_cache.get_entity_calls
        assert result == entity

    def test_service_delegates_traverse_1hop_to_cache(self, db_session, mock_cache):
        """Test that service delegates traverse_1hop to injected cache."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        entity_id = uuid4()
        related_entity = Entity(id=uuid4(), text="Related", type="PERSON", confidence=0.8, mention_count=3)
        mock_cache.set_relationships(entity_id, "mentions", [related_entity])

        # Call service method
        result = service.traverse_1hop(entity_id, "mentions")

        # Verify cache method was called
        assert (entity_id, "mentions") in mock_cache.get_relationships_calls
        assert result == [related_entity]

    def test_service_delegates_invalidate_entity_to_cache(self, db_session, mock_cache):
        """Test that service delegates invalidate_entity to injected cache."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        entity_id = uuid4()

        # Call service method
        service.invalidate_entity(entity_id)

        # Verify cache method was called
        assert entity_id in mock_cache.invalidate_entity_calls

    def test_service_delegates_get_cache_stats_to_cache(self, db_session, mock_cache):
        """Test that service delegates get_cache_stats to injected cache."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        # Call service method
        stats = service.get_cache_stats()

        # Verify cache method was called
        assert mock_cache.stats_calls == 1
        assert stats['hits'] == 10
        assert stats['misses'] == 5

    def test_service_caches_entity_on_retrieval(self, db_session, mock_cache):
        """Test that service calls set_entity after database retrieval (cache miss)."""
        service = KnowledgeGraphService(db_session, cache=mock_cache)

        entity_id = uuid4()

        # Call get_entity (will be cache miss, service should cache result)
        # NOTE: Current service implementation returns None for DB stub
        # This test validates the *pattern* - would work with real DB
        result = service.get_entity(entity_id)

        # Verify cache was checked
        assert entity_id in mock_cache.get_entity_calls

        # In real implementation with DB, would verify set_entity was called
        # For now, just verify the cache check happened


class TestServiceBackwardCompatibility:
    """Tests for backward compatibility (default cache behavior)."""

    @pytest.fixture
    def db_session(self):
        """Mock database session."""
        return None

    def test_service_creates_default_cache_when_none_provided(self, db_session):
        """Test that service creates KnowledgeGraphCache if no cache provided."""
        from src.knowledge_graph.cache import KnowledgeGraphCache

        service = KnowledgeGraphService(db_session)

        # Verify default cache was created
        assert isinstance(service._cache, KnowledgeGraphCache)

    def test_service_respects_cache_config_for_default_cache(self, db_session):
        """Test that service uses cache_config when creating default cache."""
        from src.knowledge_graph.cache_config import CacheConfig

        config = CacheConfig(max_entities=1000, max_relationship_caches=2000)
        service = KnowledgeGraphService(db_session, cache_config=config)

        # Verify config was applied
        assert service._cache.max_entities == 1000
        assert service._cache.max_relationship_caches == 2000
```

**Test Coverage**:
- ✓ Service accepts injected cache via constructor
- ✓ Service delegates all cache operations to injected cache
- ✓ Mock cache tracks method calls for verification
- ✓ Backward compatibility: default cache creation still works
- ✓ Cache config respected when creating default cache

**Effort**: 1.5 hours (mock implementation + comprehensive tests)

---

### Redis Cache Implementation (Future Phase 3)

**File**: `src/knowledge_graph/redis_cache.py` (FUTURE - NOT REQUIRED FOR PHASE 1)

**Purpose**: Demonstrate protocol-based cache swapping (reference implementation for future).

```python
"""Redis-based cache implementation for knowledge graph (FUTURE)."""

from typing import Optional, List
from uuid import UUID
import redis
import json

from src.knowledge_graph.cache import Entity, CacheStats
from src.knowledge_graph.cache_protocol import CacheProtocol


class RedisCache:
    """Redis-backed cache for knowledge graph entities and relationships.

    Implements CacheProtocol using Redis as storage backend.
    Enables distributed caching across multiple service instances.

    Features:
    - TTL-based expiration (default 1 hour)
    - JSON serialization for Entity objects
    - Atomic operations for thread-safety
    - Hit/miss tracking via Redis INCR
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 3600,
        max_entities: int = 5000,
        max_relationship_caches: int = 10000,
    ):
        """Initialize Redis cache.

        Args:
            redis_client: Configured Redis client
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_entities: Logical limit for monitoring (not enforced by Redis)
            max_relationship_caches: Logical limit for monitoring
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self._max_entities = max_entities
        self._max_relationship_caches = max_relationship_caches

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get cached entity from Redis."""
        key = f"entity:{entity_id}"
        cached = self.redis.get(key)

        if cached:
            self.redis.incr("cache:hits")
            return Entity(**json.loads(cached))
        else:
            self.redis.incr("cache:misses")
            return None

    def set_entity(self, entity: Entity) -> None:
        """Cache entity in Redis with TTL."""
        key = f"entity:{entity.id}"
        self.redis.setex(
            key,
            self.ttl,
            json.dumps({
                "id": str(entity.id),
                "text": entity.text,
                "type": entity.type,
                "confidence": entity.confidence,
                "mention_count": entity.mention_count,
            })
        )

    def get_relationships(self, entity_id: UUID, rel_type: str) -> Optional[List[Entity]]:
        """Get cached relationships from Redis."""
        key = f"rel:{entity_id}:{rel_type}"
        cached = self.redis.get(key)

        if cached:
            self.redis.incr("cache:hits")
            entities_data = json.loads(cached)
            return [Entity(**e) for e in entities_data]
        else:
            self.redis.incr("cache:misses")
            return None

    def set_relationships(self, entity_id: UUID, rel_type: str, entities: List[Entity]) -> None:
        """Cache relationships in Redis with TTL."""
        key = f"rel:{entity_id}:{rel_type}"
        entities_data = [
            {
                "id": str(e.id),
                "text": e.text,
                "type": e.type,
                "confidence": e.confidence,
                "mention_count": e.mention_count,
            }
            for e in entities
        ]
        self.redis.setex(key, self.ttl, json.dumps(entities_data))

    def invalidate_entity(self, entity_id: UUID) -> None:
        """Invalidate entity and relationships from Redis."""
        # Delete entity
        self.redis.delete(f"entity:{entity_id}")

        # Delete all relationship caches (use SCAN for pattern matching)
        for key in self.redis.scan_iter(f"rel:{entity_id}:*"):
            self.redis.delete(key)

    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None:
        """Invalidate specific relationship from Redis."""
        key = f"rel:{entity_id}:{rel_type}"
        self.redis.delete(key)

    def clear(self) -> None:
        """Clear all cache entries from Redis."""
        # WARNING: Clears ALL keys in Redis database (use carefully)
        for key in self.redis.scan_iter("entity:*"):
            self.redis.delete(key)
        for key in self.redis.scan_iter("rel:*"):
            self.redis.delete(key)

    def stats(self) -> CacheStats:
        """Get cache statistics from Redis counters."""
        hits = int(self.redis.get("cache:hits") or 0)
        misses = int(self.redis.get("cache:misses") or 0)

        # Redis doesn't track evictions directly (TTL-based expiration)
        # Would need separate tracking or approximation

        return CacheStats(
            hits=hits,
            misses=misses,
            evictions=0,  # Not tracked in Redis
            size=0,  # Would need DBSIZE, but may include non-cache keys
            max_size=self._max_entities + self._max_relationship_caches,
        )

    @property
    def max_entities(self) -> int:
        """Return max entities limit."""
        return self._max_entities

    @property
    def max_relationship_caches(self) -> int:
        """Return max relationship caches limit."""
        return self._max_relationship_caches


# Usage example (future integration)
"""
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_cache = RedisCache(redis_client, ttl_seconds=3600)

service = KnowledgeGraphService(db_session, cache=redis_cache)
# Service now uses Redis for caching, no code changes required!
"""
```

**Integration Path** (Future):
1. Install Redis client: `pip install redis`
2. Configure Redis connection in environment config
3. Create `RedisCache` instance
4. Pass to `KnowledgeGraphService` constructor
5. Zero service code changes required (protocol satisfaction)

**Effort**: 2 hours implementation (Phase 3 only)

---

### Testing Strategy for DI Refactor

**Test Coverage Matrix**:

| Test Case | File | Purpose | Effort |
|-----------|------|---------|--------|
| Protocol definition | `test_cache_protocol.py` | Verify protocol signature matches current cache | 30 min |
| Service DI | `test_service_di.py` | Verify service accepts/uses injected cache | 1 hour |
| Mock cache tracking | `test_service_di.py` | Verify service delegates to cache correctly | 30 min |
| Backward compatibility | `test_service_di.py` | Verify default cache creation still works | 30 min |
| Type checking | CI/mypy | Verify protocol satisfaction at type-check time | 15 min |

**Total Testing Effort**: 2.5 hours

---

### DI Refactor Timeline

**Sequenced Implementation**:

1. **Define Protocol** (45 min)
   - Create `cache_protocol.py`
   - Document all method signatures
   - Add thread-safety requirements

2. **Update Service** (30 min)
   - Import `CacheProtocol`
   - Add type hints to constructor
   - Update docstrings

3. **Add Cache Properties** (15 min)
   - Add `@property` decorators to `KnowledgeGraphCache`
   - Ensure protocol satisfaction

4. **Create Mock Cache** (1 hour)
   - Implement `MockCache` with call tracking
   - Add assertion helpers

5. **Write DI Tests** (1.5 hours)
   - Test service DI acceptance
   - Test method delegation
   - Test backward compatibility

6. **Validate Type Checking** (15 min)
   - Run mypy/pyright
   - Fix any type errors

**Total DI Implementation**: 3.75 hours

**Success Criteria**:
- ✓ All existing tests pass (no regressions)
- ✓ New DI tests pass (100% coverage of injection paths)
- ✓ Type checking passes (mypy/pyright)
- ✓ Backward compatibility maintained (default cache creation)

---

## Issue 8: Concurrent Cache Stress Tests

### Problem Statement

**Current Testing Gap**:
- Existing test: `test_concurrent_gets_dont_raise` - only 10 threads, basic smoke test
- No stress tests for 100+ thread scenarios
- No tests for mixed read/write contention
- No tests for cache invalidation race conditions
- No tests for LRU eviction under concurrency

**Risk**:
- **Production Concurrency**: System may handle 100+ concurrent requests in production
- **Race Conditions**: Undetected bugs in lock management could cause:
  - Cache corruption (stale data)
  - Incorrect eviction (data loss)
  - Deadlocks (service hangs)
  - Inconsistent hit/miss tracking

**Performance Requirements**:
- Cache hit latency: <2 microseconds (P95)
- Cache miss + DB latency: <10ms (P95 for 1-hop)
- Hit rate under load: >80%
- No deadlocks or race conditions under 100+ thread concurrency

---

### Solution: Comprehensive Concurrent Stress Tests

**Approach**: Implement 10-15 realistic stress tests covering:
1. High concurrency read-only (100+ threads)
2. High concurrency write/invalidation (100+ threads)
3. Mixed read/write contention (50/50 split)
4. Bidirectional relationship invalidation races
5. LRU eviction under concurrency
6. Performance validation under load

**Testing Framework**: `threading` module + `ThreadPoolExecutor` for controlled concurrency

---

### Stress Test Specifications

#### Test File Structure

**File**: `tests/knowledge_graph/test_cache_concurrent_stress.py` (NEW)

```python
"""Concurrent stress tests for KnowledgeGraphCache thread-safety.

These tests verify cache behavior under high concurrency (100+ threads)
with realistic workloads including mixed reads/writes, invalidations,
and LRU eviction contention.

Performance targets:
- Cache hit: <2 microseconds P95
- No deadlocks or race conditions
- Hit rate >80% under load
- Cache size bounded at max_entities/max_relationships
"""

import pytest
import threading
import time
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity, CacheStats
```

---

#### Test Category 1: High Concurrency Read-Only

**Purpose**: Verify cache handles 100+ concurrent reads without deadlocks or performance degradation.

```python
class TestHighConcurrencyReads:
    """Stress tests for concurrent read-only workloads."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create cache with large capacity for stress testing."""
        return KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

    @pytest.fixture
    def sample_entity(self) -> Entity:
        """Create sample entity for testing."""
        return Entity(
            id=uuid4(),
            text="High Concurrency Test Entity",
            type="PERSON",
            confidence=0.95,
            mention_count=100,
        )

    def test_concurrent_reads_100_threads_same_entity(
        self, cache: KnowledgeGraphCache, sample_entity: Entity
    ):
        """Test 100 threads reading same entity (hot cache scenario).

        Validates:
        - No deadlocks with single lock contention
        - All reads return correct entity
        - Cache hit tracking accurate
        - Performance: <2us per read (P95)
        """
        # Setup: cache entity once
        cache.set_entity(sample_entity)

        results: List[Entity] = []
        errors: List[Exception] = []
        latencies: List[float] = []

        def read_entity():
            """Read entity and track latency."""
            try:
                start = time.perf_counter()
                result = cache.get_entity(sample_entity.id)
                end = time.perf_counter()

                latencies.append((end - start) * 1_000_000)  # Convert to microseconds
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent reads
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(read_entity) for _ in range(100)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Validate results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100
        assert all(r.id == sample_entity.id for r in results)

        # Validate cache stats
        stats = cache.stats()
        assert stats.hits == 100  # All reads should be hits
        assert stats.misses == 0

        # Validate performance (P95 <2us)
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.2f}us exceeds 2us target"

        print(f"  P50: {latencies[len(latencies)//2]:.2f}us")
        print(f"  P95: {p95_latency:.2f}us")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.2f}us")

    def test_concurrent_reads_100_threads_different_entities(
        self, cache: KnowledgeGraphCache
    ):
        """Test 100 threads reading different entities (distributed load).

        Validates:
        - No lock contention with different keys
        - Correct entity returned for each thread
        - Cache hit rate accurate
        """
        # Setup: cache 100 different entities
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        results: List[tuple[UUID, Entity]] = []
        errors: List[Exception] = []

        def read_entity(entity: Entity):
            """Read specific entity."""
            try:
                result = cache.get_entity(entity.id)
                results.append((entity.id, result))
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent reads (each thread reads different entity)
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(read_entity, entity) for entity in entities]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0
        assert len(results) == 100

        # Verify each thread got correct entity
        for expected_id, actual_entity in results:
            assert actual_entity.id == expected_id

        # Validate cache stats
        stats = cache.stats()
        assert stats.hits == 100
        assert stats.misses == 0

    def test_concurrent_relationship_reads_100_threads(
        self, cache: KnowledgeGraphCache
    ):
        """Test 100 threads reading relationships (1-hop traversal load).

        Validates:
        - Relationship cache handles high concurrency
        - All reads return correct relationship lists
        - Performance maintained under load
        """
        # Setup: cache relationships for 10 entities (10 threads per entity)
        source_entities = [uuid4() for _ in range(10)]
        related_entities = [
            Entity(id=uuid4(), text=f"Related {i}", type="PERSON", confidence=0.8, mention_count=i)
            for i in range(5)
        ]

        for source_id in source_entities:
            cache.set_relationships(source_id, "mentions", related_entities)

        results: List[List[Entity]] = []
        errors: List[Exception] = []

        def read_relationships(source_id: UUID):
            """Read relationships for entity."""
            try:
                result = cache.get_relationships(source_id, "mentions")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent reads (10 threads per entity)
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for source_id in source_entities:
                for _ in range(10):  # 10 threads per entity
                    futures.append(executor.submit(read_relationships, source_id))

            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0
        assert len(results) == 100
        assert all(len(r) == 5 for r in results)  # All have 5 related entities
```

**Effort**: 1 hour (3 read stress tests + performance validation)

---

#### Test Category 2: High Concurrency Write/Invalidation

**Purpose**: Verify cache handles 100+ concurrent writes and invalidations without corruption.

```python
class TestHighConcurrencyWrites:
    """Stress tests for concurrent write/invalidation workloads."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create cache for write stress testing."""
        return KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

    def test_concurrent_entity_writes_100_threads(self, cache: KnowledgeGraphCache):
        """Test 100 threads writing different entities concurrently.

        Validates:
        - No race conditions in entity cache
        - All entities successfully cached
        - No data corruption
        """
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]

        errors: List[Exception] = []

        def write_entity(entity: Entity):
            """Write entity to cache."""
            try:
                cache.set_entity(entity)
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent writes
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(write_entity, entity) for entity in entities]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify all entities were cached correctly
        for entity in entities:
            cached = cache.get_entity(entity.id)
            assert cached is not None
            assert cached.id == entity.id
            assert cached.text == entity.text

        stats = cache.stats()
        assert stats.size >= 100  # At least 100 entities cached

    def test_concurrent_invalidations_100_threads_different_entities(
        self, cache: KnowledgeGraphCache
    ):
        """Test 100 threads invalidating different entities concurrently.

        Validates:
        - No race conditions in invalidation
        - All entities successfully invalidated
        - No cross-contamination (wrong entity invalidated)
        """
        # Setup: cache 100 entities
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        errors: List[Exception] = []

        def invalidate_entity(entity_id: UUID):
            """Invalidate entity from cache."""
            try:
                cache.invalidate_entity(entity_id)
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent invalidations
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(invalidate_entity, entity.id) for entity in entities]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify all entities were invalidated
        for entity in entities:
            cached = cache.get_entity(entity.id)
            assert cached is None, f"Entity {entity.id} should be invalidated"

        stats = cache.stats()
        assert stats.size == 0  # All entities invalidated

    def test_concurrent_invalidations_same_entity_100_threads(
        self, cache: KnowledgeGraphCache
    ):
        """Test 100 threads invalidating SAME entity concurrently.

        Validates:
        - No deadlocks with high lock contention
        - Idempotent invalidation (no errors if already invalidated)
        - Cache remains consistent
        """
        # Setup: cache single entity
        entity = Entity(id=uuid4(), text="Shared Entity", type="PERSON", confidence=0.9, mention_count=100)
        cache.set_entity(entity)

        errors: List[Exception] = []

        def invalidate_entity():
            """Invalidate entity (idempotent)."""
            try:
                cache.invalidate_entity(entity.id)
            except Exception as e:
                errors.append(e)

        # Execute 100 concurrent invalidations of SAME entity
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(invalidate_entity) for _ in range(100)]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify entity is invalidated
        cached = cache.get_entity(entity.id)
        assert cached is None
```

**Effort**: 45 minutes (3 write/invalidation stress tests)

---

#### Test Category 3: Mixed Read/Write Contention

**Purpose**: Verify cache consistency under realistic mixed workloads.

```python
class TestMixedReadWriteContention:
    """Stress tests for concurrent mixed read/write workloads."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create cache for mixed workload testing."""
        return KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

    def test_concurrent_mixed_read_write_50_50_split(self, cache: KnowledgeGraphCache):
        """Test 50 reader threads + 50 writer threads on shared entities.

        Validates:
        - No cache corruption under contention
        - Reads return valid data (never corrupted)
        - Writes complete successfully
        - Cache remains consistent after workload
        """
        # Setup: initial entities
        initial_entities = [
            Entity(id=uuid4(), text=f"Initial {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(10)
        ]
        for entity in initial_entities:
            cache.set_entity(entity)

        read_count = threading.Lock()
        write_count = threading.Lock()
        successful_reads = 0
        successful_writes = 0
        errors: List[Exception] = []

        def reader():
            """Read random entities repeatedly."""
            nonlocal successful_reads
            try:
                for _ in range(20):  # Each reader does 20 reads
                    import random
                    entity = random.choice(initial_entities)
                    result = cache.get_entity(entity.id)

                    # Validate: either None (invalidated) or correct entity
                    if result is not None:
                        assert result.id == entity.id

                    with read_count:
                        successful_reads += 1
            except Exception as e:
                errors.append(e)

        def writer():
            """Write new entities repeatedly."""
            nonlocal successful_writes
            try:
                for i in range(20):  # Each writer does 20 writes
                    new_entity = Entity(
                        id=uuid4(),
                        text=f"New {i}",
                        type="PERSON",
                        confidence=0.8,
                        mention_count=i,
                    )
                    cache.set_entity(new_entity)

                    with write_count:
                        successful_writes += 1
            except Exception as e:
                errors.append(e)

        # Execute mixed workload: 50 readers + 50 writers
        with ThreadPoolExecutor(max_workers=100) as executor:
            reader_futures = [executor.submit(reader) for _ in range(50)]
            writer_futures = [executor.submit(writer) for _ in range(50)]

            for future in as_completed(reader_futures + writer_futures):
                future.result()

        # Validate results
        assert len(errors) == 0
        assert successful_reads == 50 * 20  # 50 readers × 20 reads
        assert successful_writes == 50 * 20  # 50 writers × 20 writes

        # Cache should have initial + new entities (may have evictions)
        stats = cache.stats()
        assert stats.size > 0  # Cache not empty

    def test_concurrent_read_invalidate_race_condition(self, cache: KnowledgeGraphCache):
        """Test simultaneous reads and invalidations (race condition scenario).

        Validates:
        - No crashes when reading during invalidation
        - Reads return either valid entity or None (never corrupted)
        - Invalidations complete successfully
        """
        # Setup: cache 10 entities
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(10)
        ]
        for entity in entities:
            cache.set_entity(entity)

        errors: List[Exception] = []
        read_results: List[Entity | None] = []

        def reader(entity_id: UUID):
            """Read entity repeatedly."""
            try:
                for _ in range(50):  # 50 reads per thread
                    result = cache.get_entity(entity_id)
                    # Validate: either None or correct entity (never corrupted)
                    if result is not None:
                        assert result.id == entity_id
                    read_results.append(result)
            except Exception as e:
                errors.append(e)

        def invalidator(entity_id: UUID):
            """Invalidate entity repeatedly."""
            try:
                for _ in range(50):  # 50 invalidations per thread
                    cache.invalidate_entity(entity_id)
            except Exception as e:
                errors.append(e)

        # Execute: 5 readers + 5 invalidators per entity (100 threads total)
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for entity in entities:
                # 5 readers per entity
                for _ in range(5):
                    futures.append(executor.submit(reader, entity.id))
                # 5 invalidators per entity
                for _ in range(5):
                    futures.append(executor.submit(invalidator, entity.id))

            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify all reads returned valid data (either None or correct entity)
        # No corrupted data
        for result in read_results:
            if result is not None:
                assert result.id in [e.id for e in entities]
```

**Effort**: 45 minutes (2 mixed workload stress tests)

---

#### Test Category 4: Bidirectional Relationship Invalidation

**Purpose**: Verify cache handles complex invalidation cascades under concurrency.

```python
class TestBidirectionalInvalidation:
    """Stress tests for relationship invalidation under concurrency."""

    @pytest.fixture
    def cache(self) -> KnowledgeGraphCache:
        """Create cache for invalidation testing."""
        return KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

    def test_concurrent_bidirectional_invalidation_race(self, cache: KnowledgeGraphCache):
        """Test concurrent invalidation of bidirectional relationships.

        Scenario: A ↔ B relationship
        - Thread 1: Invalidate A (should clear A→B and B→A caches)
        - Thread 2: Invalidate B (should clear B→A and A→B caches)

        Validates:
        - No deadlocks with bidirectional invalidation
        - Both relationship caches cleared
        - No partial invalidation (atomicity)
        """
        # Setup: A ↔ B bidirectional relationship
        entity_a = Entity(id=uuid4(), text="Entity A", type="PERSON", confidence=0.9, mention_count=10)
        entity_b = Entity(id=uuid4(), text="Entity B", type="PERSON", confidence=0.9, mention_count=10)

        cache.set_entity(entity_a)
        cache.set_entity(entity_b)
        cache.set_relationships(entity_a.id, "mentions", [entity_b])
        cache.set_relationships(entity_b.id, "mentions", [entity_a])

        errors: List[Exception] = []

        def invalidate_a():
            """Invalidate entity A (triggers cascade)."""
            try:
                for _ in range(10):
                    cache.invalidate_entity(entity_a.id)
            except Exception as e:
                errors.append(e)

        def invalidate_b():
            """Invalidate entity B (triggers cascade)."""
            try:
                for _ in range(10):
                    cache.invalidate_entity(entity_b.id)
            except Exception as e:
                errors.append(e)

        # Execute concurrent invalidations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures_a = [executor.submit(invalidate_a) for _ in range(10)]
            futures_b = [executor.submit(invalidate_b) for _ in range(10)]

            for future in as_completed(futures_a + futures_b):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify both entities and relationships invalidated
        assert cache.get_entity(entity_a.id) is None
        assert cache.get_entity(entity_b.id) is None
        assert cache.get_relationships(entity_a.id, "mentions") is None
        assert cache.get_relationships(entity_b.id, "mentions") is None

    def test_concurrent_read_during_cascade_invalidation(self, cache: KnowledgeGraphCache):
        """Test reads during cascade invalidation (A→B, B→C chain).

        Scenario: A→B→C relationship chain
        - Thread 1-50: Read relationships A→B
        - Thread 51-100: Invalidate entity B (cascades to A→B and B→C)

        Validates:
        - No crashes during cascade
        - Reads return valid or None (never partially invalidated state)
        """
        # Setup: A→B→C chain
        entity_a = Entity(id=uuid4(), text="Entity A", type="PERSON", confidence=0.9, mention_count=10)
        entity_b = Entity(id=uuid4(), text="Entity B", type="PERSON", confidence=0.9, mention_count=10)
        entity_c = Entity(id=uuid4(), text="Entity C", type="PERSON", confidence=0.9, mention_count=10)

        cache.set_entity(entity_a)
        cache.set_entity(entity_b)
        cache.set_entity(entity_c)
        cache.set_relationships(entity_a.id, "mentions", [entity_b])
        cache.set_relationships(entity_b.id, "mentions", [entity_c])

        errors: List[Exception] = []
        read_results: List[List[Entity] | None] = []

        def reader():
            """Read A→B relationship repeatedly."""
            try:
                for _ in range(20):
                    result = cache.get_relationships(entity_a.id, "mentions")
                    # Validate: either None or valid list (never corrupted)
                    if result is not None:
                        assert isinstance(result, list)
                        if len(result) > 0:
                            assert result[0].id == entity_b.id
                    read_results.append(result)
            except Exception as e:
                errors.append(e)

        def invalidator():
            """Invalidate entity B (cascade invalidation)."""
            try:
                for _ in range(20):
                    cache.invalidate_entity(entity_b.id)
            except Exception as e:
                errors.append(e)

        # Execute: 50 readers + 50 invalidators
        with ThreadPoolExecutor(max_workers=100) as executor:
            reader_futures = [executor.submit(reader) for _ in range(50)]
            invalidator_futures = [executor.submit(invalidator) for _ in range(50)]

            for future in as_completed(reader_futures + invalidator_futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify all reads returned valid data (no corruption)
        for result in read_results:
            if result is not None:
                assert isinstance(result, list)
```

**Effort**: 30 minutes (2 bidirectional invalidation tests)

---

#### Test Category 5: LRU Eviction Under Concurrency

**Purpose**: Verify LRU eviction remains bounded and correct under concurrent load.

```python
class TestLRUEvictionConcurrency:
    """Stress tests for LRU eviction under concurrent load."""

    def test_concurrent_lru_eviction_bounded_size(self):
        """Test 100 threads inserting entities beyond cache capacity.

        Validates:
        - Cache size remains bounded at max_entities
        - Evictions occur correctly (oldest entries removed)
        - No data loss for non-evicted entries
        - No race conditions in eviction logic
        """
        cache = KnowledgeGraphCache(max_entities=100, max_relationship_caches=100)

        # 100 threads each insert 10 entities = 1000 total inserts
        # Cache max = 100, so 900 evictions expected

        entities_per_thread = 10
        num_threads = 100

        inserted_entities: List[Entity] = []
        errors: List[Exception] = []

        def insert_entities(thread_id: int):
            """Insert multiple entities."""
            try:
                thread_entities = []
                for i in range(entities_per_thread):
                    entity = Entity(
                        id=uuid4(),
                        text=f"Thread {thread_id} Entity {i}",
                        type="PERSON",
                        confidence=0.9,
                        mention_count=i,
                    )
                    cache.set_entity(entity)
                    thread_entities.append(entity)

                # Track entities from this thread
                with threading.Lock():
                    inserted_entities.extend(thread_entities)
            except Exception as e:
                errors.append(e)

        # Execute concurrent inserts
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_entities, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify cache size bounded at max
        stats = cache.stats()
        assert stats.size <= 100, f"Cache size {stats.size} exceeds max 100"

        # Verify evictions occurred
        expected_evictions = (num_threads * entities_per_thread) - 100
        assert stats.evictions >= expected_evictions, \
            f"Expected ≥{expected_evictions} evictions, got {stats.evictions}"

        # Verify last 100 entities are still cached (most recent)
        last_100_entities = inserted_entities[-100:]
        cached_count = sum(1 for e in last_100_entities if cache.get_entity(e.id) is not None)

        # Allow some variance due to concurrency (should be >80% of last 100)
        assert cached_count >= 80, \
            f"Expected ≥80 of last 100 entities cached, got {cached_count}"

    def test_concurrent_mixed_entity_relationship_eviction(self):
        """Test LRU eviction with both entities and relationships under load.

        Validates:
        - Entity and relationship caches bounded independently
        - Evictions work correctly for both cache types
        - No interference between entity and relationship evictions
        """
        cache = KnowledgeGraphCache(max_entities=50, max_relationship_caches=50)

        errors: List[Exception] = []

        def insert_entities_and_relationships(thread_id: int):
            """Insert entities with relationships."""
            try:
                for i in range(10):
                    # Insert entity
                    entity = Entity(
                        id=uuid4(),
                        text=f"Thread {thread_id} Entity {i}",
                        type="PERSON",
                        confidence=0.9,
                        mention_count=i,
                    )
                    cache.set_entity(entity)

                    # Insert relationship
                    related = Entity(
                        id=uuid4(),
                        text=f"Related {i}",
                        type="PERSON",
                        confidence=0.8,
                        mention_count=i,
                    )
                    cache.set_relationships(entity.id, "mentions", [related])
            except Exception as e:
                errors.append(e)

        # Execute: 20 threads × 10 inserts = 200 entities + 200 relationships
        # Expect: 150 entity evictions + 150 relationship evictions
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(insert_entities_and_relationships, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        # Validate results
        assert len(errors) == 0

        # Verify cache sizes bounded
        stats = cache.stats()
        assert stats.size <= 100, f"Total cache size {stats.size} exceeds max 100"

        # Verify evictions occurred for both types
        assert stats.evictions >= 150, f"Expected ≥150 evictions, got {stats.evictions}"
```

**Effort**: 30 minutes (2 LRU eviction stress tests)

---

### Stress Test Summary

**Total Tests**: 15 stress tests across 5 categories

| Category | Test Count | Effort | Key Validations |
|----------|-----------|--------|-----------------|
| High Concurrency Reads | 3 | 1 hour | No deadlocks, <2us P95, correct data |
| High Concurrency Writes | 3 | 45 min | No corruption, all writes succeed |
| Mixed Read/Write | 2 | 45 min | Consistency under contention |
| Bidirectional Invalidation | 2 | 30 min | Cascade correctness, no partial state |
| LRU Eviction | 2 | 30 min | Bounded size, correct eviction |

**Total Implementation Effort**: 3.5 hours

---

### Performance Validation Framework

**File**: `tests/knowledge_graph/test_cache_load.py` (NEW)

**Purpose**: Measure and validate performance under realistic load.

```python
"""Load tests for cache performance validation under concurrency."""

import pytest
import time
import statistics
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import List

from src.knowledge_graph.cache import KnowledgeGraphCache, Entity


class TestCacheLoadPerformance:
    """Performance validation tests under concurrent load."""

    @pytest.mark.performance
    def test_search_reranking_load_50_concurrent_requests(self):
        """Simulate 50 concurrent search + reranking requests.

        Scenario: Each search triggers 1-hop + 2-hop graph traversal
        Performance targets:
        - 1-hop: P95 <10ms
        - 2-hop: P95 <50ms
        - Cache hit rate: >80%
        """
        cache = KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

        # Setup: graph with 100 entities, 2-hop relationships
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        # Setup 1-hop and 2-hop relationships
        for i, entity in enumerate(entities[:-2]):
            # 1-hop: entity[i] → entity[i+1]
            cache.set_relationships(entity.id, "mentions", [entities[i+1]])
            # 2-hop: entity[i] → entity[i+1] → entity[i+2]
            cache.set_relationships(entities[i+1].id, "mentions", [entities[i+2]])

        latencies_1hop: List[float] = []
        latencies_2hop: List[float] = []

        def simulate_search(entity_idx: int):
            """Simulate search + 1-hop + 2-hop traversal."""
            entity = entities[entity_idx]

            # 1-hop traversal
            start = time.perf_counter()
            one_hop = cache.get_relationships(entity.id, "mentions")
            end = time.perf_counter()
            latencies_1hop.append((end - start) * 1000)  # milliseconds

            # 2-hop traversal (if 1-hop exists)
            if one_hop and len(one_hop) > 0:
                start = time.perf_counter()
                two_hop = cache.get_relationships(one_hop[0].id, "mentions")
                end = time.perf_counter()
                latencies_2hop.append((end - start) * 1000)  # milliseconds

        # Execute 50 concurrent searches (multiple passes for warm-up)
        with ThreadPoolExecutor(max_workers=50) as executor:
            # Warm-up pass (prime cache)
            futures = [executor.submit(simulate_search, i % 98) for i in range(50)]
            for future in futures:
                future.result()

            # Measurement pass
            latencies_1hop.clear()
            latencies_2hop.clear()

            futures = [executor.submit(simulate_search, i % 98) for i in range(50)]
            for future in futures:
                future.result()

        # Validate performance targets
        latencies_1hop.sort()
        latencies_2hop.sort()

        p95_1hop = latencies_1hop[int(len(latencies_1hop) * 0.95)] if latencies_1hop else 0
        p95_2hop = latencies_2hop[int(len(latencies_2hop) * 0.95)] if latencies_2hop else 0

        print(f"\n1-hop traversal:")
        print(f"  P50: {statistics.median(latencies_1hop):.2f}ms")
        print(f"  P95: {p95_1hop:.2f}ms")
        print(f"  P99: {latencies_1hop[int(len(latencies_1hop)*0.99)]:.2f}ms")

        print(f"\n2-hop traversal:")
        print(f"  P50: {statistics.median(latencies_2hop):.2f}ms")
        print(f"  P95: {p95_2hop:.2f}ms")
        print(f"  P99: {latencies_2hop[int(len(latencies_2hop)*0.99)]:.2f}ms")

        # Validate targets (cache hit scenario, in-memory)
        assert p95_1hop < 0.1, f"1-hop P95 {p95_1hop:.2f}ms exceeds 0.1ms target (cache hit)"
        assert p95_2hop < 0.2, f"2-hop P95 {p95_2hop:.2f}ms exceeds 0.2ms target (cache hit)"

        # Validate cache hit rate
        stats = cache.stats()
        hit_rate = stats.hits / (stats.hits + stats.misses) * 100 if (stats.hits + stats.misses) > 0 else 0
        assert hit_rate > 80, f"Cache hit rate {hit_rate:.1f}% below 80% target"

    @pytest.mark.performance
    def test_cache_hit_rate_under_hot_entity_load(self):
        """Test cache hit rate for hot entities (80/20 rule).

        Scenario: 100 concurrent requests, 80% target 20 hot entities
        Performance target: >90% hit rate on hot entities
        """
        cache = KnowledgeGraphCache(max_entities=1000, max_relationship_caches=1000)

        # Setup: 100 entities, 20 are "hot"
        all_entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]
        hot_entities = all_entities[:20]  # First 20 are hot

        for entity in all_entities:
            cache.set_entity(entity)

        hits_before = cache.stats().hits

        def access_entity():
            """Access entities following 80/20 rule."""
            import random
            if random.random() < 0.8:
                # 80% access hot entities
                entity = random.choice(hot_entities)
            else:
                # 20% access cold entities
                entity = random.choice(all_entities)

            cache.get_entity(entity.id)

        # Execute 1000 accesses from 100 concurrent threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(access_entity) for _ in range(1000)]
            for future in futures:
                future.result()

        # Validate hit rate
        stats = cache.stats()
        new_hits = stats.hits - hits_before
        hit_rate = new_hits / 1000 * 100

        print(f"\nCache hit rate: {hit_rate:.1f}%")
        print(f"Total hits: {new_hits}")

        assert hit_rate > 90, f"Hit rate {hit_rate:.1f}% below 90% target"

    @pytest.mark.performance
    def test_throughput_mixed_workload(self):
        """Measure throughput for mixed read/write workload.

        Scenario: 40% reads + 40% cache hits + 20% writes
        Performance target: >10,000 ops/sec
        """
        cache = KnowledgeGraphCache(max_entities=10000, max_relationship_caches=20000)

        # Setup: pre-cache 100 entities
        entities = [
            Entity(id=uuid4(), text=f"Entity {i}", type="PERSON", confidence=0.9, mention_count=i)
            for i in range(100)
        ]
        for entity in entities:
            cache.set_entity(entity)

        total_ops = 0
        start_time = time.perf_counter()

        def mixed_operations():
            """Perform mixed read/write operations."""
            nonlocal total_ops
            import random

            for _ in range(100):  # 100 ops per thread
                op_type = random.random()

                if op_type < 0.4:
                    # 40% reads (cache miss)
                    cache.get_entity(uuid4())
                elif op_type < 0.8:
                    # 40% reads (cache hit)
                    entity = random.choice(entities)
                    cache.get_entity(entity.id)
                else:
                    # 20% writes
                    new_entity = Entity(
                        id=uuid4(),
                        text=f"New {random.randint(0, 1000)}",
                        type="PERSON",
                        confidence=0.9,
                        mention_count=1,
                    )
                    cache.set_entity(new_entity)

                total_ops += 1

        # Execute mixed workload from 100 threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(100)]
            for future in futures:
                future.result()

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        throughput = total_ops / elapsed

        print(f"\nThroughput: {throughput:.0f} ops/sec")
        print(f"Total operations: {total_ops}")
        print(f"Elapsed time: {elapsed:.2f}s")

        # Validate throughput target
        assert throughput > 10_000, f"Throughput {throughput:.0f} ops/sec below 10,000 target"
```

**Effort**: 1.5 hours (3 load tests + performance measurement)

---

## Implementation Timeline

### Sequenced Implementation Plan

**Phase 1: Dependency Injection (3.75 hours)**
1. Define `CacheProtocol` (45 min)
2. Update `KnowledgeGraphService` constructor (30 min)
3. Add cache properties to `KnowledgeGraphCache` (15 min)
4. Create `MockCache` implementation (1 hour)
5. Write DI tests (1.5 hours)
6. Validate type checking (15 min)

**Phase 2: Stress Tests (3.5 hours)**
1. Test Category 1 - High Concurrency Reads (1 hour)
2. Test Category 2 - High Concurrency Writes (45 min)
3. Test Category 3 - Mixed Read/Write (45 min)
4. Test Category 4 - Bidirectional Invalidation (30 min)
5. Test Category 5 - LRU Eviction (30 min)

**Phase 3: Load Testing (1.5 hours)**
1. Search/reranking load test (45 min)
2. Hot entity cache hit rate test (30 min)
3. Throughput mixed workload test (15 min)

**Total Implementation Time**: 8.75 hours

**Parallelization Opportunity**:
- Phase 1 (DI) and Phase 2 (Stress Tests) can run in parallel (different files)
- Phase 3 (Load Tests) depends on Phase 2 (needs stress test patterns)

**With Parallelization**: ~5.5 hours wall-clock time (DI + Stress Tests in parallel, then Load Tests)

---

## Success Criteria

### Dependency Injection Refactor (Issue 6)

**Functional Requirements**:
- ✓ `CacheProtocol` defines complete interface contract
- ✓ `KnowledgeGraphService` accepts `CacheProtocol` via constructor
- ✓ `KnowledgeGraphCache` satisfies `CacheProtocol` (type-checked)
- ✓ Backward compatibility maintained (default cache creation)
- ✓ Mock cache injection works in tests

**Testing Requirements**:
- ✓ All existing tests pass (no regressions)
- ✓ New DI tests pass (8+ tests covering injection paths)
- ✓ Type checking passes (mypy/pyright)
- ✓ Mock cache tracks method calls correctly

**Documentation Requirements**:
- ✓ Protocol interface documented with thread-safety requirements
- ✓ Service constructor examples show DI usage
- ✓ Redis migration roadmap documented

---

### Concurrent Cache Stress Tests (Issue 8)

**Functional Requirements**:
- ✓ 15+ stress tests covering 100+ thread scenarios
- ✓ No deadlocks or race conditions detected
- ✓ Cache consistency maintained under all workloads
- ✓ LRU eviction bounded correctly
- ✓ Bidirectional invalidation atomic

**Performance Requirements**:
- ✓ Cache hit latency: <2us P95 (concurrent reads)
- ✓ 1-hop traversal: <10ms P95 under load (with cache hits <0.1ms)
- ✓ 2-hop traversal: <50ms P95 under load (with cache hits <0.2ms)
- ✓ Cache hit rate: >80% under realistic load
- ✓ Throughput: >10,000 ops/sec (mixed workload)

**Testing Requirements**:
- ✓ All stress tests pass (15+ tests)
- ✓ Load tests pass (3+ tests)
- ✓ Performance targets met (measured in CI)
- ✓ No flaky tests (reliable under 1000+ runs)

---

## Risk Assessment

### Dependency Injection Refactor

**Low Risk**:
- **Backward Compatible**: All existing code continues working
- **Additive Changes**: No breaking API changes
- **Type-Checked**: Compile-time verification of protocol satisfaction

**Mitigation**:
- Run full test suite before/after changes
- Validate type checking with mypy/pyright
- Review all constructor call sites

---

### Concurrent Stress Tests

**Medium Risk**:
- **Test Flakiness**: Concurrency tests may be non-deterministic
- **Performance Variability**: CI environment may have different perf characteristics
- **Resource Contention**: 100+ threads may strain test runner

**Mitigation**:
- Use deterministic test patterns (avoid sleep/random timing)
- Run tests multiple times in CI (detect flakiness)
- Use performance ranges instead of absolute targets
- Add `@pytest.mark.performance` for optional performance tests
- Monitor CI resources (add retry logic if needed)

---

## Future Enhancements (Post-Phase 1)

### Redis Cache Implementation (Phase 3)
- Implement `RedisCache` class satisfying `CacheProtocol`
- Add Redis integration tests
- Document deployment patterns (single-instance vs distributed)
- Benchmark Redis vs in-memory performance
- **Effort**: 2-3 hours

### Cache Warming Strategies
- Background thread for pre-caching hot entities
- Query log analysis for cache pre-population
- **Effort**: 4-6 hours

### Advanced Metrics
- Prometheus metrics export (hit rate, latency histograms)
- Cache efficiency monitoring (bytes per hit)
- **Effort**: 3-4 hours

### Adaptive Cache Sizing
- Dynamic max_entities based on memory pressure
- Automatic TTL adjustment based on hit rate
- **Effort**: 6-8 hours

---

## Appendix A: File Changes Summary

### New Files Created

1. **src/knowledge_graph/cache_protocol.py** (45 min)
   - Protocol definition for cache interface
   - Thread-safety documentation
   - 150 lines

2. **tests/knowledge_graph/test_service_di.py** (1.5 hours)
   - Mock cache implementation
   - DI acceptance tests
   - Backward compatibility tests
   - 300 lines

3. **tests/knowledge_graph/test_cache_concurrent_stress.py** (3.5 hours)
   - 15+ stress tests for concurrency
   - 100+ thread scenarios
   - Mixed workload tests
   - 800 lines

4. **tests/knowledge_graph/test_cache_load.py** (1.5 hours)
   - Load testing framework
   - Performance validation
   - Throughput measurement
   - 300 lines

5. **docs/roadmap/redis-migration-plan.md** (30 min)
   - Redis cache implementation guide
   - Integration examples
   - Deployment considerations
   - 100 lines

### Modified Files

1. **src/knowledge_graph/graph_service.py** (30 min)
   - Import `CacheProtocol`
   - Type hint `cache` parameter
   - Update docstrings
   - 10 lines changed

2. **src/knowledge_graph/cache.py** (15 min)
   - Add `@property` decorators for `max_entities`, `max_relationship_caches`
   - 10 lines changed

### Total Code Volume
- **New code**: ~1,650 lines (protocols, tests, documentation)
- **Modified code**: ~20 lines (type hints, properties)
- **Documentation**: ~500 lines (roadmaps, docstrings)

---

## Appendix B: Dependencies

### Blocker Dependencies

**Issue 6 (DI Refactor)**:
- **Requires**: Blocker 3 (Repository Integration) complete
- **Rationale**: DI refactor should include repository injection alongside cache injection
- **Can Start**: Protocol definition and planning (this document)
- **Cannot Start**: Service constructor changes until repository pattern finalized

**Issue 8 (Stress Tests)**:
- **Requires**: None (can start immediately)
- **Rationale**: Tests validate existing cache implementation
- **Can Parallelize**: With DI refactor (different files)

### Parallel Work Opportunities

**Timeline with Parallelization**:
```
Week 1:
├─ Developer A: DI Refactor (3.75 hours) ─────────────────┐
│  - Protocol definition                                   │
│  - Service constructor updates                           │
│  - Mock cache + tests                                    │
│                                                           ├─> Complete
├─ Developer B: Stress Tests (3.5 hours) ─────────────────┤
│  - High concurrency tests                                │
│  - Mixed workload tests                                  │
│  - Bidirectional invalidation tests                      │
│                                                           │
└─ Developer C: Load Tests (1.5 hours) ───────────────────┘
   - Performance validation framework
   - Throughput measurements

Total Wall-Clock Time: 5.5 hours (with 2 developers in parallel)
```

---

## Appendix C: Testing Matrix

### Test Coverage by Category

| Category | Test Count | Thread Count | Duration | Purpose |
|----------|-----------|--------------|----------|---------|
| **DI Tests** | 8 | N/A | Fast | Verify injection works |
| **Read Stress** | 3 | 100 | 5-10s | No deadlocks, perf |
| **Write Stress** | 3 | 100 | 5-10s | No corruption |
| **Mixed Workload** | 2 | 100 | 10-20s | Consistency |
| **Invalidation** | 2 | 20-100 | 5-10s | Cascade correctness |
| **LRU Eviction** | 2 | 100 | 10-15s | Bounded size |
| **Load Tests** | 3 | 50-100 | 30-60s | Performance validation |

**Total Test Count**: 23 tests
**Total Execution Time**: ~5 minutes (sequential), ~2 minutes (parallel with pytest-xdist)

### CI Integration

**Recommended pytest markers**:
```python
@pytest.mark.concurrency      # All concurrency tests (run always)
@pytest.mark.stress          # Stress tests (100+ threads, run in nightly)
@pytest.mark.performance     # Performance validation (run in release)
```

**CI Configuration**:
```yaml
# Fast CI (PR validation)
pytest -m "not stress and not performance"  # ~1 minute

# Nightly CI (comprehensive)
pytest -m "concurrency or stress"           # ~3 minutes

# Release CI (full validation)
pytest                                      # ~5 minutes
```

---

## Document Metadata

**Document Version**: 1.0
**Author**: Claude Code (Architecture Planning Agent)
**Date**: 2025-11-09
**Status**: Planning Complete - Ready for Implementation
**Approval Required**: Yes (architect review before implementation)
**Estimated Implementation Time**: 8.75 hours (sequential), 5.5 hours (parallel)
**Priority**: High (Blocker for Phase 2 Redis migration, Production thread-safety verification)

---

**End of Planning Document**
