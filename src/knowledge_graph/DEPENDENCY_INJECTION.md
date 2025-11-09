# Dependency Injection in Knowledge Graph

## Overview

The Knowledge Graph system uses **constructor injection** for cache and repository dependencies, following the **Dependency Inversion Principle** from SOLID architecture.

**Key Benefits:**
- **Testability**: Inject mock implementations for fast unit tests
- **Flexibility**: Swap implementations without code changes
- **Scalability**: Easy migration from LRU to Redis cache
- **Maintainability**: Clear separation of concerns

## Architecture

### Before Dependency Injection

```python
# ❌ Hardcoded dependencies (tightly coupled)
class KnowledgeGraphService:
    def __init__(self, db_pool):
        self.cache = KnowledgeGraphCache()  # Cannot swap!
        self.repo = KnowledgeGraphQueryRepository(db_pool)
```

**Problems:**
- Cannot test without real cache
- Cannot swap to Redis without modifying service code
- Violates Dependency Inversion Principle

### After Dependency Injection

```python
# ✅ Injectable dependencies (loosely coupled)
class KnowledgeGraphService:
    def __init__(
        self,
        db_pool,
        cache: Optional[CacheProtocol] = None,
        query_repo: Optional[KnowledgeGraphQueryRepository] = None
    ):
        self.cache = cache or KnowledgeGraphCache()
        self.repo = query_repo or KnowledgeGraphQueryRepository(db_pool)
```

**Benefits:**
- Can inject MockCache for testing
- Can swap to RedisCache in production
- Follows Dependency Inversion Principle
- Backward compatible (defaults to LRU cache)

## CacheProtocol Interface

All cache implementations must implement the `CacheProtocol`:

```python
from src.knowledge_graph.cache_protocol import CacheProtocol, Entity

class CacheProtocol(Protocol):
    """Interface for cache implementations."""

    def get_entity(self, entity_id: UUID) -> Optional[Entity]: ...
    def set_entity(self, entity: Entity) -> None: ...
    def get_relationships(self, entity_id: UUID, rel_type: str) -> Optional[List[Entity]]: ...
    def set_relationships(self, entity_id: UUID, rel_type: str, entities: List[Entity]) -> None: ...
    def invalidate_entity(self, entity_id: UUID) -> None: ...
    def invalidate_relationships(self, entity_id: UUID, rel_type: str) -> None: ...
    def clear(self) -> None: ...
    def stats(self) -> CacheStats: ...
```

**Protocol Benefits:**
- Structural subtyping (no explicit inheritance)
- Type-safe with mypy --strict
- Documents cache contract
- Enables multiple implementations

## Usage Examples

### 1. Default Configuration (LRU Cache)

```python
from src.knowledge_graph.graph_service import KnowledgeGraphService

# Uses default KnowledgeGraphCache (LRU)
service = KnowledgeGraphService(db_pool)
```

**When to use:**
- Development environment
- Single-server deployments
- <10k entities in hot path

### 2. Custom Cache Configuration

```python
from src.knowledge_graph.cache import KnowledgeGraphCache
from src.knowledge_graph.graph_service import KnowledgeGraphService

# Custom LRU cache sizes
cache = KnowledgeGraphCache(
    max_entities=20000,
    max_relationship_caches=40000
)
service = KnowledgeGraphService(db_pool, cache=cache)
```

**When to use:**
- Large knowledge graphs (>10k entities)
- High cache hit rate workloads
- Performance tuning

### 3. Testing with Mock Cache

```python
from tests.knowledge_graph.test_service_di import MockCache
from src.knowledge_graph.graph_service import KnowledgeGraphService

# Inject mock cache for testing
mock_cache = MockCache()
service = KnowledgeGraphService(db_pool, cache=mock_cache)

# Perform operations
service.get_entity(entity_id)

# Verify cache was used
assert entity_id in mock_cache.get_calls
```

**Benefits:**
- Fast unit tests (no database I/O)
- Verify cache interaction patterns
- Test cache invalidation logic

### 4. Future: Redis Cache

```python
# When RedisCache is implemented
from src.knowledge_graph.redis_cache import RedisCache

redis_cache = RedisCache(redis_client)
service = KnowledgeGraphService(db_pool, cache=redis_cache)
```

**When to use:**
- Multi-server deployments
- Shared cache across instances
- Large-scale production systems

### 5. Service Factory Pattern

```python
from src.knowledge_graph.service_factory import ServiceFactory

# Environment-based configuration
service = ServiceFactory.create_service(
    db_pool,
    cache_type='memory'  # or 'redis' when available
)

# From environment variables
service = create_from_environment(db_pool)
```

**When to use:**
- Configuration-driven deployments
- Multiple environments (dev/staging/prod)
- Centralized service creation

## Testing Strategy

### Unit Testing with Mocks

```python
def test_service_caches_entity():
    """Verify service caches entity after database fetch."""
    mock_cache = MockCache()
    mock_repo = MockRepository()

    service = KnowledgeGraphService(
        db_pool=MockDatabasePool(),
        cache=mock_cache,
        query_repo=mock_repo
    )

    # First call: cache miss, fetch from DB
    entity_id = uuid4()
    service.get_entity(entity_id)

    # Verify cache get was called
    assert entity_id in mock_cache.get_calls
```

### Integration Testing with Real Cache

```python
def test_service_with_real_cache():
    """Integration test with real LRU cache."""
    cache = KnowledgeGraphCache(max_entities=100)
    service = KnowledgeGraphService(db_pool, cache=cache)

    # Test with real cache behavior
    entity = Entity(...)
    cache.set_entity(entity)

    result = service.get_entity(entity.id)
    assert result is not None
```

## Implementation Details

### Dependency Flow

```
KnowledgeGraphService
├── cache: CacheProtocol (injected or default)
│   ├── KnowledgeGraphCache (default, in-memory LRU)
│   ├── RedisCache (future, distributed)
│   └── MockCache (testing)
└── query_repo: KnowledgeGraphQueryRepository (injected or default)
    ├── KnowledgeGraphQueryRepository (default, PostgreSQL)
    └── MockRepository (testing)
```

### Cache Abstraction Layers

```
Application Layer
    ↓
KnowledgeGraphService (uses CacheProtocol)
    ↓
CacheProtocol Interface (defines contract)
    ↓
Concrete Implementations:
├── KnowledgeGraphCache (LRU, thread-safe OrderedDict)
├── RedisCache (future, distributed cache)
└── MockCache (testing, in-memory dict)
```

## Migration Guide

### From Hardcoded Cache to DI

**Before:**
```python
service = KnowledgeGraphService(db_pool)
# Cache is hardcoded inside service
```

**After:**
```python
# Option 1: Use default (backward compatible)
service = KnowledgeGraphService(db_pool)

# Option 2: Inject custom cache
cache = KnowledgeGraphCache(max_entities=10000)
service = KnowledgeGraphService(db_pool, cache=cache)

# Option 3: Use factory
service = ServiceFactory.create_service(db_pool, cache_type='memory')
```

### Adding Redis Cache (Future)

1. **Implement RedisCache**:
```python
# src/knowledge_graph/redis_cache.py
class RedisCache:
    """Redis-backed cache implementing CacheProtocol."""

    def __init__(self, redis_client):
        self.redis = redis_client

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        # Implement Redis get logic
        ...

    def set_entity(self, entity: Entity) -> None:
        # Implement Redis set logic
        ...

    # Implement other CacheProtocol methods...
```

2. **Update ServiceFactory**:
```python
elif cache_type == 'redis':
    from src.knowledge_graph.redis_cache import RedisCache
    redis_client = kwargs.get('redis_client')
    cache = RedisCache(redis_client)
    return KnowledgeGraphService(db_pool, cache=cache)
```

3. **Use in production**:
```python
import redis
redis_client = redis.Redis(host='localhost', port=6379)
service = ServiceFactory.create_service(db_pool, cache_type='redis', redis_client=redis_client)
```

## Best Practices

### 1. Always Use Protocols for Abstractions

```python
# ✅ Good: Depend on protocol
def __init__(self, cache: CacheProtocol): ...

# ❌ Bad: Depend on concrete class
def __init__(self, cache: KnowledgeGraphCache): ...
```

### 2. Provide Sensible Defaults

```python
# ✅ Good: Optional with default
def __init__(self, cache: Optional[CacheProtocol] = None):
    self.cache = cache or KnowledgeGraphCache()

# ❌ Bad: Always required
def __init__(self, cache: CacheProtocol):
    self.cache = cache  # Forces caller to provide
```

### 3. Document DI Benefits

```python
class KnowledgeGraphService:
    """Service with dependency injection.

    Benefits:
    - Testability: Inject mock cache
    - Flexibility: Swap LRU → Redis
    - Scalability: Configuration-driven deployment
    """
```

### 4. Use Factory for Complex Creation

```python
# ✅ Good: Use factory for environment-based config
service = ServiceFactory.create_service(db_pool, cache_type='memory')

# ❌ Bad: Manual construction in many places
cache = KnowledgeGraphCache(max_entities=5000, max_relationship_caches=10000)
service = KnowledgeGraphService(db_pool, cache=cache)
```

## Performance Considerations

### Cache Performance Characteristics

| Cache Type | Latency | Throughput | Scalability |
|------------|---------|------------|-------------|
| LRU (in-memory) | <2μs | 500k ops/sec | Single server |
| Redis (local) | 0.5-1ms | 100k ops/sec | Multi-server |
| Redis (remote) | 2-5ms | 50k ops/sec | Distributed |

**Recommendation:**
- Development: LRU cache
- Production (single server): LRU cache
- Production (multi-server): Redis cache

### Memory Usage

```python
# LRU cache memory estimation
# Entity: ~200 bytes (UUID + text + metadata)
# Relationship cache: ~1KB (list of entities)

max_entities = 10000  # ~2 MB
max_relationship_caches = 20000  # ~20 MB
# Total: ~22 MB for cache
```

## References

- **CacheProtocol**: `src/knowledge_graph/cache_protocol.py`
- **KnowledgeGraphCache**: `src/knowledge_graph/cache.py`
- **ServiceFactory**: `src/knowledge_graph/service_factory.py`
- **Test Suite**: `tests/knowledge_graph/test_service_di.py`

## Related Patterns

- **Dependency Inversion Principle** (SOLID): High-level modules depend on abstractions
- **Strategy Pattern**: Swap cache implementations at runtime
- **Factory Pattern**: Centralized service creation
- **Protocol Pattern**: Structural subtyping for interfaces
