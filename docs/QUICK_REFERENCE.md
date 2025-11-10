# Knowledge Graph & Search Components - Quick Reference

## Core Concepts

### Knowledge Graph
- **Nodes**: Entities (PERSON, ORG, PRODUCT, GPE, etc.)
- **Edges**: Relationships (hierarchical, mentions-in-document, similar-to)
- **Storage**: SQLAlchemy ORM with PostgreSQL backend
- **Caching**: LRU cache for hot-path optimization

### Search Pipeline
1. Query → Strategy Router (vector/bm25/hybrid)
2. Execute searches in parallel
3. Merge with RRF (if hybrid)
4. Apply multi-factor boosts
5. Optional cross-encoder reranking
6. Filter by score threshold & top_k

### Scoring Strategy
```
Initial Score (Vector: 0-1, BM25: 0-∞)
    ↓
RRF Merging (0-1)
    ↓
Multi-factor Boosting (+5-15%)
    ↓
Cross-encoder Reranking (optional, 0-1)
    ↓
Final Score (0-1)
```

---

## Key Files & Classes

### Knowledge Graph

| File | Class | Purpose |
|------|-------|---------|
| `models.py` | `KnowledgeEntity` | Entity node with embeddings |
| `models.py` | `EntityRelationship` | Typed directed edge |
| `models.py` | `EntityMention` | Provenance tracking |
| `graph_service.py` | `KnowledgeGraphService` | Service layer + cache |
| `query_repository.py` | `KnowledgeGraphQueryRepository` | SQL traversal queries |
| `cache.py` | `KnowledgeGraphCache` | LRU cache (OrderedDict) |

### Search

| File | Class | Purpose |
|------|-------|---------|
| `hybrid_search.py` | `HybridSearch` | Main orchestrator |
| `rrf.py` | `RRFScorer` | Reciprocal Rank Fusion |
| `cross_encoder_reranker.py` | `CrossEncoderReranker` | Pair-wise relevance |
| `boosting.py` | `BoostingSystem` | Multi-factor boosts |
| `vector_search.py` | `VectorSearch` | Semantic similarity |
| `bm25_search.py` | `BM25Search` | Full-text search |

### MCP Tools

| File | Function | Purpose |
|------|----------|---------|
| `find_vendor_info.py` | `find_vendor_info()` | Graph traversal via MCP |
| `semantic_search.py` | `semantic_search()` | Hybrid search via MCP |

---

## Common Patterns

### Using Knowledge Graph Service

```python
from src.knowledge_graph.graph_service import KnowledgeGraphService

service = KnowledgeGraphService(db_pool)

# Get entity
entity = service.get_entity(entity_id)

# Get 1-hop relationships (cached)
related = service.traverse_1hop(entity_id, "hierarchical")

# Get 2-hop relationships (not cached)
two_hop = service.traverse_2hop(entity_id)

# Get cache statistics
stats = service.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
```

### Using Hybrid Search

```python
from src.search.hybrid_search import HybridSearch

hybrid = HybridSearch(db_pool, settings, logger)

# Auto-routing (strategy selected based on query)
results = hybrid.search("JWT authentication", top_k=10)

# Explicit strategy
results = hybrid.search(
    "authentication best practices",
    strategy="hybrid",  # vector, bm25, or hybrid
    min_score=0.3,
    boosts=BoostWeights(vendor=0.15, recency=0.05)
)

# With explanation
results, explanation = hybrid.search_with_explanation("OAuth2")
print(f"Strategy: {explanation.strategy}")
print(f"Confidence: {explanation.strategy_confidence:.0%}")

# With performance profiling
results, profile = hybrid.search_with_profile("API design")
print(f"Total: {profile.total_time_ms:.1f}ms")
print(f"Vector: {profile.vector_search_time_ms:.1f}ms")
print(f"BM25: {profile.bm25_search_time_ms:.1f}ms")
print(f"RRF: {profile.merging_time_ms:.1f}ms")
```

### Using Cross-Encoder Reranker

```python
from src.search.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankerConfig
)

config = RerankerConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="auto",
    batch_size=32,
    top_k=5,
    base_pool_size=50,
    adaptive_sizing=True
)

reranker = CrossEncoderReranker(config=config)
reranker.load_model()  # Lazy loading

# Rerank hybrid search results
results = hybrid.search("authentication", top_k=100)
reranked = reranker.rerank(
    query="authentication best practices",
    search_results=results,
    top_k=5,
    min_confidence=0.3
)

for result in reranked:
    print(f"{result.rank}. {result.source_file}")
    print(f"   Confidence: {result.confidence:.3f}")
```

### Using RRF Scorer

```python
from src.search.rrf import RRFScorer

scorer = RRFScorer(k=60)

# Merge two sources
merged = scorer.merge_results(
    vector_results,
    bm25_results,
    weights=(0.6, 0.4)  # 60% vector, 40% BM25
)

# Merge multiple sources
merged = scorer.fuse_multiple(
    results_by_source={
        "vector": vector_results,
        "bm25": bm25_results,
        "keyword": keyword_results,
    },
    weights={"vector": 0.5, "bm25": 0.3, "keyword": 0.2}
)
```

---

## Configuration

### Graph Service Cache Config
```python
from src.knowledge_graph.cache_config import CacheConfig

config = CacheConfig(
    max_entities=1000,              # Max cached entities
    max_relationship_caches=500,    # Max relationship caches
)

service = KnowledgeGraphService(db_pool, cache_config=config)
```

### Search Configuration
```python
from src.search.config import get_search_config

config = get_search_config()

# RRF settings
config.rrf.k = 60  # RRF k parameter

# Boost weights
config.boosts.vendor = 0.15
config.boosts.doc_type = 0.10
config.boosts.recency = 0.05
config.boosts.entity = 0.10
config.boosts.topic = 0.08
```

---

## Performance Targets

### Graph Operations
- Entity cache hit: <2μs
- 1-hop query (P50): <5ms
- 1-hop query (P95): <10ms
- 2-hop query (P50): <20ms
- 2-hop query (P95): <50ms

### Search Operations
- Vector search: <100ms
- BM25 search: <50ms
- RRF merging (100+100): <50ms
- Boosting (100 results): <10ms
- Reranking (50 pairs): <100ms
- **End-to-end hybrid**: P50 <300ms, P95 <500ms

---

## Troubleshooting

### Low Cache Hit Rate
- Increase `max_entities` in CacheConfig
- Check query patterns - are you querying same entities?
- Monitor with `service.get_cache_stats()`

### Slow Reranking
- Reduce `base_pool_size` (defaults to 50)
- Disable `adaptive_sizing` for fixed pool
- Use smaller batch_size (32 is default, try 16)

### Unbalanced Results (too many from one source)
- Adjust RRF weights (default 0.6 vector, 0.4 BM25)
- Try different RRF k values (default 60, range 1-1000)
- Check if queries favor one strategy

### Graph Queries Slow
- Verify database indexes exist (should be auto-created)
- Check min_confidence parameter (default 0.7)
- Consider reducing max_results parameter

---

## Entity Types & Relationships

### Entity Types (from spaCy NER)
- `PERSON`: People, fictional characters
- `ORG`: Organizations, companies
- `PRODUCT`: Products, technologies, services
- `GPE`: Countries, cities, states
- `EVENT`: Named events
- `FACILITY`: Buildings, infrastructure
- `LAW`: Laws, regulations
- `LANGUAGE`: Languages
- `DATE`, `TIME`, `MONEY`, `PERCENT`

### Relationship Types
- `hierarchical`: Parent/child, creator/creation
- `mentions-in-document`: Co-occurrence in same chunk
- `similar-to`: Semantic similarity (bidirectional)

---

## Debugging & Monitoring

### Get Search Explanation
```python
results, explanation = hybrid.search_with_explanation(query)
print(f"Query: {explanation.query}")
print(f"Strategy: {explanation.strategy}")
print(f"Confidence: {explanation.strategy_confidence:.0%}")
print(f"Reason: {explanation.strategy_reason}")
print(f"Vector results: {explanation.vector_results_count}")
print(f"BM25 results: {explanation.bm25_results_count}")
print(f"Merged: {explanation.merged_results_count}")
print(f"Final: {explanation.final_results_count}")
print(f"Boosts: {explanation.boosts_applied}")
```

### Monitor Cache Performance
```python
stats = service.get_cache_stats()
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Evictions: {stats['evictions']}")
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Size: {stats['size']} / {stats['max_size']}")
```

### Profile Search Performance
```python
results, profile = hybrid.search_with_profile(query)
print(f"Total: {profile.total_time_ms:.1f}ms")
print(f"  Routing: {profile.routing_time_ms:.1f}ms")
print(f"  Vector: {profile.vector_search_time_ms:.1f}ms")
print(f"  BM25: {profile.bm25_search_time_ms:.1f}ms")
print(f"  Merging: {profile.merging_time_ms:.1f}ms")
print(f"  Boosting: {profile.boosting_time_ms:.1f}ms")
print(f"  Filtering: {profile.filtering_time_ms:.1f}ms")
```

---

## Related Documentation

- Full Analysis: `docs/CODEBASE_SEARCH_REPORT.md`
- Graph Migration: `src/knowledge_graph/migrations/001_create_knowledge_graph.py`
- Search Config: `src/search/config.py`
- Response Formatting: `src/mcp/response_formatter.py`

