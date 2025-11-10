# Knowledge Graph Implementation Search Report

## Executive Summary

This codebase implements a sophisticated **hybrid search system** with integrated **knowledge graph** functionality, **reranking algorithms**, and **multi-factor scoring strategies**. The architecture combines vector similarity search, BM25 full-text search, and knowledge graph traversal with intelligent result ranking.

---

## 1. KNOWLEDGE GRAPH IMPLEMENTATION

### Core Files

#### `src/knowledge_graph/models.py` (326 lines)
**Purpose**: SQLAlchemy ORM models for knowledge graph entities and relationships

**Key Models**:

1. **KnowledgeEntity** - Entity nodes in the graph
   - Attributes: `id` (UUID), `text`, `entity_type`, `confidence`, `mention_count`
   - Entity types: PERSON, ORG, PRODUCT, GPE, EVENT, FACILITY, LAW, LANGUAGE, DATE, TIME, MONEY, PERCENT
   - Relationships: `relationships_from`, `relationships_to`, `mentions`
   - Indexes: text, entity_type, mention_count

2. **EntityRelationship** - Directed edges between entities
   - Attributes: `source_entity_id`, `target_entity_id`, `relationship_type`, `confidence`, `relationship_weight`, `is_bidirectional`
   - Relationship types:
     - `hierarchical` - Parent/child, creator/creation
     - `mentions-in-document` - Co-occurrence
     - `similar-to` - Semantic similarity
   - Unique constraint: (source, target, type)

3. **EntityMention** - Provenance tracking
   - Attributes: `entity_id`, `document_id`, `chunk_id`, `mention_text`, `offset_start`, `offset_end`
   - Purpose: Tracks where entities appear in documents

#### `src/knowledge_graph/query_repository.py` (~150 lines)
**Purpose**: Optimized SQL queries for graph traversal

**Result Classes**:
```python
RelatedEntity(id, text, entity_type, confidence, relationship_type, relationship_confidence)
TwoHopEntity(id, text, entity_type, confidence, relationship_type, intermediate_entity, path_confidence)
BidirectionalEntity(id, text, entity_type, outbound/inbound_rel_types, max_confidence, distance)
EntityMention(chunk_id, document_id, mention_text, document_category, mention_confidence)
```

**Query Methods**:
- `traverse_1hop()` - Direct relationships (P50 <5ms, P95 <10ms)
- `traverse_2hop()` - Two-hop paths
- `traverse_bidirectional()` - Incoming + outgoing relationships
- `traverse_with_type_filter()` - Filter by target entity types
- `get_entity_mentions()` - Find where entity appears

#### `src/knowledge_graph/graph_service.py` (418 lines)
**Purpose**: High-level service layer with LRU cache for graph queries

**Key Features**:
- **Dependency Injection**: Pluggable cache and repository implementations
- **LRU Cache**: In-memory caching with configurable size
- **Hot Path Optimization**: <2μs cache hit, 5-20ms cache miss
- **Cache Hit Rate**: Target >80% for 1-hop queries

**Methods**:
```python
get_entity(entity_id)              # Get single entity with cache
traverse_1hop(entity_id, rel_type) # 1-hop with cache
traverse_2hop(entity_id, rel_type) # 2-hop (always queried)
traverse_bidirectional()            # Bidirectional traversal
get_mentions(entity_id)             # Find document mentions
traverse_with_type_filter()         # Type-filtered traversal
get_cache_stats()                   # Cache hit/miss stats
```

**Cache Implementation**: 
- Type: `KnowledgeGraphCache` (OrderedDict-based LRU)
- Configurable max_entities and max_relationship_caches
- Automatic eviction, stats tracking

---

## 2. RERANKING LOGIC

### Cross-Encoder Reranking

#### `src/search/cross_encoder_reranker.py` (849 lines)
**Purpose**: Pair-wise relevance scoring using HuggingFace cross-encoder

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 6 layers, 384 hidden dims
- Trained on MS MARCO dataset
- Fast, accurate pair-wise relevance

**Key Components**:

1. **RerankerConfig** - Centralized configuration
   - `model_name`: HuggingFace identifier
   - `device`: "auto" (GPU/CPU detection), "cuda", "cpu"
   - `batch_size`: Default 32
   - `min_confidence`: Confidence threshold (0-1)
   - `top_k`: Results to return (default 5)
   - `base_pool_size`: Base candidates (default 50)
   - `max_pool_size`: Max candidates (default 100)
   - `adaptive_sizing`: Enable query-aware pool sizing
   - `complexity_constants`: Tunable complexity calculation weights

2. **QueryAnalysis** - Query complexity detection
   - Analyzes: length, keyword_count, operators, quoted_phrases
   - Computes: complexity score (0-1), query_type (short/medium/long/complex)

3. **CandidateSelector** - Adaptive pool sizing
   ```python
   analyze_query(query)            # Complexity analysis
   calculate_pool_size(analysis)   # Adaptive sizing
   select(results, pool_size)      # Top-K selection
   ```
   
   **Complexity Formula**:
   ```
   complexity = min(1.0, 
       (keyword_count / 10.0) * 0.6 +
       (0.2 if operators else 0) +
       (0.2 if quotes else 0)
   )
   ```
   
   **Pool Size Formula**:
   ```
   pool_size = base * (1.0 + complexity * 1.2)
   capped to [5, max_pool_size]
   ```

4. **CrossEncoderReranker** - Main reranking engine
   ```python
   load_model()            # Lazy-load cross-encoder
   score_pairs()           # Batch pair scoring
   rerank(query, results)  # Full reranking pipeline
   ```
   
   **Reranking Pipeline**:
   1. Select adaptive candidate pool
   2. Score all pairs (query + document)
   3. Normalize scores via sigmoid (0-1)
   4. Filter by confidence threshold
   5. Select top-K by confidence
   6. Rerank with updated metadata

**Performance Targets**:
- Model loading: <5s
- Batch inference: <100ms for 50 pairs
- Pool calculation: <1ms
- Overall reranking: <200ms

#### `src/search/reranker_protocol.py`
**Purpose**: Protocol/interface for reranker implementations
- Enables alternative reranker implementations
- Used by HybridSearch for pluggable reranking

---

## 3. RESULT RANKING & SCORING

### Reciprocal Rank Fusion (RRF)

#### `src/search/rrf.py` (383 lines)
**Purpose**: Merge vector and BM25 results using RRF algorithm

**Algorithm**: `score = 1 / (k + rank)`
- Treats different scoring scales uniformly (rank-based, not score-based)
- Reduces outlier impacts
- Naturally deduplicates results
- Reference: Cormack et al., 2009

**Key Classes**:

1. **RRFScorer**
   ```python
   __init__(k=60)                    # K parameter (default 60)
   merge_results(vector, bm25)       # Merge two sources
   fuse_multiple(results_by_source)  # Merge 3+ sources
   ```

2. **Merge Algorithm**:
   - Calculate RRF score for each result: `1 / (k + rank)`
   - Build maps: `chunk_id -> (result, rrf_score)`
   - Deduplicate results appearing in both sources
   - Combine with weights: `vector_weight=0.6, bm25_weight=0.4`
   - Sort by combined score descending
   - Rerank with 1-indexed positions

3. **Score Combination**:
   ```python
   # For results in both sources
   combined_score = (vector_score * 0.6) + (bm25_score * 0.4)
   
   # For results in one source
   combined_score = rrf_score * source_weight
   ```

**Performance Target**: <50ms for merging 100 results from each source

### Multi-Factor Boosting System

#### `src/search/boosting.py` (100+ lines)
**Purpose**: Content-aware boosts based on multiple relevance factors

**Boost Factors**:
1. **Vendor Matching** (+15%): Document vendor matches query context
2. **Document Type** (+10%): Doc type matches query intent (api_docs, guide, kb_article, code_sample, reference)
3. **Recency** (+5%): Recent documents (<30 days)
4. **Entity Matching** (+10%): Query entities found in document
5. **Topic Matching** (+8%): Document topic matches query topic

**Known Topics**:
- `authentication`: JWT, OAuth, SAML, MFA, tokens
- `api_design`: REST, GraphQL, webhooks, versioning
- `data_handling`: Database, storage, caching, migration
- `deployment`: Docker, Kubernetes, CI/CD, scaling
- `optimization`: Performance, latency, benchmarking
- `error_handling`: Error codes, exceptions, debugging

**Boost Constants**:
- Cumulative but clamped to max 1.0
- Preserves relative ranking while amplifying relevant results
- Performance target: <10ms for 100 results

#### `src/search/boost_optimizer.py` & `src/search/boost_strategies.py`
**Purpose**: Extensible boost strategy framework
- `BoostStrategy` ABC for custom implementations
- `BoostStrategyFactory` for strategy registration
- Support for multiple boost implementations

---

## 4. HYBRID SEARCH INTEGRATION

#### `src/search/hybrid_search.py` (797 lines)
**Purpose**: Unified orchestration of all search components

**Architecture**:
```
Query
  ↓
QueryRouter (auto-select strategy)
  ↓
┌─────────────────────────┐
│ Vector Search (parallel)│ BM25 Search (parallel)
└─────────────────────────┘
  ↓
RRF Merging
  ↓
BoostingSystem (multi-factor)
  ↓
Final Filtering (top_k, min_score)
  ↓
Results (ranked)
```

**Search Methods**:
1. `search()` - Basic search with auto-routing
2. `search_with_explanation()` - Returns routing/ranking explanation
3. `search_with_profile()` - Returns performance profiling data

**Strategies**:
- `"vector"`: Vector similarity only
- `"bm25"`: Full-text search only
- `"hybrid"`: Combined with RRF merging
- `None`: Auto-select via QueryRouter

**Performance Targets**:
- Vector search: <100ms
- BM25 search: <50ms
- RRF merging: <50ms
- Boosting: <10ms
- Filtering: <5ms
- **End-to-end (hybrid)**: P50 <300ms, P95 <500ms

---

## 5. KNOWLEDGE GRAPH IN MCP TOOLS

#### `src/mcp/tools/find_vendor_info.py` (200+ lines)
**Purpose**: MCP tool to retrieve vendor information from knowledge graph

**Progressive Disclosure Response Modes**:
1. **ids_only** (~100-500 tokens): Vendor ID + counts
2. **metadata** (~2-4K tokens): IDs + statistics + distributions (DEFAULT)
3. **preview** (~5-10K tokens): Metadata + top 5 entities + top 5 relationships
4. **full** (~10-50K+ tokens): Complete vendor graph (max 100 entities, 500 relationships)

**Workflow**:
1. Find vendor by name (case-insensitive exact match)
2. Traverse 1-hop relationships to find related entities
3. Format results based on response_mode
4. Apply field filtering if requested
5. Support pagination with cursor-based encoding

**Caching**:
- Cache key: `vendor:hash(vendor_name, response_mode)`
- 300s TTL
- Progressive loading to minimize token usage

---

## 6. SCORING SUMMARY

### Score Types

| Type | Source | Range | Purpose |
|------|--------|-------|---------|
| `similarity_score` | Vector search | 0-1 | Semantic similarity |
| `bm25_score` | BM25 search | 0-∞ | Term frequency match |
| `hybrid_score` | RRF merger | 0-1 | Combined ranking |
| `confidence` | Cross-encoder | 0-1 | Relevance confidence |

### Scoring Pipeline

1. **Initial Scoring**:
   - Vector: Cosine similarity → [0, 1]
   - BM25: Term frequency → [0, ∞]

2. **Merging** (RRF):
   - Rank each source: 1, 2, 3, ...
   - Calculate RRF: `1/(k+rank)`
   - Combine: `v_score * 0.6 + bm25_score * 0.4`
   - Normalize: Clamp to [0, 1]

3. **Boosting** (Multi-factor):
   - Apply up to 5 boost factors
   - Each adds 5-15% to score
   - Cumulative but capped at 1.0

4. **Reranking** (Optional):
   - Cross-encoder scores query-document pairs
   - Sigmoid normalization → [0, 1]
   - Filter by confidence threshold
   - Select top-K

5. **Final Filtering**:
   - Apply min_score threshold
   - Limit to top_k
   - Rerank with 1-indexed positions

---

## 7. FILE STRUCTURE SUMMARY

```
src/
├── knowledge_graph/
│   ├── models.py                 # ORM models (Entity, Relationship, Mention)
│   ├── graph_service.py          # Service layer with LRU cache
│   ├── query_repository.py       # Optimized SQL traversal queries
│   ├── cache.py                  # LRU cache implementation
│   ├── cache_protocol.py         # Cache interface (DI)
│   ├── cache_config.py           # Cache configuration
│   └── migrations/               # Database schema migrations
│
├── search/
│   ├── hybrid_search.py          # Main orchestration (797 lines)
│   ├── rrf.py                    # Reciprocal Rank Fusion (383 lines)
│   ├── cross_encoder_reranker.py # Cross-encoder reranking (849 lines)
│   ├── reranker_protocol.py      # Reranker interface
│   ├── boosting.py               # Multi-factor boosting
│   ├── boost_optimizer.py        # Boost optimization
│   ├── boost_strategies.py       # Strategy pattern for boosts
│   ├── vector_search.py          # Vector similarity search
│   ├── bm25_search.py            # BM25 full-text search
│   ├── query_router.py           # Strategy selection
│   ├── results.py                # SearchResult dataclass
│   └── config.py                 # Search configuration
│
└── mcp/
    └── tools/
        ├── find_vendor_info.py   # Knowledge graph MCP tool
        └── semantic_search.py    # Semantic search MCP tool
```

---

## 8. KEY INTEGRATION POINTS

1. **Vector ↔ BM25**: RRFScorer merges results
2. **Merged ↔ Boosting**: BoostingSystem applies factors
3. **Boosted ↔ Reranking**: CrossEncoderReranker reranks
4. **Graph ↔ Search**: find_vendor_info uses graph queries
5. **All ↔ MCP Tools**: Wrapped for MCP protocol compliance

---

## 9. PERFORMANCE CHARACTERISTICS

| Operation | P50 | P95 | Notes |
|-----------|-----|-----|-------|
| Entity cache hit | <2μs | <2μs | In-memory OrderedDict |
| 1-hop graph query | <5ms | <10ms | With database indexes |
| 2-hop graph query | <20ms | <50ms | No caching |
| Vector search | <100ms | - | Embedding lookup + similarity |
| BM25 search | <50ms | - | Index scan + scoring |
| RRF merging | <50ms | - | 100 results each source |
| Boosting | <10ms | - | 100 results |
| Cross-encoder scoring | <100ms | - | 50 pairs, batch inference |
| Hybrid search end-to-end | <300ms | <500ms | Full pipeline |

---

## 10. EXTENSIBILITY POINTS

1. **Cache Implementations**: Swap LRU → Redis via CacheProtocol
2. **Reranker Implementations**: Custom via RerankerProtocol
3. **Boost Strategies**: Custom via BoostStrategy ABC
4. **Query Routing**: Custom strategies in QueryRouter
5. **Entity Types/Relations**: Add to EntityTypeEnum, RelationshipTypeEnum in models
6. **Graph Traversal**: Add methods to KnowledgeGraphQueryRepository

