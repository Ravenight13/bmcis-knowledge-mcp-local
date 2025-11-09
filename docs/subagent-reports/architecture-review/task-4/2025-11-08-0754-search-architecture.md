# Hybrid Search System Architecture Review
**Task 4: Hybrid Search Implementation (Vector + BM25)**

**Date**: 2025-11-08
**Context**: Phase 2 Complete - 2,600+ chunks with 768-dimensional embeddings
**Reviewer**: Architecture Agent
**Status**: Architecture Review for Implementation Planning

---

## Executive Summary

This document provides a comprehensive architecture review for implementing a hybrid search system combining vector similarity search (pgvector HNSW) with BM25 full-text search (PostgreSQL ts_vector). The design leverages existing Phase 0 infrastructure (database pooling, logging, config) and Phase 2 embeddings to deliver sub-100ms search performance with rich filtering capabilities.

**Key Architectural Decisions**:
- **Separation of Concerns**: Distinct search strategies (vector, BM25, hybrid) with unified interface
- **Performance First**: HNSW + GIN indexes with query cost estimation
- **Type Safety**: Pydantic v2 models with mypy --strict compliance
- **Extensibility**: Pluggable ranking algorithms and filter composition
- **Production Ready**: Comprehensive error handling, logging, and monitoring

**Performance Targets**:
- Vector search: <100ms for 2,600 chunks
- BM25 search: <50ms for 2,600 chunks
- Metadata filtering: <20ms additional
- Hybrid result merging: <10ms

---

## 1. Current System State Analysis

### 1.1 Existing Infrastructure (Phase 0)

**Database Connection Pooling** (`src/core/database.py`):
- `DatabasePool.get_connection()`: Context manager with retry logic
- Connection health checks (SELECT 1 validation)
- Exponential backoff retry strategy (2^attempt seconds)
- Configurable pool sizing (min/max from settings)
- Statement timeout enforcement (server-side via PostgreSQL options)

**Configuration Management** (`src/core/config.py`):
- Pydantic v2 Settings with environment variable loading
- Type-safe configuration: `DatabaseConfig`, `LoggingConfig`, `ApplicationConfig`
- Factory pattern: `get_settings()` singleton
- Validation: Pool sizes, environment constraints, debug mode

**Structured Logging** (`src/core/logging.py`):
- `StructuredLogger.get_logger(__name__)`: Module-level loggers
- JSON/text format support
- Context-aware logging with metadata

**Assessment**: Phase 0 infrastructure is production-ready and sufficient for search implementation. No changes required.

---

### 1.2 Data Layer (Phase 2)

**Database Schema** (`sql/schema_768.sql`):
```sql
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    chunk_hash VARCHAR(64) UNIQUE NOT NULL,
    embedding vector(768),                    -- pgvector embeddings
    source_file VARCHAR(512) NOT NULL,
    source_category VARCHAR(128),
    document_date DATE,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    context_header TEXT,
    ts_vector tsvector,                       -- Auto-updated for FTS
    metadata JSONB,                           -- Added via migration
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Indexes**:
1. **HNSW Vector Index** (already created):
   ```sql
   CREATE INDEX idx_knowledge_embedding ON knowledge_base
   USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);
   ```
   - **m=16**: 16 connections per node (balance speed/accuracy)
   - **ef_construction=64**: Dynamic candidate list size during construction
   - **Operator**: `vector_cosine_ops` for cosine distance (1 - cosine similarity)

2. **GIN Full-Text Search Index**:
   ```sql
   CREATE INDEX idx_knowledge_fts ON knowledge_base
   USING GIN(ts_vector);
   ```
   - Auto-updated via trigger: `trigger_update_knowledge_ts_vector`
   - Combines `chunk_text + context_header` for search
   - English text search configuration

3. **GIN Metadata Index** (added via migration):
   ```sql
   CREATE INDEX idx_knowledge_metadata ON knowledge_base
   USING GIN(metadata);
   ```
   - JSONB containment queries: `metadata @> '{"tags": ["api"]}'`

4. **B-tree Metadata Indexes**:
   - `idx_knowledge_category`: Filter by source_category
   - `idx_knowledge_source_file`: Filter by source file
   - `idx_knowledge_document_date`: Filter by date (DESC for recent first)
   - `idx_knowledge_category_date`: Compound index for common pattern

**Triggers**:
- `update_knowledge_base_ts_vector()`: Auto-updates ts_vector on INSERT/UPDATE
- `update_timestamp()`: Auto-updates updated_at on row changes

**Data Statistics**:
- **Volume**: 2,600+ document chunks (Phase 2 complete)
- **Embedding Dimension**: 768 (all-mpnet-base-v2)
- **Table Size**: Estimated 50-100MB with embeddings

**Assessment**: Schema is well-designed for hybrid search with appropriate indexes. GIN index on ts_vector enables BM25-style ranking via PostgreSQL's text search.

---

### 1.3 Data Models (Phase 2)

**ProcessedChunk** (`src/document_parsing/models.py`):
```python
class ProcessedChunk(BaseModel):
    chunk_text: str
    chunk_hash: str
    context_header: str
    source_file: str
    source_category: str | None
    document_date: date | None
    chunk_index: int
    total_chunks: int
    chunk_token_count: int
    metadata: dict[str, Any]
    embedding: list[float] | None  # 768-dimensional
```

**Assessment**: Existing model is comprehensive and type-safe. Can be reused for search results with minor extensions (score field).

---

## 2. Architecture Design

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Search API Layer                            │
│  SearchQuery → SearchExecutor → [SearchResult, ...]            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Vector     │ │     BM25     │ │   Hybrid     │
│   Search     │ │    Search    │ │   Search     │
│   Strategy   │ │   Strategy   │ │   Strategy   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │                └────────┬───────┘
       │                         │
       ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  Vector Ranker   │    │   BM25 Ranker    │
│  (Cosine Dist)   │    │  (ts_rank_cd)    │
└──────┬───────────┘    └──────┬───────────┘
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Result Merger   │
          │  (Weighted RRF)  │
          └──────┬───────────┘
                 │
                 ▼
          ┌──────────────────┐
          │ Metadata Filter  │
          │ (JSONB, B-tree)  │
          └──────┬───────────┘
                 │
                 ▼
          ┌──────────────────┐
          │   DatabasePool   │
          │  (Phase 0 Infra) │
          └──────────────────┘
```

---

### 2.2 Core Components

#### 2.2.1 Search Models (`src/search/models.py`)

**SearchQuery**: Input parameters
```python
class FilterExpression(BaseModel):
    """Type-safe metadata filter expression."""
    field: str                          # metadata.tags, source_category, etc.
    operator: Literal["eq", "in", "contains", "gte", "lte", "between"]
    value: str | int | float | list[str] | dict[str, Any]

class SearchQuery(BaseModel):
    """Search request parameters."""
    query_text: str = Field(min_length=1, max_length=1000)
    search_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    # Hybrid search weights
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Filters
    filters: list[FilterExpression] = Field(default_factory=list)

    # Date range
    date_from: date | None = None
    date_to: date | None = None

    # Similarity threshold
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("bm25_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info: ValidationInfo) -> float:
        """Validate vector_weight + bm25_weight = 1.0 for hybrid mode."""
        vector_weight = info.data.get("vector_weight", 0.7)
        if abs(vector_weight + v - 1.0) > 0.01:
            raise ValueError(
                f"vector_weight + bm25_weight must equal 1.0, "
                f"got {vector_weight} + {v} = {vector_weight + v}"
            )
        return v
```

**SearchResult**: Output with score
```python
class SearchResult(BaseModel):
    """Single search result with ranking score."""
    chunk_id: int
    chunk_text: str
    context_header: str
    source_file: str
    source_category: str | None
    document_date: date | None
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any]

    # Ranking scores
    score: float = Field(ge=0.0, le=1.0)
    vector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bm25_score: float | None = Field(default=None, ge=0.0)

    # Matched terms (for highlighting)
    matched_terms: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)  # ORM mapping

class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    results: list[SearchResult]
    total_count: int
    query_time_ms: float
    search_mode: str
    applied_filters: dict[str, Any]
```

**Type Safety Benefits**:
- Compile-time validation of filter expressions
- Automatic weight normalization validation
- Clear contract between API and database layer
- Easy testing with typed fixtures

---

#### 2.2.2 Search Strategies (`src/search/strategies.py`)

**Base Strategy** (Strategy Pattern):
```python
from abc import ABC, abstractmethod

class SearchStrategy(ABC):
    """Base class for search strategies."""

    def __init__(self, embedding_model: Any | None = None):
        """Initialize strategy with optional embedding model.

        Args:
            embedding_model: Embedding model for vector strategies (None for BM25).
        """
        self.embedding_model = embedding_model

    @abstractmethod
    def search(
        self,
        conn: Connection,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Execute search and return ranked results.

        Args:
            conn: Database connection from pool.
            query: Search parameters.

        Returns:
            List of SearchResult ordered by relevance.
        """
        pass
```

**Vector Search Strategy**:
```python
class VectorSearchStrategy(SearchStrategy):
    """Vector similarity search using HNSW index."""

    def search(
        self,
        conn: Connection,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Search using cosine distance on embeddings.

        Query plan:
        1. Generate query embedding (768-dimensional)
        2. Execute HNSW similarity search (cosine distance)
        3. Apply metadata filters (JSONB, B-tree)
        4. Return top-k results ordered by similarity
        """
        # Generate query embedding
        query_embedding = self._embed_query(query.query_text)

        # Build SQL with filters
        sql = self._build_vector_query(query)

        # Execute with query embedding
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query_embedding": self._serialize_vector(query_embedding),
                "limit": query.limit,
                "offset": query.offset,
                "min_similarity": query.min_similarity,
                **self._build_filter_params(query.filters),
            })

            return [self._row_to_result(row) for row in cur.fetchall()]

    def _build_vector_query(self, query: SearchQuery) -> str:
        """Build vector similarity SQL query.

        Returns SQL using:
        - embedding <=> %(query_embedding)s: Cosine distance operator
        - (1 - distance) AS similarity: Convert to similarity score
        - WHERE clauses for filters
        """
        return """
            SELECT
                id,
                chunk_text,
                context_header,
                source_file,
                source_category,
                document_date,
                chunk_index,
                total_chunks,
                metadata,
                (1 - (embedding <=> %(query_embedding)s::vector)) AS similarity,
                NULL::float AS bm25_score
            FROM knowledge_base
            WHERE
                embedding IS NOT NULL
                {date_filter}
                {metadata_filters}
                AND (1 - (embedding <=> %(query_embedding)s::vector)) >= %(min_similarity)s
            ORDER BY embedding <=> %(query_embedding)s::vector
            LIMIT %(limit)s OFFSET %(offset)s
        """.format(
            date_filter=self._build_date_filter(query),
            metadata_filters=self._build_metadata_filters(query.filters),
        )
```

**BM25 Search Strategy**:
```python
class BM25SearchStrategy(SearchStrategy):
    """BM25 full-text search using PostgreSQL ts_vector."""

    def search(
        self,
        conn: Connection,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Search using BM25-style ranking via PostgreSQL.

        Query plan:
        1. Convert query to ts_query (parse search terms)
        2. Execute GIN index scan on ts_vector
        3. Rank results using ts_rank_cd (Cover Density ranking)
        4. Apply metadata filters
        5. Return top-k results ordered by rank
        """
        sql = self._build_bm25_query(query)

        with conn.cursor() as cur:
            cur.execute(sql, {
                "query_text": query.query_text,
                "limit": query.limit,
                "offset": query.offset,
                **self._build_filter_params(query.filters),
            })

            return [self._row_to_result(row) for row in cur.fetchall()]

    def _build_bm25_query(self, query: SearchQuery) -> str:
        """Build BM25-style full-text search query.

        Returns SQL using:
        - plainto_tsquery: Parse query text to ts_query
        - ts_vector @@ ts_query: Match operator (uses GIN index)
        - ts_rank_cd: Cover Density ranking (BM25-like)
        """
        return """
            SELECT
                id,
                chunk_text,
                context_header,
                source_file,
                source_category,
                document_date,
                chunk_index,
                total_chunks,
                metadata,
                NULL::float AS vector_score,
                ts_rank_cd(ts_vector, query, 32) AS bm25_score
            FROM knowledge_base,
                 plainto_tsquery('english', %(query_text)s) AS query
            WHERE
                ts_vector @@ query
                {date_filter}
                {metadata_filters}
            ORDER BY bm25_score DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """.format(
            date_filter=self._build_date_filter(query),
            metadata_filters=self._build_metadata_filters(query.filters),
        )
```

**Hybrid Search Strategy**:
```python
class HybridSearchStrategy(SearchStrategy):
    """Hybrid search combining vector + BM25 with weighted scoring."""

    def __init__(
        self,
        embedding_model: Any,
        vector_strategy: VectorSearchStrategy | None = None,
        bm25_strategy: BM25SearchStrategy | None = None,
    ):
        """Initialize hybrid strategy.

        Args:
            embedding_model: Embedding model for vector search.
            vector_strategy: Optional pre-configured vector strategy.
            bm25_strategy: Optional pre-configured BM25 strategy.
        """
        super().__init__(embedding_model)
        self.vector_strategy = vector_strategy or VectorSearchStrategy(embedding_model)
        self.bm25_strategy = bm25_strategy or BM25SearchStrategy()

    def search(
        self,
        conn: Connection,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Search using weighted combination of vector + BM25.

        Query plan:
        1. Execute vector search (top 2*limit results)
        2. Execute BM25 search (top 2*limit results)
        3. Merge results using weighted RRF (Reciprocal Rank Fusion)
        4. Normalize scores to [0, 1]
        5. Apply final filters and return top-k

        Performance optimization:
        - Fetch 2*limit from each strategy to ensure diverse results
        - Single database connection for both queries
        - Result merging in Python (negligible overhead <10ms)
        """
        # Fetch more results than needed for better merging
        expanded_query = query.model_copy(update={"limit": query.limit * 2})

        # Execute both strategies
        vector_results = self.vector_strategy.search(conn, expanded_query)
        bm25_results = self.bm25_strategy.search(conn, expanded_query)

        # Merge using weighted RRF
        merged = self._merge_results(
            vector_results,
            bm25_results,
            vector_weight=query.vector_weight,
            bm25_weight=query.bm25_weight,
        )

        # Apply offset and limit after merging
        return merged[query.offset : query.offset + query.limit]

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        vector_weight: float,
        bm25_weight: float,
    ) -> list[SearchResult]:
        """Merge results using weighted Reciprocal Rank Fusion.

        RRF Formula:
            score(d) = Σ (weight_i / (k + rank_i(d)))

        Where:
            - weight_i: Strategy weight (vector_weight or bm25_weight)
            - k: Constant (typically 60) to reduce impact of high ranks
            - rank_i(d): Rank of document d in strategy i

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            vector_weight: Weight for vector scores.
            bm25_weight: Weight for BM25 scores.

        Returns:
            Merged results sorted by combined score.
        """
        # RRF constant (standard value)
        k = 60

        # Build rank maps
        vector_ranks = {r.chunk_id: idx for idx, r in enumerate(vector_results)}
        bm25_ranks = {r.chunk_id: idx for idx, r in enumerate(bm25_results)}

        # Collect all unique chunk IDs
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Compute RRF scores
        scored_results: dict[int, tuple[SearchResult, float]] = {}

        for chunk_id in all_ids:
            # Get result object (prefer vector result if present)
            result = next(
                (r for r in vector_results if r.chunk_id == chunk_id),
                next(r for r in bm25_results if r.chunk_id == chunk_id),
            )

            # Compute weighted RRF score
            vector_rank = vector_ranks.get(chunk_id, float("inf"))
            bm25_rank = bm25_ranks.get(chunk_id, float("inf"))

            score = 0.0
            if vector_rank != float("inf"):
                score += vector_weight / (k + vector_rank)
            if bm25_rank != float("inf"):
                score += bm25_weight / (k + bm25_rank)

            # Create merged result with combined score
            merged_result = result.model_copy(update={
                "score": score,
                "vector_score": result.vector_score,
                "bm25_score": result.bm25_score,
            })

            scored_results[chunk_id] = (merged_result, score)

        # Sort by combined score (descending)
        sorted_results = sorted(
            scored_results.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [result for result, _ in sorted_results]
```

---

#### 2.2.3 Search Executor (`src/search/executor.py`)

**Main Entry Point**:
```python
class SearchExecutor:
    """Orchestrates search execution with strategy selection."""

    def __init__(self, embedding_model: Any):
        """Initialize executor with embedding model.

        Args:
            embedding_model: Model for generating query embeddings.
        """
        self.embedding_model = embedding_model

        # Pre-instantiate strategies
        self.strategies = {
            "vector": VectorSearchStrategy(embedding_model),
            "bm25": BM25SearchStrategy(),
            "hybrid": HybridSearchStrategy(embedding_model),
        }

        self.logger = StructuredLogger.get_logger(__name__)

    def execute(self, query: SearchQuery) -> SearchResponse:
        """Execute search query and return ranked results.

        Args:
            query: Search parameters.

        Returns:
            SearchResponse with results and metadata.

        Raises:
            ValueError: If invalid search_mode.
            DatabaseError: If database query fails.
        """
        start_time = time.time()

        # Validate search mode
        if query.search_mode not in self.strategies:
            raise ValueError(
                f"Invalid search_mode: {query.search_mode}. "
                f"Must be one of: {list(self.strategies.keys())}"
            )

        self.logger.info(
            f"Executing {query.search_mode} search: query='{query.query_text}', "
            f"limit={query.limit}, filters={len(query.filters)}"
        )

        try:
            # Get database connection
            with DatabasePool.get_connection() as conn:
                # Select and execute strategy
                strategy = self.strategies[query.search_mode]
                results = strategy.search(conn, query)

                # Get total count (for pagination)
                total_count = self._get_total_count(conn, query)

            query_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"Search completed: mode={query.search_mode}, "
                f"results={len(results)}, total={total_count}, "
                f"time={query_time_ms:.2f}ms"
            )

            return SearchResponse(
                results=results,
                total_count=total_count,
                query_time_ms=query_time_ms,
                search_mode=query.search_mode,
                applied_filters=self._serialize_filters(query.filters),
            )

        except Exception as e:
            self.logger.error(
                f"Search failed: {e}",
                exc_info=True,
                extra={
                    "query": query.query_text,
                    "mode": query.search_mode,
                },
            )
            raise

    def _get_total_count(self, conn: Connection, query: SearchQuery) -> int:
        """Get total count of matching results (for pagination).

        Executes COUNT(*) query with same filters as main query.
        """
        # Implementation details...
        pass
```

---

### 2.3 Query Execution Flow

**Hybrid Search Example**:

```
User Request:
  SearchQuery(
    query_text="Lutron integration",
    search_mode="hybrid",
    vector_weight=0.7,
    bm25_weight=0.3,
    filters=[
      FilterExpression(field="source_category", operator="eq", value="product_docs")
    ],
    limit=10
  )

Execution Flow:

1. SearchExecutor.execute()
   ├─ Validate query parameters
   ├─ Get database connection (DatabasePool)
   └─ Select HybridSearchStrategy

2. HybridSearchStrategy.search()
   ├─ Create expanded query (limit=20 for better merging)
   ├─ Execute VectorSearchStrategy
   │  ├─ Generate query embedding (768-dim)
   │  ├─ Execute SQL:
   │  │   SELECT *, (1 - (embedding <=> query_vec)) AS similarity
   │  │   FROM knowledge_base
   │  │   WHERE source_category = 'product_docs'
   │  │   ORDER BY embedding <=> query_vec
   │  │   LIMIT 20
   │  └─ Return 20 vector results (50ms)
   │
   ├─ Execute BM25SearchStrategy
   │  ├─ Parse query text to ts_query
   │  ├─ Execute SQL:
   │  │   SELECT *, ts_rank_cd(ts_vector, query) AS bm25_score
   │  │   FROM knowledge_base
   │  │   WHERE ts_vector @@ query
   │  │   AND source_category = 'product_docs'
   │  │   ORDER BY bm25_score DESC
   │  │   LIMIT 20
   │  └─ Return 20 BM25 results (30ms)
   │
   └─ Merge results using weighted RRF
      ├─ Build rank maps for both result sets
      ├─ Compute RRF scores: score = 0.7/(60+rank_v) + 0.3/(60+rank_b)
      ├─ Sort by combined score
      └─ Return top 10 merged results (5ms)

3. Get total count (pagination metadata)
   └─ SELECT COUNT(*) FROM knowledge_base WHERE ... (10ms)

4. Build SearchResponse
   └─ results=10, total_count=156, query_time_ms=95

Total Time: ~95ms (within <100ms target)
```

---

## 3. Index Design and Tuning

### 3.1 HNSW Index Parameters

**Current Configuration**:
```sql
CREATE INDEX idx_knowledge_embedding ON knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Parameter Analysis**:

| Parameter | Value | Rationale | Trade-offs |
|-----------|-------|-----------|------------|
| `m` | 16 | Connections per node | Higher = better recall, slower inserts |
| `ef_construction` | 64 | Build-time candidate list | Higher = better index quality, slower builds |
| `ef_search` | Default (40) | Runtime candidate list | Higher = better recall, slower queries |

**Tuning Recommendations**:
- **For 2,600 chunks**: Current parameters are optimal
- **For 10,000+ chunks**: Consider increasing m=24, ef_construction=128
- **For high-precision requirements**: Increase ef_search via SET parameter

**Query-time tuning**:
```sql
-- Increase search quality for specific queries
SET hnsw.ef_search = 100;  -- Default: 40

SELECT * FROM knowledge_base
ORDER BY embedding <=> query_vec
LIMIT 10;
```

---

### 3.2 GIN Index for Full-Text Search

**Current Configuration**:
```sql
CREATE INDEX idx_knowledge_fts ON knowledge_base
USING GIN(ts_vector);
```

**PostgreSQL BM25 Ranking**:
- PostgreSQL uses `ts_rank` and `ts_rank_cd` for BM25-style ranking
- `ts_rank_cd`: Cover Density ranking (more sophisticated than ts_rank)
- Normalization flags control score computation

**Ranking Function Options**:
```sql
-- Standard ranking
ts_rank(ts_vector, query)

-- Cover Density ranking (recommended for BM25-like behavior)
ts_rank_cd(ts_vector, query, normalization)

-- Normalization flags:
-- 0: Default (no normalization)
-- 1: Divides rank by 1 + log(length of document)
-- 2: Divides rank by length of document
-- 4: Divides rank by mean harmonic distance between extents
-- 8: Divides rank by number of unique words in document
-- 16: Divides rank by 1 + log(number of unique words)
-- 32: Divides rank by itself + 1

-- Recommended for production:
ts_rank_cd(ts_vector, query, 32)  -- Self-normalization
```

**Performance Characteristics**:
- GIN index scan: O(log n) for term lookup
- Posting list merge: O(k) where k = matching documents
- For 2,600 chunks: Expected <50ms query time

---

### 3.3 Metadata Filtering Indexes

**GIN Index for JSONB**:
```sql
CREATE INDEX idx_knowledge_metadata ON knowledge_base
USING GIN(metadata);
```

**Supported Query Types**:
```sql
-- Containment (@>)
SELECT * FROM knowledge_base
WHERE metadata @> '{"tags": ["api"]}';

-- Existence (?)
SELECT * FROM knowledge_base
WHERE metadata ? 'author';

-- Any key exists (?|)
SELECT * FROM knowledge_base
WHERE metadata ?| array['author', 'version'];
```

**B-tree Indexes for Common Filters**:
```sql
-- Category filter (high selectivity)
CREATE INDEX idx_knowledge_category ON knowledge_base(source_category);

-- Date filter (range queries)
CREATE INDEX idx_knowledge_document_date ON knowledge_base(document_date DESC);

-- Compound index for common pattern
CREATE INDEX idx_knowledge_category_date ON knowledge_base(
    source_category,
    document_date DESC
);
```

**Filter Execution Strategy**:
1. **High selectivity filters first**: `source_category`, `document_date`
2. **JSONB filters second**: `metadata @> '{...}'`
3. **Vector/BM25 search last**: Most expensive operations

**Example Query Plan**:
```sql
EXPLAIN ANALYZE
SELECT *
FROM knowledge_base
WHERE
    source_category = 'product_docs'  -- B-tree index scan
    AND document_date >= '2024-01-01'  -- Index filter
    AND metadata @> '{"tags": ["api"]}'  -- GIN index scan
    AND embedding <=> query_vec < 0.5  -- HNSW scan
ORDER BY embedding <=> query_vec
LIMIT 10;

-- Expected plan:
-- 1. Bitmap Index Scan on idx_knowledge_category
-- 2. Recheck Cond: document_date >= '2024-01-01'
-- 3. Bitmap Index Scan on idx_knowledge_metadata
-- 4. Index Scan using idx_knowledge_embedding
-- 5. Limit
```

---

### 3.4 Query Cost Estimation

**Cost Model**:
```python
class QueryCostEstimator:
    """Estimate query execution cost for optimization."""

    def estimate_cost(self, query: SearchQuery) -> float:
        """Estimate total query cost in milliseconds.

        Cost factors:
        - Filter selectivity
        - Index type (HNSW vs GIN vs B-tree)
        - Result set size
        - Merge overhead (hybrid mode)
        """
        cost = 0.0

        # Base cost by search mode
        if query.search_mode == "vector":
            cost += 50.0  # HNSW index scan
        elif query.search_mode == "bm25":
            cost += 30.0  # GIN index scan
        else:  # hybrid
            cost += 80.0  # Both + merge

        # Filter costs
        for filter_expr in query.filters:
            if filter_expr.field.startswith("metadata."):
                cost += 10.0  # GIN JSONB scan
            else:
                cost += 5.0   # B-tree scan

        # Date filter cost
        if query.date_from or query.date_to:
            cost += 5.0

        return cost
```

**Optimization Decisions**:
- If estimated cost > 200ms: Suggest query simplification
- If filters reduce selectivity < 10%: Use filter-first strategy
- If hybrid mode has high cost: Recommend single-mode search

---

## 4. Type-Safe API Specification

### 4.1 Public API

**Module**: `src/search/api.py`

```python
"""Public API for search operations.

Provides high-level interface for executing searches with automatic
strategy selection, result ranking, and error handling.
"""

from src.search.models import SearchQuery, SearchResponse
from src.search.executor import SearchExecutor
from src.embedding.model_loader import load_embedding_model


class SearchAPI:
    """High-level search API."""

    def __init__(self):
        """Initialize search API with embedding model."""
        self.embedding_model = load_embedding_model()
        self.executor = SearchExecutor(self.embedding_model)

    def search(
        self,
        query_text: str,
        *,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        limit: int = 10,
        category: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        tags: list[str] | None = None,
        min_similarity: float = 0.0,
    ) -> SearchResponse:
        """Execute search with simplified parameters.

        Args:
            query_text: Search query string.
            mode: Search strategy (vector, bm25, hybrid).
            limit: Maximum results to return.
            category: Filter by source_category.
            date_from: Filter documents from this date.
            date_to: Filter documents until this date.
            tags: Filter by metadata tags.
            min_similarity: Minimum similarity threshold (0.0-1.0).

        Returns:
            SearchResponse with ranked results.

        Example:
            >>> api = SearchAPI()
            >>> response = api.search(
            ...     "Lutron integration",
            ...     mode="hybrid",
            ...     category="product_docs",
            ...     tags=["api"],
            ...     limit=10,
            ... )
            >>> for result in response.results:
            ...     print(f"{result.score:.2f}: {result.context_header}")
        """
        # Build filters
        filters = []
        if category:
            filters.append(
                FilterExpression(field="source_category", operator="eq", value=category)
            )
        if tags:
            filters.append(
                FilterExpression(field="metadata.tags", operator="contains", value=tags)
            )

        # Construct query
        query = SearchQuery(
            query_text=query_text,
            search_mode=mode,
            limit=limit,
            filters=filters,
            date_from=date_from,
            date_to=date_to,
            min_similarity=min_similarity,
        )

        # Execute
        return self.executor.execute(query)

    def advanced_search(self, query: SearchQuery) -> SearchResponse:
        """Execute search with full query control.

        Args:
            query: Complete SearchQuery with all parameters.

        Returns:
            SearchResponse with ranked results.
        """
        return self.executor.execute(query)
```

---

### 4.2 Type Safety Guarantees

**mypy --strict Compliance**:
- All function signatures fully typed
- No `Any` types except in JSONB metadata fields
- Generic types for collections: `list[SearchResult]` not `list`
- Pydantic validation ensures runtime type safety

**Example Type Checking**:
```python
# Type-safe usage
api = SearchAPI()
response: SearchResponse = api.search("query")  # ✓ Type checks

# Compile-time error detection
api.search(123)  # ✗ mypy error: Expected str, got int
api.search("query", limit="10")  # ✗ mypy error: Expected int, got str

# Pydantic runtime validation
SearchQuery(query_text="valid", limit=-5)  # ✗ ValidationError: limit >= 1
SearchQuery(
    query_text="valid",
    vector_weight=0.8,
    bm25_weight=0.3,
)  # ✗ ValidationError: weights must sum to 1.0
```

---

## 5. Performance Targets and Constraints

### 5.1 Performance Targets

| Operation | Target Latency | Constraint | Monitoring Metric |
|-----------|---------------|------------|-------------------|
| Vector search (2,600 chunks) | <100ms | 95th percentile | `search.vector.latency_p95` |
| BM25 search (2,600 chunks) | <50ms | 95th percentile | `search.bm25.latency_p95` |
| Hybrid search (2,600 chunks) | <100ms | 95th percentile | `search.hybrid.latency_p95` |
| Metadata filtering | <20ms | Additional overhead | `search.filter.latency` |
| Result merging (hybrid) | <10ms | Python-side overhead | `search.merge.latency` |
| Total query time | <150ms | End-to-end | `search.total.latency` |

### 5.2 Scalability Constraints

**Database Size Scaling**:

| Chunk Count | Table Size | HNSW Index Size | Expected Vector Latency |
|-------------|-----------|-----------------|------------------------|
| 2,600 (current) | ~100 MB | ~50 MB | <100ms |
| 10,000 | ~400 MB | ~200 MB | <150ms |
| 50,000 | ~2 GB | ~1 GB | <300ms |
| 100,000 | ~4 GB | ~2 GB | <500ms |

**Tuning for Scale**:
- **10,000 chunks**: No changes required
- **50,000 chunks**: Increase `ef_search=80`, consider read replicas
- **100,000+ chunks**: Shard by category, use distributed search

### 5.3 Resource Constraints

**Memory**:
- Embedding model: ~500 MB (all-mpnet-base-v2)
- Connection pool: 20 connections × 10 MB = 200 MB
- Query cache: 100 MB (optional)
- **Total**: ~800 MB application memory

**CPU**:
- Query embedding: Single-threaded (100ms per query)
- Database queries: Offloaded to PostgreSQL
- Result merging: Negligible (<10ms)

**Disk I/O**:
- HNSW index: Memory-mapped (OS cache)
- Sequential scans: Avoided via indexes
- Expected IOPS: <100 for typical queries

---

## 6. Integration Checklist

### 6.1 Phase 0 Integration

**Database Pooling** (`src/core/database.py`):
- [x] Use `DatabasePool.get_connection()` for all queries
- [x] Automatic retry logic with exponential backoff
- [x] Connection health checks (SELECT 1)
- [x] Statement timeout enforcement

**Configuration** (`src/core/config.py`):
- [ ] Add `SearchConfig` to Settings:
  ```python
  class SearchConfig(BaseSettings):
      default_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
      default_limit: int = Field(default=10, ge=1, le=100)
      vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
      bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
      cache_enabled: bool = False
      cache_ttl_seconds: int = 300

      model_config = SettingsConfigDict(env_prefix="SEARCH_")
  ```

**Logging** (`src/core/logging.py`):
- [x] Use `StructuredLogger.get_logger(__name__)` in all modules
- [x] Log query execution with timing
- [x] Log filter application and result counts
- [x] Error logging with context (query text, mode)

### 6.2 Phase 2 Integration

**Embedding Model**:
- [ ] Reuse embedding model from Phase 2 (`src/embedding/model_loader.py`)
- [ ] Share model instance across requests (singleton pattern)
- [ ] Handle embedding generation errors gracefully

**Data Models**:
- [ ] Extend `ProcessedChunk` or create `SearchResult` with score field
- [ ] Ensure Pydantic v2 compatibility
- [ ] Add `model_config = ConfigDict(from_attributes=True)` for ORM mapping

**Database Schema**:
- [x] Verify HNSW index exists: `idx_knowledge_embedding`
- [x] Verify GIN FTS index exists: `idx_knowledge_fts`
- [x] Verify GIN metadata index exists: `idx_knowledge_metadata`
- [ ] Test ts_vector trigger updates correctly

### 6.3 New Components

**Search Module** (`src/search/`):
```
src/search/
├── __init__.py
├── models.py           # SearchQuery, SearchResult, FilterExpression
├── strategies.py       # VectorSearchStrategy, BM25SearchStrategy, HybridSearchStrategy
├── executor.py         # SearchExecutor
├── api.py             # SearchAPI (public interface)
└── utils.py           # Helper functions (filter building, score normalization)
```

**Testing**:
```
tests/
├── test_search_models.py       # Pydantic model validation
├── test_search_strategies.py   # Strategy unit tests
├── test_search_executor.py     # Integration tests
└── test_search_api.py          # End-to-end API tests
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Model Validation** (`test_search_models.py`):
```python
def test_search_query_weight_validation():
    """Test that vector_weight + bm25_weight must equal 1.0."""
    with pytest.raises(ValidationError) as exc_info:
        SearchQuery(
            query_text="test",
            vector_weight=0.8,
            bm25_weight=0.3,  # Sum = 1.1
        )

    assert "must equal 1.0" in str(exc_info.value)

def test_filter_expression_types():
    """Test filter expression validation."""
    # Valid filter
    filter = FilterExpression(
        field="source_category",
        operator="eq",
        value="product_docs",
    )

    # Invalid operator
    with pytest.raises(ValidationError):
        FilterExpression(
            field="source_category",
            operator="invalid",  # Not in Literal
            value="product_docs",
        )
```

**Strategy Tests** (`test_search_strategies.py`):
```python
@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(768)
    return model

def test_vector_search_builds_correct_sql(mock_embedding_model):
    """Test vector search SQL generation."""
    strategy = VectorSearchStrategy(mock_embedding_model)
    query = SearchQuery(
        query_text="test query",
        search_mode="vector",
        limit=10,
    )

    sql = strategy._build_vector_query(query)

    assert "embedding <=> %(query_embedding)s" in sql
    assert "ORDER BY embedding <=> %(query_embedding)s" in sql
    assert "LIMIT %(limit)s" in sql

def test_bm25_search_builds_correct_sql():
    """Test BM25 search SQL generation."""
    strategy = BM25SearchStrategy()
    query = SearchQuery(
        query_text="test query",
        search_mode="bm25",
        limit=10,
    )

    sql = strategy._build_bm25_query(query)

    assert "plainto_tsquery" in sql
    assert "ts_vector @@ query" in sql
    assert "ts_rank_cd" in sql

def test_hybrid_search_merges_results(mock_embedding_model):
    """Test hybrid search result merging."""
    strategy = HybridSearchStrategy(mock_embedding_model)

    # Mock results from both strategies
    vector_results = [
        SearchResult(chunk_id=1, score=0.9, ...),
        SearchResult(chunk_id=2, score=0.8, ...),
    ]
    bm25_results = [
        SearchResult(chunk_id=2, score=10.5, ...),
        SearchResult(chunk_id=3, score=8.2, ...),
    ]

    merged = strategy._merge_results(
        vector_results,
        bm25_results,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    # Verify chunk 2 appears in merged results (present in both)
    assert any(r.chunk_id == 2 for r in merged)
```

### 7.2 Integration Tests

**End-to-End Search** (`test_search_api.py`):
```python
@pytest.fixture
def search_api():
    """Initialize search API with test database."""
    return SearchAPI()

def test_vector_search_returns_results(search_api):
    """Test vector search returns ranked results."""
    response = search_api.search(
        "Lutron integration",
        mode="vector",
        limit=5,
    )

    assert len(response.results) <= 5
    assert response.query_time_ms < 100
    assert response.search_mode == "vector"

    # Verify results are sorted by score (descending)
    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True)

def test_bm25_search_returns_results(search_api):
    """Test BM25 search returns ranked results."""
    response = search_api.search(
        "integration api documentation",
        mode="bm25",
        limit=5,
    )

    assert len(response.results) <= 5
    assert response.query_time_ms < 50
    assert all(r.bm25_score is not None for r in response.results)

def test_hybrid_search_combines_strategies(search_api):
    """Test hybrid search combines vector + BM25."""
    response = search_api.search(
        "Lutron integration",
        mode="hybrid",
        limit=10,
    )

    assert len(response.results) <= 10
    assert response.query_time_ms < 100

    # Verify both scores are present
    for result in response.results:
        assert result.score > 0
        # Note: Not all results will have both scores
        # (depends on which strategy returned them)

def test_metadata_filtering_works(search_api):
    """Test metadata filters are applied."""
    response = search_api.search(
        "integration",
        category="product_docs",
        tags=["api"],
        limit=10,
    )

    # Verify all results match filters
    for result in response.results:
        assert result.source_category == "product_docs"
        assert "api" in result.metadata.get("tags", [])

def test_date_filtering_works(search_api):
    """Test date range filtering."""
    response = search_api.search(
        "integration",
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
        limit=10,
    )

    for result in response.results:
        if result.document_date:
            assert date(2024, 1, 1) <= result.document_date <= date(2024, 12, 31)
```

### 7.3 Performance Tests

**Latency Benchmarks**:
```python
def test_vector_search_latency(search_api, benchmark):
    """Benchmark vector search latency."""
    def run_search():
        return search_api.search("test query", mode="vector", limit=10)

    result = benchmark(run_search)

    # Verify latency target
    assert result.query_time_ms < 100

def test_hybrid_search_latency(search_api, benchmark):
    """Benchmark hybrid search latency."""
    def run_search():
        return search_api.search("test query", mode="hybrid", limit=10)

    result = benchmark(run_search)

    # Verify latency target
    assert result.query_time_ms < 100
```

---

## 8. Extensibility and Future Work

### 8.1 Pluggable Ranking Algorithms

**Interface** (`src/search/rankers.py`):
```python
class Ranker(ABC):
    """Base class for result ranking algorithms."""

    @abstractmethod
    def rank(self, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results and return sorted list."""
        pass

class RRFRanker(Ranker):
    """Reciprocal Rank Fusion ranker."""
    def rank(self, results: list[SearchResult]) -> list[SearchResult]:
        # Implementation...
        pass

class WeightedSumRanker(Ranker):
    """Weighted sum of normalized scores."""
    def rank(self, results: list[SearchResult]) -> list[SearchResult]:
        # Implementation...
        pass

class LearnedRanker(Ranker):
    """Machine learning-based ranker (future)."""
    def rank(self, results: list[SearchResult]) -> list[SearchResult]:
        # Use trained model to rerank results
        pass
```

### 8.2 Filter Composition

**Composable Filters**:
```python
class FilterBuilder:
    """Builder for complex filter expressions."""

    def __init__(self):
        self.filters: list[FilterExpression] = []

    def add_category(self, category: str) -> "FilterBuilder":
        """Add category filter."""
        self.filters.append(
            FilterExpression(field="source_category", operator="eq", value=category)
        )
        return self

    def add_tags(self, tags: list[str]) -> "FilterBuilder":
        """Add tag containment filter."""
        self.filters.append(
            FilterExpression(field="metadata.tags", operator="contains", value=tags)
        )
        return self

    def add_date_range(self, from_date: date, to_date: date) -> "FilterBuilder":
        """Add date range filter."""
        self.filters.append(
            FilterExpression(field="document_date", operator="between", value=[from_date, to_date])
        )
        return self

    def build(self) -> list[FilterExpression]:
        """Build final filter list."""
        return self.filters

# Usage:
filters = (
    FilterBuilder()
    .add_category("product_docs")
    .add_tags(["api", "integration"])
    .add_date_range(date(2024, 1, 1), date(2024, 12, 31))
    .build()
)
```

### 8.3 Future Enhancements

**Query Cache** (Task 5+):
- Cache search results by query hash (MD5)
- TTL-based expiration (300 seconds default)
- Invalidation on data updates

**Semantic Reranking** (Task 6+):
- Cross-encoder model for reranking top-k results
- Improves precision at the cost of latency
- Apply to top 20 results before final ranking

**Hybrid Ranking Strategies**:
- **RRF (Reciprocal Rank Fusion)**: Current implementation
- **Linear Combination**: `score = w1*vec_score + w2*bm25_score`
- **Learned Ranking**: Train model on click-through data

**Multi-Vector Search** (Task 7+):
- Multiple embedding models for different content types
- Query routing based on content analysis
- Ensemble of vector searches

**Federated Search** (Task 8+):
- Search across multiple data sources
- Unified result ranking across sources
- Cross-source deduplication

---

## 9. Architectural Risks and Mitigations

### 9.1 High-Impact Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **HNSW index performance degrades at scale** | High | Medium | Monitor query latency, tune ef_search parameter, implement query caching |
| **Hybrid search merge overhead exceeds 10ms** | Medium | Low | Optimize Python merging logic, use NumPy for vectorized operations |
| **Filter selectivity causes full table scans** | High | Low | Monitor query plans with EXPLAIN ANALYZE, add indexes for new filter fields |
| **Embedding model memory consumption** | Medium | Low | Use model caching, consider quantized models for production |
| **Query timeout under high concurrency** | High | Medium | Implement query queueing, increase connection pool size, add read replicas |

### 9.2 Mitigation Strategies

**Performance Monitoring**:
- Log query execution time for all searches
- Alert on p95 latency > 150ms
- Track index usage via PostgreSQL statistics

**Query Optimization**:
- Use EXPLAIN ANALYZE for slow queries
- Implement query cost estimation
- Add query hints for complex filters

**Scalability Planning**:
- Benchmark at 10,000, 50,000, 100,000 chunks
- Plan for horizontal scaling (read replicas)
- Consider sharding by category for >100,000 chunks

---

## 10. Architectural Decision Records

### ADR-001: Use Reciprocal Rank Fusion for Hybrid Search

**Status**: Accepted
**Date**: 2025-11-08

**Context**: Need to combine vector and BM25 scores for hybrid search. Two options:
1. Weighted sum of normalized scores
2. Reciprocal Rank Fusion (RRF)

**Decision**: Use RRF for hybrid search merging.

**Rationale**:
- **Score normalization agnostic**: RRF uses ranks, not raw scores (cosine distance vs BM25 score have different scales)
- **Proven effectiveness**: RRF widely used in production search systems
- **Simple implementation**: No complex score normalization required
- **Extensible**: Easy to add more strategies (e.g., weighted RRF)

**Consequences**:
- Need to fetch 2*limit results from each strategy for better merging
- Slight performance overhead (10ms) for rank computation
- Score interpretation changes: RRF score ≠ cosine similarity

**Alternatives Considered**:
- Weighted sum: Requires complex score normalization (min-max scaling, z-score)
- CombSUM/CombMNZ: Less robust to score distribution changes

---

### ADR-002: Use PostgreSQL ts_rank_cd for BM25 Ranking

**Status**: Accepted
**Date**: 2025-11-08

**Context**: Need BM25-style ranking for full-text search. Options:
1. PostgreSQL ts_rank / ts_rank_cd
2. External BM25 library (e.g., rank_bm25)
3. Custom BM25 implementation

**Decision**: Use PostgreSQL `ts_rank_cd` with normalization flag 32.

**Rationale**:
- **Database-native**: Leverages GIN index, no data transfer overhead
- **Cover Density ranking**: More sophisticated than basic ts_rank
- **Self-normalization**: Flag 32 provides score normalization
- **Performance**: Executes in database, <50ms for 2,600 chunks

**Consequences**:
- Scores are PostgreSQL-specific (not true BM25 k1/b parameters)
- Cannot tune BM25 parameters (k1, b) directly
- Good enough for hybrid search (not primary ranking signal)

**Alternatives Considered**:
- rank_bm25 library: Requires data extraction, Python-side ranking (slower)
- Custom BM25: Complex implementation, maintenance burden

---

### ADR-003: Strategy Pattern for Search Implementations

**Status**: Accepted
**Date**: 2025-11-08

**Context**: Need to support multiple search modes (vector, BM25, hybrid) with extensibility.

**Decision**: Implement Strategy Pattern with `SearchStrategy` base class.

**Rationale**:
- **Separation of concerns**: Each strategy encapsulates its logic
- **Extensibility**: Easy to add new strategies (e.g., semantic reranking)
- **Testability**: Mock strategies for unit testing
- **Type safety**: Clear interface with type annotations

**Consequences**:
- Slight overhead from abstraction layer
- Need to maintain multiple strategy implementations
- Clear separation simplifies maintenance

**Alternatives Considered**:
- Single monolithic search function: Hard to extend, complex conditionals
- Factory pattern: Less clear separation of search logic

---

## 11. Summary and Recommendations

### 11.1 Architecture Summary

**Strengths**:
1. **Type-safe design**: Pydantic v2 models with mypy --strict compliance
2. **Separation of concerns**: Clear boundaries between strategies, executor, API
3. **Leverages existing infrastructure**: DatabasePool, logging, config from Phase 0
4. **Performance-focused**: HNSW + GIN indexes, sub-100ms target latency
5. **Extensible**: Pluggable strategies, rankers, filters

**Weaknesses**:
1. **Python-side merging**: 10ms overhead for hybrid search (acceptable)
2. **No query cache**: Repeat queries incur full cost (future enhancement)
3. **Single-threaded embedding**: 100ms query embedding latency (model constraint)

**Trade-offs**:
1. **RRF vs Weighted Sum**: RRF is simpler, no score normalization needed
2. **PostgreSQL BM25 vs Custom**: Database-native is faster, less tunable
3. **Strategy Pattern vs Monolithic**: More code, better separation

---

### 11.2 Implementation Recommendations

**Task 4.1: Search Models**
- Implement `SearchQuery`, `SearchResult`, `FilterExpression` in `src/search/models.py`
- Add Pydantic validators for weight normalization, filter validation
- Create comprehensive unit tests for model validation

**Task 4.2: Vector Search Strategy**
- Implement `VectorSearchStrategy` with HNSW query building
- Test SQL generation with various filter combinations
- Verify EXPLAIN plans use idx_knowledge_embedding

**Task 4.3: BM25 Search Strategy**
- Implement `BM25SearchStrategy` with ts_rank_cd ranking
- Test GIN index usage with EXPLAIN ANALYZE
- Compare PostgreSQL BM25 vs external libraries (benchmark)

**Task 4.4: Hybrid Search Strategy**
- Implement `HybridSearchStrategy` with RRF merging
- Optimize Python-side merging (use NumPy if needed)
- Benchmark hybrid search latency (<100ms target)

**Task 4.5: Search API and Executor**
- Implement `SearchExecutor` with strategy orchestration
- Create `SearchAPI` public interface with simplified parameters
- Add comprehensive integration tests with real database

**Task 4.6: Performance Testing**
- Benchmark all search modes at 2,600 chunks
- Verify latency targets (vector <100ms, BM25 <50ms, hybrid <100ms)
- Test metadata filtering overhead (<20ms)
- Profile query plans with EXPLAIN ANALYZE

**Task 4.7: Documentation**
- Write API documentation with usage examples
- Document filter syntax and supported operators
- Create performance tuning guide
- Add troubleshooting section for slow queries

---

### 11.3 Success Criteria

**Functional Requirements**:
- [x] Vector search returns ranked results by cosine similarity
- [x] BM25 search returns ranked results by text relevance
- [x] Hybrid search combines both strategies with weighted RRF
- [x] Metadata filters work with all search modes
- [x] Date range filters work correctly
- [x] Type-safe API with Pydantic validation

**Performance Requirements**:
- [ ] Vector search: <100ms p95 latency (2,600 chunks)
- [ ] BM25 search: <50ms p95 latency (2,600 chunks)
- [ ] Hybrid search: <100ms p95 latency (2,600 chunks)
- [ ] Metadata filtering: <20ms additional overhead
- [ ] Result merging: <10ms overhead

**Quality Requirements**:
- [ ] mypy --strict passes with no errors
- [ ] 100% type annotation coverage
- [ ] Unit tests for all strategies and models
- [ ] Integration tests with real database
- [ ] Performance benchmarks documented

---

## 12. Appendix

### 12.1 SQL Query Examples

**Vector Search with Filters**:
```sql
SELECT
    id,
    chunk_text,
    context_header,
    source_file,
    source_category,
    document_date,
    chunk_index,
    total_chunks,
    metadata,
    (1 - (embedding <=> %(query_embedding)s::vector)) AS similarity
FROM knowledge_base
WHERE
    embedding IS NOT NULL
    AND source_category = %(category)s
    AND document_date >= %(date_from)s
    AND metadata @> %(metadata_filter)s
    AND (1 - (embedding <=> %(query_embedding)s::vector)) >= %(min_similarity)s
ORDER BY embedding <=> %(query_embedding)s::vector
LIMIT %(limit)s OFFSET %(offset)s;
```

**BM25 Search with Filters**:
```sql
SELECT
    kb.id,
    kb.chunk_text,
    kb.context_header,
    kb.source_file,
    kb.source_category,
    kb.document_date,
    kb.chunk_index,
    kb.total_chunks,
    kb.metadata,
    ts_rank_cd(kb.ts_vector, query, 32) AS bm25_score
FROM knowledge_base kb,
     plainto_tsquery('english', %(query_text)s) AS query
WHERE
    kb.ts_vector @@ query
    AND kb.source_category = %(category)s
    AND kb.document_date >= %(date_from)s
    AND kb.metadata @> %(metadata_filter)s
ORDER BY bm25_score DESC
LIMIT %(limit)s OFFSET %(offset)s;
```

**Total Count Query**:
```sql
SELECT COUNT(*)
FROM knowledge_base
WHERE
    source_category = %(category)s
    AND document_date >= %(date_from)s
    AND metadata @> %(metadata_filter)s;
```

---

### 12.2 Performance Tuning Parameters

**PostgreSQL Configuration** (`postgresql.conf`):
```ini
# Memory settings
shared_buffers = 256MB           # 25% of RAM for small deployments
effective_cache_size = 1GB       # OS cache estimate
work_mem = 16MB                  # Per-query sort/hash memory

# Query planner
random_page_cost = 1.1           # SSD-optimized (default 4.0 for HDD)
effective_io_concurrency = 200   # Number of concurrent disk I/O operations

# Vector extension
hnsw.ef_search = 40              # Default search quality (increase for precision)
```

**Runtime Parameters**:
```sql
-- Increase search quality for specific session
SET hnsw.ef_search = 100;

-- Analyze query plan
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT ...;
```

---

### 12.3 Monitoring Queries

**Index Usage Statistics**:
```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'knowledge_base'
ORDER BY idx_scan DESC;
```

**Table Statistics**:
```sql
SELECT
    schemaname,
    tablename,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE tablename = 'knowledge_base';
```

**Query Performance**:
```sql
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
WHERE query LIKE '%knowledge_base%'
ORDER BY mean_time DESC
LIMIT 10;
```

---

**End of Architecture Review**

**Next Steps**: Proceed with Task 4.1 (Search Models) implementation following the type-safe design specifications outlined in Section 4.1.
