# BMCIS Knowledge MCP Local - Implementation Roadmap & Test Strategy

**Project:** bmcis-knowledge-mcp-local
**Created:** November 7, 2025, 18:14
**Purpose:** Comprehensive implementation roadmap with atomic phases and testing strategy
**Target:** 90%+ search accuracy, <300ms latency, full knowledge graph capabilities

---

## Executive Summary

This roadmap structures the development of a local PostgreSQL + pgvector system that replicates and improves upon the production Neon database for semantic search with knowledge graph capabilities. The implementation is divided into 6 atomic phases, each delivering measurable value and building upon the previous phase's foundation.

**Key Principles:**
- Each phase is independently buildable and testable
- Exit criteria are observable and measurable
- Dependencies flow strictly left-to-right (no circular dependencies)
- Test coverage grows incrementally (unit → integration → e2e)
- Validation against Neon production system is continuous

**Technology Stack:**
- PostgreSQL 16 + pgvector extension
- Python 3.11+, FastMCP framework
- sentence-transformers (all-mpnet-base-v2)
- psycopg2, tiktoken, pydantic

**Data Source:**
- 343 markdown files from bmcis-knowledge-mcp
- Expected output: ~2,600 chunks with 768-dimensional embeddings

---

## Development Phases Overview

```
Phase 0: Foundation
    ↓ (Database ready, config validated, utilities tested)
Phase 1: Data Layer
    ↓ (Ingestion working, chunks generated, embeddings created)
Phase 2: Search Layer
    ↓ (Vector + BM25 + Hybrid search operational)
Phase 3: Graph Layer
    ↓ (Entity extraction, relationship storage working)
Phase 4: Validation Layer
    ↓ (Cross-check with Neon, accuracy benchmarks established)
Phase 5: Reranking & Optimization
    ↓ (Cross-encoder, boosting, performance tuned)
Phase 6: Integration
    ↓ (FastMCP server, APIs, full e2e testing complete)
```

---

## Phase 0: Foundation Setup

**Goal:** Establish database infrastructure, configuration management, and core utilities

**Entry Criteria:**
- PostgreSQL 16 installed locally
- Python 3.11+ environment available
- Access to production Neon database credentials (read-only)
- Source code repository cloned

**Tasks:**

1. **Database Initialization** [CRITICAL]
   - Install PostgreSQL 16 with pgvector extension
   - Create `bmcis_knowledge_local` database
   - Verify pgvector version >= 0.5.0 (HNSW support required)
   - Create database user with appropriate permissions
   - **Acceptance:** `SELECT * FROM pg_extension WHERE extname = 'vector'` returns row

2. **Schema Creation** [CRITICAL]
   - Execute `sql/schema_768.sql` to create `knowledge_base` table
   - Create all indexes (HNSW, GIN, B-tree)
   - Create triggers (tsvector update, updated_at auto-update)
   - Validate table structure matches production schema
   - **Acceptance:** `\d knowledge_base` shows all columns, indexes, triggers

3. **Configuration Management** [HIGH]
   - Create `config/settings.py` with Pydantic BaseSettings
   - Support environment variables (.env file)
   - Configure database connection strings (local + Neon)
   - Set embedding model parameters (768-dim, all-mpnet-base-v2)
   - Define chunking strategy (512 tokens, 20% overlap)
   - **Acceptance:** Can load config from .env, validate all required fields

4. **Core Utilities** [MEDIUM]
   - Implement database connection pooling (psycopg2)
   - Create logging utilities (structured logging with timestamps)
   - Build error handling base classes
   - Implement retry logic for database operations
   - **Acceptance:** Can connect to both local and Neon databases, logs are structured

5. **Development Environment** [MEDIUM]
   - Set up virtual environment with requirements.txt
   - Configure pytest with coverage reporting
   - Set up pre-commit hooks (black, ruff, mypy)
   - Create Makefile for common operations
   - **Acceptance:** `make test` runs successfully, coverage report generated

**Exit Criteria:**
- PostgreSQL local database running with pgvector extension
- All tables, indexes, triggers created and validated
- Configuration loaded successfully from environment
- Database connections established (local + Neon)
- Test suite infrastructure working (0 tests passing is OK)

**Delivers:**
- Working local PostgreSQL database
- Configuration system for all subsequent phases
- Database utilities for data access
- Test infrastructure ready for Phase 1

**Estimated Duration:** 2-3 days

---

## Phase 1: Data Layer (Ingestion, Chunking, Embeddings)

**Goal:** Implement complete data ingestion pipeline from markdown files to database chunks with embeddings

**Entry Criteria:**
- Phase 0 complete (database ready, config working)
- Access to 343 markdown files from bmcis-knowledge-mcp
- sentence-transformers library installed

**Tasks:**

1. **File Discovery & Categorization** [HIGH]
   - Implement recursive directory scanner for .md files
   - Extract category from directory path structure
   - Extract metadata from filename patterns
   - Build file manifest with metadata
   - **Acceptance:** Can scan directory, return list of 343 files with categories

2. **Document Loading** [HIGH]
   - Implement markdown file reader
   - Parse frontmatter (if exists) for metadata
   - Extract section headers (##, ###) for context
   - Build metadata dictionary per document
   - **Acceptance:** Can load any markdown file, extract headers and metadata

3. **Token-Based Chunking** [CRITICAL]
   - Implement tiktoken-based chunker (cl100k_base tokenizer)
   - Implement OPTIMAL strategy: 512 tokens, 20% overlap (102 tokens)
   - Calculate chunk indices and total counts
   - Preserve document boundaries (no cross-document chunks)
   - **Acceptance:** 1 sample file → expected number of chunks with ~512 tokens each

4. **Contextual Header Generation** [MEDIUM]
   - Prepend semantic context to each chunk
   - Format: `[CONTEXT: ...]\n[VENDOR: ...]\n[CATEGORY: ...]\nOriginal content`
   - Include section name, vendor, category, date
   - **Acceptance:** Chunk has header prepended, increases searchability

5. **Embedding Generation** [CRITICAL]
   - Load sentence-transformers model (all-mpnet-base-v2)
   - Implement sync embedding function (for ingestion)
   - Implement async embedding function (for MCP server)
   - Validate output dimensions (768)
   - Implement model caching (singleton pattern)
   - **Acceptance:** "test query" → 768-dimensional vector in <50ms (cached)

6. **Database Insertion** [CRITICAL]
   - Implement batch insertion for chunks
   - Store content, embedding, source_path, metadata, timestamps
   - Validate JSONB metadata structure
   - Handle duplicate detection (re-ingestion safety)
   - Implement progress tracking and logging
   - **Acceptance:** 10 sample files → chunks in database with embeddings

7. **Full Ingestion Pipeline** [CRITICAL]
   - Orchestrate: discovery → load → chunk → embed → insert
   - Implement transaction management (rollback on failure)
   - Generate ingestion statistics (files, chunks, tokens, duration)
   - Create ingestion report
   - **Acceptance:** All 343 files → 2,600 chunks in database, stats report generated

**Exit Criteria:**
- All 343 markdown files processed successfully
- ~2,600 chunks inserted into `knowledge_base` table
- All chunks have valid 768-dimensional embeddings
- All indexes built and operational
- Ingestion statistics report generated
- Can re-run ingestion safely (idempotent)

**Delivers:**
- Complete ingestion pipeline (file → chunks → embeddings → database)
- 2,600+ searchable chunks with embeddings
- Metadata extraction and storage
- Foundation for search functionality

**Estimated Duration:** 4-5 days

---

## Phase 2: Search Layer (Vector, BM25, Hybrid + RRF)

**Goal:** Implement core search functionality with vector, keyword, and hybrid approaches

**Entry Criteria:**
- Phase 1 complete (database populated with embeddings)
- HNSW index operational
- GIN full-text index operational

**Tasks:**

1. **Vector Search Implementation** [CRITICAL]
   - Implement cosine similarity search using HNSW index
   - Query: `embedding <=> query_embedding`
   - Convert distance to similarity score: `1 - distance`
   - Return top N results with similarity scores
   - **Acceptance:** Query "Lutron pricing" → top 20 results in <30ms

2. **BM25 Keyword Search** [CRITICAL]
   - Implement full-text search using GIN index
   - Use `ts_rank()` for BM25-style scoring
   - Query against `content_tsv` column
   - Return top N results with BM25 scores
   - **Acceptance:** Query "Lutron pricing" → top 20 results in <40ms

3. **Reciprocal Rank Fusion (RRF)** [HIGH]
   - Implement RRF algorithm: `score = 1/(k+rank_vec) + 1/(k+rank_bm25)`
   - Use k=60 (RERANKING_FUSION_K)
   - Deduplicate results from both paths
   - Re-rank by combined RRF score
   - **Acceptance:** Hybrid search returns merged + re-ranked results

4. **Metadata Filtering** [MEDIUM]
   - Implement JSONB filtering for vendor, category, team_member
   - Support case-insensitive matching
   - Use array containment operators
   - Combine with vector/BM25 search
   - **Acceptance:** Filter by vendor="Lutron" reduces results appropriately

5. **Search Result Formatting** [MEDIUM]
   - Implement SearchResult Pydantic model
   - Include: id, content, similarity, source_path, metadata, score_type
   - Implement SearchResponse wrapper (query, results, total)
   - **Acceptance:** Search returns properly formatted JSON-serializable results

6. **Search Configuration** [LOW]
   - Implement configurable limits (RERANKING_VECTOR_LIMIT=20, etc.)
   - Toggle hybrid search on/off
   - Configure RRF k parameter
   - **Acceptance:** Can change config, behavior changes accordingly

**Exit Criteria:**
- Vector search operational (<30ms for top 20)
- BM25 search operational (<40ms for top 20)
- Hybrid RRF search operational (<100ms total)
- Metadata filtering works correctly
- Search results properly formatted
- All search modes tested and benchmarked

**Delivers:**
- 3 search modes: vector-only, BM25-only, hybrid RRF
- Metadata filtering capability
- Latency benchmarks established
- Foundation for reranking layer

**Estimated Duration:** 3-4 days

---

## Phase 3: Graph Layer (Entity Extraction, Relationships)

**Goal:** Add knowledge graph capabilities with entity extraction and relationship storage

**Entry Criteria:**
- Phase 2 complete (search working)
- Access to NER model or API (spaCy, Hugging Face, or LLM-based)
- Graph schema designed

**Tasks:**

1. **Entity Extraction Implementation** [HIGH]
   - Choose entity extraction approach (spaCy en_core_web_sm or similar)
   - Extract entities: PERSON, ORG (vendors), PRODUCT, GPE (locations)
   - Store entities with types, confidence scores
   - Link entities to source chunks
   - **Acceptance:** Extract entities from 10 sample chunks with 80%+ precision

2. **Graph Schema Creation** [CRITICAL]
   - Create `entities` table (id, name, type, metadata)
   - Create `relationships` table (id, entity1_id, entity2_id, relation_type, chunk_id)
   - Create `entity_chunks` junction table
   - Add indexes for entity lookup
   - **Acceptance:** Tables created, can insert sample entities and relationships

3. **Relationship Extraction** [MEDIUM]
   - Implement co-occurrence detection (entities in same chunk)
   - Extract explicit relationships from text patterns
   - Store relationship metadata (confidence, context)
   - **Acceptance:** Detect "John Smith manages Lutron" → relationship record

4. **Entity Resolution** [LOW]
   - Deduplicate entities ("Lutron" vs "Lutron Electronics")
   - Normalize entity names
   - Link variants to canonical form
   - **Acceptance:** "Lutron" and "Lutron Electronics" → same entity ID

5. **Graph Query Interface** [MEDIUM]
   - Implement entity lookup by name/type
   - Implement relationship traversal (1-hop, 2-hop)
   - Return entity subgraphs
   - **Acceptance:** Query "Lutron" → related vendors, team members, products

6. **Graph Integration with Search** [HIGH]
   - Boost search results containing query entities
   - Expand search with related entities
   - Include entity metadata in search responses
   - **Acceptance:** Search "Lutron" surfaces related vendor content

**Exit Criteria:**
- Entities extracted from all 2,600 chunks
- Relationships stored in graph tables
- Entity resolution working (90%+ accuracy on manual spot-check)
- Graph queries operational
- Search integrated with entity boosting

**Delivers:**
- Knowledge graph with entities and relationships
- Entity-aware search
- Foundation for advanced query expansion
- Network analysis capabilities

**Estimated Duration:** 5-6 days

---

## Phase 4: Validation Layer (Cross-Check with Neon)

**Goal:** Validate local system against production Neon database for accuracy and consistency

**Entry Criteria:**
- Phase 2 complete (search operational)
- Read-only access to production Neon database
- Test query dataset available

**Tasks:**

1. **Query Dataset Creation** [HIGH]
   - Create 50 test queries covering:
     - Vendor queries (15): "Lutron pricing", "Crestron integration"
     - Team queries (10): "KAM responsibilities", "VP Sales territories"
     - Product queries (15): "Lighting control systems"
     - Strategic queries (10): "Market trends", "competitive analysis"
   - Include expected results (manual annotation)
   - **Acceptance:** 50 queries with expected top-3 results annotated

2. **Dual-Query Harness** [CRITICAL]
   - Implement query executor for local database
   - Implement query executor for Neon database
   - Run same query against both systems
   - Collect results, timings, scores
   - **Acceptance:** Can run same query on local + Neon, compare results

3. **Accuracy Metrics** [CRITICAL]
   - Implement Relevance@K (K=1,3,5)
   - Implement Mean Reciprocal Rank (MRR)
   - Implement Normalized Discounted Cumulative Gain (NDCG)
   - Compare local vs Neon results
   - **Acceptance:** Calculate metrics for 50 queries, generate report

4. **Latency Benchmarking** [HIGH]
   - Measure p50, p95, p99 latencies (local vs Neon)
   - Break down by search stage (embedding, vector, BM25, RRF)
   - Identify bottlenecks
   - **Acceptance:** Latency report with percentiles, local meets <300ms SLA

5. **Consistency Validation** [MEDIUM]
   - Compare chunk counts (local vs Neon)
   - Validate embedding dimensions match
   - Check metadata completeness
   - Verify index coverage
   - **Acceptance:** <1% variance in chunk counts, 100% embedding dimension match

6. **Regression Testing** [MEDIUM]
   - Create baseline performance snapshot
   - Detect regressions in future changes
   - Alert on accuracy drops >5%
   - **Acceptance:** Baseline established, can detect regressions

**Exit Criteria:**
- 50 test queries executed on both systems
- Accuracy metrics: Relevance@5 >= 85% (local vs Neon ground truth)
- Latency: p50 < 250ms, p95 < 300ms
- Consistency: chunk counts within 1%, metadata >95% complete
- Regression detection system operational

**Delivers:**
- Validation framework for continuous testing
- Accuracy benchmarks vs production
- Latency baselines
- Confidence in local system parity

**Estimated Duration:** 3-4 days

---

## Phase 5: Reranking & Optimization (Cross-Encoder, Boosting)

**Goal:** Implement advanced reranking and optimization to exceed production accuracy

**Entry Criteria:**
- Phase 2 complete (hybrid search working)
- Phase 4 complete (baseline metrics established)
- Cross-encoder model available (cross-encoder/ms-marco-MiniLM-L-6-v2)

**Tasks:**

1. **Cross-Encoder Integration** [CRITICAL]
   - Load cross-encoder model (ms-marco-MiniLM-L-6-v2)
   - Implement two-stage pipeline:
     - Stage 1: Hybrid RRF → top 20 candidates
     - Stage 2: Cross-encoder → top 5 final results
   - Cache model for fast subsequent queries
   - **Acceptance:** Query returns top 5 cross-encoder reranked results in <200ms

2. **Multi-Factor Boosting** [HIGH]
   - Implement metadata match boost (vendor, category) +15%
   - Implement document type boost (analyses, status) +10%
   - Implement recency boost (<30 days) +5%
   - Implement entity match boost +10%
   - Implement topic match boost +8%
   - Apply multiplicative boosts to RRF scores
   - **Acceptance:** Boosted results show improved relevance in spot-checks

3. **Query Expansion (Conditional)** [LOW]
   - Implement acronym expansion dictionary
   - Test with ENABLE_QUERY_EXPANSION flag
   - Benchmark accuracy with/without expansion
   - **Acceptance:** If accuracy improves >3%, enable; else disable

4. **Performance Optimization** [MEDIUM]
   - Implement query result caching (LRU cache, TTL=5min)
   - Optimize batch embedding generation
   - Tune PostgreSQL (work_mem, effective_cache_size)
   - Profile slow queries
   - **Acceptance:** p50 latency reduced by 10-15%, cache hit rate >30%

5. **A/B Testing Framework** [LOW]
   - Implement config toggles for reranking strategies
   - Log search results for offline analysis
   - Compare reranking variants (RRF only vs RRF+boost vs RRF+boost+CE)
   - **Acceptance:** Can run A/B test on 50 queries, compare metrics

6. **Accuracy Validation** [CRITICAL]
   - Re-run Phase 4 validation suite
   - Target: Relevance@5 >= 90% (up from 85% baseline)
   - Target: MRR >= 0.85
   - Compare to production Neon metrics
   - **Acceptance:** Local system meets or exceeds Neon accuracy

**Exit Criteria:**
- Cross-encoder reranking operational (<200ms)
- Multi-factor boosting applied correctly
- Performance optimizations reduce latency 10-15%
- Accuracy targets met: Relevance@5 >= 90%
- A/B testing framework ready for future experiments

**Delivers:**
- Production-grade search accuracy (90%+ relevance)
- Optimized latency (<250ms p50)
- Flexible reranking strategies
- Foundation for continuous improvement

**Estimated Duration:** 4-5 days

---

## Phase 6: Integration (FastMCP Server, APIs, E2E Testing)

**Goal:** Integrate all components into FastMCP server with full end-to-end testing

**Entry Criteria:**
- Phases 1-5 complete (all capabilities operational)
- FastMCP framework installed
- API design spec available

**Tasks:**

1. **FastMCP Server Setup** [CRITICAL]
   - Create FastMCP server application
   - Implement `semantic_search` tool
   - Implement async request handling
   - Configure CORS and authentication (Cloudflare Access)
   - **Acceptance:** Server starts, responds to health check

2. **API Endpoint Implementation** [CRITICAL]
   - Implement `/search` endpoint (semantic_search)
   - Implement `/entities` endpoint (entity lookup)
   - Implement `/graph` endpoint (relationship queries)
   - Implement `/health` endpoint (system status)
   - Validate request/response schemas (Pydantic)
   - **Acceptance:** All endpoints return valid JSON responses

3. **Error Handling & Logging** [HIGH]
   - Implement structured error responses
   - Add request/response logging
   - Track latency metrics
   - Implement rate limiting (optional)
   - **Acceptance:** Errors return proper HTTP codes + messages, logs are queryable

4. **End-to-End Testing** [CRITICAL]
   - Implement e2e test suite:
     - Test semantic_search with 10 representative queries
     - Test metadata filtering
     - Test entity/graph queries
     - Test error cases (empty query, invalid limit)
   - Run against live local server
   - **Acceptance:** All e2e tests pass, covering 80%+ user flows

5. **Performance Testing** [HIGH]
   - Load test with concurrent requests (10 users)
   - Measure throughput (queries/second)
   - Validate latency under load (p95 < 300ms)
   - Identify concurrency bottlenecks
   - **Acceptance:** System handles 10 concurrent users without degradation

6. **Deployment Documentation** [MEDIUM]
   - Document deployment to Railway.app
   - Create environment variable checklist
   - Document database migration steps
   - Create rollback procedure
   - **Acceptance:** Can deploy from scratch following docs

7. **Client Integration** [MEDIUM]
   - Test MCP server connection from Claude Desktop
   - Validate semantic_search tool appears
   - Run sample queries from Claude interface
   - Document usage examples
   - **Acceptance:** Claude Desktop can call semantic_search successfully

**Exit Criteria:**
- FastMCP server running with all endpoints operational
- E2E tests passing (80%+ coverage of user flows)
- Performance tests validate <300ms p95 latency under load
- Deployment to Railway.app successful
- Claude Desktop integration tested and documented
- Production-ready system

**Delivers:**
- Production-ready FastMCP server
- Full API suite (search, entities, graph)
- Comprehensive e2e test coverage
- Deployment-ready system
- Client integration validated

**Estimated Duration:** 4-5 days

---

## Test Strategy

### Test Pyramid Structure

```
                    ┌─────────────┐
                    │  E2E Tests  │  10%  (40 tests)
                    │  (Slow)     │
                    └─────────────┘
                ┌───────────────────┐
                │ Integration Tests │  30%  (120 tests)
                │   (Medium)        │
                └───────────────────┘
        ┌───────────────────────────────┐
        │      Unit Tests               │  60%  (240 tests)
        │       (Fast)                  │
        └───────────────────────────────┘

Total: ~400 tests
```

**Rationale:**
- **Unit tests (60%):** Fast feedback, test individual functions in isolation
- **Integration tests (30%):** Test module interactions (database, embeddings, search)
- **E2E tests (10%):** Test full user flows (slow but critical for confidence)

---

### Coverage Requirements

**Overall Target:** 85% line coverage minimum

**Per-Module Breakdown:**

| Module | Coverage Target | Rationale |
|--------|----------------|-----------|
| `config.py` | 95% | Critical system configuration |
| `database.py` | 90% | Core data access layer |
| `embeddings.py` | 90% | Critical for search quality |
| `chunking.py` | 90% | Critical for data quality |
| `search.py` | 95% | Core business logic |
| `graph.py` | 85% | Knowledge graph queries |
| `server.py` | 85% | API endpoints |
| `utils.py` | 80% | Helper functions |
| `models.py` | 100% | Data models (easy to test) |

**Measurement:**
- Use pytest-cov for coverage reports
- Run `make coverage` to generate HTML report
- Block PRs with <85% coverage

---

### Critical Test Scenarios

#### Phase 0: Foundation
1. **Database Connection**
   - Can connect to local PostgreSQL
   - Can connect to Neon (read-only)
   - Connection pooling works correctly
   - Retry logic handles transient failures

2. **Configuration Loading**
   - Load from .env file
   - Override with environment variables
   - Validate required fields present
   - Error on invalid config

3. **Schema Validation**
   - All tables created
   - All indexes exist
   - Triggers fire correctly
   - Constraints enforced

#### Phase 1: Data Layer
1. **File Discovery**
   - Scan directory recursively
   - Filter .md files correctly
   - Extract category from path
   - Handle missing files gracefully

2. **Chunking**
   - 512-token chunks created
   - 20% overlap applied correctly
   - Chunk boundaries respected
   - Edge cases: tiny files (<512 tokens), huge files (>10K tokens)

3. **Embedding Generation**
   - Correct 768 dimensions
   - Consistent embeddings (same input → same output)
   - Model caching works
   - Handles empty/invalid input

4. **Ingestion Pipeline**
   - All 343 files processed
   - No duplicate chunks
   - Metadata extracted correctly
   - Re-ingestion is idempotent

#### Phase 2: Search Layer
1. **Vector Search**
   - Returns results sorted by similarity
   - Handles empty results
   - Respects limit parameter
   - Latency <30ms

2. **BM25 Search**
   - Returns keyword-relevant results
   - Handles multi-word queries
   - Respects limit parameter
   - Latency <40ms

3. **Hybrid RRF**
   - Merges vector + BM25 results
   - Deduplicates correctly
   - RRF scoring accurate
   - Latency <100ms

4. **Metadata Filtering**
   - Filter by vendor (case-insensitive)
   - Filter by category
   - Combine with search correctly
   - No results when filter too restrictive

#### Phase 3: Graph Layer
1. **Entity Extraction**
   - Extract PERSON, ORG, PRODUCT entities
   - Confidence scores reasonable
   - Link to source chunks
   - Handle chunks with no entities

2. **Relationship Extraction**
   - Co-occurrence relationships created
   - Explicit relationships parsed
   - No duplicate relationships
   - Relationship metadata stored

3. **Entity Resolution**
   - Variants linked to canonical form
   - Case-insensitive matching
   - Partial matching (fuzzy)
   - Manual overrides supported

4. **Graph Queries**
   - Lookup entity by name
   - Traverse 1-hop relationships
   - Return subgraphs
   - Handle disconnected entities

#### Phase 4: Validation Layer
1. **Dual-Query Testing**
   - Run query on local + Neon
   - Results formatted consistently
   - Timing collected accurately
   - Handle network errors

2. **Accuracy Metrics**
   - Relevance@K calculated correctly
   - MRR calculated correctly
   - NDCG calculated correctly
   - Baseline vs new results compared

3. **Latency Benchmarking**
   - p50, p95, p99 calculated
   - Per-stage breakdown accurate
   - Outliers identified
   - Trends tracked over time

#### Phase 5: Reranking & Optimization
1. **Cross-Encoder**
   - Model loads correctly
   - Scores 20 candidates in <200ms
   - Scores are [0, 1] range
   - Ranking changes improve relevance

2. **Boosting**
   - Metadata boost applied correctly
   - Document type boost applied
   - Recency boost applied
   - Boosts are multiplicative

3. **Caching**
   - Cache hits reduce latency
   - Cache invalidation works
   - TTL respected
   - Cache size bounded (LRU eviction)

#### Phase 6: Integration
1. **FastMCP Server**
   - Server starts successfully
   - Handles concurrent requests
   - CORS configured correctly
   - Authentication works (if enabled)

2. **API Endpoints**
   - /search returns valid results
   - /entities returns entity data
   - /graph returns relationships
   - /health returns status

3. **Error Handling**
   - Invalid query → 400 Bad Request
   - Database down → 503 Service Unavailable
   - Rate limit → 429 Too Many Requests
   - Internal error → 500 with details

4. **End-to-End Flows**
   - Search → results → correct
   - Entity lookup → graph traversal → correct
   - Metadata filter → reduced results → correct
   - Error case → proper error → correct

---

### Test Generation Guidelines

#### Unit Tests: Focus on Pure Functions
- Test each function with 3-5 representative inputs
- Test edge cases: empty, null, very large
- Test error cases: invalid input, exceptions
- Use fixtures for common test data
- Mock external dependencies (database, models)

**Example: Test `chunk_document` function**
```python
def test_chunk_document_normal_size():
    # Test 2048-token document → 5 chunks
    ...

def test_chunk_document_tiny():
    # Test 100-token document → 1 chunk
    ...

def test_chunk_document_empty():
    # Test empty document → raises ValueError
    ...
```

#### Integration Tests: Focus on Module Interactions
- Test database read/write operations
- Test embedding generation + storage
- Test search pipeline (embedding → query → results)
- Use real database (test DB, not prod)
- Use real models (cached for speed)

**Example: Test search pipeline**
```python
def test_vector_search_integration(test_db):
    # Insert 10 sample chunks
    # Query "Lutron pricing"
    # Verify results contain expected chunks
    # Verify latency < 50ms
    ...
```

#### E2E Tests: Focus on User Workflows
- Test complete flows (API request → response)
- Test against running server (localhost)
- Validate response schemas
- Measure end-to-end latency
- Use realistic test data

**Example: Test semantic search e2e**
```python
def test_semantic_search_e2e(mcp_client):
    # Call semantic_search("Lutron pricing", limit=5)
    # Verify 5 results returned
    # Verify all results have required fields
    # Verify similarity scores descending
    # Verify latency < 300ms
    ...
```

---

### Quality Gates

#### Pre-Commit Checks (Local)
Run before every commit:
```bash
make lint          # black, ruff, mypy
make test          # pytest (unit + integration)
make coverage      # ensure ≥85% coverage
```

**Enforcement:** Use pre-commit hooks to block commits if any fail

#### Pre-Phase Validation (Manual)
Run before declaring phase complete:
```bash
make test-phase-N  # phase-specific test suite
make benchmark     # latency benchmarks
make validate      # against Neon (Phase 4+)
```

**Enforcement:** Phase exit criteria must be met and documented

#### Pre-Deployment Checks (CI/CD)
Run before deploying to Railway:
```bash
make test-all      # all tests (unit + integration + e2e)
make benchmark     # latency SLA validation
make security      # dependency scanning
make load-test     # concurrent user simulation
```

**Enforcement:** All checks must pass before deploy

---

### Performance Benchmarks (Quality Gates)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Latency (p50)** | <250ms | 50 test queries, median time |
| **Latency (p95)** | <300ms | 50 test queries, 95th percentile |
| **Latency (p99)** | <500ms | 50 test queries, 99th percentile |
| **Throughput** | ≥3 QPS | Concurrent user load test |
| **Accuracy (Relevance@5)** | ≥90% | 50 annotated queries |
| **Accuracy (MRR)** | ≥0.85 | 50 annotated queries |
| **Coverage** | ≥85% | pytest-cov line coverage |
| **Database Size** | <3GB | PostgreSQL disk usage |
| **Index Build Time** | <15min | Full ingestion pipeline |

**Monitoring:** Track metrics over time, alert on regressions

---

### Validation Accuracy Thresholds

**Critical Metrics (Block deployment if not met):**
- Relevance@5 >= 90% (at least 45/50 test queries have relevant result in top 5)
- MRR >= 0.85 (mean reciprocal rank across 50 queries)
- Latency p95 < 300ms (95% of queries complete in <300ms)

**Warning Metrics (Investigate but don't block):**
- Relevance@5 between 85-90%
- MRR between 0.80-0.85
- Latency p95 between 300-400ms

**Failure Metrics (Stop deployment, investigate):**
- Relevance@5 < 85%
- MRR < 0.80
- Latency p95 > 400ms

---

## Testing Tools & Infrastructure

### Required Libraries
```python
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.1

# Load testing
locust>=2.15.0

# Code quality
black>=23.7.0
ruff>=0.0.280
mypy>=1.4.1
pre-commit>=3.3.3
```

### Test Organization
```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_embeddings.py
│   ├── test_chunking.py
│   ├── test_search.py
│   ├── test_graph.py
│   └── test_utils.py
├── integration/
│   ├── test_ingestion_pipeline.py
│   ├── test_search_pipeline.py
│   ├── test_graph_pipeline.py
│   └── test_database_operations.py
├── e2e/
│   ├── test_semantic_search.py
│   ├── test_entity_queries.py
│   ├── test_graph_queries.py
│   └── test_error_handling.py
├── validation/
│   ├── test_accuracy_vs_neon.py
│   ├── test_latency_benchmarks.py
│   └── test_consistency.py
├── fixtures/
│   ├── sample_documents.py
│   ├── test_queries.py
│   └── expected_results.py
└── conftest.py  # Shared fixtures and config
```

---

## Success Criteria Summary

**Phase 0 Success:**
- ✓ Database operational with pgvector
- ✓ All indexes created
- ✓ Config system working
- ✓ Test infrastructure ready

**Phase 1 Success:**
- ✓ 343 files → 2,600 chunks
- ✓ All embeddings generated (768-dim)
- ✓ Ingestion pipeline tested
- ✓ Can re-run ingestion safely

**Phase 2 Success:**
- ✓ Vector search <30ms
- ✓ BM25 search <40ms
- ✓ Hybrid RRF <100ms
- ✓ All search modes tested

**Phase 3 Success:**
- ✓ Entities extracted from all chunks
- ✓ Relationships stored
- ✓ Graph queries working
- ✓ Search integrated with entities

**Phase 4 Success:**
- ✓ 50 test queries executed (local + Neon)
- ✓ Relevance@5 >= 85%
- ✓ Latency p95 < 300ms
- ✓ Consistency validated

**Phase 5 Success:**
- ✓ Cross-encoder operational (<200ms)
- ✓ Boosting applied correctly
- ✓ Relevance@5 >= 90%
- ✓ Performance optimized

**Phase 6 Success:**
- ✓ FastMCP server operational
- ✓ All APIs tested
- ✓ E2E tests passing
- ✓ Claude Desktop integration working
- ✓ Production-ready deployment

---

## Risk Mitigation

### High-Risk Areas

1. **Embedding Model Performance**
   - Risk: Model too slow, latency SLA missed
   - Mitigation: Use model caching, batch processing, GPU if available
   - Fallback: Use smaller model (384-dim) if latency critical

2. **Database Index Build Time**
   - Risk: HNSW index build takes >1 hour for 2,600 vectors
   - Mitigation: Build indexes in parallel, tune ef_construction
   - Fallback: Use IVFFlat if HNSW too slow

3. **Accuracy vs Neon**
   - Risk: Local system doesn't match Neon accuracy
   - Mitigation: Phase 4 validation catches early, iterate on reranking
   - Fallback: Use Neon as ground truth, tune local system

4. **Knowledge Graph Complexity**
   - Risk: Entity extraction too inaccurate, graph not useful
   - Mitigation: Start with simple co-occurrence, add sophistication later
   - Fallback: Defer graph layer to post-MVP

### Dependency Risks

1. **PostgreSQL pgvector Version**
   - Risk: Older pgvector lacks HNSW support
   - Mitigation: Verify pgvector >= 0.5.0 in Phase 0
   - Fallback: Use IVFFlat index

2. **sentence-transformers Model Availability**
   - Risk: Model download fails, offline scenarios
   - Mitigation: Cache model in Docker image, document manual download
   - Fallback: Use OpenAI embeddings API

3. **FastMCP Framework Changes**
   - Risk: Breaking changes in FastMCP library
   - Mitigation: Pin version in requirements.txt
   - Fallback: Use standard FastAPI

---

## Roadmap Visualization

```
Timeline (Estimated):

Week 1: Phase 0 (Foundation)
Week 2: Phase 1 (Data Layer)
Week 3: Phase 2 (Search Layer)
Week 4: Phase 3 (Graph Layer)
Week 5: Phase 4 (Validation)
Week 6: Phase 5 (Reranking)
Week 7: Phase 6 (Integration)

Total: 7 weeks (35 days)

Critical Path:
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5 → Phase 6

Parallel Work Opportunities:
- Phase 3 (Graph) can start after Phase 1 (data available)
- Test writing can happen throughout (TDD approach)
- Documentation can be written in parallel with implementation

Milestones:
- Week 2: Data ingestion working
- Week 3: Basic search operational
- Week 5: Validation complete, accuracy benchmarked
- Week 6: Production-grade accuracy achieved
- Week 7: Deployed and integrated with Claude
```

---

## Next Steps

**Immediate Actions:**
1. Review and approve this roadmap
2. Set up development environment (Phase 0, Task 5)
3. Create project board with tasks from all phases
4. Begin Phase 0: Foundation Setup

**Task Master Integration:**
1. Parse this roadmap into Task Master tasks
2. Create task hierarchy matching phase structure
3. Set dependencies between tasks
4. Begin tracking progress through phases

**Documentation:**
1. Create Phase 0 implementation guide
2. Document environment setup
3. Create testing templates
4. Document deployment procedures

---

**Report Generated:** November 7, 2025, 18:14
**Roadmap Version:** 1.0
**Status:** Ready for Task Master parsing and implementation
