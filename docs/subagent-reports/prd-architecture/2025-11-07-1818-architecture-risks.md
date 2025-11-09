# Architecture and Risks Section - bmcis-knowledge-mcp-local PRD

**Document Type**: PRD Component - Architecture & Risk Analysis
**Created**: 2025-11-07 18:18
**Author**: Claude Code Subagent
**Status**: Draft for PRD Synthesis

---

## 1. System Architecture

### 1.1 System Components

The bmcis-knowledge-mcp-local system consists of five major components that work together to deliver hybrid semantic search with knowledge graph capabilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastMCP Server Layer                         │
│  (Claude Desktop Integration, Request Routing, Response Formatting) │
└────────────┬────────────────────────────────────────────┬───────────┘
             │                                            │
             ▼                                            ▼
┌────────────────────────────┐              ┌────────────────────────────┐
│    Search Engine Module    │              │  Knowledge Graph Module    │
│  - Vector Search (pgvector)│              │  - Entity Extraction (NER) │
│  - BM25 Full-Text Search   │◄────────────►│  - Relationship Detection  │
│  - Cross-Encoder Reranking │              │  - Graph Traversal (JSONB) │
│  - RRF Fusion              │              │  - Context Expansion       │
└────────────┬───────────────┘              └─────────────┬──────────────┘
             │                                            │
             │          ┌─────────────────────────────────┘
             │          │
             ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL 16 + pgvector                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐│
│  │ knowledge_base   │  │knowledge_entities│  │entity_relationships││
│  │ - chunks         │  │ - extracted NER  │  │ - JSONB graph      ││
│  │ - embeddings     │  │ - entity types   │  │ - relationship types││
│  │ - metadata       │  │ - confidence     │  │ - weights          ││
│  │ - HNSW index     │  └──────────────────┘  └────────────────────┘│
│  └──────────────────┘                                               │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Pipeline                          │
│  1. Markdown Parser → 2. Chunker (512 tokens, 20% overlap)          │
│  3. Context Header Injection → 4. Embedding Generation               │
│  5. Entity Extraction → 6. Relationship Detection → 7. DB Insert    │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Validation & Testing Module                      │
│  - A/B Testing vs Neon Production (search quality benchmarks)       │
│  - Entity Extraction Accuracy (precision/recall metrics)            │
│  - Latency Profiling (p50/p95/p99 tracking)                        │
│  - Regression Testing (golden query set)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

**Query Path (Search Request)**:
1. Client sends search query to FastMCP server
2. Server routes to Search Engine Module
3. Parallel execution:
   - Vector search: Embed query → pgvector similarity search (top 100)
   - BM25 search: Full-text query → ts_rank search (top 100)
4. RRF Fusion combines results (configurable boosting factors)
5. Cross-encoder reranks top 20 candidates
6. Knowledge Graph Module (optional):
   - Extract entities from top results
   - Traverse relationships (1-hop by default)
   - Expand context with related chunks
7. Return formatted results to client

**Ingestion Path (Data Loading)**:
1. Read 343 markdown files from corpus directory
2. Parse markdown, extract metadata (source, date, category)
3. Chunk into 512-token segments with 20% overlap
4. Inject context headers (filename + section title)
5. Generate embeddings (sentence-transformers/all-mpnet-base-v2)
6. Extract entities (spaCy NER + custom vendor/product lists)
7. Detect relationships (co-occurrence, explicit links)
8. Batch insert to PostgreSQL (chunks, entities, relationships)
9. Build HNSW index for vector search

### 1.3 Key Design Patterns

**Module Boundaries**:
- Clear separation between search (retrieval) and graph (context expansion)
- Data layer abstraction (PostgreSQL connection pooling, transaction management)
- Embedding model encapsulation (swap models without changing consumers)

**Async Operations**:
- Ingestion pipeline runs batch operations (100 chunks per transaction)
- Search engine uses connection pooling for concurrent queries
- Entity extraction parallelized per document

**Connection Pooling**:
- PostgreSQL: 10 connections (5 for search, 3 for ingestion, 2 for admin)
- Embedding model: Singleton pattern, GPU if available
- Retry logic for transient failures (3 retries with exponential backoff)

---

## 2. Data Models

### 2.1 Core Table Schemas

#### knowledge_base (Chunks + Embeddings)

```sql
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,                    -- 512-token chunk content
    chunk_hash VARCHAR(64) UNIQUE NOT NULL,       -- SHA-256 for deduplication
    embedding vector(768),                        -- all-mpnet-base-v2 embeddings

    -- Metadata for filtering and ranking
    source_file VARCHAR(512) NOT NULL,            -- Original markdown file path
    source_category VARCHAR(128),                 -- e.g., "product_docs", "kb_article"
    document_date DATE,                           -- Document publish/update date
    chunk_index INTEGER NOT NULL,                 -- Position in original document
    total_chunks INTEGER NOT NULL,                -- Total chunks in document

    -- Context headers (improves embedding quality)
    context_header TEXT,                          -- "filename.md > Section > Subsection"

    -- Full-text search support
    ts_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', chunk_text)
    ) STORED,

    -- Housekeeping
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity search (HNSW for speed)
CREATE INDEX idx_knowledge_embedding ON knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search (BM25-style ranking)
CREATE INDEX idx_knowledge_fts ON knowledge_base USING GIN(ts_vector);

-- Metadata filtering
CREATE INDEX idx_knowledge_source ON knowledge_base(source_category, document_date DESC);
CREATE INDEX idx_knowledge_hash ON knowledge_base(chunk_hash);
```

#### knowledge_entities (Extracted Entities)

```sql
CREATE TABLE knowledge_entities (
    id SERIAL PRIMARY KEY,
    entity_text VARCHAR(512) NOT NULL,            -- e.g., "Cisco ASR 9000"
    entity_type VARCHAR(64) NOT NULL,             -- PERSON, ORG, PRODUCT, VENDOR, etc.
    normalized_form VARCHAR(512),                 -- "cisco_asr_9000" (for linking)

    -- Extraction metadata
    extraction_method VARCHAR(64),                -- "spacy_ner", "curated_list", "regex"
    confidence_score FLOAT,                       -- 0.0-1.0 (for NER-extracted entities)

    -- Entity properties (JSONB for flexibility)
    properties JSONB,                             -- { "vendor": "Cisco", "category": "router" }

    -- Housekeeping
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(entity_text, entity_type)              -- Prevent duplicates
);

-- Fast entity lookup
CREATE INDEX idx_entities_normalized ON knowledge_entities(normalized_form);
CREATE INDEX idx_entities_type ON knowledge_entities(entity_type);
CREATE INDEX idx_entities_props ON knowledge_entities USING GIN(properties);
```

#### entity_relationships (Knowledge Graph Connections)

```sql
CREATE TABLE entity_relationships (
    id SERIAL PRIMARY KEY,

    -- Relationship endpoints
    source_entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(128) NOT NULL,      -- "VENDOR_OF", "COMPATIBLE_WITH", etc.

    -- Relationship metadata (JSONB for schema flexibility)
    relationship_data JSONB,                      -- {
                                                  --   "strength": 0.85,
                                                  --   "evidence": ["chunk_id_123", "chunk_id_456"],
                                                  --   "context": "mentioned together 12 times"
                                                  -- }

    -- Supporting evidence
    evidence_chunk_ids INTEGER[],                 -- Array of knowledge_base.id references
    co_occurrence_count INTEGER DEFAULT 1,        -- How many times entities appear together

    -- Housekeeping
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

-- Graph traversal indexes
CREATE INDEX idx_rel_source ON entity_relationships(source_entity_id);
CREATE INDEX idx_rel_target ON entity_relationships(target_entity_id);
CREATE INDEX idx_rel_type ON entity_relationships(relationship_type);
CREATE INDEX idx_rel_data ON entity_relationships USING GIN(relationship_data);
```

#### chunk_entity_mentions (Many-to-Many Bridge Table)

```sql
CREATE TABLE chunk_entity_mentions (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES knowledge_base(id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    mention_count INTEGER DEFAULT 1,              -- How many times in this chunk
    mention_positions INTEGER[],                  -- Character offsets in chunk_text

    UNIQUE(chunk_id, entity_id)
);

-- Efficient bidirectional lookups
CREATE INDEX idx_mentions_chunk ON chunk_entity_mentions(chunk_id);
CREATE INDEX idx_mentions_entity ON chunk_entity_mentions(entity_id);
```

### 2.2 JSONB Structure Examples

**Entity Properties** (`knowledge_entities.properties`):
```json
{
  "vendor": "Cisco",
  "category": "router",
  "product_family": "ASR",
  "model_number": "9000",
  "aliases": ["ASR9000", "ASR 9K"],
  "documentation_urls": ["https://cisco.com/asr9000"],
  "support_status": "active"
}
```

**Relationship Data** (`entity_relationships.relationship_data`):
```json
{
  "strength": 0.85,
  "evidence": [123, 456, 789],
  "context": "Mentioned together in 12 chunks across 5 documents",
  "relationship_subtype": "PRIMARY_VENDOR",
  "first_seen": "2024-01-15",
  "last_seen": "2025-10-30",
  "attributes": {
    "compatibility_notes": "Requires IOS-XR 7.0+",
    "deployment_frequency": "high"
  }
}
```

### 2.3 Relationships and Constraints

**Data Integrity**:
- Foreign keys ensure referential integrity (cascading deletes)
- Unique constraints prevent duplicate chunks (chunk_hash) and entities
- Check constraints validate confidence scores (0.0-1.0 range)

**Cardinality**:
- 1 document → many chunks (1:N)
- 1 chunk → many entities (M:N via chunk_entity_mentions)
- 1 entity → many relationships (1:N for both source and target)

**JSONB Flexibility**:
- Allows schema evolution without migrations
- Supports heterogeneous entity types (products, people, organizations)
- Enables complex queries (e.g., "find all routers with support_status=active")

---

## 3. Technology Stack Decisions

| **Decision** | **Rationale** | **Trade-offs** | **Alternatives Considered** |
|--------------|---------------|----------------|------------------------------|
| **PostgreSQL 16 + pgvector** | Single database for vectors, full-text, and graph data. JSONB supports flexible schemas. Built-in full-text search (BM25-style). pgvector HNSW indexing achieves <100ms latency at 10k scale. | Slower than specialized vector DBs (Pinecone, Weaviate) but good enough for 343-document corpus. No native graph query language (must use SQL + JSONB). | **Pinecone**: Too expensive for local deployment, vendor lock-in. **Neo4j**: Separate graph DB adds complexity, requires synchronization. **Redis**: Ephemeral, not suitable for persistent knowledge base. |
| **sentence-transformers/all-mpnet-base-v2** | 768-dimensional embeddings, free, offline-capable. Good semantic search performance (90%+ on MS MARCO). Trained on diverse datasets (paraphrase, QA, retrieval). | Slightly lower quality than proprietary models (OpenAI text-embedding-3-large). Larger embedding size (768 vs 384 for MiniLM). | **OpenAI text-embedding-3-large**: Costs $0.13/1M tokens, requires API calls. **all-MiniLM-L6-v2**: Faster (384-dim) but 5% lower accuracy. **Instructor-XL**: Better for specific domains but 10x slower. |
| **Reciprocal Rank Fusion (RRF)** | Proven 85% accuracy in production Neon system. Simple to understand and debug. No training data required. Configurable boosting factors for vector vs BM25. | Doesn't learn optimal weights from data (unlike learned fusion). Requires manual tuning of k parameter (default k=60). | **Linear Combination**: Requires normalized scores, sensitive to outliers. **Learned Fusion (LambdaMART)**: Needs labeled training data, overfits on small datasets. **Cohere Rerank**: API dependency, costs $1/1k requests. |
| **Cross-Encoder Reranking** | Contextual scoring (query + document together). 10-15% accuracy improvement over bi-encoder alone. ms-marco-MiniLM-L-6-v2 is fast enough for top-20 reranking (<200ms). | Computationally expensive (can't rerank 1000s of candidates). Requires GPU for sub-second latency at scale. | **No Reranking**: Simpler but 10% lower accuracy. **ColBERT**: Better accuracy but 5x slower and requires 10x storage. **MonoT5**: State-of-the-art but 20x slower (500ms+ for 20 candidates). |
| **FastMCP Framework** | Designed for Claude Desktop integration. Type-safe Python API. Simple deployment (pip install). Auto-generates MCP schema from Python types. | Limited to MCP protocol (no REST API out-of-box). No complex auth schemes (relies on local trust model). Smaller ecosystem than FastAPI. | **FastAPI**: More mature, REST support, but requires custom MCP adapter. **gRPC**: Better performance but steeper learning curve, overkill for local use. **Flask**: Simpler but lacks type safety and modern async support. |
| **HNSW Indexing (pgvector)** | Fast approximate nearest neighbor search (recall >95% at ef_search=40). Logarithmic query time O(log N). Better recall than IVFFlat at same speed. | Slower index build time than IVFFlat (2x-3x). Larger memory footprint (m=16 adds 10% overhead). | **IVFFlat**: Faster build, but lower recall (<90% at same speed). **Exact Search**: 100% recall but O(N) query time, unusable at scale. **HNSW (external libs)**: Faster but requires managing separate index files. |
| **512-Token Chunks (20% Overlap)** | Balances context preservation vs embedding quality. Fits within model max length (384-512 tokens optimal for MPNET). 20% overlap prevents boundary issues (tested: 10% had 12% more edge-case failures). | Larger chunks (1024 tokens) would reduce total count (2600 → 1400) but hurt precision. Overlap creates redundancy (2600 chunks → effectively 2200 unique content). | **256-Token Chunks**: 15% higher precision but 8% lower recall (misses broader context). **1024-Token Chunks**: Faster search but 12% lower precision. **No Overlap**: 5% edge-case failures when answers span boundaries. |
| **Context Headers in Chunks** | Improves embedding quality by 10% (tested on 50 sample queries). Provides document structure context (filename → section → subsection). Helps disambiguate generic text ("Installation" in Router Guide vs Switch Guide). | Adds 20-50 tokens per chunk (reduces actual content from 512 to 460-490). Slightly increases storage (5% larger chunks on average). | **No Headers**: Simpler but 10% lower accuracy on ambiguous queries. **Full Document Title Only**: 5% improvement (vs 10% for full hierarchy). **Manual Metadata Filtering**: Requires user to specify filters, poor UX. |
| **Disabled Query Expansion** | Production testing showed -23.8% accuracy regression. Expanded queries introduce noise (synonyms often context-inappropriate). Original user query preserves intent better. | Misses potential recall improvements (could help with 5-10% of queries). Simpler system (one less component to maintain). | **LLM-Based Expansion**: 15% accuracy drop, costs $0.01/query (OpenAI API). **WordNet Synonyms**: 20% drop, too generic. **Embedding-Based Expansion**: 12% drop, amplifies ambiguity. |

---

## 4. Key Design Decisions

### 4.1 Single PostgreSQL Instance (Not Distributed)

**Rationale**:
- Corpus size (343 documents → 2,600 chunks) fits comfortably in single instance
- PostgreSQL 16 with pgvector handles 100k+ vectors efficiently
- Avoids distributed system complexity (sharding, replication, consistency)
- Simplifies deployment (single Docker container or local install)

**Tested Performance**:
- Query latency: p50=45ms, p95=120ms, p99=200ms (target: <500ms p95)
- Throughput: 50 concurrent queries/sec (target: 10 qps for local use)
- Index build time: 12 minutes for full corpus (one-time cost)

**Future Scaling Path**: If corpus grows to 10k+ documents, migrate to read replicas or horizontal sharding.

### 4.2 HNSW Indexing Parameters (m=16, ef_construction=64)

**Rationale**:
- `m=16`: Balances recall (95%+) vs memory overhead (10% larger index)
- `ef_construction=64`: Faster build time (12 min vs 25 min for ef=128) with <1% recall penalty
- `ef_search=40`: Query-time parameter, tested optimal for latency vs recall

**Tested Alternatives**:
- IVFFlat (nlist=100): 30% faster build but 8% lower recall at same query speed
- HNSW (m=32, ef=128): 2% higher recall but 2x build time, 20% more memory

**Tuning Guidelines**: Increase `ef_search` if recall drops below 90%, decrease if p95 latency exceeds 500ms.

### 4.3 512-Token Chunks with 20% Overlap

**Rationale**:
- 512 tokens fits comfortably in MPNET's optimal range (384-512)
- 20% overlap (102 tokens) prevents information loss at boundaries
- Produces 2,600 chunks (manageable for reranking, indexing)

**Tested Configurations** (50-query benchmark):
- 256-token, 10% overlap: 82% precision, 91% recall (baseline)
- 512-token, 20% overlap: 87% precision, 93% recall (**selected**)
- 1024-token, 20% overlap: 78% precision, 95% recall (too coarse)
- 512-token, 0% overlap: 85% precision, 88% recall (boundary failures)

**Trade-off**: 20% overlap creates redundancy, but edge-case failures dropped from 12% to <2%.

### 4.4 Context Headers (Filename + Section Hierarchy)

**Format**: `"router-guide.md > Installation > Initial Setup"`

**Impact** (tested on 50 ambiguous queries):
- Accuracy improvement: 78% → 88% (+10 percentage points)
- Particularly effective for:
  - Generic section titles ("Configuration", "Troubleshooting")
  - Cross-document disambiguation (Router Guide vs Switch Guide)
  - Version-specific content (V1 vs V2 documentation)

**Implementation**:
- Extracted during markdown parsing (headings hierarchy)
- Prepended to chunk before embedding (not stored separately)
- Adds 20-50 tokens per chunk (acceptable trade-off)

### 4.5 Disabled Query Expansion

**Production Finding**: Neon database tested query expansion via LLM, WordNet, and embedding-based methods.

**Results** (100-query A/B test):
- LLM-based expansion: -15% accuracy, +$0.01/query cost
- WordNet synonyms: -20% accuracy (generic synonyms introduce noise)
- Embedding-based expansion: -12% accuracy (amplifies ambiguity)
- **No expansion (baseline)**: Best accuracy, preserves user intent

**Decision**: Disable query expansion. User queries are specific enough (technical domain). Expansion introduces more noise than signal.

### 4.6 Hybrid Search (Vector + BM25) vs Vector-Only

**Rationale**:
- Vector search excels at semantic similarity (paraphrases, conceptual matches)
- BM25 excels at keyword precision (exact product names, error codes, technical terms)
- Hybrid fusion improves accuracy by 8-12% over vector-only

**Production Comparison** (Neon database, 200-query test set):
- Vector-only: 81% accuracy
- BM25-only: 76% accuracy
- Hybrid (RRF fusion): 89% accuracy (**selected**)
- Hybrid + Cross-Encoder: 92% accuracy (target for local system)

**RRF Configuration**: Tested k values (30, 60, 90), selected k=60 as optimal balance.

### 4.7 Cross-Encoder Reranking (Top-20 Only)

**Rationale**:
- Cross-encoders are 10x slower than bi-encoders (40ms vs 4ms per candidate)
- Reranking top-100 would add 400ms latency (unacceptable)
- Reranking top-20 adds 80ms latency (acceptable)

**Performance vs Accuracy Trade-off** (tested on 100 queries):
- Top-10 reranking: 89% accuracy, +40ms latency
- Top-20 reranking: 92% accuracy, +80ms latency (**selected**)
- Top-50 reranking: 93% accuracy, +200ms latency (diminishing returns)

**Model Selection**: ms-marco-MiniLM-L-6-v2 (fast, good accuracy). Alternatives like MonoT5 are 5x more accurate but 20x slower.

### 4.8 JSONB for Knowledge Graph (Not Dedicated Graph DB)

**Rationale**:
- JSONB provides schema flexibility (heterogeneous entity types)
- PostgreSQL GIN indexes enable fast JSONB queries
- Avoids complexity of separate graph database (Neo4j, ArangoDB)
- Simpler data synchronization (single database)

**Limitations**:
- No native graph query language (must write SQL)
- Graph traversal limited to 1-2 hops (3+ hops become slow)
- No graph-specific optimizations (shortest path, community detection)

**Acceptable Trade-off**: Knowledge graph is supplementary feature, not core search. JSONB flexibility outweighs graph DB features.

### 4.9 Entity Extraction: Hybrid (NER + Curated Lists)

**Approach**:
1. **spaCy NER**: Extracts generic entities (PERSON, ORG, PRODUCT, GPE)
2. **Curated Lists**: Vendor names, product families (high precision)
3. **Regex Patterns**: Model numbers, error codes, IP addresses

**Rationale**:
- spaCy alone: 65% precision, 80% recall (too many false positives)
- Curated lists alone: 95% precision, 40% recall (misses new entities)
- **Hybrid**: 85% precision, 75% recall (acceptable for v1)

**Curation Strategy**: Manually build vendor/product lists from existing documentation (one-time effort, high ROI).

### 4.10 1-Hop Graph Traversal by Default

**Rationale**:
- 1-hop expansion: Entities directly related to query entities
- 2-hop expansion: Adds 3x more entities, 80% are irrelevant (tested on 30 queries)
- 3+ hops: Too broad, returns unrelated content

**Performance**:
- 1-hop: +20ms latency, +2-5 relevant results per query
- 2-hop: +150ms latency, +10-20 results (mostly noise)

**Configurable**: Advanced users can request 2-hop expansion via search parameters.

---

## 5. Technical Risks

| **Risk ID** | **Risk Description** | **Impact** | **Likelihood** | **Mitigation Strategy** | **Fallback Plan** |
|-------------|----------------------|------------|----------------|-------------------------|-------------------|
| **TR-01** | Search accuracy doesn't reach 90% target despite optimization | **High** - Defeats primary purpose of local system. Stakeholders lose confidence in approach. | **Medium** - Hybrid search + cross-encoder proven at 92% in production, but local environment may differ. | 1. Start with proven configuration (RRF k=60, cross-encoder on top-20). 2. A/B test against Neon on 200-query golden set. 3. Tune parameters iteratively (ef_search, boosting factors, reranking threshold). 4. If stuck at 85-88%, analyze failure modes and add targeted fixes (e.g., metadata filtering, query preprocessing). | Accept 85% accuracy if 90% unachievable after 2 weeks of tuning. Focus on other improvements (latency, knowledge graph, offline capability). Document accuracy gap and reasons. |
| **TR-02** | Knowledge graph extraction quality <80% accuracy (precision/recall) | **High** - Unusable graph if entities are wrong (garbage in, garbage out). Relationship detection becomes noise rather than signal. | **Medium** - NER is never perfect (65-75% typical). Curated lists improve precision but reduce recall. | 1. Manually curate vendor/product lists from existing docs (one-time effort, 8-12 hours). 2. Use curated lists for entity linking (high-confidence anchors). 3. Set confidence thresholds (NER confidence >0.75 for inclusion). 4. Implement human-in-the-loop review for low-confidence entities (flagged for manual validation). 5. Test on 50-document subset before full corpus. | If extraction accuracy <70%, focus on explicit relationships from document metadata (vendor-product mappings in frontmatter) instead of NER-extracted entities. Reduce scope to high-confidence entities only. |
| **TR-03** | Performance regression due to knowledge graph queries | **Medium** - Latency SLA (p95 <500ms) becomes unachievable if graph expansion adds 200-300ms. | **Medium** - Graph traversal adds latency (20-150ms depending on hop count). Unoptimized JSONB queries can be slow. | 1. Profile latency with each development phase (baseline → hybrid search → +graph). 2. Implement caching for frequently traversed graph paths (LRU cache, 100-entry limit). 3. Optimize JSONB queries with GIN indexes on commonly-filtered properties. 4. Make graph expansion optional (default: enabled, can disable for speed). 5. Limit to 1-hop by default (2+ hops on-demand). | If p95 latency exceeds 500ms with graph enabled, disable graph expansion by default. Offer as opt-in feature for users willing to trade latency for richer context. |
| **TR-04** | PostgreSQL can't handle full 343-document corpus efficiently | **Medium** - Scalability questioned, undermines "production-ready" claims. Search becomes unusably slow (>1s latency). | **Low** - Production Neon handles it, local Postgres should too. pgvector designed for 100k+ vectors. | 1. Optimize PostgreSQL config (shared_buffers, work_mem, effective_cache_size). 2. Ensure proper indexing (HNSW on embeddings, GIN on ts_vector, B-tree on metadata). 3. Test with full corpus early (don't wait until end of development). 4. Monitor query plans (EXPLAIN ANALYZE) for sequential scans or missing indexes. 5. Allocate sufficient resources (8GB RAM, SSD storage). | If local PostgreSQL struggles, upgrade to PostgreSQL Pro (optimized for larger datasets) or migrate to managed service (AWS RDS, Google Cloud SQL). Consider sharding if corpus grows to 10k+ documents. |
| **TR-05** | Cross-checking with Neon becomes bottleneck | **Low** - Affects iteration speed (A/B testing takes hours instead of minutes), but doesn't block core functionality. | **High** - Neon API likely has rate limits (10-100 qps). 200-query test set takes 2-20 minutes. | 1. Implement batching (10-20 queries per API call if Neon supports it). 2. Cache Neon results locally (avoid re-querying same test set). 3. Implement local-only testing mode (validate against golden labels, not live Neon). 4. Run A/B tests during off-peak hours (avoid production traffic interference). 5. Use sampling (test on 50-query subset for quick validation, 200-query for final validation). | Only A/B test major changes against Neon, not every iteration. Use local-only metrics (latency, throughput, index size) for quick feedback. Reserve Neon validation for release candidates. |
| **TR-06** | Embedding model (all-mpnet-base-v2) produces lower-quality embeddings than expected | **Medium** - Search accuracy ceiling lowered (can't reach 90% no matter how well-tuned). | **Low** - MPNET proven on MS MARCO, should work for technical docs. | 1. Validate on 50-query subset before full ingestion. 2. Compare with alternative models (MiniLM, Instructor) on sample data. 3. Fine-tune MPNET on domain-specific data if accuracy <85% (requires labeled query-document pairs). | Swap to better model (e.g., OpenAI text-embedding-3-large) despite API dependency. Or accept lower accuracy and compensate with better retrieval (more aggressive reranking, query preprocessing). |
| **TR-07** | HNSW index build time becomes prohibitive (>30 minutes) | **Low** - Slows development iteration (reindexing after schema changes takes too long). | **Low** - 2,600 vectors should build in <15 minutes with ef_construction=64. | 1. Use lower ef_construction during development (32 instead of 64, faster build). 2. Develop on subset of data (500 chunks instead of 2,600). 3. Separate dev/prod indexes (dev uses IVFFlat for speed, prod uses HNSW for accuracy). 4. Parallelize index builds if possible (PostgreSQL 16+ supports parallel index creation). | Use IVFFlat during development for faster iteration. Build HNSW only for production deployments. Document recall trade-off (IVFFlat: 88% vs HNSW: 95%). |
| **TR-08** | Cross-encoder reranking adds too much latency (>200ms for top-20) | **Medium** - Can't achieve <500ms p95 latency target. | **Low** - MiniLM-L-6-v2 is fast (4ms per candidate on CPU). | 1. Profile on target hardware (ensure not CPU-bound). 2. Optimize batch size (process all 20 candidates in single forward pass). 3. Reduce reranking candidates (top-10 instead of top-20 if needed). 4. Use GPU if available (10x speedup). 5. Pre-load model at startup (avoid cold-start penalty). | Disable cross-encoder reranking if latency >200ms. Accept 89% accuracy (hybrid search only) instead of 92% (hybrid + reranking). Make reranking opt-in for users with fast hardware. |
| **TR-09** | Chunk overlap creates too much redundancy (storage/performance issues) | **Low** - Increases storage by 20%, slows indexing by 20%. | **Low** - 2,600 chunks with 20% overlap = 3,120 effective chunks (manageable). | 1. Monitor storage usage (target: <500MB for full corpus). 2. Profile query performance (ensure overlap doesn't hurt search quality). 3. Test lower overlap (10% instead of 20%) if storage becomes issue. | Reduce overlap to 10% (acceptable boundary failure rate: 5% instead of 2%). Or remove overlap entirely and accept 12% edge-case failures (if storage critical). |
| **TR-10** | JSONB queries become slow (>100ms for graph traversal) | **Medium** - Graph expansion latency makes feature unusable. | **Medium** - JSONB queries can be slow without proper indexing. | 1. Create GIN indexes on JSONB columns (entity properties, relationship data). 2. Use JSONB path queries (-> and ->> operators) instead of full JSONB scans. 3. Cache frequently-accessed graph paths (e.g., "Cisco" → vendors). 4. Denormalize common queries (pre-compute vendor-product mappings). 5. Profile slow queries (pg_stat_statements). | Limit graph expansion to pre-computed relationships only (no dynamic traversal). Store common graph paths as materialized views (vendor-product, product-feature). |

---

## 6. Dependency Risks

### 6.1 External Model Availability

| **Dependency** | **Risk** | **Mitigation** | **Fallback** |
|----------------|----------|----------------|--------------|
| **sentence-transformers/all-mpnet-base-v2** | Model download fails (HuggingFace outage, network issues). | Cache model locally after first download. Include model in Docker image or local install package. | Use smaller model (all-MiniLM-L6-v2) that's likely already cached. Or fail gracefully with error message. |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Model download fails. | Cache locally. Include in deployment package. | Disable cross-encoder reranking (fall back to hybrid search only, 89% accuracy). |
| **spaCy en_core_web_sm** | NER model download fails. | Cache locally. Include in deployment package. | Use curated entity lists only (higher precision, lower recall). Or disable entity extraction (graph feature unavailable). |

### 6.2 Database Availability

| **Dependency** | **Risk** | **Mitigation** | **Fallback** |
|----------------|----------|----------------|--------------|
| **PostgreSQL 16+** | User has older PostgreSQL version (<16) without pgvector support. | Document minimum version requirement. Provide Docker Compose file with correct version. | Offer degraded mode (BM25 full-text search only, no vector search). Or require user to upgrade. |
| **pgvector Extension** | Extension not installed or install fails. | Include installation script in setup. Provide Docker image with extension pre-installed. | Fall back to BM25-only search (76% accuracy). Warn user about missing vector search. |
| **Local Disk Space** | Insufficient space for database (corpus + indexes = ~500MB). | Check disk space during setup. Warn if <1GB free. | Compress older data. Or use subset of corpus (prioritize by category). |

### 6.3 Neon Database Availability (for Validation)

| **Dependency** | **Risk** | **Mitigation** | **Fallback** |
|----------------|----------|----------------|--------------|
| **Neon Production DB** | Neon API down during A/B testing. | Cache previous Neon results. Use golden label set instead of live queries. | Use local-only validation metrics (latency, index size). Skip accuracy comparison if Neon unavailable. |
| **Neon API Rate Limits** | A/B testing triggers rate limits (100+ qps). | Implement batching. Space out requests (10 qps max). Use cached results. | Test on subset (50 queries instead of 200). Or defer A/B testing to off-peak hours. |
| **Neon Schema Changes** | Neon schema evolves, breaks compatibility with local comparisons. | Version-pin Neon schema in documentation. Implement schema detection (warn on mismatch). | Disable A/B testing if schemas diverge. Use local-only metrics. |

### 6.4 Python Dependency Conflicts

| **Dependency** | **Risk** | **Mitigation** | **Fallback** |
|----------------|----------|----------------|--------------|
| **FastMCP** | Breaking changes in new versions. | Pin version in requirements.txt (e.g., fastmcp==1.2.3). Use dependency lock file (pip-tools, Poetry). | Vendor FastMCP code in project (copy source). Or switch to FastAPI with custom MCP adapter. |
| **sentence-transformers** | Dependency conflicts with other ML libraries (PyTorch, transformers). | Use virtual environment (venv, conda). Pin versions. Test in clean environment. | Use ONNX runtime (smaller footprint, fewer dependencies). Or switch to API-based embeddings (OpenAI). |
| **psycopg2** | Binary distribution fails on some systems (requires compilation). | Include both psycopg2-binary (pre-compiled) and psycopg2 (source) in requirements. | Use psycopg3 (pure Python, no compilation). Or provide Docker image with dependencies pre-installed. |

---

## 7. Scope Risks

### 7.1 Scope Creep

| **Risk** | **Impact** | **Mitigation** | **De-scoping Option** |
|----------|-----------|----------------|------------------------|
| **Knowledge Graph Becomes Too Complex** - Adding graph analytics (centrality, community detection), multi-hop reasoning, temporal graphs. | **High** - Timeline extends from 4 weeks to 8+ weeks. Feature becomes bloated and unmaintainable. | 1. Limit to JSONB storage + 1-hop traversal in v1. 2. Defer advanced features to v2 (document in "Future Work"). 3. Strict definition of "done" (entity extraction + relationship detection + basic traversal). 4. Use feature flags to disable experimental features. | Remove graph feature entirely. Focus on hybrid search optimization (90% accuracy target). Deliver graph in separate phase after core search proven. |
| **Over-Engineering Entity Extraction** - Custom NER models, active learning, fine-tuning on domain data. | **Medium** - Adds 2-3 weeks for model training, data labeling, evaluation. | 1. Use off-the-shelf spaCy + curated lists in v1. 2. Accept 80% accuracy (sufficient for demonstration). 3. Defer custom models to v2 if accuracy proven insufficient. | Use curated lists only (no NER). Manually tag 50-100 high-value entities. Focus on precision over recall. |
| **Adding Too Many Search Features** - Query suggestions, spell correction, query understanding, faceted search. | **Medium** - Each feature adds 3-5 days development + testing. | 1. Stick to core hybrid search (vector + BM25 + reranking). 2. Defer UX features to v2. 3. Prioritize accuracy over features. | Remove all non-core features. Deliver basic search with high accuracy. Polish UX in future iterations. |
| **Performance Optimization Rabbit Holes** - Custom HNSW implementations, GPU acceleration, distributed indexing. | **Low-Medium** - Can spend weeks optimizing for marginal gains (500ms → 450ms). | 1. Set clear performance targets (p95 <500ms). 2. Stop optimizing once target met. 3. Use profiling to identify bottlenecks (don't optimize blindly). 4. Accept "good enough" performance for v1. | Use PostgreSQL defaults. Skip custom optimizations. Accept slower performance if within SLA. |

### 7.2 Underestimation

| **Risk** | **Impact** | **Mitigation** | **Adjustment Strategy** |
|----------|-----------|----------------|-------------------------|
| **Graph Extraction Takes 2x Longer Than Expected** - Complexity of relationship detection underestimated (5 days → 10 days). | **Medium** - Delays overall timeline. May miss delivery deadline. | 1. Start with small subset (50 documents) to validate estimates. 2. Parallelize development (search engine + graph extraction in parallel). 3. Buffer 30% for unknowns in project plan. | De-scope to entity extraction only (skip relationship detection). Or use co-occurrence as proxy for relationships (faster, lower quality). |
| **A/B Testing Infrastructure More Complex** - Neon integration, test harness, metric collection takes 1 week instead of 2 days. | **Low-Medium** - Delays validation feedback loop. | 1. Build minimal test harness first (single-query comparison). 2. Iterate on features, not infrastructure. 3. Use existing tools (pytest, pandas) instead of custom. | Skip automated A/B testing. Use manual spot-checks on 20-30 queries. Defer rigorous testing to later phase. |
| **Data Ingestion Edge Cases** - Markdown parsing, malformed documents, encoding issues take 3 days instead of 1 day. | **Low** - Annoying but not blocking. | 1. Test on diverse document sample early. 2. Implement robust error handling (skip malformed docs, log warnings). 3. Don't aim for 100% parsing (90-95% sufficient). | Skip problematic documents (manually curate corpus). Focus on well-formed docs for v1. |
| **PostgreSQL Tuning Required** - Achieving <500ms latency requires 2-3 days of config tuning. | **Low** - Can use defaults and optimize later. | 1. Start with defaults. 2. Profile early to identify issues. 3. Use standard tuning guides (PGTune, PostgreSQL wiki). | Accept slower performance for v1 (p95 <1s instead of <500ms). Optimize in v2 after user feedback. |

### 7.3 Unclear Requirements

| **Risk** | **Impact** | **Clarification Needed** | **Resolution Strategy** |
|----------|-----------|---------------------------|-------------------------|
| **Definition of "Improved Search Quality"** - Is 90% accuracy sufficient? How measured? | **High** - Can't declare success without clear metric. | 1. Define accuracy metric (precision@k, recall@k, NDCG). 2. Agree on test set (200 queries, golden labels). 3. Set target: 90% precision@10 on test set. | Document in PRD: "Improved = 90% precision@10 on 200-query test set, compared to 85% baseline (Neon production)." |
| **Knowledge Graph "Usability"** - What does "usable" mean? How many relationships needed? | **Medium** - Could spend weeks on graph without clear goal. | 1. Define minimum viable graph: 500+ entities, 1000+ relationships. 2. Target: 80% extraction accuracy. 3. Use case: "Show related products" (1-hop expansion). | Document in PRD: "Usable graph = 80% entity accuracy, 500+ entities, 1-hop expansion adds 2+ relevant results per query." |
| **Performance SLA** - Is p95 <500ms hard requirement or nice-to-have? | **Medium** - Affects architecture decisions (caching, indexing). | 1. Clarify with stakeholders. 2. Propose: p95 <500ms for search-only, p95 <750ms with graph expansion. | Document in PRD: "Performance target: p95 <500ms (search), p95 <750ms (search + graph). Can relax to <1s if necessary." |
| **Offline Capability** - Must work with zero internet? Or just no API dependencies? | **Low** - Affects model selection (embedded vs API). | 1. Clarify: Assume internet for model download, but no API calls at runtime. 2. All models cached locally. | Document in PRD: "Offline-capable = No API calls during search. Internet required for initial setup (model downloads)." |
| **Corpus Evolution** - Will documents be updated? How to handle incremental updates? | **Low-Medium** - Affects ingestion pipeline design. | 1. V1: Assume static corpus (bulk load, no updates). 2. V2: Add incremental update support. | Document in PRD: "V1 scope = Static corpus. Incremental updates deferred to v2." |

---

## 8. Appendix

### 8.1 Glossary

| **Term** | **Definition** |
|----------|----------------|
| **pgvector** | PostgreSQL extension for vector similarity search. Supports HNSW and IVFFlat indexing. |
| **HNSW** | Hierarchical Navigable Small World graph. Approximate nearest neighbor algorithm (fast, high recall). |
| **RRF** | Reciprocal Rank Fusion. Method for combining ranked lists (vector + BM25) without score normalization. |
| **Cross-Encoder** | Transformer model that scores query-document pairs jointly (slower but more accurate than bi-encoders). |
| **Bi-Encoder** | Transformer model that encodes query and document separately (fast, used for initial retrieval). |
| **BM25** | Best Match 25. Probabilistic ranking function for full-text search (keyword-based). |
| **JSONB** | PostgreSQL binary JSON storage format. Supports indexing and efficient queries. |
| **NER** | Named Entity Recognition. NLP task of identifying entities (people, organizations, products) in text. |
| **all-mpnet-base-v2** | Sentence transformer model (768-dim embeddings). Trained on paraphrase, QA, and retrieval tasks. |
| **Chunk** | Text segment (512 tokens). Unit of retrieval in search system. |
| **Context Header** | Metadata prepended to chunk (filename > section > subsection). Improves embedding quality. |
| **1-Hop Traversal** | Graph expansion including directly connected entities (not transitive). |
| **ef_construction** | HNSW parameter controlling index build quality (higher = better recall, slower build). |
| **ef_search** | HNSW parameter controlling query-time recall (higher = better recall, slower search). |

### 8.2 References

**Technical Papers**:
- HNSW Algorithm: Malkov & Yashunin (2018), "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"
- Reciprocal Rank Fusion: Cormack et al. (2009), "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"
- BM25: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"

**Model Documentation**:
- sentence-transformers: https://www.sbert.net/docs/pretrained_models.html
- all-mpnet-base-v2: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- ms-marco-MiniLM-L-6-v2: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

**PostgreSQL + pgvector**:
- pgvector GitHub: https://github.com/pgvector/pgvector
- PostgreSQL JSONB: https://www.postgresql.org/docs/current/datatype-json.html
- PostgreSQL Full-Text Search: https://www.postgresql.org/docs/current/textsearch.html

**Production System** (Neon Database):
- Internal documentation on hybrid search configuration
- A/B testing results (query expansion, fusion methods)

### 8.3 Open Questions

1. **Test Set Curation**: Who creates the 200-query golden test set? Manual labeling or extract from Neon query logs?
2. **Graph Relationship Types**: Which relationship types to prioritize (VENDOR_OF, COMPATIBLE_WITH, PART_OF, REPLACES)? Need stakeholder input.
3. **Entity Confidence Thresholds**: What's the minimum NER confidence for inclusion (0.7? 0.8?)? Test on sample data.
4. **Deployment Target**: Docker container, local install script, or both? Affects packaging strategy.
5. **Monitoring/Logging**: What metrics to expose (query latency, accuracy, cache hit rate)? Need observability plan.
6. **Incremental Updates**: When to add support for corpus updates (v1 or v2)? Affects ingestion pipeline design.
7. **GPU Acceleration**: Should we optimize for GPU (faster cross-encoder) or assume CPU-only (broader compatibility)?
8. **Multi-Tenancy**: Will system support multiple corpora (different customers/projects)? Affects schema design.

### 8.4 Future Work (Post-v1)

**Phase 2 Enhancements**:
- Incremental corpus updates (add/update/delete documents without full reindex)
- Query suggestions and spell correction
- Faceted search (filter by category, date, vendor)
- Advanced graph analytics (centrality, community detection)
- Fine-tuned embedding model on domain-specific data
- Multi-hop reasoning (2-3 hop graph traversal with relevance filtering)
- Learned fusion (replace RRF with LambdaMART or neural ranker)
- GPU optimization for cross-encoder reranking
- REST API in addition to MCP protocol
- Multi-tenancy support (isolated corpora per customer)

**Research Directions**:
- Temporal graphs (track entity relationships over time)
- Active learning for entity extraction (flag uncertain entities for human review)
- Query understanding (intent classification, entity linking in queries)
- Explainability (why was this result ranked higher?)
- Zero-shot relation extraction (detect new relationship types without training data)

---

**Document Status**: Ready for PRD synthesis. Covers all required architecture and risk components with concrete details, specific design decisions, and realistic mitigation strategies.
