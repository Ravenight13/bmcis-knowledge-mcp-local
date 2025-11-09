# BMCIS Knowledge MCP - Data Models Analysis - Complete Index

**Analysis Date:** November 7, 2025  
**Project:** bmcis-knowledge-mcp at `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp`  
**Scope:** Comprehensive data models, database design, data flow, vector storage, and query patterns

---

## Document Overview

This analysis consists of 3 comprehensive documents totaling 2,680 lines:

### 1. DATA_MODELS_ANALYSIS.md (1,023 lines, 30KB)
**Comprehensive technical deep-dive**

Primary sections:
- **Section 1: Data Models** - Core Pydantic models (SearchResult, DocumentChunk, VendorInfo, TeamMember)
- **Section 2: Database Design** - PostgreSQL schema, 7-column structure, 5-index strategy
- **Section 3: Data Flow** - Complete ingestion pipeline (7 steps) + query processing (7 stages)
- **Section 4: Vector & Semantic Storage** - all-mpnet-base-v2 (768-dim), pgvector, HNSW indexing
- **Section 5: Query Patterns & Search** - 4 query types, 6 configuration settings, quality metrics
- **Section 6: Architectural Decisions** - Why PostgreSQL+pgvector, hybrid search, token chunking
- **Section 7: Data Consistency** - Constraints, triggers, validation mechanisms
- **Section 8: Performance & Scalability** - Current capacity, query performance, bottlenecks
- **Section 9: Summary Table** - Index components and dimensions
- **Section 10: KPIs** - Metrics dashboard (data stats, latency, quality)
- **Section 11: Future Enhancements** - Query expansion, filtering, learnable weights

**Best For:** Detailed technical understanding, architectural decisions, full pipeline analysis

---

### 2. DATA_MODELS_QUICK_REFERENCE.md (321 lines, 7.3KB)
**Quick lookup and implementation guide**

Primary sections:
- **Core Entities** - knowledge_base table structure, SearchResult model, DocumentChunk model
- **Search Pipeline Stages** - 7-stage visual pipeline with key operations
- **Key Configuration** - All tunable parameters with current values
- **Metadata Schema** - JSONB structure with example
- **Query Patterns** - 4 patterns with processing flow
- **Performance Metrics** - Latency table, throughput, concurrent users
- **Data Statistics** - 343 files, 2,600 chunks, 768 dimensions
- **Search Quality Results** - Comparison table (vector vs BM25 vs hybrid vs cross-encoder)
- **Files to Modify** - Quick reference for implementation changes
- **Query Expansion Note** - Why it's disabled and alternative
- **Useful Queries** - 5 SQL examples (by path, by vendor, vector search, BM25, recent)
- **Integration Points** - Input→Processing→Storage→Retrieval→Output flow

**Best For:** Quick implementation reference, configuration tuning, SQL queries

---

### 3. DATA_FLOW_DIAGRAM.md (668 lines, 42KB)
**ASCII diagrams and visual flowcharts**

Primary sections:
- **Section 1: System Architecture** - Claude Desktop → Cloudflare → Railway → PostgreSQL diagram
- **Section 2: Query Processing Pipeline** - 7-stage detailed flowchart with timing
  - Stage 1: Input validation
  - Stage 2: Query expansion (disabled)
  - Stage 3: Parallel search (vector + BM25)
  - Stage 4: Reciprocal Rank Fusion
  - Stage 5: Multi-factor boosting
  - Stage 6: Cross-encoder reranking
  - Stage 7: Response formatting
- **Section 3: Data Ingestion Pipeline** - 7-step process from files to indexed database
- **Section 4: Index Strategy Visualization** - All 5 indexes with performance characteristics
- **Section 5: Embedding Model Architecture** - MPNet model structure
- **Section 6: Metadata Filtering** - JSONB containment example with logic
- **Section 7: Score Evolution** - Score transformation through all stages

**Best For:** Visual understanding, implementation walkthrough, debugging latency

---

## Quick Navigation by Topic

### If you want to understand...

**...the database schema**
- → DATA_MODELS_ANALYSIS.md Section 2 (Database Design)
- → DATA_MODELS_QUICK_REFERENCE.md (Core Entities)

**...how data flows through the system**
- → DATA_FLOW_DIAGRAM.md (All 3 flow diagrams)
- → DATA_MODELS_ANALYSIS.md Section 3 (Data Flow)

**...the search pipeline**
- → DATA_FLOW_DIAGRAM.md Section 2 (Query Processing - 7 stages)
- → DATA_MODELS_QUICK_REFERENCE.md (Search Pipeline Stages)
- → DATA_MODELS_ANALYSIS.md Section 5 (Query Patterns)

**...vector embeddings and indexes**
- → DATA_MODELS_ANALYSIS.md Section 4 (Vector & Semantic Storage)
- → DATA_FLOW_DIAGRAM.md Section 4 (Index Strategy)
- → DATA_FLOW_DIAGRAM.md Section 5 (Embedding Model)

**...query patterns and examples**
- → DATA_MODELS_QUICK_REFERENCE.md (Query Patterns + Useful Queries)
- → DATA_MODELS_ANALYSIS.md Section 5 (Query Patterns)

**...performance characteristics**
- → DATA_MODELS_QUICK_REFERENCE.md (Performance Metrics)
- → DATA_MODELS_ANALYSIS.md Section 8 (Performance & Scalability)
- → DATA_FLOW_DIAGRAM.md Section 2 (Latency breakdown in flow)

**...why architectural decisions were made**
- → DATA_MODELS_ANALYSIS.md Section 6 (Architectural Decisions)

**...how to configure the system**
- → DATA_MODELS_QUICK_REFERENCE.md (Key Configuration)
- → DATA_MODELS_ANALYSIS.md Section 5.5 (Search Configuration)

**...metadata structure**
- → DATA_MODELS_ANALYSIS.md Section 2.3 (JSONB Metadata Schema)
- → DATA_MODELS_QUICK_REFERENCE.md (Metadata Schema)
- → DATA_FLOW_DIAGRAM.md Section 6 (Metadata Filtering)

---

## Key Metrics Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Data Volume** | 2,600 chunks | 343 markdown files |
| **Database Size** | 2.5GB | Neon Launch tier (3GB) |
| **Embedding Dimension** | 768 | all-mpnet-base-v2 |
| **Vector Data Size** | 7.8MB | 3KB per embedding |
| **Query Latency (p50)** | 270ms | SLA <300ms ✓ |
| **Query Latency (p95)** | 320ms | Peak performance |
| **Search Quality** | 92% | Relevance@5 (with cross-encoder) |
| **Throughput** | 3-4 q/s | Single-threaded |
| **Concurrent Users** | 27 | Target capacity |

---

## Architecture Overview

```
CLAUDE DESKTOP (User)
    ↓ semantic_search query
CLOUDFLARE ACCESS (Microsoft 365 SSO)
    ↓ authenticated request
RAILWAY.APP (FastMCP Server)
    ├─ Stage 1: Validate
    ├─ Stage 2: Expand [DISABLED]
    ├─ Stage 3: Parallel search (Vector + BM25)
    ├─ Stage 4: RRF fusion
    ├─ Stage 5: Boost
    ├─ Stage 6: Cross-encode
    └─ Stage 7: Format
    ↓ query & parameters
NEON POSTGRESQL
    ├─ HNSW index (vector similarity)
    ├─ GIN index (full-text BM25)
    ├─ GIN index (metadata filtering)
    ├─ B-tree indexes (path, timestamp)
    └─ knowledge_base table (2,600 rows)
    ↓ results
RESPONSE (top 5 results with metadata)
```

---

## Implementation Checklist

When implementing changes to the system:

- [ ] Modify src/config.py for feature flags and weights
- [ ] Update src/models.py if adding response fields
- [ ] Modify src/reranker.py for search logic changes
- [ ] Update src/embeddings.py for model changes
- [ ] Adjust src/chunking.py for chunking strategy changes
- [ ] Run ingestion (scripts/ingest_sample_files.py)
- [ ] Create indexes (sql/schema_768.sql)
- [ ] Test search quality with test suite

---

## Search Quality Benchmarks

Current system performance (empirical testing):

| Approach | Relevance@5 | MRR | Coverage |
|----------|------------|-----|----------|
| Vector only | 72% | 0.68 | 70% |
| BM25 only | 68% | 0.62 | 75% |
| Hybrid (Vector+BM25 RRF) | 85% | 0.81 | 90% |
| + Cross-Encoder Reranking | **92%** | **0.88** | **95%** |

**Note:** Query expansion DISABLED (-23.8% regression)

---

## Configuration Deep Dive

Key tunable parameters (all in src/config.py):

```python
# Embedding model
embedding_model = "all-mpnet-base-v2"  # 768-dim
embedding_dimension = 768

# Chunking strategy
chunk_size_tokens = 512                 # tokens per chunk
chunk_overlap_percent = 20              # 20% overlap

# Hybrid search
ENABLE_HYBRID_RERANKING = True
RERANKING_VECTOR_LIMIT = 20             # top-20 from vector
RERANKING_BM25_LIMIT = 20               # top-20 from BM25
RERANKING_FUSION_K = 60                 # RRF parameter

# Cross-encoder reranking
ENABLE_CROSS_ENCODER_RERANKING = True
CROSS_ENCODER_CANDIDATE_LIMIT = 20      # stage 1 candidates
CROSS_ENCODER_FINAL_LIMIT = 5           # stage 2 results

# Query expansion
ENABLE_QUERY_EXPANSION = False          # DISABLED

# Boosting weights
BOOST_METADATA_MATCH = 0.15             # +15% for vendor match
BOOST_DOCUMENT_TYPE = 0.10              # +10% for analysis/status
BOOST_RECENCY = 0.05                    # +5% for recent (<30 days)
BOOST_ENTITY_MATCH = 0.10               # +10% for entity match
BOOST_TOPIC_MATCH = 0.08                # +8% for topic match
```

---

## File References

**Schema & Database:**
- `/sql/schema_768.sql` - Complete PostgreSQL schema
- `/sql/hybrid_reranking_schema.sql` - Additional BM25 setup

**Source Code:**
- `/src/models.py` - Pydantic models (SearchResult, etc.)
- `/src/database.py` - Database operations & connection pooling
- `/src/embeddings.py` - Embedding generation (cached)
- `/src/reranker.py` - Hybrid search & cross-encoder
- `/src/chunking.py` - Token-based document chunking
- `/src/query_expansion.py` - Query expansion (currently disabled)
- `/src/server.py` - FastMCP server definition
- `/src/config.py` - Configuration & settings

**Ingestion:**
- `/scripts/ingest_sample_files.py` - Data ingestion pipeline
- `/scripts/analyze_corpus_stats.py` - Corpus analysis

**Testing:**
- `/tests/test_database.py` - Database tests
- `/tests/test_search_enhancements.py` - Search quality tests
- `/tests/test_cross_encoder.py` - Cross-encoder tests

---

## Next Steps for Development

### If you want to improve search quality:
1. Read: DATA_MODELS_ANALYSIS.md Section 5 (Query Patterns)
2. Review: Current boosting weights in DATA_MODELS_QUICK_REFERENCE.md
3. Modify: src/config.py boost parameters
4. Test: Run search quality benchmark
5. Commit: Document changes & rationale

### If you want to add new features:
1. Read: DATA_MODELS_ANALYSIS.md Section 2 (Schema)
2. Plan: What new metadata fields needed?
3. Update: sql/schema_768.sql
4. Modify: src/models.py & src/config.py
5. Test: Run ingestion & verify indexes

### If you want to optimize latency:
1. Read: DATA_MODELS_ANALYSIS.md Section 8 (Performance)
2. Profile: Which stage is slowest?
3. Reference: DATA_FLOW_DIAGRAM.md Section 2 (Timing)
4. Implement: Caching, quantization, or parameter tuning
5. Benchmark: Measure p50 & p95 latency

---

## Related Documents in Project

Additional context available:
- `/CLAUDE.md` - Project guidelines
- `/docs/SEARCH_QUALITY_REPORT.md` - Detailed quality analysis
- `/docs/SEARCH_IMPROVEMENTS.md` - Enhancement experiments
- `/docs/INGESTION_WORKFLOW.md` - Ingestion details
- `/session-handoffs/` - Previous work context

---

## Document Statistics

| Document | Lines | Size | Focus |
|----------|-------|------|-------|
| DATA_MODELS_ANALYSIS.md | 1,023 | 30KB | Deep technical |
| DATA_MODELS_QUICK_REFERENCE.md | 321 | 7.3KB | Implementation |
| DATA_FLOW_DIAGRAM.md | 668 | 42KB | Visual flows |
| **Total** | **2,680** | **79.3KB** | Complete coverage |

---

**Last Updated:** November 7, 2025  
**Status:** Complete & Production Ready  
**Version:** 1.0 (768-dimensional embeddings)
