# BMCIS Knowledge MCP - Data Models Quick Reference

## Core Entities

### knowledge_base (PostgreSQL Table)
```
id              → SERIAL PRIMARY KEY
content         → TEXT (with contextual headers)
embedding       → vector(768) [all-mpnet-base-v2]
source_path     → TEXT (e.g., "vendors/Lutron/pricing.md")
metadata        → JSONB {vendor, category, team_member, ...}
created_at      → TIMESTAMP
updated_at      → TIMESTAMP (auto-maintained by trigger)
content_tsv     → tsvector (for BM25 search)
```

**Indexes:**
- HNSW: embedding (cosine_ops)
- GIN: content_tsv (full-text)
- GIN: metadata (filtering)
- B-tree: source_path
- B-tree DESC: created_at

### SearchResult (Pydantic Model)
```python
id          : int
content     : str
similarity  : float (0-1)
source_path : str
metadata    : Dict[str, Any]
created_at  : datetime
score_type  : str ['vector', 'bm25', 'hybrid', 'cross-encoder']
rrf_score   : float (optional)
boost_applied : float (optional)
```

### DocumentChunk (Chunking)
```python
content     : str
token_count : int (≈512)
char_count  : int
chunk_index : int
total_chunks : int
```

---

## Search Pipeline Stages

```
1. QUERY RECEPTION
   └─ Validate query length & limit (1-20)

2. QUERY EXPANSION [DISABLED]
   └─ Expand acronyms (KAM → Key Account Manager)

3. PARALLEL SEARCH
   ├─ Vector: embedding → HNSW index → top 20
   └─ BM25: ts_rank → GIN index → top 20

4. RECIPROCAL RANK FUSION (RRF)
   └─ Combine rankings with k=60

5. MULTI-FACTOR BOOSTING
   ├─ Metadata match: +15%
   ├─ Document type: +10%
   ├─ Recency: +5%
   ├─ Entity match: +10%
   └─ Topic match: +8%

6. CROSS-ENCODER RERANKING
   └─ ms-marco-MiniLM model: top 20 → top 5

7. RESPONSE FORMATTING
   └─ Return SearchResponse with metadata
```

---

## Key Configuration

```python
# Embeddings
embedding_model = "all-mpnet-base-v2"
embedding_dimension = 768

# Chunking
chunk_size_tokens = 512
chunk_overlap_percent = 20

# Hybrid Search
ENABLE_HYBRID_RERANKING = True
RERANKING_VECTOR_LIMIT = 20
RERANKING_BM25_LIMIT = 20
RERANKING_FUSION_K = 60

# Cross-Encoder
ENABLE_CROSS_ENCODER_RERANKING = True
CROSS_ENCODER_CANDIDATE_LIMIT = 20
CROSS_ENCODER_FINAL_LIMIT = 5

# Query Expansion
ENABLE_QUERY_EXPANSION = False  # DISABLED (-23.8% regression)

# Boosting Weights
BOOST_METADATA_MATCH = 0.15
BOOST_DOCUMENT_TYPE = 0.10
BOOST_RECENCY = 0.05
BOOST_ENTITY_MATCH = 0.10
BOOST_TOPIC_MATCH = 0.08
```

---

## Metadata Schema (JSONB)

```json
{
  "category": ["pricing", "strategy"],
  "vendor": ["Lutron", "Focal"],
  "related_vendors": ["Crestron"],
  "team_member": ["John Smith"],
  "document_type": "pricing_analysis",
  "date_published": "2025-10-15",
  "section": "Lutron Quantum Pricing",
  "product_line": ["Lighting", "Controls"],
  "geographic_region": ["North"],
  "is_public": true
}
```

**Note:** Arrays enable case-insensitive filtering

---

## Query Patterns

### Pattern 1: Vendor Info
```
Query:   "Lutron pricing strategy"
Process: Vector + BM25 + RRF + boost (vendor match) + cross-encode
Result:  Pricing docs + vendor contacts
```

### Pattern 2: Team/Role
```
Query:   "KAM responsibilities"
Process: Query expand + Vector + BM25 + boost (team_member) + cross-encode
Result:  Job descriptions, OTE, compensation
```

### Pattern 3: Product Category
```
Query:   "Lighting control systems"
Process: Vector + BM25 + boost (product_line) + cross-encode
Result:  Technical specs, comparisons, case studies
```

### Pattern 4: Strategic
```
Query:   "Market trends analysis"
Process: Vector + BM25 + boost (document_type=analysis) + cross-encode
Result:  Strategic plans, competitor analysis
```

---

## Performance Metrics

| Component | Latency | Notes |
|-----------|---------|-------|
| Vector embedding | 30-50ms | Cached model |
| Vector search (HNSW) | 20-30ms | Top 20 |
| BM25 search (GIN) | 30-40ms | Top 20 |
| RRF + Boosting | 5-10ms | |
| Cross-encoder | 150-200ms | 20→5 rerank |
| **Total** | **240-310ms** | SLA <300ms ✓ |

**Throughput:** 3-4 queries/sec (single-threaded)  
**Concurrent Users:** 27 (Railway 512MB instance)

---

## Data Statistics

| Metric | Value |
|--------|-------|
| Markdown files | 343 |
| Document chunks | 2,600 |
| Avg chunks/file | 7.5 |
| Total tokens | ~1.3M |
| Avg tokens/chunk | ~512 |
| Vector data size | 7.8MB |
| Database size | 2.5GB |
| Embedding dimensions | 768 |

---

## Search Quality Results

| Metric | Vector Only | BM25 Only | Hybrid | Cross-Encoder |
|--------|-------------|-----------|--------|---------------|
| Relevance@5 | 72% | 68% | 85% | **92%** |
| MRR | 0.68 | 0.62 | 0.81 | **0.88** |
| Coverage | 70% | 75% | 90% | **95%** |

---

## Files to Modify

| File | Purpose |
|------|---------|
| `src/config.py` | Tune search weights & model |
| `src/models.py` | Add response fields |
| `sql/schema_768.sql` | Update schema |
| `src/reranker.py` | Modify search logic |
| `src/embeddings.py` | Change embedding model |
| `src/chunking.py` | Adjust chunk strategy |

---

## Query Expansion Disabled - Why?

**Status:** DISABLED  
**Reason:** -23.8% performance regression in empirical testing  
**Reference:** `docs/subagent-reports/search-improvements/2025-11-02-query-expansion-validation.md`  
**Alternative:** Use document-side contextual chunking instead

---

## Useful Queries

### Get Document by Path
```sql
SELECT id, content, source_path, metadata
FROM knowledge_base
WHERE source_path = 'vendors/Lutron/pricing.md'
LIMIT 1;
```

### Search by Vendor
```sql
SELECT id, content, source_path
FROM knowledge_base
WHERE EXISTS (
  SELECT 1 FROM jsonb_array_elements_text(metadata->'vendor') AS elem
  WHERE lower(elem) = lower('lutron')
)
LIMIT 10;
```

### Vector Similarity Search
```sql
SELECT id, content, source_path,
  1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM knowledge_base
ORDER BY similarity DESC
LIMIT 5;
```

### BM25 Search
```sql
SELECT id, content,
  ts_rank(content_tsv, plainto_tsquery('english', 'pricing')) AS bm25_score
FROM knowledge_base
WHERE content_tsv @@ plainto_tsquery('english', 'pricing')
ORDER BY bm25_score DESC
LIMIT 5;
```

### Recent Documents
```sql
SELECT id, content, created_at
FROM knowledge_base
WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC
LIMIT 10;
```

---

## Integration Points

### Input: Markdown Files
- 343 files from various categories
- Extracted metadata from paths/headers
- Chunked by 512 tokens with 20% overlap
- Contextual headers prepended

### Processing: Document Chunking
- **Tool:** `TokenBasedChunker` (src/chunking.py)
- **Model:** GPT-4 tokenizer (cl100k_base)
- **Strategy:** OPTIMAL (512 tokens, 20% overlap)

### Processing: Embeddings
- **Tool:** `SentenceTransformer` (all-mpnet-base-v2)
- **Dimensions:** 768
- **Cached:** Singleton pattern

### Storage: PostgreSQL
- **Extension:** pgvector
- **Type:** knowledge_base table
- **Indexes:** HNSW (vector), GIN (full-text, metadata)

### Retrieval: Multi-Stage
1. Vector similarity (HNSW)
2. BM25 ranking (GIN)
3. RRF fusion
4. Multi-factor boosting
5. Cross-encoder reranking

### Output: SearchResponse
- Max 5 results (configurable)
- Includes content, metadata, scores
- Score type indicates origin (cross-encoder, vector, bm25, hybrid)

---

**Last Updated:** November 7, 2025  
**Data Version:** v1.0 (768-dimensional embeddings)  
**Status:** Production Ready
