# BMCIS Knowledge MCP - Data Flow & Architecture Diagrams

## 1. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLAUDE DESKTOP                          │
│                   (27 Sales Team Members)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ semantic_search(query)
                               │
                      ┌────────▼────────┐
                      │   Cloudflare    │
                      │   Access (SSO)  │
                      │ Microsoft 365   │
                      └────────┬────────┘
                               │
                      ┌────────▼────────────────────────────┐
                      │    Railway.app (512MB Instance)    │
                      │  knowledge.bmcis.net               │
                      │                                    │
                      │  ┌──────────────────────────────┐  │
                      │  │  FastMCP Server (Python)     │  │
                      │  │  - semantic_search           │  │
                      │  │  - Hybrid search pipeline    │  │
                      │  │  - Cross-encoder reranking  │  │
                      │  └──────────────────────────────┘  │
                      └────────┬────────────────────────────┘
                               │
                      ┌────────▼─────────────────────────────┐
                      │  Neon PostgreSQL (Launch Tier)      │
                      │  neon.tech                          │
                      │                                     │
                      │  ┌───────────────────────────────┐  │
                      │  │   knowledge_base (2,600 rows) │  │
                      │  │                               │  │
                      │  │  - id (SERIAL PRIMARY KEY)    │  │
                      │  │  - content (TEXT)             │  │
                      │  │  - embedding (vector(768))    │  │
                      │  │  - metadata (JSONB)           │  │
                      │  │  - source_path (TEXT)         │  │
                      │  │  - created_at (TIMESTAMP)     │  │
                      │  │  - content_tsv (tsvector)     │  │
                      │  │                               │  │
                      │  │  INDEXES:                      │  │
                      │  │  - HNSW (embedding)           │  │
                      │  │  - GIN (content_tsv)          │  │
                      │  │  - GIN (metadata)             │  │
                      │  │  - B-tree (source_path)       │  │
                      │  │  - B-tree (created_at DESC)   │  │
                      │  └───────────────────────────────┘  │
                      └────────────────────────────────────┘
```

---

## 2. Query Processing Pipeline

```
USER QUERY
"What's our Lutron pricing strategy?"
         │
         │
    ┌────▼──────────────────────────────────────────┐
    │  STAGE 1: INPUT VALIDATION                    │
    │  - Check query not empty                      │
    │  - Check limit 1-20 (default 5)               │
    │  - Validate constraints                       │
    └────┬──────────────────────────────────────────┘
         │ Valid
         │
    ┌────▼──────────────────────────────────────────┐
    │  STAGE 2: QUERY EXPANSION [DISABLED]          │
    │  - Would expand "KAM" → "Key Account Manager"│
    │  - But: -23.8% regression (empirical test)   │
    │  - Alternative: use contextual chunking      │
    └────┬──────────────────────────────────────────┘
         │ Expanded query (or original if disabled)
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  STAGE 3: PARALLEL SEARCH (Hybrid Approach)                  │
    │                                                               │
    │  ┌─────────────────────────┐  ┌──────────────────────────┐   │
    │  │  VECTOR SEARCH PATH     │  │  BM25 SEARCH PATH       │   │
    │  │                         │  │                          │   │
    │  │ 1. Generate embedding   │  │ 1. Tokenize query       │   │
    │  │    "pricing strategy"   │  │    "pricing" "strategy" │   │
    │  │    (768-dim vector)     │  │                         │   │
    │  │                         │  │ 2. ts_rank() scoring    │   │
    │  │ 2. Query HNSW index     │  │    (English stemming)   │   │
    │  │    cosine similarity    │  │                         │   │
    │  │    ↓                    │  │ 3. GIN index lookup     │   │
    │  │ 3. Return top 20 by     │  │    content_tsv          │   │
    │  │    similarity score     │  │    ↓                    │   │
    │  │                         │  │ 4. Return top 20 by     │   │
    │  │ ┌─────────────────────┐ │  │    BM25 score           │   │
    │  │ │ id  similarity      │ │  │                         │   │
    │  │ │ 42  0.92           │ │  │ ┌─────────────────────┐ │   │
    │  │ │ 45  0.88           │ │  │ │ id  bm25_score     │ │   │
    │  │ │ 51  0.85           │ │  │ │ 42  15.3           │ │   │
    │  │ │ ... (20 total)     │ │  │ │ 67  14.1           │ │   │
    │  │ └─────────────────────┘ │  │ │ 45  13.8           │ │   │
    │  │                         │  │ │ ... (20 total)     │ │   │
    │  │ Latency: 20-30ms        │  │ │                     │ │   │
    │  │ (HNSW index lookup)     │  │ │ Latency: 30-40ms    │ │   │
    │  │                         │  │ │ (GIN index lookup)  │ │   │
    │  │                         │  │ └─────────────────────┘ │   │
    │  └─────────────────────────┘  └──────────────────────────┘   │
    │                                                               │
    │  STAGE 3 OUTPUT: 40 candidates (20 from each path)          │
    └────┬──────────────────────────────────────────────────────────┘
         │ 40 candidates with dual rankings
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  STAGE 4: RECIPROCAL RANK FUSION (RRF)                      │
    │                                                              │
    │  Inputs:                                                    │
    │  ┌──────────────────────┐  ┌──────────────────────────┐   │
    │  │ Vector Results       │  │ BM25 Results            │   │
    │  ├──────────────────────┤  ├──────────────────────────┤   │
    │  │ 1. id=42 (rank 1)   │  │ 1. id=67 (rank 1)       │   │
    │  │ 2. id=45 (rank 2)   │  │ 2. id=42 (rank 3)       │   │
    │  │ 3. id=51 (rank 3)   │  │ 3. id=45 (rank 5)       │   │
    │  └──────────────────────┘  └──────────────────────────┘   │
    │                                                              │
    │  RRF Formula: score = 1/(k+r_vec) + 1/(k+r_bm25)           │
    │  where k=60, r=rank                                        │
    │                                                              │
    │  Example (id=42):                                           │
    │  score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.032         │
    │                                                              │
    │  RRF Output (dedup + resort):                              │
    │  ┌────────────────────────────┐                            │
    │  │ id  rrf_score             │                            │
    │  │ 42  0.032                 │                            │
    │  │ 45  0.029                 │                            │
    │  │ 67  0.027                 │                            │
    │  │ 51  0.023                 │                            │
    │  │ ... (20 total)            │                            │
    │  └────────────────────────────┘                            │
    │                                                              │
    │  Latency: 5ms                                              │
    └────┬──────────────────────────────────────────────────────────┘
         │ 20 deduplicated + re-ranked results
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  STAGE 5: MULTI-FACTOR BOOSTING                             │
    │                                                              │
    │  Apply boost multipliers based on metadata:                │
    │                                                              │
    │  For each result:                                           │
    │    base_score = rrf_score                                   │
    │    IF metadata.vendor contains "Lutron" THEN                │
    │      base_score *= (1 + 0.15) = 1.15x  [boost_metadata]   │
    │    IF document_type == "pricing_analysis" THEN              │
    │      base_score *= (1 + 0.10) = 1.10x  [boost_doc_type]   │
    │    IF created_at > 30 days ago THEN                        │
    │      base_score *= (1 + 0.05) = 1.05x  [boost_recency]    │
    │    IF query contains entity in metadata THEN                │
    │      base_score *= (1 + 0.10) = 1.10x  [boost_entity]     │
    │    IF metadata.topic matches query THEN                    │
    │      base_score *= (1 + 0.08) = 1.08x  [boost_topic]      │
    │                                                              │
    │  Example (id=42):                                           │
    │    0.032 × 1.15 × 1.10 × 1.05 × 1.10 = 0.048             │
    │                                                              │
    │  Boosted Results:                                           │
    │  ┌────────────────────────────┐                            │
    │  │ id  boosted_score boost    │                            │
    │  │ 42  0.048         1.50x    │                            │
    │  │ 45  0.041         1.41x    │                            │
    │  │ 67  0.027         1.00x    │                            │
    │  │ ... (20 total)             │                            │
    │  └────────────────────────────┘                            │
    │                                                              │
    │  Latency: 2-5ms                                            │
    └────┬──────────────────────────────────────────────────────────┘
         │ 20 results with boosted scores
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  STAGE 6: CROSS-ENCODER RERANKING                           │
    │                                                              │
    │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2                │
    │  Training: MS MARCO passage ranking                         │
    │  Size: 90MB (MiniLM, 6 layers)                              │
    │                                                              │
    │  Input: Query + top 20 documents (pairs)                    │
    │  "pricing strategy" + chunk_42_content                      │
    │  "pricing strategy" + chunk_45_content                      │
    │  ... (20 pairs)                                             │
    │                                                              │
    │  Processing:                                                │
    │  - Feed query + document through model                      │
    │  - Output: relevance score [0, 1] per pair                  │
    │  - Sort by cross-encoder score descending                   │
    │                                                              │
    │  Cross-Encoder Scores:                                      │
    │  ┌────────────────────────────────┐                         │
    │  │ id  ce_score  original_boosted │                         │
    │  │ 42  0.94      0.048            │ ← Reranked to #1       │
    │  │ 51  0.92      0.023            │ ← Reranked to #2       │
    │  │ 45  0.87      0.041            │ ← Reranked to #3       │
    │  │ 67  0.82      0.027            │ ← Reranked to #4       │
    │  │ 38  0.78      0.035            │ ← Reranked to #5       │
    │  │ ... (remaining 15 not shown)   │                         │
    │  └────────────────────────────────┘                         │
    │                                                              │
    │  Latency: 150-200ms                                        │
    └────┬──────────────────────────────────────────────────────────┘
         │ Top 5 final results
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  STAGE 7: RESPONSE FORMATTING                               │
    │                                                              │
    │  Convert to SearchResult objects:                          │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │ {                                                   │   │
    │  │   "id": 42,                                         │   │
    │  │   "content": "Lutron quantum series pricing...",   │   │
    │  │   "similarity": 0.94,                               │   │
    │  │   "source_path": "vendors/Lutron/pricing.md",       │   │
    │  │   "metadata": {                                     │   │
    │  │     "vendor": ["Lutron"],                           │   │
    │  │     "category": ["pricing"],                        │   │
    │  │     "document_type": "pricing_analysis"             │   │
    │  │   },                                                │   │
    │  │   "created_at": "2025-10-15T10:30:00",             │   │
    │  │   "score_type": "cross-encoder",                    │   │
    │  │   "rrf_score": 0.032,                               │   │
    │  │   "boost_applied": 1.50                             │   │
    │  │ }                                                   │   │
    │  │ ... (5 total results)                               │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                                                              │
    │  Latency: <1ms                                             │
    └────┬──────────────────────────────────────────────────────────┘
         │
         │
    ┌────▼──────────────────────────────────────────────────────────┐
    │  RESPONSE RETURNED TO CLAUDE DESKTOP                         │
    │                                                              │
    │  SearchResponse:                                            │
    │  {                                                          │
    │    "query": "What's our Lutron pricing strategy?",         │
    │    "results": [ ... 5 SearchResult objects ... ],         │
    │    "total": 5                                              │
    │  }                                                          │
    │                                                              │
    │  TOTAL LATENCY: 240-310ms                                 │
    │  - Stage 1 (validation):      <1ms                        │
    │  - Stage 2 (expansion):       <5ms [DISABLED]             │
    │  - Stage 3 (search):          80-100ms                    │
    │  - Stage 4 (RRF):             5ms                         │
    │  - Stage 5 (boosting):        2-5ms                       │
    │  - Stage 6 (cross-encode):    150-200ms                   │
    │  - Stage 7 (format):          <1ms                        │
    └────────────────────────────────────────────────────────────┘
```

---

## 3. Data Ingestion Pipeline

```
MARKDOWN FILES (343 files)
├── 00_ESSENTIAL_FOR_CLAUDE/
├── 01_EXECUTIVE_SUMMARIES/
├── 02_MEMORY_BANKS/ (Vendors)
├── 03_TEAM_PROFILES/
├── 04_MARKET_STATE/
├── 06_STRATEGIC_PLANS/
├── 08_HISTORICAL_DATA/
├── 09_TECHNICAL_INFRASTRUCTURE/
└── etc.
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 1: FILE DISCOVERY & CATEGORIZATION │
    │                                          │
    │  - Scan directory recursively            │
    │  - Filter *.md files                     │
    │  - Extract category from path            │
    │  - Extract vendor/section from filename  │
    │                                          │
    │  Output: List[Path] with metadata        │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 2: DOCUMENT LOADING                │
    │                                          │
    │  For each file:                          │
    │  - Read content                          │
    │  - Extract frontmatter (if exists)       │
    │  - Parse headers (## Section Name)       │
    │  - Build metadata dict                   │
    │                                          │
    │  Example metadata:                       │
    │  {                                       │
    │    "category": ["pricing"],              │
    │    "vendor": ["Lutron"],                 │
    │    "section": "Quantum Series",          │
    │    "date_published": "2025-10-15"        │
    │  }                                       │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 3: TOKEN-BASED CHUNKING            │
    │                                          │
    │  Strategy: OPTIMAL                       │
    │  - Chunk size: 512 tokens                │
    │  - Overlap: 20% (102 tokens)             │
    │                                          │
    │  Tokenizer: tiktoken (cl100k_base)       │
    │  (GPT-4 tokenizer, proxy for S-BERT)    │
    │                                          │
    │  Example:                                │
    │  Lutron pricing document (2,048 tokens)  │
    │  → 5 chunks of ~512 tokens each         │
    │  with 102-token overlap                  │
    │                                          │
    │  Output:                                 │
    │  [                                       │
    │    DocumentChunk(                        │
    │      content="...",                      │
    │      token_count=512,                    │
    │      char_count=3104,                    │
    │      chunk_index=0,                      │
    │      total_chunks=5                      │
    │    ),                                    │
    │    ... (4 more chunks)                   │
    │  ]                                       │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 4: CONTEXTUAL HEADERS              │
    │                                          │
    │  Prepend semantic context to each chunk: │
    │                                          │
    │  [CONTEXT: Lutron Quantum Pricing]      │
    │  [VENDOR: Lutron]                        │
    │  [CATEGORY: pricing]                     │
    │  [SECTION: Q4 2025 Strategy]            │
    │  [DATE: 2025-10-15]                     │
    │  Original chunk content...               │
    │                                          │
    │  Benefits:                               │
    │  - Embeds semantic context               │
    │  - Improves search relevance             │
    │  - Enables category-based search         │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 5: EMBEDDING GENERATION            │
    │                                          │
    │  Model: all-mpnet-base-v2                │
    │  - Dimensions: 768                       │
    │  - Parameters: 110M                      │
    │  - Speed: ~30-50ms per chunk             │
    │                                          │
    │  Process:                                │
    │  - Tokenize chunk+context                │
    │  - Feed through model                    │
    │  - Output 768-dim vector                 │
    │  - Convert to list[float]                │
    │                                          │
    │  Output per chunk:                       │
    │  embedding = [0.12, -0.34, 0.56, ...]  │
    │             (768 floating point values)  │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 6: DATABASE INSERTION              │
    │                                          │
    │  For each chunk:                         │
    │  INSERT INTO knowledge_base (             │
    │    id,            -- auto-increment      │
    │    content,       -- chunk + headers     │
    │    embedding,     -- vector(768)         │
    │    source_path,   -- "vendors/Lutron..." │
    │    metadata,      -- JSONB               │
    │    created_at,    -- NOW()              │
    │    updated_at     -- NOW()              │
    │  ) VALUES (...)                         │
    │                                          │
    │  Triggers fire:                          │
    │  - knowledge_base_tsvector_trigger()     │
    │    → Populates content_tsv column        │
    │    → English stemming for BM25           │
    │                                          │
    │  Result:                                 │
    │  2,600 rows inserted                     │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────┐
    │  STEP 7: INDEX CREATION                  │
    │                                          │
    │  Parallel index builds:                  │
    │                                          │
    │  1. HNSW Vector Index:                   │
    │     CREATE INDEX ... USING hnsw           │
    │     (m=16, ef_construction=64)           │
    │     Time: ~5-10 minutes                  │
    │                                          │
    │  2. GIN Full-Text Index:                 │
    │     CREATE INDEX ... USING gin           │
    │     ON content_tsv                       │
    │     Time: ~2-3 minutes                   │
    │                                          │
    │  3. GIN Metadata Index:                  │
    │     CREATE INDEX ... USING gin           │
    │     ON metadata                          │
    │     Time: ~1 minute                      │
    │                                          │
    │  4. B-tree Indexes:                      │
    │     - source_path                        │
    │     - created_at DESC                    │
    │     Time: <1 minute each                 │
    │                                          │
    │  Total Index Time: ~10-15 minutes        │
    └───┬──────────────────────────────────────┘
        │
        │
    ┌───▼──────────────────────────────────────────┐
    │  DATABASE READY FOR QUERIES                 │
    │                                              │
    │  Metrics:                                   │
    │  - 343 files ingested                       │
    │  - 2,600 chunks created                     │
    │  - ~1.3M total tokens                       │
    │  - ~7.8MB vector data                       │
    │  - Database size: ~2.5GB                    │
    │  - Indexes: 5 total (HNSW, 3×GIN, 2×B-tree)│
    │  - Ready for semantic search!               │
    └────────────────────────────────────────────┘
```

---

## 4. Index Strategy Visualization

```
KNOWLEDGE_BASE TABLE (2,600 rows)

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Column Indexes:                                            │
│                                                              │
│  PRIMARY KEY:                                              │
│  ┌──────────────┐                                           │
│  │ id (SERIAL)  │────> Auto-increment (1-2600)             │
│  └──────────────┘                                           │
│                                                              │
│  VECTOR INDEX (HNSW):                                      │
│  ┌──────────────────────┐                                   │
│  │ embedding            │ vector(768)                        │
│  │ (HNSW cosin_ops)     │                                   │
│  │                      │ Algorithm: Hierarchical           │
│  │ idx_knowledge_base   │ Navigable Small World             │
│  │ _embedding_hnsw      │ m=16, ef_construction=64         │
│  └──────────────────────┘                                   │
│     ↓                                                        │
│  Query: "pricing strategy"                                  │
│  → Embedding(768-dim) → HNSW search → Top 20 results       │
│  Latency: 20-30ms                                          │
│                                                              │
│  FULL-TEXT INDEX (GIN):                                    │
│  ┌──────────────────────┐                                   │
│  │ content_tsv          │ tsvector                           │
│  │ (GIN)                │                                   │
│  │                      │ Algorithm: Generalized           │
│  │ idx_content_tsv      │ Inverted Index                    │
│  │                      │ English stemming                  │
│  └──────────────────────┘                                   │
│     ↓                                                        │
│  Query: "pricing strategy"                                  │
│  → ts_rank() → GIN lookup → Top 20 results                 │
│  Latency: 30-40ms                                          │
│                                                              │
│  METADATA INDEX (GIN):                                     │
│  ┌──────────────────────┐                                   │
│  │ metadata             │ JSONB                             │
│  │ (GIN)                │                                   │
│  │                      │ Fast array/object containment     │
│  │ idx_knowledge_base   │ lookup                            │
│  │ _metadata            │                                   │
│  └──────────────────────┘                                   │
│     ↓                                                        │
│  Query: Filter by vendor="Lutron"                          │
│  → JSONB array containment → Filtered results             │
│  Latency: <5ms                                             │
│                                                              │
│  SOURCE PATH INDEX (B-tree):                               │
│  ┌──────────────────────┐                                   │
│  │ source_path          │ TEXT                              │
│  │ (B-tree)             │                                   │
│  │                      │ Standard binary tree              │
│  │ idx_knowledge_base   │                                   │
│  │ _source_path         │                                   │
│  └──────────────────────┘                                   │
│     ↓                                                        │
│  Query: Get document by exact path                         │
│  → B-tree lookup → Single result                          │
│  Latency: <2ms                                             │
│                                                              │
│  TIMESTAMP INDEX (B-tree DESC):                            │
│  ┌──────────────────────┐                                   │
│  │ created_at DESC      │ TIMESTAMP                         │
│  │ (B-tree)             │                                   │
│  │                      │ Descending order for recent-first │
│  │ idx_knowledge_base   │                                   │
│  │ _created_at          │                                   │
│  └──────────────────────┘                                   │
│     ↓                                                        │
│  Query: Recent documents (recency boost)                   │
│  → B-tree lookup → Sorted by created_at DESC             │
│  Latency: <2ms                                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Query Performance Summary:
┌────────────────────────────────────────────────────────────┐
│ Index Type      │ Algorithm      │ Latency │ Use Case     │
├────────────────────────────────────────────────────────────┤
│ HNSW            │ Graph-based    │ 20-30ms │ Vector sim   │
│ GIN (full-text) │ Inverted idx   │ 30-40ms │ BM25 keyword │
│ GIN (metadata)  │ Inverted idx   │ <5ms    │ Filtering    │
│ B-tree (path)   │ Binary tree    │ <2ms    │ Exact lookup │
│ B-tree (time)   │ Binary tree    │ <2ms    │ Time range   │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Embedding Model Architecture

```
INPUT TEXT
"Lutron pricing strategy"
       │
       │
   ┌───▼──────────────────────────────┐
   │  SentenceTransformer             │
   │  all-mpnet-base-v2               │
   │                                  │
   │  Architecture: MPNet              │
   │  Parameters: ~110M                │
   │  Hidden size: 768                 │
   │  Num layers: 12                   │
   │  Num attention heads: 12          │
   │  Max seq length: 514 tokens       │
   └───┬──────────────────────────────┘
       │
       │  [Mean pooling over sequence]
       │
       ▼
   768-DIMENSIONAL VECTOR
   [0.12, -0.34, 0.56, ..., -0.23]
       │
       └──> Stored in PostgreSQL as vector(768)
           Stored as float32 array
           Size: 3,072 bytes per embedding
```

---

## 6. Metadata Filtering Example

```
Query + Filters:
"Lutron pricing" + vendor="Lutron"

METADATA COLUMN (JSONB):
{
  "category": ["pricing", "strategy"],
  "vendor": ["Lutron"],
  "section": "Quantum Pricing",
  "team_member": ["John Smith"],
  "document_type": "pricing_analysis",
  "date_published": "2025-10-15"
}

SQL FILTER LOGIC:
WHERE EXISTS (
  SELECT 1 FROM jsonb_array_elements_text(metadata->'vendor') AS elem
  WHERE lower(elem) = lower('Lutron')
)

RESULT:
✓ Row included (matches filter)

Non-matching example:
{
  "vendor": ["Crestron"],
  ...
}
↓
✗ Row excluded (vendor not Lutron)
```

---

## 7. Scoring Evolution Through Pipeline

```
EXAMPLE DOCUMENT (id=42):
Original content: "Lutron quantum series pricing Q4 2025..."

┌─────────────────────────────────────────────────────┐
│ STAGE 3: VECTOR SEARCH                              │
│ Cosine similarity: 0.92                             │
│ (from HNSW index)                                   │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 3: BM25 SEARCH                                │
│ BM25 score: 15.3                                    │
│ (from GIN full-text index)                          │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 4: RECIPROCAL RANK FUSION                     │
│ Vector rank: 1                                      │
│ BM25 rank: 3                                        │
│ RRF score: 1/(60+1) + 1/(60+3) = 0.0317            │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 5: MULTI-FACTOR BOOSTING                      │
│ Base score: 0.0317                                  │
│                                                     │
│ Boost factors:                                      │
│ × (1 + 0.15) = 1.15x  [vendor="Lutron" matches]   │
│ × (1 + 0.10) = 1.10x  [doc_type="pricing_anal"]   │
│ × (1 + 0.05) = 1.05x  [recent < 30 days]         │
│ × (1 + 0.10) = 1.10x  [entity match]              │
│ × (1 + 0.08) = 1.08x  [topic match]               │
│                                                     │
│ Total multiplier: 1.15 × 1.10 × 1.05 × 1.10 ×    │
│                  1.08 = 1.50x                     │
│                                                     │
│ Boosted score: 0.0317 × 1.50 = 0.0476            │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 6: CROSS-ENCODER RERANKING                    │
│ Input pairs:                                        │
│ ("pricing strategy", "Lutron quantum series...") │
│                                                     │
│ Cross-Encoder Score: 0.94                          │
│ (higher = more relevant)                           │
│                                                     │
│ Result: Moved from rank 5 → rank 1                │
└─────────────────────────────────────────────────────┘
                      ↓
         ✓ FINAL RESULT (Rank 1/5)
```

---

**Last Updated:** November 7, 2025  
**Status:** Production Architecture (v1.0)
