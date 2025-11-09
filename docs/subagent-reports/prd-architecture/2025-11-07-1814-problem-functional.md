# bmcis-knowledge-mcp-local: Problem Statement & Functional Decomposition

**Project:** bmcis-knowledge-mcp-local
**Analysis Date:** November 7, 2025
**Document Type:** PRD - Problem Statement & Functional Requirements
**RPG Methodology:** Problem → Users → Metrics → Capabilities

---

## 1. Problem Statement

The BMCIS sales team (27 people) relies on the production bmcis-knowledge-mcp system running on Neon PostgreSQL to search 343 markdown documents (~2,600 chunks) for vendor information, pricing strategies, team contacts, and technical specifications. While the current system achieves 80% semantic search accuracy using hybrid search (vector + BM25 + cross-encoder reranking), this leaves 1 in 5 queries returning suboptimal results—a critical gap for sales scenarios where incorrect vendor pricing or outdated contact information can lead to lost deals or embarrassing client interactions.

The production Neon environment presents three fundamental limitations that prevent optimization and experimentation:

1. **Serverless Architecture Constraints:** Neon's serverless PostgreSQL doesn't allow fine-grained pgvector parameter tuning (HNSW ef_construction, ef_search), preventing experimentation with index optimization that could improve recall without sacrificing latency.

2. **No Safe Testing Environment:** All search quality experiments (query expansion, boosting weights, chunking strategies) must be tested directly against production data, risking degraded user experience during business hours.

3. **Limited Observability:** Neon's free tier provides minimal query performance metrics, making it difficult to identify which specific pipeline stages (embedding generation, HNSW search, BM25 ranking, cross-encoder reranking) contribute most to latency or accuracy issues.

**Target Goal:** Achieve 90%+ semantic search accuracy (up from 80%) while maintaining <300ms p50 latency, validated through A/B testing against production Neon results. The local PostgreSQL environment must support full feature parity with production, plus knowledge graph capabilities for entity relationship mapping (vendors ↔ products ↔ team members).

**Success Criteria:** The local system becomes the primary development and optimization platform, with changes promoted to Neon only after local validation proves >5% accuracy improvement and <10% latency regression.

---

## 2. Target Users

### Primary Users

**Sales Team (27 people)**
- **Need:** Fast, accurate access to vendor information, pricing strategies, product specifications, and team contacts during client calls or proposal preparation.
- **Pain Point:** Current 80% accuracy means critical queries (e.g., "Lutron Quantum pricing tier 3" or "Who handles Focal Point Northeast?") may miss the exact document or return outdated information.
- **Success Metric:** Zero incorrect vendor information incidents per quarter.

**Developers (2-3 active)**
- **Need:** Safe environment to test search quality improvements (query expansion, embedding models, chunking strategies) without risking production stability.
- **Pain Point:** Cannot iterate quickly on search improvements due to lack of local testing environment; changes require direct production deployment.
- **Success Metric:** Reduce search improvement iteration cycle from 2-3 days (production testing) to <2 hours (local validation).

### Secondary Users

**Administrators (1-2 people)**
- **Need:** Manage knowledge base ingestion, monitor search quality metrics, maintain entity relationships (vendor-product mappings).
- **Pain Point:** No centralized dashboard for search quality trends or entity coverage gaps.
- **Success Metric:** Knowledge graph coverage >80% of vendor-product relationships.

---

## 3. Success Metrics

### Primary Metrics (Must-Have)

| Metric | Baseline (Neon Production) | Target (Local System) | Measurement Method |
|--------|---------------------------|----------------------|-------------------|
| **Semantic Search Accuracy** | 80% (Relevance@5) | **90%+** | A/B test 50 queries against Neon |
| **Query Latency (p50)** | 270ms | **<300ms** | PostgreSQL query logging + Python profiling |
| **Query Latency (p95)** | 320ms | **<500ms** | Track worst-case scenarios |
| **Neon Result Parity** | N/A | **100%** | Validate identical results for same query/params |

### Secondary Metrics (Should-Have)

| Metric | Target | Purpose |
|--------|--------|---------|
| **Knowledge Graph Coverage** | >80% entity relationships | Enable "Who works with Lutron?" queries |
| **Search Result Diversity** | <30% duplicate vendors in top 5 | Avoid vendor-specific result bias |
| **Metadata Completeness** | >95% chunks have vendor/category | Improve filtering and boosting |
| **Iteration Velocity** | <2 hours for search improvement cycle | Enable rapid A/B testing |

### Tertiary Metrics (Nice-to-Have)

| Metric | Target | Purpose |
|--------|--------|---------|
| **Index Rebuild Time** | <5 minutes for full re-index | Support daily data updates |
| **Entity Extraction Accuracy** | >85% correct vendor/product extraction | Maintain graph data quality |
| **Cross-Graph Query Latency** | <500ms for 2-hop traversal | "Show Lutron products in Northeast" |

---

## 4. Capability Tree (Functional Decomposition)

**RPG Methodology Note:** Capabilities describe WHAT the system DOES (user-facing behaviors), not HOW it's structured (implementation details). Each capability enumerates features with clear inputs, outputs, and behavior.

---

### Capability 1: Semantic Search & Retrieval

**Purpose:** Enable users to find relevant knowledge base content using natural language queries, combining vector similarity and keyword matching for optimal recall.

#### Feature 1.1: Vector Similarity Search
- **Description:** Search document chunks by semantic similarity using 768-dimensional embeddings (all-mpnet-base-v2 model).
- **Inputs:**
  - User query (string, 1-500 characters)
  - Result limit (integer, 1-20, default: 5)
  - Optional metadata filters (vendor, category, team_member)
- **Outputs:**
  - Top-N SearchResult objects ranked by cosine similarity
  - Each result includes: content, similarity score (0-1), source_path, metadata, created_at
- **Behavior:**
  - Generate query embedding using cached SentenceTransformer model (30-50ms)
  - Execute HNSW index search against knowledge_base.embedding column
  - Return results sorted by descending similarity (1.0 = perfect match, 0.0 = no match)
  - Apply metadata filtering using JSONB containment before ranking
- **Quality Expectation:** Relevance@5 ≥ 72% (baseline from production)

#### Feature 1.2: BM25 Keyword Search
- **Description:** Search document chunks using full-text BM25 ranking for exact keyword matches and rare term boosting.
- **Inputs:**
  - User query (string)
  - Result limit (integer, 1-20, default: 5)
  - Language config (default: 'english')
- **Outputs:**
  - Top-N SearchResult objects ranked by BM25 score
  - Each result includes: content, bm25_score (float), source_path, metadata
- **Behavior:**
  - Convert query to ts_query using plainto_tsquery
  - Execute GIN index search against knowledge_base.content_tsv column
  - Apply ts_rank scoring with default BM25 parameters
  - Return results sorted by descending BM25 score
- **Quality Expectation:** Relevance@5 ≥ 68% (baseline from production)

#### Feature 1.3: Metadata Filtering
- **Description:** Filter search results by structured metadata (vendor, category, team_member, document_type) before ranking.
- **Inputs:**
  - Metadata filters (dict): `{"vendor": ["Lutron"], "category": ["pricing"]}`
  - Base search results (from vector or BM25)
- **Outputs:**
  - Filtered SearchResult list matching ALL specified filters
- **Behavior:**
  - Use JSONB containment operators (`@>`) for array fields
  - Case-insensitive matching using `lower()` on text values
  - Combine filters with AND logic (all must match)
  - Preserve original ranking order within filtered set
- **Performance Target:** <5ms overhead per query

---

### Capability 2: Hybrid Search (Vector + BM25 Fusion)

**Purpose:** Combine vector similarity and BM25 keyword ranking to achieve better recall than either method alone, using Reciprocal Rank Fusion (RRF).

#### Feature 2.1: Reciprocal Rank Fusion (RRF)
- **Description:** Merge ranked lists from vector search and BM25 search using RRF algorithm with configurable k parameter.
- **Inputs:**
  - Vector search results (top 20)
  - BM25 search results (top 20)
  - Fusion parameter k (default: 60)
- **Outputs:**
  - Unified ranked list with RRF scores
  - Each result tagged with score_type='hybrid'
- **Behavior:**
  - Calculate RRF score: `1 / (k + rank)` for each result in both lists
  - Sum scores for results appearing in both lists
  - Sort by descending RRF score
  - Preserve metadata and original similarity/BM25 scores
- **Quality Expectation:** Relevance@5 ≥ 85% (10% improvement over single-method)

**Example:**
```
Query: "Lutron pricing strategy"
Vector results: [doc_A (rank 1), doc_B (rank 2), doc_C (rank 5)]
BM25 results:   [doc_B (rank 1), doc_D (rank 2), doc_A (rank 3)]

RRF scores (k=60):
  doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325 ← Top result
  doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
  doc_D: 1/(60+2) = 0.0161
  doc_C: 1/(60+5) = 0.0154
```

#### Feature 2.2: Multi-Factor Boosting
- **Description:** Apply additive score boosts based on metadata matches, document type, recency, and entity/topic alignment.
- **Inputs:**
  - RRF-ranked results
  - Query metadata (optional): vendor, team_member, product_line
  - Current date (for recency calculation)
- **Outputs:**
  - Boosted SearchResult list with boost_applied field
- **Behavior:**
  - Apply vendor match boost: +15% if result.metadata.vendor matches query context
  - Apply document type boost: +10% if document_type in ['analysis', 'status_report']
  - Apply recency boost: +5% if created_at within 30 days
  - Apply entity match boost: +10% if team_member/product_line matches
  - Apply topic match boost: +8% if category aligns with query intent
  - Sum all applicable boosts (max 48% total)
  - Re-rank by boosted scores
- **Tuning:** All boost weights configurable in src/config.py

**Good Example (Feature Description):**
```
Feature: Vendor Match Boosting
- Description: Increase result score by 15% when document vendor matches query context
- Inputs: SearchResult with metadata.vendor, query context (optional vendor hint)
- Outputs: SearchResult with boost_applied=0.15 and updated final_score
- Behavior: Case-insensitive vendor name matching, applies to all vendor arrays
```

**Bad Example (Too Vague):**
```
Feature: Boosting
- Description: Make results better
- Inputs: Results
- Outputs: Better results
- Behavior: TBD
```

---

### Capability 3: Cross-Encoder Reranking

**Purpose:** Apply transformer-based query-document similarity scoring as final reranking stage to maximize precision@5.

#### Feature 3.1: Cross-Encoder Scoring
- **Description:** Use ms-marco-MiniLM cross-encoder model to score query-document pairs for final top-5 selection.
- **Inputs:**
  - Query string
  - Top 20 candidates from hybrid search + boosting
  - Final result limit (default: 5)
- **Outputs:**
  - Top-5 SearchResult objects with cross-encoder scores
  - Each result tagged with score_type='cross-encoder'
- **Behavior:**
  - Create query-document pairs: [(query, doc1.content), (query, doc2.content), ...]
  - Run cross-encoder model inference (150-200ms for 20 pairs)
  - Score range: -10 to +10 (higher = more relevant)
  - Sort by descending cross-encoder score
  - Return top 5 results
- **Quality Expectation:** Relevance@5 ≥ 92% (+7% over hybrid alone)

#### Feature 3.2: Adaptive Candidate Selection
- **Description:** Dynamically adjust cross-encoder candidate pool size based on query complexity and initial result diversity.
- **Inputs:**
  - Query complexity score (based on token count, entity count)
  - Hybrid search results (variable length)
  - Diversity threshold (default: 0.7 cosine similarity between top results)
- **Outputs:**
  - Optimized candidate pool size (10-30 documents)
- **Behavior:**
  - Simple queries (<5 tokens): Use top 10 candidates
  - Complex queries (>10 tokens): Use top 30 candidates
  - High diversity (avg similarity <0.7): Use top 15 candidates
  - Low diversity (avg similarity >0.9): Expand to top 25 candidates
- **Performance Target:** Reduce cross-encoder latency to 100-150ms (from 150-200ms)

---

### Capability 4: Knowledge Graph Management

**Purpose:** Build and query entity relationships (vendors ↔ products ↔ team members ↔ regions) to enable multi-hop semantic queries.

**Note:** Knowledge graph is implemented using PostgreSQL-native JSONB relationships, not a dedicated graph database, to maintain single-database simplicity.

#### Feature 4.1: Entity Extraction & Linking
- **Description:** Extract structured entities (vendors, products, team members, regions) from document chunks and store relationships in knowledge_entities table.
- **Inputs:**
  - Document chunk content
  - Source metadata (vendor, team_member from path/headers)
- **Outputs:**
  - Extracted entities with types: VENDOR, PRODUCT, TEAM_MEMBER, REGION
  - Relationships: (entity1, relation_type, entity2)
  - Confidence scores (0-1)
- **Behavior:**
  - Use NER model (spaCy or custom) to extract named entities
  - Match extracted entities against known vendor/product lists
  - Create bidirectional relationships: (Lutron, OFFERS, Quantum System), (Quantum System, OFFERED_BY, Lutron)
  - Store in knowledge_entities and entity_relationships tables
  - Link back to source chunks via chunk_id foreign key
- **Quality Target:** >85% extraction accuracy (validated against manual annotations)

**Entity Schema Example:**
```sql
CREATE TABLE knowledge_entities (
  id SERIAL PRIMARY KEY,
  entity_name TEXT NOT NULL,
  entity_type VARCHAR(50), -- VENDOR, PRODUCT, TEAM_MEMBER, REGION
  metadata JSONB,           -- {aliases: [], confidence: 0.95}
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE entity_relationships (
  id SERIAL PRIMARY KEY,
  source_entity_id INT REFERENCES knowledge_entities(id),
  target_entity_id INT REFERENCES knowledge_entities(id),
  relation_type VARCHAR(100), -- OFFERS, WORKS_WITH, LOCATED_IN
  confidence FLOAT,
  source_chunk_id INT REFERENCES knowledge_base(id),
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### Feature 4.2: Graph-Enhanced Search
- **Description:** Expand search results to include related entities and their associated documents.
- **Inputs:**
  - Base search results (from semantic search)
  - Expansion depth (1-hop or 2-hop, default: 1)
  - Relation types to follow (e.g., OFFERS, WORKS_WITH)
- **Outputs:**
  - Expanded SearchResult list including related entity documents
  - Relationship path metadata (e.g., "Found via: Lutron → OFFERS → Quantum")
- **Behavior:**
  - Extract entities from top search results
  - Query entity_relationships for 1-hop neighbors
  - Retrieve associated chunks via chunk_id foreign keys
  - Merge with original results, preserving ranking
  - Tag expanded results with relationship_path metadata
- **Performance Target:** <500ms for 2-hop traversal + chunk retrieval

**Example Query:**
```
User Query: "Who handles Lutron in the Northeast?"
Step 1: Semantic search finds "Lutron pricing strategy" document
Step 2: Extract entity: "Lutron" (VENDOR)
Step 3: Graph query: (Lutron, MANAGED_BY, ?) → "John Smith" (TEAM_MEMBER)
Step 4: Graph query: (John Smith, COVERS, ?) → "Northeast" (REGION)
Step 5: Return: "John Smith manages Lutron accounts in Northeast region"
```

#### Feature 4.3: Entity Deduplication & Canonicalization
- **Description:** Merge duplicate entities (e.g., "Lutron", "Lutron Electronics") into canonical records with aliases.
- **Inputs:**
  - Raw extracted entities
  - Similarity threshold (default: 0.85 Levenshtein similarity)
- **Outputs:**
  - Canonical entity records with metadata.aliases arrays
- **Behavior:**
  - Compare all entity pairs within same entity_type
  - Merge entities exceeding similarity threshold
  - Store all variants in metadata.aliases
  - Update all relationships to point to canonical entity_id
- **Quality Target:** <5% duplicate entity rate

---

### Capability 5: Cross-Checking & Validation

**Purpose:** Validate local system accuracy against production Neon results to ensure safe promotion of search improvements.

#### Feature 5.1: A/B Search Comparison
- **Description:** Execute identical queries against both local PostgreSQL and production Neon, comparing results for parity.
- **Inputs:**
  - Test query
  - Expected result source (local or Neon)
  - Comparison depth (top-5 or top-20)
- **Outputs:**
  - Comparison report: overlap percentage, rank correlation, unique results
- **Behavior:**
  - Execute query on local system → results_local
  - Execute query on Neon (via MCP client) → results_neon
  - Calculate metrics:
    - Overlap@5: |results_local[:5] ∩ results_neon[:5]| / 5
    - Rank correlation: Kendall's tau on overlapping results
    - Unique to local: results_local - results_neon
    - Unique to Neon: results_neon - results_local
  - Flag queries with <80% overlap for manual review
- **Automation:** Run nightly against 50-query test set

#### Feature 5.2: Accuracy Regression Testing
- **Description:** Maintain golden dataset of query-expected_result pairs, validate search quality after each change.
- **Inputs:**
  - Golden dataset (50 queries with manually annotated top-5 results)
  - Current system search implementation
- **Outputs:**
  - Accuracy metrics: Precision@5, Recall@5, MRR, NDCG@5
  - Pass/fail threshold (>90% accuracy = pass)
- **Behavior:**
  - Execute all 50 golden queries
  - Compare results against annotated expected results
  - Calculate metrics:
    - Precision@5: |retrieved ∩ expected| / 5
    - Recall@5: |retrieved ∩ expected| / |expected|
    - MRR: Mean reciprocal rank of first relevant result
    - NDCG@5: Normalized discounted cumulative gain
  - Fail build if accuracy drops >5% from baseline
- **Integration:** Run as pre-commit hook or CI/CD pipeline

---

### Capability 6: Data Ingestion & Chunking

**Purpose:** Transform raw markdown documents into indexed, searchable chunks with extracted metadata and embeddings.

#### Feature 6.1: Document Parsing & Metadata Extraction
- **Description:** Parse markdown files, extract structured metadata from paths/headers/frontmatter.
- **Inputs:**
  - Markdown file path (e.g., "vendors/Lutron/pricing.md")
  - File content (markdown text)
- **Outputs:**
  - Parsed metadata dict: {vendor, category, team_member, document_type, date_published}
  - Cleaned content (HTML/markdown formatting removed)
- **Behavior:**
  - Extract vendor from path: vendors/{vendor_name}/...
  - Extract category from path: {category}/...
  - Parse YAML frontmatter if present
  - Extract team_member from headers (regex: "Contact: {name}")
  - Infer document_type from filename patterns (pricing_*, analysis_*, status_*)
  - Clean markdown: remove headers, links, code blocks (preserve readable text)
- **Quality Target:** >95% metadata completeness

#### Feature 6.2: Token-Based Chunking with Overlap
- **Description:** Split documents into 512-token chunks with 20% overlap, prepending contextual headers.
- **Inputs:**
  - Document content (cleaned markdown)
  - Source metadata (vendor, category, section)
  - Chunk size (default: 512 tokens)
  - Overlap percentage (default: 20%)
- **Outputs:**
  - DocumentChunk objects with content, token_count, chunk_index, total_chunks
  - Each chunk prepended with context: "Document: {vendor} - {category} | Section: {section}\n\n{content}"
- **Behavior:**
  - Tokenize using GPT-4 tokenizer (cl100k_base)
  - Create chunks of 512 tokens with 102-token overlap (20% of 512)
  - Prepend contextual header (vendor, category, section) to each chunk
  - Maintain chunk_index and total_chunks for reconstruction
- **Quality Target:** 95% of chunks within 480-544 token range

**Example:**
```
Original Document: "vendors/Lutron/pricing.md" (2,500 tokens)

Chunk 1 (tokens 0-512):
  "Document: Lutron - Pricing | Section: Overview

   Lutron offers tiered pricing based on project size..."

Chunk 2 (tokens 410-922): [102-token overlap with Chunk 1]
  "Document: Lutron - Pricing | Section: Overview

   ...project size. Tier 1 (under $50k) receives 15% discount..."
```

#### Feature 6.3: Embedding Generation
- **Description:** Generate 768-dimensional vector embeddings for each chunk using all-mpnet-base-v2 model.
- **Inputs:**
  - DocumentChunk content (with contextual headers)
- **Outputs:**
  - 768-dimensional float vector
- **Behavior:**
  - Load cached SentenceTransformer model (all-mpnet-base-v2)
  - Encode chunk content (includes contextual headers)
  - Normalize vector to unit length
  - Cache model in memory (singleton pattern)
- **Performance:** 30-50ms per chunk, batch processing for full ingestion

#### Feature 6.4: Database Insertion with Indexing
- **Description:** Insert chunks, embeddings, and metadata into PostgreSQL knowledge_base table with automatic indexing.
- **Inputs:**
  - DocumentChunk objects
  - Generated embeddings
  - Extracted metadata
- **Outputs:**
  - Inserted database rows with auto-generated IDs
  - Updated indexes (HNSW, GIN, B-tree)
- **Behavior:**
  - Insert into knowledge_base table (id, content, embedding, source_path, metadata, created_at, content_tsv)
  - Auto-update content_tsv using trigger: `to_tsvector('english', content)`
  - Auto-update updated_at trigger
  - HNSW index updates incrementally (no full rebuild needed)
  - GIN indexes update on commit
- **Performance Target:** <5 minutes for full 2,600-chunk ingestion

---

### Capability 7: Query Enhancement (Experimental)

**Purpose:** Test query transformation techniques (expansion, spell correction, entity recognition) to improve recall, with A/B validation against baseline.

#### Feature 7.1: Acronym & Term Expansion
- **Description:** Expand domain-specific acronyms and abbreviations before search (e.g., "KAM" → "Key Account Manager").
- **Inputs:**
  - Raw query string
  - Acronym dictionary (configurable)
- **Outputs:**
  - Expanded query string
- **Behavior:**
  - Match acronyms using word boundary regex: `\bKAM\b`
  - Replace with expanded form: "KAM responsibilities" → "Key Account Manager responsibilities"
  - Preserve original query in metadata for comparison
- **Status:** Currently DISABLED (caused -23.8% accuracy regression in production)
- **Future Work:** Use document-side contextual chunking instead of query-side expansion

**Note:** Query expansion is an experimental capability. Production testing showed:
- Relevance@5 dropped from 92% to 70%
- High false positive rate on ambiguous acronyms
- Better results achieved through contextual chunk headers

#### Feature 7.2: Spell Correction
- **Description:** Detect and correct spelling errors in queries using domain-specific dictionary.
- **Inputs:**
  - Query string
  - Custom dictionary (vendor names, product terms)
- **Outputs:**
  - Corrected query string
  - Confidence score (0-1)
- **Behavior:**
  - Check each term against dictionary using Levenshtein distance
  - Flag terms with distance >2 as potential misspellings
  - Suggest corrections only if confidence >0.8
  - Preserve original query for user confirmation
- **Status:** NOT IMPLEMENTED (future enhancement)

---

## 5. Non-Functional Requirements

### Performance Requirements

| Requirement | Target | Rationale |
|------------|--------|-----------|
| Search latency (p50) | <300ms | Match production SLA |
| Search latency (p95) | <500ms | Acceptable worst-case |
| Concurrent queries | 5-10 q/s | Support 27 users (avg 0.2-0.4 q/s each) |
| Database size | <10GB | Fit on local SSD |
| Index build time | <5 min | Support daily updates |

### Scalability Requirements

| Requirement | Current | Target | Notes |
|------------|---------|--------|-------|
| Document chunks | 2,600 | 5,000 | 2x growth headroom |
| Vector dimensions | 768 | 768-1024 | Support larger models |
| Entity relationships | 0 | 2,000+ | Initial graph population |
| Query test set | 0 | 50-100 | Regression testing |

### Data Quality Requirements

| Requirement | Target | Measurement |
|------------|--------|-------------|
| Metadata completeness | >95% | % chunks with vendor + category |
| Entity extraction accuracy | >85% | Manual annotation validation |
| Graph relationship coverage | >80% | % vendors with product links |
| Duplicate entity rate | <5% | Canonicalization effectiveness |

---

## 6. Good vs. Bad Feature Descriptions

### Good Example: Feature with Clear I/O/Behavior

```
Feature: Metadata Boosting
- Description: Increase search result score by 15% when document vendor matches query context
- Inputs:
  - SearchResult with metadata.vendor (array of strings)
  - Query context (optional vendor hint from user or prior query)
- Outputs:
  - SearchResult with boost_applied=0.15 and recalculated final_score
- Behavior:
  - Case-insensitive vendor name matching using lower()
  - Applies to all vendor names in metadata.vendor array
  - Additive boost (combines with other boosts)
  - No boost if query context has no vendor hint
- Example:
  Query: "pricing strategy" (no vendor) → no boost
  Query: "Lutron pricing" + result.metadata.vendor=["Lutron"] → +15% boost
```

### Bad Example: Vague Feature Description

```
Feature: Search Enhancement
- Description: Make search better
- Inputs: User query
- Outputs: Better results
- Behavior: Improve ranking using AI
```

**Problems:**
- No clear input/output specification
- "Better" is not measurable
- "Using AI" is too vague (which model? what parameters?)
- No example or expected behavior

---

## 7. Capability Priority Matrix

| Capability | Priority | Rationale | MVP Status |
|-----------|----------|-----------|-----------|
| Semantic Search & Retrieval | P0 (Critical) | Core functionality | ✅ Must have |
| Hybrid Search (Vector + BM25) | P0 (Critical) | Proven 85% accuracy | ✅ Must have |
| Cross-Encoder Reranking | P0 (Critical) | Achieves 92% target | ✅ Must have |
| Cross-Checking & Validation | P0 (Critical) | Ensures safe promotion | ✅ Must have |
| Data Ingestion & Chunking | P0 (Critical) | Populates system | ✅ Must have |
| Knowledge Graph Management | P1 (High) | Differentiator vs. Neon | ⚠️ MVP subset (extraction only) |
| Query Enhancement | P2 (Low) | Experimental, risky | ❌ Post-MVP |

---

## 8. Implementation Phasing

### Phase 1: Neon Parity (MVP)
**Goal:** Replicate exact production behavior locally
**Scope:**
- ✅ Capabilities 1-3: Semantic + Hybrid + Cross-Encoder (existing code)
- ✅ Capability 5.1: A/B validation against Neon
- ✅ Capability 6: Ingestion pipeline

**Success Criteria:** 100% result parity on 50-query test set

---

### Phase 2: Search Optimization
**Goal:** Achieve 90%+ accuracy through parameter tuning
**Scope:**
- Boost weight optimization (capability 2.2)
- HNSW index tuning (ef_search, ef_construction)
- Chunking strategy experiments (512 vs 256 tokens)
- Cross-encoder candidate pool optimization (capability 3.2)

**Success Criteria:** >5% accuracy improvement over Neon baseline

---

### Phase 3: Knowledge Graph
**Goal:** Enable multi-hop entity queries
**Scope:**
- Capability 4.1: Entity extraction & linking
- Capability 4.2: Graph-enhanced search (1-hop only)
- Capability 4.3: Entity deduplication

**Success Criteria:** >80% vendor-product relationship coverage

---

### Phase 4: Advanced Query Enhancement
**Goal:** Test query expansion alternatives
**Scope:**
- Capability 7.2: Spell correction
- Capability 7.1: Improved acronym expansion (context-aware)
- Query intent classification

**Success Criteria:** >2% accuracy improvement without latency regression

---

## 9. Technical Constraints & Assumptions

### Constraints

1. **Single Database Requirement:** All data (chunks, embeddings, entities, relationships) must reside in single PostgreSQL instance for operational simplicity.

2. **Neon API Limitations:** Cross-checking against Neon limited to MCP client rate limits (~10 queries/minute).

3. **Model Compatibility:** Must use same embedding model (all-mpnet-base-v2) as production to ensure comparable vector spaces.

4. **Local Resource Limits:** Development on MacBook (16GB RAM, M1/M2) must support full dataset + indexes.

### Assumptions

1. **Data Stability:** Source markdown files change <5% per week, allowing daily ingestion batches.

2. **Query Patterns:** User queries remain similar to historical patterns (vendor/product/team lookups, pricing questions).

3. **Graph Density:** Estimated 2,000-3,000 entity relationships extractable from current 343 documents.

4. **A/B Test Set:** 50-query golden dataset representative of actual user behavior (to be created from query logs).

---

## 10. Success Validation Plan

### Week 1: Neon Parity Validation
- [ ] Ingest full 343-document corpus locally
- [ ] Execute 50 A/B queries comparing local vs. Neon
- [ ] Achieve 100% result parity (identical top-5 for all queries)
- [ ] Measure latency: ensure p50 <300ms, p95 <500ms

### Week 2-3: Search Optimization
- [ ] Run 20 boost weight experiments
- [ ] Test 5 chunking strategies (256/512/1024 tokens, 10%/20%/30% overlap)
- [ ] Optimize HNSW parameters (ef_search: 40/80/100)
- [ ] Validate each change against A/B test set
- [ ] Document accuracy improvements in session handoffs

### Week 4: Knowledge Graph MVP
- [ ] Extract entities from 343 documents
- [ ] Build vendor-product relationships
- [ ] Implement 1-hop graph queries
- [ ] Measure entity extraction accuracy (>85% target)
- [ ] Test graph-enhanced search on 10 example queries

### Week 5: Production Promotion
- [ ] Select top 3 search improvements (>5% accuracy gain)
- [ ] Document changes in PRD addendum
- [ ] Create Neon migration plan
- [ ] Run final A/B validation
- [ ] Deploy to Neon staging environment

---

## 11. Appendix: Reference Materials

### Source Analysis Documents
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/docs/reference/ANALYSIS_INDEX.md` - Complete analysis overview
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/docs/architecture/DATA_MODELS_QUICK_REFERENCE.md` - Schema and metrics
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/docs/architecture/DATA_MODELS_ANALYSIS.md` - Deep technical analysis
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/docs/architecture/DATA_FLOW_DIAGRAM.md` - Pipeline visualizations

### Production System (bmcis-knowledge-mcp)
- Location: `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp`
- Database: Neon PostgreSQL (serverless)
- Current Accuracy: 80% (Relevance@5)
- Current Latency: 270ms (p50), 320ms (p95)

### Key Metrics Summary
- 343 markdown files → 2,600 chunks
- 768-dimensional embeddings (all-mpnet-base-v2)
- 7.8MB vector data size
- 2.5GB total database size
- 27 concurrent users target
- 3-4 queries/sec throughput

---

**Document Status:** Draft v1.0
**Next Steps:** Review with development team, validate capability priorities, begin Phase 1 implementation
**Last Updated:** November 7, 2025
