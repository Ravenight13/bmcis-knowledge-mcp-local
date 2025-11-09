# BMCIS Knowledge MCP - Structural Decomposition & Dependency Graph

**Project:** bmcis-knowledge-mcp-local
**Date:** November 7, 2025
**Purpose:** Define repository structure, module boundaries, and task dependency ordering for Task Master
**Methodology:** RPG (Recursive Problem Generation) - Structural Analysis Phase

---

## 1. REPOSITORY STRUCTURE

### 1.1 Proposed Directory Organization

```
bmcis-knowledge-mcp-local/
├── src/
│   ├── core/                      # Foundation layer (Phase 0)
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── types.py               # Type definitions and schemas
│   │   ├── exceptions.py          # Custom exception hierarchy
│   │   └── logging.py             # Logging configuration
│   │
│   ├── database/                  # Data layer (Phase 1)
│   │   ├── __init__.py
│   │   ├── connection.py          # PostgreSQL connection management
│   │   ├── schema.py              # Schema definitions & migrations
│   │   └── operations.py          # CRUD operations wrapper
│   │
│   ├── embeddings/                # Embedding layer (Phase 1)
│   │   ├── __init__.py
│   │   ├── generator.py           # Embedding generation (all-mpnet-base-v2)
│   │   ├── cache.py               # Model caching & lifecycle
│   │   └── validator.py           # Dimension validation
│   │
│   ├── chunking/                  # Document processing (Phase 1)
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # Token-based chunking (tiktoken)
│   │   ├── strategies.py          # Chunking strategies (OPTIMAL, etc.)
│   │   └── context.py             # Contextual header generation
│   │
│   ├── search/                    # Search engine (Phase 2)
│   │   ├── __init__.py
│   │   ├── vector.py              # Vector similarity search (HNSW)
│   │   ├── bm25.py                # BM25 full-text search (GIN)
│   │   ├── hybrid.py              # RRF fusion logic
│   │   └── boost.py               # Multi-factor boosting
│   │
│   ├── reranking/                 # Cross-encoder layer (Phase 2)
│   │   ├── __init__.py
│   │   ├── cross_encoder.py       # ms-marco-MiniLM-L-6-v2 integration
│   │   └── pipeline.py            # Two-stage ranking pipeline
│   │
│   ├── graph/                     # Knowledge graph (Phase 3)
│   │   ├── __init__.py
│   │   ├── entities.py            # Entity extraction & storage
│   │   ├── relationships.py       # Relationship mapping (JSONB)
│   │   ├── topics.py              # Topic hierarchy
│   │   └── query.py               # Graph traversal queries
│   │
│   ├── ingestion/                 # Data pipeline (Phase 2)
│   │   ├── __init__.py
│   │   ├── loader.py              # Markdown file discovery & loading
│   │   ├── metadata.py            # Metadata extraction
│   │   ├── pipeline.py            # End-to-end ingestion orchestration
│   │   └── stats.py               # Ingestion statistics tracking
│   │
│   ├── validation/                # Truth verification (Phase 3)
│   │   ├── __init__.py
│   │   ├── neon_connector.py      # Neon production DB connection
│   │   ├── comparator.py          # Local vs Neon comparison
│   │   └── metrics.py             # Accuracy metrics (90%+ target)
│   │
│   ├── query/                     # Query processing (Phase 2)
│   │   ├── __init__.py
│   │   ├── parser.py              # Query understanding
│   │   ├── expansion.py           # Query expansion (DISABLED but ready)
│   │   └── router.py              # Query routing to appropriate search
│   │
│   └── server/                    # API/MCP layer (Phase 4)
│       ├── __init__.py
│       ├── mcp_server.py          # FastMCP server implementation
│       ├── handlers.py            # Request handlers
│       └── auth.py                # Authentication (Cloudflare SSO)
│
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests by module
│   │   ├── test_config.py
│   │   ├── test_embeddings.py
│   │   ├── test_chunking.py
│   │   ├── test_search.py
│   │   └── ...
│   ├── integration/               # Integration tests
│   │   ├── test_pipeline.py
│   │   ├── test_search_flow.py
│   │   └── test_graph_queries.py
│   └── fixtures/                  # Test data
│       ├── sample_docs/
│       └── expected_outputs/
│
├── scripts/                       # Utilities & tools
│   ├── migrate_schema.py          # Database migrations
│   ├── ingest_documents.py        # Manual ingestion trigger
│   ├── validate_accuracy.py       # Neon comparison script
│   └── benchmark_search.py        # Performance benchmarking
│
├── sql/                           # Database schemas
│   ├── schema_768.sql             # Main schema (vector(768))
│   ├── indexes.sql                # Index creation (HNSW, GIN, B-tree)
│   └── migrations/                # Schema version migrations
│
├── docs/                          # Documentation
│   ├── architecture/              # Architecture decisions
│   ├── api/                       # API documentation
│   └── subagent-reports/          # Analysis reports (this file)
│
├── .env.example                   # Environment template
├── pyproject.toml                 # Python dependencies & config
├── requirements.txt               # Pinned dependencies
└── README.md                      # Project overview
```

---

## 2. MODULE DEFINITIONS

### 2.1 Foundation Layer (Phase 0)

#### Module: `core/config`
- **Capability Mapping:** Configuration Management (from functional decomposition)
- **Responsibility:** Single source of truth for all application settings
- **File Structure:**
  - `config.py`: Settings class with environment variable loading
- **Exports:**
  - `Settings`: Pydantic settings model
  - `get_settings()`: Singleton settings accessor
- **Dependencies:** NONE (foundation layer)
- **Key Settings:**
  - Database connection strings
  - Model names & dimensions
  - Search parameters (RERANKING_FUSION_K, boost factors)
  - Feature flags (ENABLE_HYBRID_RERANKING, ENABLE_CROSS_ENCODER_RERANKING)

#### Module: `core/types`
- **Capability Mapping:** Type System Foundation
- **Responsibility:** Define all data models, schemas, and type contracts
- **File Structure:**
  - `types.py`: Pydantic models for SearchResult, SearchResponse, DocumentChunk, etc.
- **Exports:**
  - `SearchResult`: Response model
  - `SearchResponse`: Aggregated response
  - `DocumentChunk`: Chunking dataclass
  - `VendorInfo`, `TeamMember`: Domain models
- **Dependencies:** NONE (foundation layer)

#### Module: `core/exceptions`
- **Capability Mapping:** Error Handling Foundation
- **Responsibility:** Define application-specific exception hierarchy
- **File Structure:**
  - `exceptions.py`: Exception classes
- **Exports:**
  - `BMCISException`: Base exception
  - `DatabaseConnectionError`: DB errors
  - `EmbeddingGenerationError`: Model errors
  - `SearchQueryError`: Query errors
- **Dependencies:** NONE (foundation layer)

#### Module: `core/logging`
- **Capability Mapping:** Observability Foundation
- **Responsibility:** Centralized logging configuration
- **File Structure:**
  - `logging.py`: Logging setup & formatters
- **Exports:**
  - `get_logger()`: Logger factory
  - `setup_logging()`: Application-wide logging config
- **Dependencies:** NONE (foundation layer)

---

### 2.2 Core Data Layer (Phase 1)

#### Module: `database/connection`
- **Capability Mapping:** Database Connectivity
- **Responsibility:** PostgreSQL connection pool management & health checks
- **File Structure:**
  - `connection.py`: psycopg3 async connection pool
- **Exports:**
  - `get_connection()`: Connection pool accessor
  - `init_db()`: Initialize database & extensions (pgvector)
  - `health_check()`: Database availability check
- **Dependencies:**
  - `core/config` (connection strings)
  - `core/exceptions` (DatabaseConnectionError)
  - `core/logging` (logging)

#### Module: `database/schema`
- **Capability Mapping:** Schema Management
- **Responsibility:** Define & migrate database schema
- **File Structure:**
  - `schema.py`: Schema definitions, DDL generation
- **Exports:**
  - `create_tables()`: Create knowledge_base table
  - `create_indexes()`: Create HNSW, GIN, B-tree indexes
  - `migrate()`: Apply schema migrations
- **Dependencies:**
  - `database/connection` (execute DDL)
  - `core/config` (embedding dimensions)

#### Module: `embeddings/generator`
- **Capability Mapping:** Embedding Generation
- **Responsibility:** Generate 768-dimensional embeddings using sentence-transformers
- **File Structure:**
  - `generator.py`: Model loading & inference
- **Exports:**
  - `generate_embedding()`: Async embedding generation
  - `generate_embedding_sync()`: Sync embedding generation
  - `batch_generate_embeddings()`: Batch processing
- **Dependencies:**
  - `core/config` (model name, dimensions)
  - `core/exceptions` (EmbeddingGenerationError)
  - `embeddings/cache` (model caching)

#### Module: `embeddings/cache`
- **Capability Mapping:** Model Lifecycle Management
- **Responsibility:** Cache sentence-transformer model in memory
- **File Structure:**
  - `cache.py`: Singleton model cache with LRU eviction
- **Exports:**
  - `get_model()`: Cached model accessor
  - `clear_cache()`: Explicit cache invalidation
- **Dependencies:**
  - `core/config` (model name)

#### Module: `chunking/tokenizer`
- **Capability Mapping:** Document Chunking
- **Responsibility:** Token-based document chunking with tiktoken
- **File Structure:**
  - `tokenizer.py`: Tokenization & chunking logic
- **Exports:**
  - `chunk_document()`: Split document into token-based chunks
  - `count_tokens()`: Token counting utility
- **Dependencies:**
  - `core/config` (chunk_size_tokens, chunk_overlap_percent)
  - `core/types` (DocumentChunk)

#### Module: `chunking/strategies`
- **Capability Mapping:** Chunking Strategy Selection
- **Responsibility:** Define & apply chunking strategies (OPTIMAL, PRECISE, etc.)
- **File Structure:**
  - `strategies.py`: ChunkingStrategy enum & application logic
- **Exports:**
  - `ChunkingStrategy`: Enum of strategies
  - `apply_strategy()`: Apply strategy to document
- **Dependencies:**
  - `chunking/tokenizer` (chunk_document)
  - `core/types` (DocumentChunk)

#### Module: `chunking/context`
- **Capability Mapping:** Contextual Augmentation
- **Responsibility:** Prepend contextual headers to chunks
- **File Structure:**
  - `context.py`: Context header generation
- **Exports:**
  - `add_context_header()`: Prepend context to chunk
  - `extract_metadata()`: Extract metadata from document
- **Dependencies:**
  - `core/types` (DocumentChunk)

---

### 2.3 Search Layer (Phase 2)

#### Module: `search/vector`
- **Capability Mapping:** Vector Similarity Search
- **Responsibility:** Execute cosine similarity search using HNSW index
- **File Structure:**
  - `vector.py`: Vector search queries (pgvector)
- **Exports:**
  - `execute_vector_search()`: Query HNSW index for top-k results
- **Dependencies:**
  - `database/connection` (query execution)
  - `embeddings/generator` (generate query embedding)
  - `core/config` (RERANKING_VECTOR_LIMIT)
  - `core/types` (SearchResult)

#### Module: `search/bm25`
- **Capability Mapping:** BM25 Keyword Search
- **Responsibility:** Execute BM25 full-text search using GIN index
- **File Structure:**
  - `bm25.py`: BM25 search queries (ts_rank)
- **Exports:**
  - `execute_bm25_search()`: Query GIN index for top-k results
- **Dependencies:**
  - `database/connection` (query execution)
  - `core/config` (RERANKING_BM25_LIMIT)
  - `core/types` (SearchResult)

#### Module: `search/hybrid`
- **Capability Mapping:** Hybrid Search Fusion
- **Responsibility:** Combine vector + BM25 results using Reciprocal Rank Fusion
- **File Structure:**
  - `hybrid.py`: RRF fusion algorithm
- **Exports:**
  - `reciprocal_rank_fusion()`: Fuse two result sets
  - `deduplicate_results()`: Remove duplicate results
- **Dependencies:**
  - `search/vector` (vector results)
  - `search/bm25` (BM25 results)
  - `core/config` (RERANKING_FUSION_K)
  - `core/types` (SearchResult)

#### Module: `search/boost`
- **Capability Mapping:** Multi-Factor Boosting
- **Responsibility:** Apply dynamic boost multipliers based on metadata
- **File Structure:**
  - `boost.py`: Boosting logic
- **Exports:**
  - `apply_boosts()`: Apply all boost factors
  - `calculate_metadata_boost()`: Vendor/category boost
  - `calculate_recency_boost()`: Time-based boost
- **Dependencies:**
  - `core/config` (BOOST_METADATA_MATCH, BOOST_RECENCY, etc.)
  - `core/types` (SearchResult)

#### Module: `reranking/cross_encoder`
- **Capability Mapping:** Cross-Encoder Reranking
- **Responsibility:** Rerank results using cross-encoder model
- **File Structure:**
  - `cross_encoder.py`: ms-marco-MiniLM-L-6-v2 integration
- **Exports:**
  - `rerank()`: Rerank results using cross-encoder
  - `get_cross_encoder_model()`: Cached model accessor
- **Dependencies:**
  - `core/config` (CROSS_ENCODER_CANDIDATE_LIMIT, CROSS_ENCODER_FINAL_LIMIT)
  - `core/types` (SearchResult)

#### Module: `reranking/pipeline`
- **Capability Mapping:** Two-Stage Ranking
- **Responsibility:** Orchestrate vector/BM25 → RRF → boost → cross-encoder pipeline
- **File Structure:**
  - `pipeline.py`: End-to-end search pipeline
- **Exports:**
  - `execute_search_pipeline()`: Full search flow
- **Dependencies:**
  - `search/hybrid` (RRF fusion)
  - `search/boost` (boosting)
  - `reranking/cross_encoder` (final reranking)
  - `core/config` (all search settings)
  - `core/types` (SearchResponse)

---

### 2.4 Ingestion Layer (Phase 2)

#### Module: `ingestion/loader`
- **Capability Mapping:** Document Discovery & Loading
- **Responsibility:** Scan directory, load markdown files
- **File Structure:**
  - `loader.py`: File discovery & reading
- **Exports:**
  - `discover_files()`: Recursively find .md files
  - `load_document()`: Read file content
- **Dependencies:**
  - `core/logging` (logging)

#### Module: `ingestion/metadata`
- **Capability Mapping:** Metadata Extraction
- **Responsibility:** Extract metadata from file path, frontmatter, content
- **File Structure:**
  - `metadata.py`: Metadata parsing
- **Exports:**
  - `extract_metadata()`: Parse metadata from document
  - `categorize_by_path()`: Infer category from file path
- **Dependencies:**
  - `core/types` (metadata schemas)

#### Module: `ingestion/pipeline`
- **Capability Mapping:** End-to-End Ingestion
- **Responsibility:** Orchestrate load → chunk → embed → insert pipeline
- **File Structure:**
  - `pipeline.py`: Ingestion orchestration
- **Exports:**
  - `ingest_documents()`: Full ingestion flow
  - `ingest_single_file()`: Ingest one file
- **Dependencies:**
  - `ingestion/loader` (discover & load)
  - `ingestion/metadata` (extract metadata)
  - `chunking/strategies` (chunk documents)
  - `chunking/context` (add context headers)
  - `embeddings/generator` (generate embeddings)
  - `database/connection` (insert chunks)
  - `ingestion/stats` (track metrics)

#### Module: `ingestion/stats`
- **Capability Mapping:** Ingestion Monitoring
- **Responsibility:** Track ingestion performance metrics
- **File Structure:**
  - `stats.py`: Statistics tracking
- **Exports:**
  - `IngestionStats`: Dataclass for stats
  - `track_ingestion()`: Context manager for tracking
- **Dependencies:**
  - `core/types` (IngestionStats dataclass)

---

### 2.5 Query Processing Layer (Phase 2)

#### Module: `query/parser`
- **Capability Mapping:** Query Understanding
- **Responsibility:** Parse & validate user queries
- **File Structure:**
  - `parser.py`: Query parsing & validation
- **Exports:**
  - `parse_query()`: Validate & normalize query
  - `validate_limit()`: Validate result limit
- **Dependencies:**
  - `core/config` (max_search_results)
  - `core/exceptions` (SearchQueryError)

#### Module: `query/expansion`
- **Capability Mapping:** Query Expansion
- **Responsibility:** Expand acronyms, synonyms (DISABLED but ready)
- **File Structure:**
  - `expansion.py`: Query expansion logic
- **Exports:**
  - `expand_query()`: Expand query with synonyms/acronyms
  - `load_expansion_dictionary()`: Load expansion mappings
- **Dependencies:**
  - `core/config` (ENABLE_QUERY_EXPANSION)

#### Module: `query/router`
- **Capability Mapping:** Query Routing
- **Responsibility:** Route query to appropriate search strategy
- **File Structure:**
  - `router.py`: Query routing logic
- **Exports:**
  - `route_query()`: Determine search strategy
- **Dependencies:**
  - `query/parser` (parse query)
  - `core/config` (feature flags)

---

### 2.6 Knowledge Graph Layer (Phase 3)

#### Module: `graph/entities`
- **Capability Mapping:** Entity Extraction
- **Responsibility:** Extract & store entities (vendors, team members, products)
- **File Structure:**
  - `entities.py`: Entity extraction & storage
- **Exports:**
  - `extract_entities()`: NER-based entity extraction
  - `store_entity()`: Store entity in JSONB
- **Dependencies:**
  - `database/connection` (entity storage)
  - `core/types` (VendorInfo, TeamMember)

#### Module: `graph/relationships`
- **Capability Mapping:** Relationship Mapping
- **Responsibility:** Define & query entity relationships
- **File Structure:**
  - `relationships.py`: Relationship graph (JSONB-based)
- **Exports:**
  - `create_relationship()`: Link two entities
  - `query_relationships()`: Traverse relationship graph
- **Dependencies:**
  - `graph/entities` (entity access)
  - `database/connection` (JSONB queries)

#### Module: `graph/topics`
- **Capability Mapping:** Topic Hierarchy
- **Responsibility:** Build & navigate topic hierarchies
- **File Structure:**
  - `topics.py`: Topic hierarchy management
- **Exports:**
  - `build_topic_hierarchy()`: Construct topic tree
  - `query_by_topic()`: Topic-based search
- **Dependencies:**
  - `database/connection` (topic storage)

#### Module: `graph/query`
- **Capability Mapping:** Graph Traversal
- **Responsibility:** Execute graph queries (e.g., "find all docs related to Lutron pricing")
- **File Structure:**
  - `query.py`: Graph query execution
- **Exports:**
  - `traverse_graph()`: Execute graph traversal
  - `find_related_documents()`: Find docs via graph
- **Dependencies:**
  - `graph/entities` (entity access)
  - `graph/relationships` (relationship queries)
  - `search/vector` (semantic augmentation)

---

### 2.7 Validation Layer (Phase 3)

#### Module: `validation/neon_connector`
- **Capability Mapping:** Neon Production DB Access
- **Responsibility:** Connect to Neon PostgreSQL for truth verification
- **File Structure:**
  - `neon_connector.py`: Neon connection & query
- **Exports:**
  - `connect_to_neon()`: Establish Neon connection
  - `query_neon()`: Execute query on Neon DB
- **Dependencies:**
  - `core/config` (Neon connection string)
  - `core/exceptions` (DatabaseConnectionError)

#### Module: `validation/comparator`
- **Capability Mapping:** Local vs Neon Comparison
- **Responsibility:** Compare local search results against Neon ground truth
- **File Structure:**
  - `comparator.py`: Result comparison logic
- **Exports:**
  - `compare_results()`: Compare local vs Neon results
  - `calculate_overlap()`: Compute result overlap
- **Dependencies:**
  - `validation/neon_connector` (Neon queries)
  - `reranking/pipeline` (local search)
  - `validation/metrics` (accuracy calculation)

#### Module: `validation/metrics`
- **Capability Mapping:** Accuracy Metrics
- **Responsibility:** Calculate search accuracy (90%+ target)
- **File Structure:**
  - `metrics.py`: Accuracy metrics calculation
- **Exports:**
  - `calculate_accuracy()`: Compute accuracy score
  - `generate_report()`: Generate validation report
- **Dependencies:**
  - `core/types` (SearchResult)

---

### 2.8 Server Layer (Phase 4)

#### Module: `server/mcp_server`
- **Capability Mapping:** FastMCP Integration
- **Responsibility:** Expose semantic_search via FastMCP protocol
- **File Structure:**
  - `mcp_server.py`: FastMCP server setup
- **Exports:**
  - `app`: FastMCP application instance
  - `semantic_search()`: MCP tool handler
- **Dependencies:**
  - `reranking/pipeline` (search execution)
  - `query/router` (query routing)
  - `core/config` (all settings)
  - `server/auth` (authentication)

#### Module: `server/handlers`
- **Capability Mapping:** Request Handling
- **Responsibility:** Handle MCP requests & responses
- **File Structure:**
  - `handlers.py`: Request/response handlers
- **Exports:**
  - `handle_search_request()`: Process search request
  - `format_response()`: Format SearchResponse
- **Dependencies:**
  - `reranking/pipeline` (search execution)
  - `core/types` (SearchResponse)

#### Module: `server/auth`
- **Capability Mapping:** Authentication
- **Responsibility:** Cloudflare Access SSO integration
- **File Structure:**
  - `auth.py`: Authentication middleware
- **Exports:**
  - `verify_token()`: Verify Cloudflare JWT
  - `require_auth()`: Authentication decorator
- **Dependencies:**
  - `core/config` (auth settings)
  - `core/exceptions` (AuthenticationError)

---

## 3. DEPENDENCY GRAPH

### 3.1 Topological Ordering (Foundation → Application)

```
PHASE 0: FOUNDATION LAYER (No dependencies)
┌────────────────────────────────────────────┐
│ core/config                                │
│ core/types                                 │
│ core/exceptions                            │
│ core/logging                               │
└────────────────────────────────────────────┘
                    ↓
PHASE 1: CORE DATA LAYER (Depends on Phase 0)
┌────────────────────────────────────────────┐
│ database/connection    → core/*            │
│ database/schema        → database/connection, core/config │
│ embeddings/cache       → core/config       │
│ embeddings/generator   → core/*, embeddings/cache │
│ chunking/tokenizer     → core/config, core/types │
│ chunking/strategies    → chunking/tokenizer, core/types │
│ chunking/context       → core/types        │
└────────────────────────────────────────────┘
                    ↓
PHASE 2: SEARCH & INGESTION LAYER (Depends on Phase 0, 1)
┌────────────────────────────────────────────┐
│ search/vector          → database/connection, embeddings/generator, core/* │
│ search/bm25            → database/connection, core/* │
│ search/hybrid          → search/vector, search/bm25, core/* │
│ search/boost           → core/config, core/types │
│ reranking/cross_encoder → core/config, core/types │
│ reranking/pipeline     → search/hybrid, search/boost, reranking/cross_encoder │
│                                                    │
│ ingestion/loader       → core/logging      │
│ ingestion/metadata     → core/types        │
│ ingestion/stats        → core/types        │
│ ingestion/pipeline     → ingestion/loader, ingestion/metadata, chunking/*, │
│                          embeddings/generator, database/connection │
│                                                    │
│ query/parser           → core/config, core/exceptions │
│ query/expansion        → core/config       │
│ query/router           → query/parser, core/config │
└────────────────────────────────────────────┘
                    ↓
PHASE 3: GRAPH & VALIDATION LAYER (Depends on Phase 0, 1, 2)
┌────────────────────────────────────────────┐
│ graph/entities         → database/connection, core/types │
│ graph/relationships    → graph/entities, database/connection │
│ graph/topics           → database/connection │
│ graph/query            → graph/entities, graph/relationships, search/vector │
│                                                    │
│ validation/neon_connector → core/config, core/exceptions │
│ validation/metrics     → core/types        │
│ validation/comparator  → validation/neon_connector, reranking/pipeline, │
│                          validation/metrics │
└────────────────────────────────────────────┘
                    ↓
PHASE 4: SERVER LAYER (Depends on all previous phases)
┌────────────────────────────────────────────┐
│ server/auth            → core/config, core/exceptions │
│ server/handlers        → reranking/pipeline, core/types │
│ server/mcp_server      → reranking/pipeline, query/router, │
│                          server/auth, core/config │
└────────────────────────────────────────────┘
```

---

### 3.2 Explicit Dependency Matrix

| Module | Dependencies |
|--------|--------------|
| **PHASE 0: Foundation** | |
| `core/config` | NONE |
| `core/types` | NONE |
| `core/exceptions` | NONE |
| `core/logging` | NONE |
| **PHASE 1: Core Data** | |
| `database/connection` | `core/config`, `core/exceptions`, `core/logging` |
| `database/schema` | `database/connection`, `core/config` |
| `embeddings/cache` | `core/config` |
| `embeddings/generator` | `core/config`, `core/exceptions`, `embeddings/cache` |
| `chunking/tokenizer` | `core/config`, `core/types` |
| `chunking/strategies` | `chunking/tokenizer`, `core/types` |
| `chunking/context` | `core/types` |
| **PHASE 2: Search & Ingestion** | |
| `search/vector` | `database/connection`, `embeddings/generator`, `core/config`, `core/types` |
| `search/bm25` | `database/connection`, `core/config`, `core/types` |
| `search/hybrid` | `search/vector`, `search/bm25`, `core/config`, `core/types` |
| `search/boost` | `core/config`, `core/types` |
| `reranking/cross_encoder` | `core/config`, `core/types` |
| `reranking/pipeline` | `search/hybrid`, `search/boost`, `reranking/cross_encoder`, `core/config`, `core/types` |
| `ingestion/loader` | `core/logging` |
| `ingestion/metadata` | `core/types` |
| `ingestion/stats` | `core/types` |
| `ingestion/pipeline` | `ingestion/loader`, `ingestion/metadata`, `ingestion/stats`, `chunking/strategies`, `chunking/context`, `embeddings/generator`, `database/connection` |
| `query/parser` | `core/config`, `core/exceptions` |
| `query/expansion` | `core/config` |
| `query/router` | `query/parser`, `core/config` |
| **PHASE 3: Graph & Validation** | |
| `graph/entities` | `database/connection`, `core/types` |
| `graph/relationships` | `graph/entities`, `database/connection` |
| `graph/topics` | `database/connection` |
| `graph/query` | `graph/entities`, `graph/relationships`, `search/vector` |
| `validation/neon_connector` | `core/config`, `core/exceptions` |
| `validation/metrics` | `core/types` |
| `validation/comparator` | `validation/neon_connector`, `reranking/pipeline`, `validation/metrics` |
| **PHASE 4: Server** | |
| `server/auth` | `core/config`, `core/exceptions` |
| `server/handlers` | `reranking/pipeline`, `core/types` |
| `server/mcp_server` | `reranking/pipeline`, `query/router`, `server/auth`, `core/config` |

---

### 3.3 Critical Path for Minimum Viable Product (MVP)

**Goal:** Enable basic semantic search (no graph, no validation)

```
MVP Critical Path (Phases 0, 1, 2):

1. core/config                   [Phase 0 - 0 dependencies]
2. core/types                    [Phase 0 - 0 dependencies]
3. core/exceptions               [Phase 0 - 0 dependencies]
4. core/logging                  [Phase 0 - 0 dependencies]
   ↓
5. database/connection           [Phase 1 - depends: 1, 3, 4]
6. database/schema               [Phase 1 - depends: 1, 5]
7. embeddings/cache              [Phase 1 - depends: 1]
8. embeddings/generator          [Phase 1 - depends: 1, 3, 7]
9. chunking/tokenizer            [Phase 1 - depends: 1, 2]
10. chunking/strategies          [Phase 1 - depends: 2, 9]
11. chunking/context             [Phase 1 - depends: 2]
    ↓
12. search/vector                [Phase 2 - depends: 1, 2, 5, 8]
13. search/bm25                  [Phase 2 - depends: 1, 2, 5]
14. search/hybrid                [Phase 2 - depends: 1, 2, 12, 13]
15. search/boost                 [Phase 2 - depends: 1, 2]
16. reranking/cross_encoder      [Phase 2 - depends: 1, 2]
17. reranking/pipeline           [Phase 2 - depends: 1, 2, 14, 15, 16]
18. ingestion/loader             [Phase 2 - depends: 4]
19. ingestion/metadata           [Phase 2 - depends: 2]
20. ingestion/stats              [Phase 2 - depends: 2]
21. ingestion/pipeline           [Phase 2 - depends: 5, 8, 10, 11, 18, 19, 20]
22. query/parser                 [Phase 2 - depends: 1, 3]
23. query/router                 [Phase 2 - depends: 1, 22]
    ↓
24. server/auth                  [Phase 4 - depends: 1, 3]
25. server/handlers              [Phase 4 - depends: 2, 17]
26. server/mcp_server            [Phase 4 - depends: 1, 17, 23, 24]
```

**Total MVP modules:** 26 (excludes graph & validation layers)

---

### 3.4 Full Feature Set Critical Path

**Goal:** Complete system with graph & validation

```
Full Critical Path (All Phases):

MVP (modules 1-26) +

27. graph/entities               [Phase 3 - depends: 2, 5]
28. graph/relationships          [Phase 3 - depends: 5, 27]
29. graph/topics                 [Phase 3 - depends: 5]
30. graph/query                  [Phase 3 - depends: 12, 27, 28]
31. validation/neon_connector    [Phase 3 - depends: 1, 3]
32. validation/metrics           [Phase 3 - depends: 2]
33. validation/comparator        [Phase 3 - depends: 17, 31, 32]
```

**Total modules:** 33

---

## 4. VALIDATION CHECKLIST

### 4.1 Circular Dependency Check

```
✓ No circular dependencies detected
✓ All dependencies point toward foundation (core/*)
✓ Phase 0 modules have no dependencies
✓ Each phase only depends on previous phases
✓ No module depends on modules in later phases
```

### 4.2 Single Responsibility Check

```
✓ Each module has one clear responsibility
✓ No module spans multiple functional capabilities
✓ Module boundaries align with data flow diagrams
✓ Public interfaces are minimal and well-defined
```

### 4.3 Dependency Depth Analysis

| Module | Max Dependency Depth | Dependency Chain |
|--------|---------------------|-----------------|
| `core/*` | 0 | NONE |
| `database/connection` | 1 | core/* |
| `embeddings/generator` | 2 | core/* → embeddings/cache |
| `search/vector` | 2 | core/* → database/connection, embeddings/generator |
| `search/hybrid` | 3 | core/* → database/* → search/vector, search/bm25 |
| `reranking/pipeline` | 4 | core/* → database/* → search/* → reranking/cross_encoder |
| `server/mcp_server` | 5 | core/* → database/* → search/* → reranking/* → query/router |

**Max depth:** 5 (server layer)
**Status:** ✓ Acceptable (typical for layered architecture)

### 4.4 Phase Isolation Check

```
✓ Phase 0 modules can be tested independently
✓ Phase 1 modules can be tested with Phase 0 only
✓ Phase 2 modules can be tested with Phase 0, 1 only
✓ Phase 3 modules can be tested with Phase 0, 1, 2 only
✓ Phase 4 modules require all previous phases
```

---

## 5. TASK ORDERING GUIDANCE FOR TASK MASTER

### 5.1 Recommended Implementation Order

#### Sprint 1: Foundation (Phase 0)
```
Task 1.0: Implement core/config
  - Pydantic settings model
  - Environment variable loading
  - Feature flags
  - Validation

Task 1.1: Implement core/types
  - SearchResult, SearchResponse
  - DocumentChunk dataclass
  - VendorInfo, TeamMember
  - ChunkingStrategy enum

Task 1.2: Implement core/exceptions
  - Exception hierarchy
  - Custom exception classes
  - Error messages

Task 1.3: Implement core/logging
  - Logger setup
  - Formatters
  - Log levels
```

#### Sprint 2: Core Data Layer (Phase 1)
```
Task 2.0: Implement database/connection
  - PostgreSQL connection pool
  - Health checks
  - Connection lifecycle

Task 2.1: Implement database/schema
  - knowledge_base table DDL
  - Index creation (HNSW, GIN, B-tree)
  - Migrations

Task 2.2: Implement embeddings/cache
  - Model caching
  - LRU eviction
  - Cache invalidation

Task 2.3: Implement embeddings/generator
  - all-mpnet-base-v2 integration
  - Batch processing
  - Dimension validation

Task 2.4: Implement chunking/tokenizer
  - tiktoken integration
  - Token-based chunking
  - Overlap calculation

Task 2.5: Implement chunking/strategies
  - ChunkingStrategy enum
  - Strategy application
  - OPTIMAL strategy (512 tokens, 20% overlap)

Task 2.6: Implement chunking/context
  - Context header generation
  - Metadata extraction
  - Header prepending
```

#### Sprint 3: Search & Ingestion (Phase 2)
```
Task 3.0: Implement search/vector
  - HNSW index queries
  - Cosine similarity search
  - Top-k retrieval

Task 3.1: Implement search/bm25
  - GIN index queries
  - ts_rank scoring
  - Top-k retrieval

Task 3.2: Implement search/hybrid
  - RRF fusion algorithm
  - Deduplication
  - Rank aggregation

Task 3.3: Implement search/boost
  - Multi-factor boosting
  - Metadata boost
  - Recency boost

Task 3.4: Implement reranking/cross_encoder
  - ms-marco-MiniLM-L-6-v2 integration
  - Two-stage ranking
  - Top-5 selection

Task 3.5: Implement reranking/pipeline
  - End-to-end search flow
  - Stage orchestration
  - Error handling

Task 3.6: Implement ingestion/loader
  - File discovery
  - Markdown loading
  - Category inference

Task 3.7: Implement ingestion/metadata
  - Metadata extraction
  - Frontmatter parsing
  - Path-based categorization

Task 3.8: Implement ingestion/stats
  - Statistics tracking
  - Performance metrics
  - Reporting

Task 3.9: Implement ingestion/pipeline
  - End-to-end ingestion
  - Batch processing
  - Error recovery

Task 3.10: Implement query/parser
  - Query validation
  - Limit validation
  - Normalization

Task 3.11: Implement query/expansion
  - Expansion dictionary
  - Acronym expansion
  - Synonym expansion (DISABLED)

Task 3.12: Implement query/router
  - Query routing logic
  - Feature flag handling
```

#### Sprint 4: Knowledge Graph (Phase 3)
```
Task 4.0: Implement graph/entities
  - Entity extraction
  - Entity storage (JSONB)
  - VendorInfo, TeamMember models

Task 4.1: Implement graph/relationships
  - Relationship creation
  - JSONB-based graph
  - Traversal queries

Task 4.2: Implement graph/topics
  - Topic hierarchy
  - Topic-based search
  - Hierarchical queries

Task 4.3: Implement graph/query
  - Graph traversal
  - Related document queries
  - Semantic augmentation
```

#### Sprint 5: Validation (Phase 3)
```
Task 5.0: Implement validation/neon_connector
  - Neon connection
  - Production DB queries
  - Connection pooling

Task 5.1: Implement validation/metrics
  - Accuracy calculation
  - Relevance metrics
  - Report generation

Task 5.2: Implement validation/comparator
  - Result comparison
  - Overlap calculation
  - 90%+ accuracy verification
```

#### Sprint 6: Server & Deployment (Phase 4)
```
Task 6.0: Implement server/auth
  - Cloudflare Access SSO
  - JWT verification
  - Authentication middleware

Task 6.1: Implement server/handlers
  - Request handling
  - Response formatting
  - Error handling

Task 6.2: Implement server/mcp_server
  - FastMCP server setup
  - semantic_search tool
  - MCP protocol compliance

Task 6.3: Deploy to Railway
  - Docker containerization
  - Environment configuration
  - Health checks
```

---

### 5.2 Dependency Constraints for Task Master

**Critical Rule:** A task can only be started if ALL its dependencies are completed.

#### Examples:

```
✗ INVALID: Start Task 3.0 (search/vector) before Task 2.3 (embeddings/generator)
  Reason: search/vector depends on embeddings/generator

✓ VALID: Start Task 3.0 (search/vector) after Tasks 1.0, 1.1, 2.0, 2.3 complete
  Reason: All dependencies satisfied

✓ VALID: Start Tasks 3.0 (search/vector) and 3.1 (search/bm25) in parallel
  Reason: No dependency between them (both depend on Phase 1)

✗ INVALID: Start Task 3.2 (search/hybrid) before Task 3.0 and 3.1 complete
  Reason: search/hybrid depends on both search/vector and search/bm25
```

---

### 5.3 Parallelization Opportunities

**Phase 0:** All tasks can run in parallel (no dependencies)
```
Task 1.0, 1.1, 1.2, 1.3 → Parallel
```

**Phase 1:** After Phase 0 completes
```
Parallel group 1: Task 2.0, 2.2, 2.4 (independent)
After 2.0: Task 2.1 (depends on database/connection)
After 2.2: Task 2.3 (depends on embeddings/cache)
Parallel group 2: Task 2.5, 2.6 (depend on 2.4 only)
```

**Phase 2:** After Phase 1 completes
```
Parallel group 1: Task 3.0, 3.1 (independent search paths)
After 3.0, 3.1: Task 3.2 (hybrid search)
Parallel group 2: Task 3.3, 3.4 (boosting & reranking)
After 3.2, 3.3, 3.4: Task 3.5 (pipeline orchestration)

Parallel stream (ingestion):
Task 3.6, 3.7, 3.8 → Task 3.9

Parallel stream (query):
Task 3.10, 3.11 → Task 3.12
```

**Phase 3:** After Phase 2 completes
```
Parallel stream (graph):
Task 4.0 → Task 4.1, 4.2 (parallel) → Task 4.3

Parallel stream (validation):
Task 5.0, 5.1 (parallel) → Task 5.2
```

**Phase 4:** After Phases 0-3 complete
```
Task 6.0 (auth) → Task 6.1, 6.2 (parallel) → Task 6.3 (deploy)
```

---

## 6. TESTING STRATEGY BY PHASE

### 6.1 Phase 0: Unit Tests Only
```
tests/unit/test_config.py       → Test settings loading, validation
tests/unit/test_types.py        → Test Pydantic models, schemas
tests/unit/test_exceptions.py   → Test exception hierarchy
tests/unit/test_logging.py      → Test logger setup
```

### 6.2 Phase 1: Unit + Integration Tests
```
Unit:
tests/unit/test_database.py     → Test connection pool, schema
tests/unit/test_embeddings.py   → Test embedding generation
tests/unit/test_chunking.py     → Test chunking strategies

Integration:
tests/integration/test_db_connect.py → Test PostgreSQL connection
tests/integration/test_embedding_db.py → Test embedding insertion
```

### 6.3 Phase 2: Unit + Integration + E2E Tests
```
Unit:
tests/unit/test_search_vector.py   → Test vector search
tests/unit/test_search_bm25.py     → Test BM25 search
tests/unit/test_search_hybrid.py   → Test RRF fusion
tests/unit/test_reranking.py       → Test cross-encoder

Integration:
tests/integration/test_search_flow.py → Test full search pipeline
tests/integration/test_ingestion.py   → Test ingestion pipeline

E2E:
tests/e2e/test_semantic_search.py → Test semantic_search end-to-end
```

### 6.4 Phase 3: Validation Tests
```
Integration:
tests/integration/test_graph_queries.py → Test graph traversal
tests/integration/test_neon_compare.py  → Test Neon comparison

Validation:
tests/validation/test_accuracy.py → Verify 90%+ accuracy vs Neon
```

### 6.5 Phase 4: Server Tests
```
Integration:
tests/integration/test_mcp_server.py → Test FastMCP server
tests/integration/test_auth.py       → Test Cloudflare SSO

E2E:
tests/e2e/test_production_flow.py → Test full production flow
```

---

## 7. FILE ORGANIZATION VALIDATION

### 7.1 Module-to-File Mapping

| Module Path | File Path |
|-------------|-----------|
| `core.config` | `src/core/config.py` |
| `core.types` | `src/core/types.py` |
| `core.exceptions` | `src/core/exceptions.py` |
| `core.logging` | `src/core/logging.py` |
| `database.connection` | `src/database/connection.py` |
| `database.schema` | `src/database/schema.py` |
| `embeddings.generator` | `src/embeddings/generator.py` |
| `embeddings.cache` | `src/embeddings/cache.py` |
| `chunking.tokenizer` | `src/chunking/tokenizer.py` |
| `chunking.strategies` | `src/chunking/strategies.py` |
| `chunking.context` | `src/chunking/context.py` |
| `search.vector` | `src/search/vector.py` |
| `search.bm25` | `src/search/bm25.py` |
| `search.hybrid` | `src/search/hybrid.py` |
| `search.boost` | `src/search/boost.py` |
| `reranking.cross_encoder` | `src/reranking/cross_encoder.py` |
| `reranking.pipeline` | `src/reranking/pipeline.py` |
| `ingestion.loader` | `src/ingestion/loader.py` |
| `ingestion.metadata` | `src/ingestion/metadata.py` |
| `ingestion.pipeline` | `src/ingestion/pipeline.py` |
| `ingestion.stats` | `src/ingestion/stats.py` |
| `query.parser` | `src/query/parser.py` |
| `query.expansion` | `src/query/expansion.py` |
| `query.router` | `src/query/router.py` |
| `graph.entities` | `src/graph/entities.py` |
| `graph.relationships` | `src/graph/relationships.py` |
| `graph.topics` | `src/graph/topics.py` |
| `graph.query` | `src/graph/query.py` |
| `validation.neon_connector` | `src/validation/neon_connector.py` |
| `validation.comparator` | `src/validation/comparator.py` |
| `validation.metrics` | `src/validation/metrics.py` |
| `server.mcp_server` | `src/server/mcp_server.py` |
| `server.handlers` | `src/server/handlers.py` |
| `server.auth` | `src/server/auth.py` |

**Status:** ✓ Clear 1:1 mapping between modules and files

---

### 7.2 Import Validation

#### Valid Import Examples:
```python
# Phase 0 modules (no imports)
# core/config.py
from pydantic_settings import BaseSettings  # External only

# Phase 1 modules (import Phase 0 only)
# database/connection.py
from core.config import get_settings        # ✓ Valid (Phase 0)
from core.exceptions import DatabaseConnectionError  # ✓ Valid (Phase 0)

# Phase 2 modules (import Phase 0, 1 only)
# search/vector.py
from database.connection import get_connection  # ✓ Valid (Phase 1)
from embeddings.generator import generate_embedding  # ✓ Valid (Phase 1)
from core.types import SearchResult  # ✓ Valid (Phase 0)

# reranking/pipeline.py
from search.hybrid import reciprocal_rank_fusion  # ✓ Valid (Phase 2)
from search.boost import apply_boosts  # ✓ Valid (Phase 2)
```

#### Invalid Import Examples:
```python
# ✗ INVALID: Phase 0 importing Phase 1
# core/config.py
from database.connection import get_connection  # ✗ Violates phase order

# ✗ INVALID: Phase 1 importing Phase 2
# embeddings/generator.py
from search.vector import execute_vector_search  # ✗ Violates phase order

# ✗ INVALID: Circular dependency
# search/vector.py
from reranking.pipeline import execute_search_pipeline  # ✗ Circular
# reranking/pipeline.py
from search.vector import execute_vector_search  # ✗ Circular (already imported above)
```

---

## 8. SUMMARY

### 8.1 Key Metrics

- **Total modules:** 33
- **Total phases:** 5 (0-4)
- **MVP modules:** 26 (excludes graph & validation)
- **Foundation modules:** 4 (Phase 0)
- **Max dependency depth:** 5 (server layer)
- **Circular dependencies:** 0 ✓
- **Module responsibility clarity:** 100% ✓

### 8.2 Critical Success Factors

1. **Strict Phase Ordering:** Never violate phase boundaries (Phase N cannot import Phase N+1)
2. **Dependency Completion:** Task cannot start until ALL dependencies complete
3. **Single Responsibility:** Each module has exactly one clear purpose
4. **Testing Isolation:** Each phase can be tested independently with previous phases
5. **Parallelization:** Leverage parallel tasks within phases for speed

### 8.3 Task Master Integration

This document provides:
- ✓ **Clear module boundaries** → Task definitions
- ✓ **Explicit dependencies** → Task ordering constraints
- ✓ **Phase structure** → Sprint planning
- ✓ **Parallelization opportunities** → Task assignment
- ✓ **Testing requirements** → Verification steps

**Recommendation:** Use Section 5.1 (Recommended Implementation Order) as primary input for Task Master PRD parsing.

---

**Document Status:** ✓ Complete
**Validation:** ✓ Passed (no circular dependencies, clear responsibilities, explicit ordering)
**Ready for:** Task Master PRD generation

