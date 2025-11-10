# Codebase Search Results Index

## Overview

This directory contains comprehensive documentation from an in-depth search of the bmcis-knowledge-mcp codebase, focusing on knowledge graph implementation, reranking logic, result ranking/scoring, and relevance metrics.

## Key Documents

### 1. **SEARCH_RESULTS_SUMMARY.txt** (Executive Summary)
**Location**: `/docs/SEARCH_RESULTS_SUMMARY.txt`
**Size**: 312 lines
**Purpose**: High-level overview of all search findings

Contains:
- Knowledge graph implementation details (files, components, performance)
- Reranking logic breakdown (cross-encoder, adaptive pool sizing)
- Result ranking and scoring pipeline (RRF, boosting, scoring hierarchy)
- Hybrid search orchestration overview
- Knowledge graph in MCP tools
- File location summary
- Key findings and integration points

**Read this for**: Quick understanding of what's implemented and where

---

### 2. **CODEBASE_SEARCH_REPORT.md** (Comprehensive Analysis)
**Location**: `/docs/CODEBASE_SEARCH_REPORT.md`
**Size**: 414 lines
**Purpose**: Detailed technical analysis of all components

Contains 10 major sections:
1. Executive Summary
2. Knowledge Graph Implementation (models, service, repository, cache)
3. Reranking Logic (cross-encoder, config, query analysis, candidate selection)
4. Result Ranking & Scoring (RRF algorithm, multi-factor boosting)
5. Hybrid Search Integration
6. Knowledge Graph in MCP Tools
7. Scoring Summary (types and pipeline)
8. File Structure Summary
9. Key Integration Points
10. Extensibility Points

**Read this for**: Deep understanding of architecture, algorithms, and integration

---

### 3. **QUICK_REFERENCE.md** (Developer Guide)
**Location**: `/docs/QUICK_REFERENCE.md`
**Size**: 324 lines
**Purpose**: Practical code examples and quick lookup

Contains:
- Core Concepts explanation
- Key Files & Classes table
- Common Patterns with code examples:
  - Using Knowledge Graph Service
  - Using Hybrid Search
  - Using Cross-Encoder Reranker
  - Using RRF Scorer
- Configuration examples
- Performance targets table
- Troubleshooting guide
- Entity types and relationships
- Debugging and monitoring code samples

**Read this for**: Code examples, configuration, troubleshooting, and developer workflows

---

## Component Breakdown

### Knowledge Graph (3 files, 894 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `src/knowledge_graph/models.py` | 326 | SQLAlchemy ORM models |
| `src/knowledge_graph/graph_service.py` | 418 | Service layer with LRU cache |
| `src/knowledge_graph/query_repository.py` | 150+ | Optimized SQL queries |

**Key Classes**: KnowledgeEntity, EntityRelationship, EntityMention, KnowledgeGraphService

---

### Search & Ranking (9 files, 2800+ lines)

| File | Lines | Purpose |
|------|-------|---------|
| `src/search/hybrid_search.py` | 797 | Main orchestration |
| `src/search/cross_encoder_reranker.py` | 849 | Pair-wise reranking |
| `src/search/rrf.py` | 383 | RRF algorithm |
| `src/search/boosting.py` | 100+ | Multi-factor boosting |
| Other search files | 600+ | Vector/BM25/routing/config |

**Key Classes**: HybridSearch, CrossEncoderReranker, RRFScorer, BoostingSystem, CandidateSelector

---

### MCP Tools (2 files, 500+ lines)
| File | Purpose |
|------|---------|
| `src/mcp/tools/find_vendor_info.py` | Knowledge graph queries via MCP |
| `src/mcp/tools/semantic_search.py` | Hybrid search via MCP |

---

## Key Technologies & Algorithms

### Data Structures
- **Knowledge Graph**: Property graph with typed edges
- **LRU Cache**: In-memory OrderedDict-based cache
- **Search Results**: SearchResult dataclass with multiple score fields

### Algorithms
- **RRF** (Reciprocal Rank Fusion): Merge vector and BM25 results
- **Cross-Encoder**: ms-marco-MiniLM-L-6-v2 for pair-wise relevance
- **Multi-Factor Boosting**: Content-aware ranking boost
- **Adaptive Pool Sizing**: Query complexity-based candidate selection

### Performance Optimizations
- Parallel vector/BM25 search (ThreadPoolExecutor)
- LRU caching with >80% target hit rate
- Progressive disclosure in MCP for token efficiency
- Lazy model loading for cross-encoder

---

## Performance Characteristics

### Graph Operations
- Cache hit: <2 microseconds
- 1-hop query: P50 <5ms, P95 <10ms
- 2-hop query: P50 <20ms, P95 <50ms

### Search Operations
- Vector search: <100ms
- BM25 search: <50ms
- RRF merging: <50ms
- Boosting: <10ms
- Reranking: <200ms
- **End-to-end**: P50 <300ms, P95 <500ms

---

## Integration Map

```
Knowledge Graph (Entity/Relationship storage)
    ↓
KnowledgeGraphService (Cached queries)
    ↓
find_vendor_info MCP Tool (Graph traversal)

Vector Search + BM25 Search (Parallel execution)
    ↓
RRFScorer (Merge with uniform scoring)
    ↓
BoostingSystem (Multi-factor ranking boost)
    ↓
CrossEncoderReranker (Pair-wise relevance refinement)
    ↓
HybridSearch (Main orchestration)
    ↓
semantic_search MCP Tool (Result delivery)
```

---

## How to Use This Documentation

### For Architecture Understanding
1. Start with **SEARCH_RESULTS_SUMMARY.txt** (5 min read)
2. Then read **CODEBASE_SEARCH_REPORT.md** sections 1-4 (20 min read)
3. Reference **QUICK_REFERENCE.md** section "Core Concepts" for details

### For Development
1. Read **QUICK_REFERENCE.md** section "Common Patterns" for code examples
2. Check "Configuration" section for how to configure components
3. Use "Troubleshooting" section when issues arise
4. Reference specific files in CODEBASE_SEARCH_REPORT.md section 8 for implementation details

### For Performance Tuning
1. Review SEARCH_RESULTS_SUMMARY.txt section 3 (Ranking & Scoring)
2. Read CODEBASE_SEARCH_REPORT.md section 9 (Performance Characteristics)
3. Use QUICK_REFERENCE.md "Troubleshooting" section for optimization tips

### For Testing/Debugging
1. Check QUICK_REFERENCE.md "Debugging & Monitoring" section
2. Use code examples to instrument your tests
3. Reference CODEBASE_SEARCH_REPORT.md sections 5-6 for cache/reranker testing

---

## Related Documentation in Repository

### Pre-existing Search Guides
- `docs/rrf-algorithm-guide.md` - Deep dive on RRF algorithm
- `docs/boost-strategies-guide.md` - Boost strategy patterns
- `docs/search-config-reference.md` - Configuration reference
- `docs/performance_optimization_roadmap.md` - Performance improvements

### Implementation Guides
- `docs/task-5-4-implementation-summary.md` - Search pipeline implementation
- `docs/task-5-refinement-orchestration.md` - Orchestration details
- `docs/task-9-2-implementation-guide.md` - MCP tool implementation

### Performance Resources
- `docs/performance_quick_reference.md` - Performance targets and metrics

---

## File Locations (Quick Reference)

**Knowledge Graph**:
```
src/knowledge_graph/
├── models.py                  # Entity, Relationship, Mention models
├── graph_service.py           # Service + LRU cache
├── query_repository.py        # SQL traversal queries
├── cache.py                   # Cache implementation
└── cache_protocol.py          # Cache interface (DI)
```

**Search Pipeline**:
```
src/search/
├── hybrid_search.py           # Main orchestrator
├── rrf.py                     # RRF merging
├── cross_encoder_reranker.py  # Reranking
├── boosting.py                # Multi-factor boosting
├── vector_search.py           # Vector similarity
└── bm25_search.py             # Full-text search
```

**MCP Tools**:
```
src/mcp/tools/
├── find_vendor_info.py        # Knowledge graph queries
└── semantic_search.py         # Hybrid search
```

---

## Search Query Used

**Search Terms**:
- "graph", "relationship", "entity", "rerank"
- "scoring", "ranking"
- "boost", "hybrid search"

**Scope**: `src/` directory, all Python files

**Strategy**: 
1. File pattern matching (glob) for graph/ranking/search files
2. Content searching (grep) for specific keywords
3. Deep analysis of 4 core implementation files
4. Integration analysis of MCP tools

---

## Document Statistics

| Document | Size | Lines | Creation |
|----------|------|-------|----------|
| SEARCH_RESULTS_SUMMARY.txt | 11KB | 312 | Nov 9, 2025 |
| CODEBASE_SEARCH_REPORT.md | 14KB | 414 | Nov 9, 2025 |
| QUICK_REFERENCE.md | 8.4KB | 324 | Nov 9, 2025 |
| **Total** | **33.4KB** | **1,050** | - |

---

## Contact & Questions

For questions about this documentation:
1. Check the specific document's relevant section
2. Reference the code files listed in section 8 of CODEBASE_SEARCH_REPORT.md
3. Review examples in QUICK_REFERENCE.md Common Patterns
4. Check pre-existing docs listed in "Related Documentation in Repository"

---

**Generated**: November 9, 2025
**Project**: bmcis-knowledge-mcp-local
**Directory**: `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/docs/`
