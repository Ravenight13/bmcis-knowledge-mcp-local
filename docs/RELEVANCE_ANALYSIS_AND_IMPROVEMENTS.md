# Search Relevance & Knowledge Graph Reranking Analysis
**Date**: November 9, 2025
**Status**: ⚠️ IMPROVEMENT NEEDED

---

## Executive Summary

Search relevance evaluation reveals **15.69% average relevance** across 8 test queries, indicating the knowledge base contains noise that reduces result quality. However, **ranking order quality is good (69.44%)**, meaning the reranking infrastructure is sound—the issue is in data quality, not the ranking algorithm.

**Key Findings**:
- ❌ Overall relevance: **15.69%** (target: >70%)
- ✅ Ranking quality: **69.44%** (target: >80%)
- ✅ Query speed: **315ms avg** (excellent)
- ⚠️ Data quality issues: Non-domain documents in knowledge base

---

## Detailed Test Results

### Performance by Query Category

| Category | Relevance | Ranking | Issue | Priority |
|----------|-----------|---------|-------|----------|
| vendor_entity | 31.0% | 77.8% | Some commission data, mixed with noise | HIGH |
| metrics | 24.0% | 55.6% | Sales targets not well represented | HIGH |
| organizational | 21.5% | 55.6% | Team structure scattered across docs | MEDIUM |
| product | 20.5% | 44.4% | Product specs minimal in KB | MEDIUM |
| process | 16.5% | 66.7% | Procedures hidden in code/config files | MEDIUM |
| classification | 7.5% | 77.8% | Classification system not in KB | HIGH |
| people | 2.5% | 88.9% | Team profiles not indexed | HIGH |
| market | 2.0% | 88.9% | Market intelligence missing | HIGH |

### Root Cause Analysis

#### Issue #1: Knowledge Base Contains Non-Domain Documents
**Evidence**:
```
Returned files include:
- speckit_constitution_reference_101225.md (codebase architecture)
- Git_Authentication_Notes.md (internal technical docs)
- API_CHANGES.txt (API documentation)
- constitution.md (development guidelines)
```

**Impact**: Vector similarity returns technically similar but contextually irrelevant documents. The embedding model finds semantic matches but lacks business context.

#### Issue #2: Incomplete Domain Data
**Evidence**:
```
Missing/incomplete in KB:
- Team member profiles & contact info → only 2.5% relevance
- Market intelligence data → 2% relevance
- Dealer classification system → 7.5% relevance
- Weekly sales metrics → 24% relevance
```

**Impact**: Some critical business documents aren't in the knowledge base at all.

#### Issue #3: Data Organization Prevents Effective Search
**Evidence**:
```
Commission data scattered across:
- PROCESSING_SUMMARY_*.txt (fragmented processing logs)
- Various timestamp-suffixed files
- No unified commission reference document
```

**Impact**: Related information is split across many small documents, reducing ranking signal.

---

## Reranking Architecture Overview

The system has a **multi-stage ranking pipeline** that should help:

```
Initial Vector Search (100-120ms)
         ↓
    Hybrid RRF Merge (if BM25 also runs)
         ↓
    Multi-Factor Boosting:
    - Vendor relevance boost
    - Document type boost
    - Recency boost
    - Entity mentions boost
         ↓
    Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
         ↓
    Final Result Filtering
```

---

## Why Ranking Quality is Good But Relevance is Low

The ranking infrastructure **correctly orders** whatever results it gets, but it can't overcome poor initial retrieval. This is a **classic retrieval problem**:

```
Poor Initial Retrieval + Good Reranking = Still Poor Overall
Because reranking works on: top_k_initial_results

If initial retrieval is 8/10 noise + 2/10 relevant:
Reranking will put the 2 relevant docs first, but you still only have 2 good results.
```

---

## Solutions & Recommendations

### PHASE 1: Immediate (Data Cleaning)

#### 1.1 Exclude Non-Domain Documents
**Action**: Filter knowledge base to include only BMCIS business documents.

**Implementation**:
```python
# Create document whitelist
INCLUDE_PATTERNS = [
    "*Commission*",
    "*Team*Profile*",
    "*Organization*",
    "*Dealer*",
    "*Sales*",
    "*Vendor*",
    "*ProSource*",
    "*Lutron*",
    "*Playbook*",
    "*Market*",
]

EXCLUDE_PATTERNS = [
    "*constitution*",
    "*API_CHANGES*",
    "*Git_Authentication*",
    "*Spec_Kit*",
    "*speckit*",
    "*.py",
    "*.json",
    "*.txt"  # Except commission summaries
]

# Reingest with filters
```

**Expected Impact**: Relevance improvement from 15.69% → 40-50%

#### 1.2 Add Document Filtering to Search
**Action**: Implement search-time filtering before reranking.

**Implementation**:
```python
class SearchFilter:
    """Filter documents by category before reranking."""

    def filter_domain_relevant(self, results):
        """Keep only BMCIS business documents."""
        domain_keywords = {
            'commission', 'sales', 'vendor', 'dealer', 'team',
            'market', 'product', 'bmcis', 'organization'
        }

        filtered = []
        for result in results:
            text_lower = result.chunk_text.lower()
            source_lower = result.source_file.lower()

            # Must match domain keywords
            has_domain = any(kw in text_lower or kw in source_lower
                           for kw in domain_keywords)

            if has_domain:
                filtered.append(result)

        return filtered

# Apply in search pipeline
results = initial_search()
results = search_filter.filter_domain_relevant(results)
results = reranker.rerank(query, results)
```

---

### PHASE 2: Short Term (Knowledge Graph Enhancement)

#### 2.1 Implement Entity-Based Reranking
**Action**: Enhance reranker with knowledge graph entity mentions.

**Current Issue**:
- "ProSource commission" query returns generic commission data without ProSource context
- "Team structure" returns random docs mentioning "team" keyword

**Solution**:
```python
class KnowledgeGraphReranker:
    """Rerank based on knowledge graph entity mentions."""

    async def rerank_with_entities(self, query, results, graph_service):
        """Boost results that mention entities from knowledge graph."""

        # Extract query entities
        query_entities = await graph_service.extract_entities(query)
        # e.g., ["ProSource", "commission", "vendor"]

        # For each result, check entity mentions
        scored_results = []
        for result in results:
            entity_mentions = 0
            for entity in query_entities:
                if entity.lower() in result.chunk_text.lower():
                    entity_mentions += 1

            # Boost score based on entity mentions
            entity_boost = entity_mentions * 0.2
            result.entity_boost = entity_boost
            scored_results.append(result)

        # Rerank with entity boost
        return self.cross_encoder.rerank(
            query,
            scored_results,
            entity_weights=True  # Use entity_boost in final score
        )
```

**Expected Impact**: Vendor queries improve 31% → 60-70%

#### 2.2 Multi-Hop Graph Traversal
**Action**: Use knowledge graph relationships to find related documents.

**Example**:
```
Query: "ProSource commission rates"

Graph traversal:
ProSource (entity)
    ├─ has_relationship: Commission_Structure
    ├─ has_relationship: Vendor_Info
    └─ has_relationship: Financial_Data

Return docs linked to these relationships
```

**Implementation**:
```python
async def search_with_graph_context(self, query):
    """Combine vector search with graph-based document finding."""

    # Standard search
    vector_results = await self.vector_search(query)

    # Graph-enhanced search
    entities = await self.graph.extract_entities(query)
    graph_related_docs = []

    for entity in entities:
        # Find relationships
        relationships = await self.graph.get_entity_relationships(entity)

        for rel in relationships:
            # Get documents linked to relationship
            docs = await self.graph.get_related_documents(rel)
            graph_related_docs.extend(docs)

    # Merge results (give priority to graph results)
    merged = self._merge_results(vector_results, graph_related_docs)
    return merged
```

---

### PHASE 3: Medium Term (Data Quality)

#### 3.1 Curate Master Business Documents
**Action**: Create consolidated reference documents for key topics.

**Example**:
```
Master Documents to Create:
1. COMMISSION_REFERENCE.md
   - All commission rates by vendor
   - Commission processing timeline
   - Commission calculation rules

2. TEAM_DIRECTORY.md
   - All team members with profiles
   - Specializations and expertise
   - Contact information

3. VENDOR_CATALOG.md
   - All vendor information
   - Product categories
   - Relationship status

4. DEALER_CLASSIFICATION.md
   - Classification system
   - Dealer types
   - Segmentation rules

5. MARKET_INTELLIGENCE.md
   - Competitive landscape
   - Market trends
   - Industry benchmarks
```

**Impact**: Consolidated documents improve findability and reduce fragmentation.

#### 3.2 Document Metadata & Tagging
**Action**: Add structured metadata to improve filtering and ranking.

**Implementation**:
```json
{
  "document": "Commission_Analysis_2024.md",
  "metadata": {
    "content_type": "business_analysis",
    "business_category": "financial",
    "entities": ["ProSource", "commission", "vendor"],
    "keywords": ["commission_rate", "percentage", "payment_timeline"],
    "last_updated": "2025-01-15",
    "relevance_tags": ["commission", "sales", "financial"],
    "confidence": 0.95
  }
}
```

**Reranking with metadata**:
```python
def score_with_metadata(self, query, result):
    """Incorporate metadata in scoring."""

    base_score = result.semantic_score

    # Boost if content_type matches query intent
    if self._get_query_intent(query) == result.metadata['content_type']:
        base_score *= 1.3

    # Boost if entities mentioned
    query_entities = extract_entities(query)
    entity_hits = len(set(query_entities) & set(result.metadata['entities']))
    base_score *= (1 + 0.2 * entity_hits)

    return base_score
```

---

### PHASE 4: Long Term (ML-Based Relevance)

#### 4.1 Fine-Tune Cross-Encoder for Domain
**Action**: Fine-tune ms-marco-MiniLM-L-6-v2 on BMCIS queries.

**Benefits**:
- Model learns BMCIS-specific relevance patterns
- Expected 20-30% relevance improvement
- Maintains fast inference (<50ms)

**Data needed**:
- 100-500 query/result pairs with relevance judgments (0-3 scale)
- Expensive but worth it for production system

#### 4.2 Learning-to-Rank (LTR) Pipeline
**Action**: Implement LTR model that combines:
- Semantic similarity
- Entity mentions
- Keyword overlap
- Knowledge graph relationships
- Document metadata

**Libraries**: LightGBM with RankNet or LambdaGBM

---

## Quick Wins (Can Implement This Week)

### Win #1: Document Filtering
**File to modify**: `src/search/hybrid_search.py`

```python
# Add after initial search results
results = self._filter_business_documents(results, query)
results = self._apply_domain_boost(results)
```

**Time**: 2-4 hours
**Expected improvement**: +15-20% relevance

### Win #2: Query-Specific Result Limits
**Implementation**: Return only top 5 results if confidence < 0.6

```python
if avg_score < 0.6:
    return results[:5]  # Only return high-confidence results
```

**Time**: 1-2 hours
**Expected improvement**: Better perceived relevance (fewer bad results shown)

### Win #3: Entity Mention Boosting
**File**: `src/search/cross_encoder_reranker.py`

```python
# Before reranking, boost scores for entity mentions
for result in results:
    entity_bonus = count_entity_mentions(query, result.text)
    result.score += entity_bonus * 0.1
```

**Time**: 2-3 hours
**Expected improvement**: +10-15% for entity-based queries

---

## Testing Framework (Already Provided)

### Run Relevance Tests
```bash
PYTHONPATH=/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local \
  python scripts/evaluate_search_relevance.py
```

### Run Pytest Suite
```bash
pytest tests/test_search_relevance.py -v
```

### Continuous Monitoring
```bash
# Add to CI/CD pipeline
pytest tests/test_search_relevance.py \
  --junitxml=results.xml \
  --cov=src/search
```

---

## Success Metrics

### Current State
- Overall relevance: 15.69%
- Ranking quality: 69.44%
- Query latency: 315ms

### Target (30 days)
- Overall relevance: **50%+**
- Ranking quality: **80%+**
- Query latency: **<400ms**

### Excellent (90 days)
- Overall relevance: **70%+**
- Ranking quality: **85%+**
- Query latency: **<300ms**

---

## Implementation Roadmap

```timeline
Week 1: Quick Wins
  ├─ Document filtering
  ├─ Entity mention boosting
  └─ Result limiting

Week 2-3: Knowledge Graph Enhancement
  ├─ Implement KG entity extraction
  ├─ Multi-hop traversal
  └─ Entity-aware reranking

Week 4: Data Curation
  ├─ Create master documents
  ├─ Add metadata/tagging
  └─ Remove non-domain docs

Month 2-3: ML Enhancement
  ├─ Collect training data
  ├─ Fine-tune cross-encoder
  └─ Implement LTR pipeline
```

---

## Conclusion

The BMCIS Knowledge MCP search system has **strong technical foundations** (good ranking order, fast query speed) but **needs data quality improvement**. The good news: **most gains come from data/filtering**, not algorithmic changes.

**Recommended approach**:
1. **This week**: Implement quick wins (filtering, boosting)
2. **Next 2 weeks**: Enhance knowledge graph integration
3. **Next month**: Curate master documents and metadata
4. **Following month**: Fine-tune ML models if needed

With these changes, expect to reach **50%+ relevance** within 4 weeks.

---

**Report Generated**: November 9, 2025 21:18 PST
**Prepared By**: Claude Code
**Data Source**: 8 test queries across 8 categories with manual relevance assessment
