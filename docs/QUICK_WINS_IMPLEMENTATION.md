# Quick Wins Implementation Guide
**Target**: Improve relevance from 15.69% → 35-40% in 1 week

---

## Quick Win #1: Document Type Filtering
**Effort**: 2-4 hours
**Expected impact**: +10-15% relevance

### Problem
Non-domain documents (code, configs, technical notes) are returned as results.

### Solution: Domain-Aware Filter

Edit `src/search/hybrid_search.py`:

```python
def _filter_business_documents(self, results: list, query: str) -> list:
    """Filter out non-business documents that don't match business context."""

    # Business document patterns
    BUSINESS_KEYWORDS = {
        "commission", "sales", "vendor", "dealer", "team",
        "market", "product", "bmcis", "organization", "customer",
        "revenue", "forecast", "metric", "kpi", "target", "strategy"
    }

    # Non-business patterns to exclude
    EXCLUDE_PATTERNS = {
        "specification", "api", "git", "authentication",
        "constitution", "code", "error", "traceback",
        "deprecated", "python", "javascript", "docker"
    }

    filtered = []
    query_lower = query.lower()

    for result in results:
        text_lower = result.chunk_text.lower()
        source_lower = result.source_file.lower()

        # Check if document has business context
        has_business_content = any(
            keyword in text_lower or keyword in source_lower
            for keyword in BUSINESS_KEYWORDS
        )

        # Check if document should be excluded
        is_non_business = any(
            pattern in text_lower or pattern in source_lower
            for pattern in EXCLUDE_PATTERNS
        )

        if has_business_content and not is_non_business:
            filtered.append(result)

    # If filtering removes too many results, fall back to original
    if len(filtered) < 3:
        return results

    return filtered


# In search() method, add after initial retrieval:
def search(self, query: str, top_k: int = 10) -> list:
    """Hybrid search with business filtering."""

    # ... existing vector/BM25 search ...
    results, stats = self._execute_search(...)

    # NEW: Apply business document filtering
    results = self._filter_business_documents(results, query)

    return results
```

### Testing
```bash
# Test with vendor query
python -c "
from src.mcp.server import get_hybrid_search
search = get_hybrid_search()
results = search.search('ProSource commission', top_k=10)
print(f'Results: {len(results)}')
for r in results[:3]:
    print(f'  - {r.source_file}: {r.chunk_text[:100]}...')
"
```

---

## Quick Win #2: Entity Mention Boosting in Reranker
**Effort**: 2-3 hours
**Expected impact**: +5-10% relevance for entity queries

### Problem
Documents mentioning query entities aren't prioritized over generic matches.

### Solution: Entity-Aware Reranking

Edit `src/search/cross_encoder_reranker.py`:

```python
def _extract_named_entities(self, text: str) -> set[str]:
    """Extract likely named entities from text."""
    import re

    # Known entities in BMCIS
    BMCIS_ENTITIES = {
        "prosource", "lutron", "legrand", "masimo", "cedia",
        "seura", "josh ai", "k-array", "acoustic innovations",
        "straight wire", "amina audio", "storm audio",
        "cliff clarke", "james copple", "wyatt shanks", "jacob hartmann"
    }

    entities = set()
    text_lower = text.lower()

    for entity in BMCIS_ENTITIES:
        if entity in text_lower:
            entities.add(entity)

    return entities


def rerank_with_entity_boost(
    self, query: str, candidates: list
) -> list[float]:
    """Rerank with entity mention boost.

    Algorithm:
    1. Get base cross-encoder scores
    2. Extract entities from query
    3. For each candidate, count entity mentions
    4. Apply entity boost to scores
    5. Return boosted + normalized scores
    """

    # Extract query entities
    query_entities = self._extract_named_entities(query)

    if not query_entities:
        # No entities in query, use standard reranking
        return self.rerank(query, candidates)

    # Get base scores from cross-encoder
    base_scores = self.rerank(query, candidates)

    # Apply entity boost
    boosted_scores = []
    for candidate, base_score in zip(candidates, base_scores):
        # Count entity mentions in candidate
        candidate_entities = self._extract_named_entities(candidate.chunk_text)
        entity_mentions = len(query_entities & candidate_entities)

        # Boost score: +10% per entity mention (max +50%)
        entity_boost = min(entity_mentions * 0.1, 0.5)
        boosted_score = base_score * (1 + entity_boost)

        boosted_scores.append(boosted_score)

    # Normalize scores back to 0-1 range
    max_score = max(boosted_scores) if boosted_scores else 1.0
    normalized = [s / max_score for s in boosted_scores]

    return normalized


# In search pipeline, use entity-aware reranking:
# results = self.reranker.rerank_with_entity_boost(query, results)
```

### Testing
```bash
python -c "
from src.search.cross_encoder_reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()

query = 'ProSource commission structure'
candidates = [
    'ProSource offers tiered commission rates by product category',
    'General vendor commission information not specific to ProSource',
    'ProSource financial performance and commission history'
]

scores = reranker.rerank_with_entity_boost(query, candidates)
print('Scores with entity boost:')
for i, (candidate, score) in enumerate(zip(candidates, scores), 1):
    print(f'{i}. [{score:.3f}] {candidate[:60]}...')
"
```

---

## Quick Win #3: Confidence-Based Result Limiting
**Effort**: 1-2 hours
**Expected impact**: +3-5% perceived relevance (fewer bad results shown)

### Problem
Low-confidence results shouldn't be shown to users.

### Solution: Adaptive Result Limiting

Edit `src/search/results.py`:

```python
def apply_confidence_filtering(
    self,
    results: list,
    query: str,
    min_confidence: float = 0.5
) -> list:
    """Return fewer results if confidence is low.

    Logic:
    - If avg_score >= 0.7: return top_k results (high confidence)
    - If avg_score >= 0.5: return top 5 (medium confidence)
    - If avg_score < 0.5: return top 3 (low confidence)
    """

    if not results:
        return results

    # Calculate average confidence
    scores = [
        getattr(r, 'score', getattr(r, 'hybrid_score', 0))
        for r in results
    ]

    # Convert strings to floats if needed
    scores = [float(s) if isinstance(s, str) else s for s in scores]

    avg_score = sum(scores) / len(scores) if scores else 0

    # Adaptive limiting
    if avg_score >= 0.7:
        return results  # High confidence: return all
    elif avg_score >= 0.5:
        return results[:5]  # Medium: return top 5
    else:
        return results[:3]  # Low: return top 3

    return results


# In search pipeline:
def search(self, query: str, top_k: int = 10) -> list:
    """Search with confidence-based limiting."""

    # ... existing search logic ...
    results = self._execute_search(...)

    # NEW: Apply confidence-based limiting
    results = self.apply_confidence_filtering(results, query)

    return results
```

### Testing
```bash
python -c "
from src.mcp.server import get_hybrid_search
search = get_hybrid_search()

# Test low-relevance query
results = search.search('market intelligence competitive analysis', top_k=10)
print(f'Results for low-confidence query: {len(results)} (should be <= 3)')

# Test high-relevance query
results = search.search('ProSource commission', top_k=10)
print(f'Results for high-confidence query: {len(results)} (should be closer to 10)')
"
```

---

## Quick Win #4: Query Expansion for Entity Queries
**Effort**: 1-2 hours
**Expected impact**: +5-8% relevance

### Problem
Entity-based queries ("ProSource") don't find all related content.

### Solution: Query Expansion

Edit `src/search/query_expansion.py`:

```python
class QueryExpander:
    """Expand queries with synonyms and related terms."""

    ENTITY_EXPANSIONS = {
        "prosource": ["ProSource", "pro-source", "prosource vendor"],
        "commission": ["commission", "commission rate", "commission structure", "payment"],
        "dealer": ["dealer", "dealer types", "dealer classification", "customer"],
        "team": ["team", "sales team", "organization", "district"],
        "lutron": ["Lutron", "lutron control", "lighting control"],
    }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""

        expanded = [query]
        query_lower = query.lower()

        for entity, expansions in self.ENTITY_EXPANSIONS.items():
            if entity in query_lower:
                # Add related terms
                expanded.extend([exp for exp in expansions if exp not in query])

        # Return combined query
        return " OR ".join(expanded)


# Usage in search:
def search(self, query: str, top_k: int = 10) -> list:
    """Search with query expansion."""

    query_expander = QueryExpander()
    expanded_query = query_expander.expand_query(query)

    # Search with expanded query
    results = self._execute_search(expanded_query, top_k)

    return results
```

### Testing
```bash
python -c "
from src.search.query_expansion import QueryExpander
expander = QueryExpander()

queries = [
    'ProSource commission',
    'Dealer classification',
    'Lutron control system'
]

for q in queries:
    expanded = expander.expand_query(q)
    print(f'Original: {q}')
    print(f'Expanded: {expanded}')
    print()
"
```

---

## Implementation Checklist

### Week 1: Implement Quick Wins

**Day 1-2: Document Filtering**
- [ ] Edit `src/search/hybrid_search.py`
- [ ] Add `_filter_business_documents()` method
- [ ] Test with 5 sample queries
- [ ] Commit changes

**Day 2-3: Entity Mention Boosting**
- [ ] Edit `src/search/cross_encoder_reranker.py`
- [ ] Add `_extract_named_entities()` method
- [ ] Add `rerank_with_entity_boost()` method
- [ ] Test with entity queries
- [ ] Commit changes

**Day 4: Confidence-Based Filtering**
- [ ] Edit `src/search/results.py`
- [ ] Add `apply_confidence_filtering()` method
- [ ] Integrate into search pipeline
- [ ] Test with various queries
- [ ] Commit changes

**Day 5: Query Expansion**
- [ ] Create `src/search/query_expansion.py`
- [ ] Implement `QueryExpander` class
- [ ] Test with entity-based queries
- [ ] Commit changes

**Day 5-6: Testing & Validation**
- [ ] Run full test suite
- [ ] Run relevance evaluation: `python scripts/evaluate_search_relevance.py`
- [ ] Document improvements
- [ ] Commit evaluation results

---

## Validation

### Before and After
```bash
# Run relevance evaluation before changes
PYTHONPATH=/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local \
  python scripts/evaluate_search_relevance.py | tee baseline.txt

# Implement quick wins...

# Run relevance evaluation after changes
PYTHONPATH=/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local \
  python scripts/evaluate_search_relevance.py | tee improved.txt

# Compare results
diff baseline.txt improved.txt
```

### Expected Results
**Baseline** (current):
- Overall relevance: 15.69%
- Vendor queries: 31%

**After Quick Wins** (target):
- Overall relevance: 30-40%
- Vendor queries: 50-60%
- Organizational queries: 40-50%

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feat/search-relevance-improvements

# Make changes and test
# Day 1
git add src/search/hybrid_search.py
git commit -m "feat: Add business document filtering"

# Day 2-3
git add src/search/cross_encoder_reranker.py
git commit -m "feat: Add entity-aware reranking"

# Day 4
git add src/search/results.py
git commit -m "feat: Add confidence-based result limiting"

# Day 5
git add src/search/query_expansion.py
git commit -m "feat: Add query expansion for entities"

# Day 6
git add docs/QUICK_WINS_IMPLEMENTATION.md
git commit -m "docs: Add quick wins implementation guide"

# Push and create PR
git push origin feat/search-relevance-improvements
```

---

## Success Criteria

| Metric | Current | Target | Success |
|--------|---------|--------|---------|
| Overall relevance | 15.69% | 30%+ | ✅ |
| Vendor queries | 31% | 50%+ | ✅ |
| Organizational | 21.5% | 40%+ | ✅ |
| Process/Procedural | 16.5% | 35%+ | ✅ |
| Query latency | 315ms | <400ms | ✅ |

---

## Notes

- **Document filtering** is the highest-impact change. Implement this first.
- **Entity boosting** helps entity-based queries significantly.
- **Confidence limiting** improves perceived quality (fewer bad results shown).
- **Query expansion** helps but is lowest priority.

**Best approach**: Implement filters + entity boosting this week, then evaluate results.

---

**Created**: November 9, 2025
**Target Completion**: November 15, 2025
**Prepared By**: Claude Code
