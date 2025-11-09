# RRF Algorithm Guide

## Overview

Reciprocal Rank Fusion (RRF) is a data fusion technique that combines rankings from multiple information retrieval systems into a single consolidated ranking. In the context of hybrid search, RRF combines results from vector similarity search and BM25 full-text search into a unified ranking.

RRF is particularly effective when:
- Multiple retrieval systems have different strengths and weaknesses
- You want to leverage both semantic and keyword-based relevance
- Results need to be reproducible and interpretable
- You want a simple, parameter-minimal approach to ranking

## Mathematical Foundation

### Core Formula

The RRF score for a document is calculated as:

```
RRF(d) = ∑ (1 / (k + rank(d)))
```

Where:
- `d` = the document
- `k` = constant parameter (default: 60)
- `rank(d)` = rank of document in a particular ranking (1-indexed)
- The sum is over all ranking systems that returned the document

### Simplified Formula for Two Systems

When combining two ranking systems (vector and BM25), the formula becomes:

```
RRF(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
```

The final normalized RRF score (in range [0, 1]) is:

```
normalized_RRF(d) = RRF(d) / max_RRF
```

Where `max_RRF` is the theoretical maximum RRF score (typically `2/(k+1)` for two systems).

## Example Calculation

Let's walk through a concrete example combining two search results:

### Setup

**Constants:**
- k = 60 (default parameter)
- Vector search returns 3 results
- BM25 search returns 3 results (with some overlap)

**Vector Search Results:**
1. Document A (rank 1)
2. Document B (rank 2)
3. Document C (rank 3)

**BM25 Search Results:**
1. Document B (rank 1)
2. Document A (rank 2)
3. Document D (rank 3)

### Calculation

**Document A:**
```
RRF(A) = 1/(60 + 1) + 1/(60 + 2)
       = 1/61 + 1/62
       = 0.01639 + 0.01613
       = 0.03252
Rank: 1st (highest)
```

**Document B:**
```
RRF(B) = 1/(60 + 2) + 1/(60 + 1)
       = 1/62 + 1/61
       = 0.01613 + 0.01639
       = 0.03252
Rank: 1st (tied with A)
```

**Document C:**
```
RRF(C) = 1/(60 + 3) + 0  (not in BM25 results)
       = 1/63
       = 0.01587
Rank: 3rd
```

**Document D:**
```
RRF(D) = 0 + 1/(60 + 3)  (not in vector results)
       = 1/63
       = 0.01587
Rank: 3rd (tied with C)
```

### Final Ranking (RRF)

1. Document A (0.03252) ✓ Appeared in both lists
2. Document B (0.03252) ✓ Appeared in both lists
3. Document C (0.01587)
4. Document D (0.01587)

**Key Insight:** Documents appearing in both ranking systems are significantly boosted (by 2x relative to single-system matches), which is the primary advantage of RRF.

## The k Parameter

The `k` constant is crucial for RRF performance:

### What k Does

The parameter `k` controls how strongly early-ranked documents are favored:

- **Small k (e.g., 10):** Gives much higher weight to top-ranked documents
  - Differences between ranks are emphasized
  - Top results dominate the final ranking
  - More sensitive to ranking variations

- **Medium k (e.g., 60):** Balanced weighting (recommended default)
  - Good balance between top results and overall ranking
  - Reduces outlier effects
  - Works well for most use cases

- **Large k (e.g., 200):** Flattens the ranking curve
  - More democratic weighting across ranks
  - Reduces impact of being #1 vs #2
  - Better for democratic consensus

### Selecting k for Your Use Case

**Hybrid Search (Default: k = 60)**
```
Rationale:
- Vector search typically returns 10-100 results
- BM25 search typically returns 10-100 results
- k=60 provides good balance between vector and BM25 weights
- Prevents top-heavy bias from either system
```

**More Vector-Heavy (k = 30-40)**
```
Use when:
- Vector embeddings are very high quality
- Query understanding is your primary strength
- You want semantic relevance to dominate
- Example: Product recommendation (semantic understanding critical)
```

**More Balanced (k = 60-80)**
```
Use when:
- Both vector and BM25 are equally reliable
- You want to avoid over-optimizing for one system
- You have diverse document types and query patterns
- Example: General knowledge base search
```

**More BM25-Heavy (k = 100-150)**
```
Use when:
- Exact keyword matches are important
- Documents have good keyword metadata
- You have many narrow, specific queries
- Vector embeddings are less reliable
- Example: Technical documentation (exact terms critical)
```

### k Parameter Guidance

The optimal k value depends on:

| Factor | Guidance |
|--------|----------|
| **Result Set Size** | Smaller k for smaller result sets (10-20 results), larger k for larger sets (100+ results) |
| **System Balance** | Use equal k for balanced systems; adjust if one system is stronger |
| **Query Type** | Use smaller k for broad queries; larger k for specific queries |
| **Domain** | Technical domains benefit from larger k (keyword emphasis); semantic domains from smaller k |
| **User Expectations** | If users expect exact keyword matches: use larger k; if semantic matches acceptable: use smaller k |

## Comparison with Other Ranking Algorithms

### RRF vs Linear Scoring

**Linear Scoring:**
```
score = w_vector * vector_score + w_bm25 * bm25_score
```

**Comparison:**

| Aspect | RRF | Linear Scoring |
|--------|-----|----------------|
| **Parameters** | 1 (k) | 2+ (weights) |
| **Interpretability** | High (rank-based) | Medium (score-based) |
| **Robustness** | Very high (uses relative ranking) | Depends on weight calibration |
| **Normalization** | Automatic | Manual (scores must be normalized) |
| **Performance** | O(n log n) sorting | O(n) linear scoring |
| **Best For** | Multi-system fusion | Known good scorers |

**Example:**

Given:
- Vector score: 0.95 (normalized 0-1)
- BM25 score: 0.42 (normalized 0-1)

Linear scoring: `0.6 * 0.95 + 0.4 * 0.42 = 0.738` (requires weight tuning)

RRF: Ranks are ranked-combined (weights implicitly depend on ranking position)

### RRF vs Multiplicative Scoring

**Multiplicative Scoring:**
```
score = vector_score * bm25_score
```

**Comparison:**

| Aspect | RRF | Multiplicative |
|--------|-----|-----------------|
| **Sensitivity to Weak Scores** | Low (uses ranks) | High (0.3 * 0.5 = 0.15) |
| **Parameter Tuning** | Minimal | None required |
| **Zero Handling** | Graceful | Problematic (0 in either kills score) |
| **Interpretability** | Excellent | Good |
| **Ideal For** | Diverse systems | Systems with complementary strengths |

### RRF vs Cross-Encoder Re-ranking

**When to Use Each:**

**RRF (Preferred for Hybrid Search)**
- Fast inference (no model calls)
- Deterministic and reproducible
- Works with any ranking systems
- Good for general-purpose search
- Performance: <1ms

**Cross-Encoder Re-ranking**
- Superior accuracy (trained model)
- Understands semantic relationships
- Can be slow (model inference)
- Requires labeled training data
- Performance: 10-100ms depending on model

**Recommended Approach:**
1. **Primary:** Use RRF for hybrid search (fast, reliable)
2. **Optional:** Add cross-encoder re-ranking for top-k results (accuracy boost)
3. **Polish:** Apply content-aware boosts for final ranking refinement

## Advantages of RRF

### 1. Robustness Across Systems

RRF works equally well whether:
- Vector embeddings are high-quality
- BM25 indices are comprehensive
- One system significantly outperforms the other

**Why:** Uses relative ranking (position) rather than absolute scores, which are harder to normalize.

### 2. Automatic Normalization

No need to normalize vector scores (0-1) to BM25 scores, or vice versa:

```python
# Without RRF (needs manual normalization)
vector_score_normalized = (vector_score - min_vector) / (max_vector - min_vector)
bm25_score_normalized = (bm25_score - min_bm25) / (max_bm25 - min_bm25)
combined = weight_vector * vector_score_normalized + weight_bm25 * bm25_score_normalized

# With RRF (automatic)
rrf_score = 1/(k + vector_rank) + 1/(k + bm25_rank)
```

### 3. Parameter Minimal

- Only requires tuning one parameter (k)
- Default k=60 works well for most scenarios
- Reduces hyperparameter search space

### 4. Explicit Diversity Boost

Documents appearing in multiple ranking systems get automatic boost:
- If document in both systems: score includes both rankings
- If document in only one system: score from single ranking
- This naturally favors documents with broad appeal

### 5. Rank-Aware Behavior

RRF understands that:
- Rank #1 position is much better than #100
- The difference between #1 and #2 is significant
- The difference between #50 and #51 is minimal

This behavior emerges naturally from the `1/(k + rank)` formula.

## Disadvantages of RRF

### 1. Ignores Score Magnitude

RRF uses ranks, not scores, so:
- Doesn't distinguish between score 0.95 and 0.50 (if both rank #1)
- Can miss cases where one system is highly confident

**Mitigation:** Combine RRF with cross-encoder re-ranking for accuracy-critical applications.

### 2. Assumes Independent Rankings

RRF assumes each system's ranking is independent:
- May not hold if both systems use same embeddings
- If both systems are correlated, information is lost

**Mitigation:** Use diverse embedding models (e.g., vector + keyword-based)

### 3. Sensitive to Missing Results

If a system doesn't return a document:
- RRF gives score 0 for that system
- Document may be penalized if it should appear in both systems

**Mitigation:** Set higher `top_k` values to ensure broader coverage.

## Implementation Details

### Basic RRF Implementation

```python
def calculate_rrf_score(
    vector_rank: int | None,
    bm25_rank: int | None,
    k: int = 60,
) -> float:
    """Calculate RRF score for a document.

    Args:
        vector_rank: Rank in vector search (1-indexed), None if not in results
        bm25_rank: Rank in BM25 search (1-indexed), None if not in results
        k: RRF constant (default 60)

    Returns:
        RRF score in range (0, 1]
    """
    score = 0.0

    if vector_rank is not None:
        score += 1.0 / (k + vector_rank)

    if bm25_rank is not None:
        score += 1.0 / (k + bm25_rank)

    # Normalize to [0, 1] range
    # Maximum possible score is 2/(k+1) for two systems
    max_rrf = 2.0 / (k + 1)
    return min(score / max_rrf, 1.0)
```

### Advanced: RRF with Deduplication

When combining results from multiple systems, deduplication is critical:

```python
def merge_results_with_rrf(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Merge results from two systems using RRF.

    1. Create unified document ID mapping
    2. Assign ranks within each system
    3. Calculate RRF score for each document
    4. Sort by RRF score
    5. Return reranked results
    """
    # Build ranking maps
    vector_ranks = {result.chunk_id: rank
                   for rank, result in enumerate(vector_results, 1)}
    bm25_ranks = {result.chunk_id: rank
                 for rank, result in enumerate(bm25_results, 1)}

    # Get all unique documents
    all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    # Calculate RRF for each
    scored_docs = [
        (
            doc_id,
            calculate_rrf_score(
                vector_ranks.get(doc_id),
                bm25_ranks.get(doc_id),
                k
            )
        )
        for doc_id in all_doc_ids
    ]

    # Sort by RRF score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return reranked results
    return [fetch_result(doc_id) for doc_id, _ in scored_docs]
```

## Performance Characteristics

### Time Complexity

- **Scoring:** O(1) per document (single calculation)
- **Total for n documents:** O(n) = linear
- **Sorting:** O(n log n) (dominated by sorting step)
- **Total:** O(n log n) for hybrid search

### Space Complexity

- **Rank maps:** O(n) for vector results + O(n) for BM25 results
- **Score storage:** O(n) for merged results
- **Total:** O(n) = linear

### Practical Performance

Typical hybrid search with RRF:
- Vector search: 50-100ms
- BM25 search: 20-50ms
- RRF ranking: 1-5ms
- **Total:** 70-155ms (with parallelization: 50-100ms)

## Tuning RRF in Production

### Step 1: Establish Baseline

```python
# Use default k=60
rrf_results = calculate_hybrid_search_with_rrf(query, k=60)
# Measure: precision@10, recall@20, user satisfaction
```

### Step 2: Test k Variations

```python
for k in [30, 45, 60, 75, 100, 150]:
    results = calculate_hybrid_search_with_rrf(query, k=k)
    # Measure quality metrics
    # Track which k performs best
```

### Step 3: Analyze Performance by Query Type

```python
# Segment queries
semantic_queries = [...]  # "find papers about X"
keyword_queries = [...]   # "what version does Y support"
mixed_queries = [...]     # combination

# Test k separately for each
for query_set in [semantic_queries, keyword_queries, mixed_queries]:
    # Find optimal k
    # Different query types may prefer different k
```

### Step 4: Domain-Specific Tuning

For specific domains:
- **APIs:** Use k=100-120 (exact keywords matter)
- **Guides:** Use k=60-80 (balanced approach)
- **Knowledge Base:** Use k=40-60 (semantic + keywords)
- **Code:** Use k=80-100 (exact syntax critical)

### Step 5: Monitor and Adjust

- Track click-through rates by rank position
- Monitor A/B test results
- Adjust k gradually (step size ±10-20)
- Re-evaluate quarterly as index changes

## Common Pitfalls and Solutions

### Pitfall 1: Unfair System Weights

**Problem:** One system returns many results, the other returns few.

**Solution:** Use consistent `top_k` for both systems:
```python
vector_results = vector_search(query, top_k=100)
bm25_results = bm25_search(query, top_k=100)
# Now ranks are comparable
```

### Pitfall 2: Ignoring Deduplication

**Problem:** Same document appears multiple times in results.

**Solution:** Deduplicate by document ID before merging:
```python
all_results = vector_results + bm25_results
unique_results = {r.chunk_id: r for r in all_results}.values()
# Use unique_results for RRF
```

### Pitfall 3: Over-Tuning k

**Problem:** k=73 is "best" for your test set, but doesn't generalize.

**Solution:** Use conservative k values:
- Stick with multiples of 10-20: 40, 60, 80, 100
- k=60 is good default for most cases
- Only tune if you have clear, consistent improvements

### Pitfall 4: Mixing Different Scoring Scales

**Problem:** Vector scores 0-1, BM25 scores 0-1000.

**Solution:** RRF uses ranks (positions), not scores:
```python
# These work equally well with RRF
vector_score: 0.95, rank: 1
bm25_score: 2150, rank: 1

# RRF combines ranks, not scores
rrf_score = 1/(k+1) + 1/(k+1) = same regardless of original scores
```

## References and Further Reading

### Academic Papers

1. Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"
   - Original RRF paper
   - Shows RRF outperforms many sophisticated methods
   - Recommended for theoretical understanding

2. Webber et al. (2010) - "Evaluation Methods for Ranking Aggregation"
   - Comprehensive evaluation of ranking fusion methods
   - Compares RRF to alternatives
   - Good for competitive analysis

### Practical Guides

1. Elastic Search Documentation - "Retrieval Augmented Generation"
   - RRF implementation in production systems
   - Performance benchmarks and tuning guide

2. Pinecone Blog - "Reciprocal Rank Fusion for Semantic Search"
   - RRF in modern vector databases
   - Real-world examples and benchmarks

### Related Techniques

- **Linear Combinations:** Simple weight-based fusion
- **Condorcet Methods:** Voting-based ranking fusion
- **Positional Weights:** Position-aware fusion
- **Cross-Encoder Models:** Learning-based re-ranking

## Summary

Reciprocal Rank Fusion is a powerful, simple method for combining multiple ranking systems:

**When to Use RRF:**
- ✅ Combining vector + BM25 search (hybrid search)
- ✅ Need simple, parameter-minimal solution
- ✅ Want reproducible, interpretable results
- ✅ Have multiple diverse ranking systems

**When to Consider Alternatives:**
- ❌ Need maximum accuracy (use cross-encoder re-ranking)
- ❌ Have only one ranking system (no fusion needed)
- ❌ Score magnitudes critical (use score-based fusion)

**Key Takeaways:**
1. RRF score = `∑ 1/(k + rank)` across systems
2. Default k=60 works well; tune for domain/query types
3. RRF is rank-based, so normalization is automatic
4. Combine RRF results with boost strategies for production search
5. Monitor performance and adjust k based on metrics

---

**Next Steps:**

- Read `boost-strategies-guide.md` for content-aware ranking refinement
- See `search-config-reference.md` for configuration details
- Review `hybrid_search.py` source for implementation details
