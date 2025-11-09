# Boost Strategies Guide

## Overview

Boost strategies provide a flexible, extensible system for content-aware ranking refinement in hybrid search. After RRF combines results from vector and BM25 search, boost strategies amplify relevance by analyzing query context and document metadata.

The boost system is built on:

1. **BoostStrategy ABC:** Abstract interface for strategy implementations
2. **Concrete Strategies:** 5 built-in strategies for common use cases
3. **BoostStrategyFactory:** Registry for dynamic strategy creation and composition
4. **Score Modification:** Cumulative boosts applied to final ranking

This guide covers each strategy, tuning guidance, custom implementations, and integration patterns.

## Architecture Overview

### Execution Flow

```
Vector Search Results    BM25 Search Results
      (rank 1-N)              (rank 1-N)
            ↓                        ↓
         RRF Ranking (combines by rank)
              ↓
         Initial Scores
              ↓
     Boost Strategy #1 (e.g., Vendor)
              ↓
     Boost Strategy #2 (e.g., Recency)
              ↓
     Boost Strategy #3 (e.g., Entity)
              ↓
      Final Ranking
```

### Strategy Interface

All boost strategies implement two methods:

```python
class BoostStrategy(ABC):
    @abstractmethod
    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Check if document qualifies for boost."""
        ...

    @abstractmethod
    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate boost multiplier [0.0, 1.0]."""
        ...
```

### Score Application

Boosts are **cumulative and clamped**:

```
base_score = 0.75 (from RRF)

Vendor boost: +0.15 → cumulative = 0.30
Recency boost: +0.05 → cumulative = 0.35
Entity boost: +0.10 → cumulative = 0.45

final_score = base_score * (1 + cumulative_boost)
            = 0.75 * (1 + 0.45)
            = 0.75 * 1.45
            = 1.0875 → clamped to [0.0, 1.0] → 1.0
```

## Built-In Strategies

### Strategy 1: Vendor Boost (+15%)

**Purpose:** Boost documents from vendors mentioned in the query.

**When to Use:**
- Users search for specific vendors/companies
- Multi-vendor knowledge base
- Vendor documentation is important

**Example Scenarios:**
- Query: "OpenAI API documentation" → Boost OpenAI docs
- Query: "AWS vs Azure comparison" → Boost both AWS and Azure docs
- Query: "How to integrate with Google APIs" → Boost Google docs

**How It Works:**

1. **Detects vendors** in query by checking against known vendor list:
   ```
   Known vendors: OpenAI, Anthropic, Google, AWS, Azure, Meta, XAI, etc.
   ```

2. **Extracts vendor** from document metadata:
   ```python
   result.metadata.get("vendor")
   # Falls back to: vendor_name, company, organization, author
   ```

3. **Boosts if match:**
   ```python
   if result_vendor.lower() in [v.lower() for v in query_vendors]:
       boost = 0.15  # +15%
   ```

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boost_factor` | 0.15 | 0.0-1.0 | Boost multiplier when vendor matches |

**Tuning Guidance:**

- **Increase to 0.20-0.25** if:
  - Vendor information is critical for relevance
  - Documents from specified vendors are significantly better
  - Users expect vendor-specific results

- **Decrease to 0.10** if:
  - Vendor is less important than content quality
  - Many cross-vendor documents are relevant
  - Want more balanced results

- **Set to 0.0** to disable vendor boosting

**Example:**

```python
from src.search.boost_strategies import BoostStrategyFactory

# Use custom vendor boost weight
vendor_boost = BoostStrategyFactory.create_strategy(
    "vendor",
    boost_factor=0.20  # Increased from default 0.15
)

# In boosting system
results = boosting.apply_boost_strategies(
    results,
    query,
    strategies=["vendor"]  # Apply only vendor boost
)
```

### Strategy 2: Document Type Boost (+10%)

**Purpose:** Boost documents that match the query's document type intent.

**When to Use:**
- Multi-type knowledge bases (guides, APIs, code samples)
- Users have implicit document type preferences
- Different result types serve different purposes

**Example Scenarios:**
- Query: "how to implement X" (keyword: "how") → Boost guides, tutorials
- Query: "API endpoint for users" → Boost API documentation
- Query: "code example for caching" → Boost code samples
- Query: "FAQ about authentication" → Boost KB articles

**Supported Document Types:**

| Type | Keywords | Example |
|------|----------|---------|
| `api_docs` | api, endpoint, request, response, parameter, method, http | OpenAI API docs, AWS API reference |
| `guide` | guide, tutorial, how to, getting started, step by step | Deployment guide, Setup tutorial |
| `kb_article` | kb, faq, frequently asked, troubleshooting, common issue | Troubleshooting guide, FAQ |
| `code_sample` | code, example, sample, implementation, github, snippet | Code examples, GitHub repos |
| `reference` | reference, spec, schema, documentation, standard | API reference, Protocol spec |

**How It Works:**

1. **Detect query intent** by scanning for type keywords:
   ```
   Query: "how to deploy on kubernetes"
   Detected keywords: "how", "deploy"
   → Type = "guide"
   ```

2. **Extract document type** from `source_category`:
   ```
   source_category="api_docs" → type = "api_docs"
   source_category="deployment_guide" → type = "guide"
   ```

3. **Boost if types match:**
   ```python
   if query_type == document_type:
       boost = 0.10  # +10%
   ```

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boost_factor` | 0.10 | 0.0-1.0 | Boost multiplier when types match |

**Tuning Guidance:**

- **Increase to 0.15-0.20** if:
  - Document type is critical for query intent
  - Users strongly prefer specific document types
  - Mixed-type results are confusing

- **Decrease to 0.05** if:
  - Document type is less important
  - High-quality content matters more than type
  - Want diverse result types

**Example:**

```python
# Create document type boost with custom weight
doc_type_boost = BoostStrategyFactory.create_strategy(
    "doc_type",
    boost_factor=0.15  # Increased for type-sensitive domain
)
```

### Strategy 3: Recency Boost (+5%)

**Purpose:** Boost recently updated documents.

**When to Use:**
- Information evolves frequently (APIs, standards, best practices)
- Fresh information is more valuable than older information
- Want to prefer up-to-date documentation

**Example Scenarios:**
- Query: "latest OpenAI models" → Boost recent documentation
- Query: "Kubernetes networking" → Boost recently updated articles
- Query: "Python best practices" → Boost recent community standards
- Query: "COVID-19 treatment information" → Heavily boost recent docs

**How It Works:**

1. **Check document age**:
   ```
   Very recent (< 7 days): 100% of boost
   Moderate (7-30 days): 70% of boost
   Old (> 30 days): 0% of boost
   ```

2. **Apply graduated boost**:
   ```python
   age_days = (today - document_date).days

   if age_days < 7:
       recency_factor = 1.0  # 100%
   elif age_days < 30:
       recency_factor = 0.7  # 70%
   else:
       recency_factor = 0.0  # 0%

   boost = 0.05 * recency_factor
   ```

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boost_factor` | 0.05 | 0.0-1.0 | Maximum boost for very recent docs |

**Thresholds** (hardcoded in strategy):

| Threshold | Days | Boost Factor | Use Case |
|-----------|------|--------------|----------|
| Very Recent | < 7 days | 1.0 (100%) | Critical for fresh information |
| Recent | 7-30 days | 0.7 (70%) | Good for moderate recency |
| Old | > 30 days | 0.0 (0%) | No boost for outdated info |

**Tuning Guidance:**

- **Increase to 0.10-0.15** if:
  - Information changes frequently (APIs, security advisories)
  - Outdated information causes problems
  - Examples: Healthcare, finance, security

- **Decrease to 0.02-0.03** if:
  - Content is relatively evergreen
  - Older content is still valid
  - Examples: Math, core concepts, history

- **Set to 0.0** if:
  - Recency doesn't matter (timeless knowledge)
  - Document dates are unreliable

**Example:**

```python
# Heavy emphasis on recency for security docs
recency_boost = BoostStrategyFactory.create_strategy(
    "recency",
    boost_factor=0.15  # Critical for security updates
)

# Minimal recency emphasis for evergreen content
recency_boost = BoostStrategyFactory.create_strategy(
    "recency",
    boost_factor=0.02  # Light touch for evergreen docs
)
```

### Strategy 4: Entity Boost (+10%)

**Purpose:** Boost documents mentioning entities (proper nouns) from the query.

**When to Use:**
- Query contains product names, company names, proper nouns
- Entity extraction improves relevance
- Want to find documents about specific "things"

**Example Scenarios:**
- Query: "How to use OpenAI GPT-4" → Boost docs mentioning "OpenAI" and "GPT-4"
- Query: "Integrate Stripe with Python" → Boost docs mentioning "Stripe" and "Python"
- Query: "Claude AI assistant capabilities" → Boost docs mentioning "Claude" and "AI"

**How It Works:**

1. **Extract entities** from query (capitalized words, multi-word phrases):
   ```python
   query = "How to use OpenAI GPT-4 with Python"
   entities = ["OpenAI", "GPT-4", "Python"]  # Capitalized words
   ```

2. **Search for entities** in document text:
   ```python
   for entity in entities:
       if entity.lower() in result.chunk_text.lower():
           matched.append(entity)
   ```

3. **Boost if entities found**:
   ```python
   if matched_entities:
       boost = 0.10  # +10%
   ```

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boost_factor` | 0.10 | 0.0-1.0 | Boost when entities are matched |

**Tuning Guidance:**

- **Increase to 0.15-0.20** if:
  - Entity mentions are strong relevance signals
  - Want to prioritize specific product/company documentation
  - Examples: SaaS integrations, multi-vendor comparisons

- **Decrease to 0.05-0.08** if:
  - Entity matches are secondary signals
  - Content quality matters more than entity mentions
  - Examples: General knowledge bases

- **Set to 0.0** if:
  - Entities are unreliable relevance signals
  - Many false positive entity extractions

**Example:**

```python
# Strong entity matching for integration documentation
entity_boost = BoostStrategyFactory.create_strategy(
    "entity",
    boost_factor=0.15  # High weight for named entities
)

# Light entity matching for general docs
entity_boost = BoostStrategyFactory.create_strategy(
    "entity",
    boost_factor=0.05  # Lower weight
)
```

### Strategy 5: Topic Boost (+8%)

**Purpose:** Boost documents related to the query's detected primary topic.

**When to Use:**
- Query has clear topic (authentication, deployment, performance)
- Topic-specific documents should rank higher
- Want semantic topic matching

**Example Scenarios:**
- Query: "JWT token validation" (topic: authentication) → Boost auth docs
- Query: "Kubernetes deployment strategy" (topic: deployment) → Boost deployment docs
- Query: "API rate limiting optimization" (topic: optimization) → Boost perf docs
- Query: "Database connection pooling" (topic: data_handling) → Boost data docs

**Supported Topics:**

| Topic | Keywords | Example Queries |
|-------|----------|-----------------|
| `authentication` | auth, login, jwt, token, oauth, saml, mfa | "JWT tokens", "OAuth implementation" |
| `api_design` | api design, rest, graphql, webhook, versioning | "GraphQL vs REST", "API versioning" |
| `data_handling` | data, database, storage, caching, indexing | "Database indexing", "Cache invalidation" |
| `deployment` | deploy, production, docker, kubernetes, ci/cd | "Kubernetes deployment", "CI/CD pipeline" |
| `optimization` | performance, latency, throughput, benchmark | "Performance optimization", "Latency tuning" |
| `error_handling` | error, exception, retry, timeout, circuit breaker | "Error handling", "Retry logic" |

**How It Works:**

1. **Detect topic** by scanning keywords:
   ```
   Query: "How to optimize database queries"
   Keywords found: "optimize" (optimization topic)
   → Topic = "optimization"
   ```

2. **Get topic keywords**:
   ```python
   topic_keywords = {
       "performance", "latency", "throughput", "efficient",
       "fast", "slow", "benchmark", "profiling"
   }
   ```

3. **Boost if keywords found in document**:
   ```python
   topic_keywords = TOPIC_KEYWORDS.get(detected_topic, [])
   if any(kw in result.chunk_text.lower() for kw in topic_keywords):
       boost = 0.08  # +8%
   ```

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boost_factor` | 0.08 | 0.0-1.0 | Boost for topic keyword matches |

**Tuning Guidance:**

- **Increase to 0.12-0.15** if:
  - Topic matching strongly correlates with relevance
  - Users benefit from topic-specific articles
  - Examples: Technical forums, problem-solving docs

- **Decrease to 0.04-0.06** if:
  - Topic is secondary signal
  - Content relevance matters more
  - Examples: General knowledge bases

**Example:**

```python
# Emphasize topic matching for technical queries
topic_boost = BoostStrategyFactory.create_strategy(
    "topic",
    boost_factor=0.12  # Higher for topic-heavy domains
)

# Light topic matching for diverse content
topic_boost = BoostStrategyFactory.create_strategy(
    "topic",
    boost_factor=0.04  # Lower for general content
)
```

## Strategy Composition

### Default Strategy Set

By default, all 5 strategies are applied:

```python
# Use all strategies with default weights
results = boosting.apply_boost_strategies(results, query)

# Equivalent to:
results = boosting.apply_boost_strategies(
    results,
    query,
    strategies=["vendor", "doc_type", "recency", "entity", "topic"]
)
```

### Custom Strategy Combinations

Apply only specific strategies:

```python
# Only vendor and recency (ignore document type, entity, topic)
results = boosting.apply_boost_strategies(
    results,
    query,
    strategies=["vendor", "recency"]
)

# Only entity and topic (for semantic-focused search)
results = boosting.apply_boost_strategies(
    results,
    query,
    strategies=["entity", "topic"]
)
```

### Use Cases for Selective Composition

| Use Case | Strategies | Rationale |
|----------|-----------|-----------|
| Vendor-heavy domain | vendor, recency | Vendor specific, freshness important |
| Type-sensitive docs | doc_type, entity | Clear doc types, entity matching |
| Semantic search | entity, topic | Emphasize semantic understanding |
| Mixed purpose | all 5 | Balanced approach for diverse queries |
| Fresh content critical | recency only | Maximum freshness emphasis |

## Custom Boost Strategies

### Creating a Custom Strategy

**Step 1: Subclass BoostStrategy**

```python
from src.search.boost_strategies import BoostStrategy
from src.search.results import SearchResult

class CodeQualityBoostStrategy(BoostStrategy):
    """Boost well-written code examples.

    Looks for indicators of code quality:
    - Presence of tests (pytest, unittest)
    - Type hints/typing
    - Documentation
    - Comments
    """

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Check if code has quality indicators."""
        quality_indicators = [
            "pytest", "unittest", "typing", "@dataclass",
            "docstring", "# ", "\"\"\"", "def test_"
        ]
        text_lower = result.chunk_text.lower()
        return any(indicator in text_lower for indicator in quality_indicators)

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Calculate boost based on quality signal strength."""
        if not self.should_boost(query, result):
            return 0.0

        # Count quality indicators for stronger boost
        text_lower = result.chunk_text.lower()
        indicator_count = sum(
            1 for ind in ["pytest", "typing", "docstring", "def test_"]
            if ind in text_lower
        )

        # More indicators = stronger boost (up to boost_factor)
        return min(self.boost_factor * (indicator_count / 4), self.boost_factor)
```

**Step 2: Register with Factory**

```python
from src.search.boost_strategies import BoostStrategyFactory

# Register the custom strategy
BoostStrategyFactory.register_strategy(
    "code_quality",
    CodeQualityBoostStrategy,
    default_factor=0.12
)
```

**Step 3: Use the Custom Strategy**

```python
# Create an instance
code_boost = BoostStrategyFactory.create_strategy("code_quality", boost_factor=0.15)

# Use in boosting system
results = boosting.apply_boost_strategies(
    results,
    query,
    strategies=["vendor", "code_quality", "recency"]
)

# Or with factory directly
all_strategies = BoostStrategyFactory.create_all_strategies()
# Now includes CodeQualityBoostStrategy
```

### Advanced: Conditional Boost Strategy

```python
class VersionMatchBoostStrategy(BoostStrategy):
    """Boost documents matching query's detected version."""

    def should_boost(self, query: str, result: SearchResult) -> bool:
        """Check if result version matches query version."""
        # Extract version from query (e.g., "v2", "3.9", "6.0")
        query_versions = self._extract_versions(query)
        result_version = self._get_version_from_metadata(result)

        if not query_versions or not result_version:
            return False

        # Check if result version is in query versions
        return any(
            qv.lower() in result_version.lower()
            for qv in query_versions
        )

    def calculate_boost(self, query: str, result: SearchResult) -> float:
        """Return full boost if version matches."""
        if self.should_boost(query, result):
            return self.boost_factor
        return 0.0

    @staticmethod
    def _extract_versions(query: str) -> list[str]:
        """Extract version numbers from query."""
        import re
        # Match patterns like v2, 3.9, 6.0, python3.10
        pattern = r'\b(?:v)?(\d+(?:\.\d+)*)\b'
        return re.findall(pattern, query)

    @staticmethod
    def _get_version_from_metadata(result: SearchResult) -> str | None:
        """Extract version from metadata."""
        if not result.metadata:
            return None
        version = result.metadata.get("version")
        return version if isinstance(version, str) else None
```

## Integration Patterns

### Pattern 1: Default Configuration

```python
from src.search.boosting import BoostingSystem

# Use default boost strategies
boosting = BoostingSystem()
results = boosting.apply_boost_strategies(results, query)
# Uses: vendor (0.15), doc_type (0.10), recency (0.05),
#       entity (0.10), topic (0.08)
```

### Pattern 2: Custom Weights

```python
from src.search.boost_strategies import BoostStrategyFactory

# Adjust strategy weights via BoostStrategyFactory
custom_factors = {
    "vendor": 0.20,      # Increased vendor importance
    "recency": 0.10,     # Increased freshness importance
    "doc_type": 0.05,    # Decreased type importance
    "entity": 0.10,
    "topic": 0.08
}

strategies = BoostStrategyFactory.create_all_strategies(custom_factors)
# Manually apply boosts...
```

### Pattern 3: Domain-Specific Strategy Sets

**For API Documentation:**
```python
# Emphasize exact document type and vendor matches
api_strategies = BoostStrategyFactory.create_all_strategies(
    custom_factors={
        "vendor": 0.20,      # Critical for multi-vendor APIs
        "doc_type": 0.15,    # APIs are well-typed
        "entity": 0.12,      # Endpoint names matter
        "recency": 0.08,     # API versions change
        "topic": 0.05
    }
)
```

**For Knowledge Base:**
```python
# Balanced approach with strong recency
kb_strategies = BoostStrategyFactory.create_all_strategies(
    custom_factors={
        "vendor": 0.10,      # Some vendor content
        "doc_type": 0.08,    # Mixed types
        "recency": 0.15,     # Recent info valued
        "entity": 0.10,
        "topic": 0.10
    }
)
```

**For Code Examples:**
```python
# Emphasize entity and quality
code_strategies = [
    BoostStrategyFactory.create_strategy("entity", 0.15),    # Code entities
    BoostStrategyFactory.create_strategy("topic", 0.10),     # Topic relevance
    BoostStrategyFactory.create_strategy("recency", 0.12)    # Recent patterns
]
```

### Pattern 4: Conditional Strategy Application

```python
def apply_context_aware_boosts(
    results: list[SearchResult],
    query: str,
    context: str
) -> list[SearchResult]:
    """Apply strategies based on query context."""

    if context == "version_specific":
        # Use version matching strategy
        strategies = ["vendor", "version_match", "recency"]
    elif context == "integration":
        # Emphasize entity matching
        strategies = ["entity", "vendor", "doc_type"]
    elif context == "learning":
        # Prefer guides and tutorials
        strategies = ["doc_type", "topic"]
    else:
        # Default balanced
        strategies = None  # All strategies

    return boosting.apply_boost_strategies(results, query, strategies)
```

## Performance Considerations

### Computational Overhead

Boost strategy execution is **very fast**:

```
Time per document:
- should_boost(): ~0.1ms (string scanning)
- calculate_boost(): ~0.01ms (calculation)
- Total per strategy: ~0.11ms

For 100 results with 5 strategies:
- Total time: 100 * 5 * 0.11ms ≈ 55ms
- As % of total search: <10%
```

### Optimization Tips

1. **Selective Strategies:**
   - Apply only relevant strategies for your domain
   - Saves proportional time

2. **Cache Extracted Features:**
   - Pre-compute vendor list, entity extraction
   - Reuse across multiple strategies

3. **Lazy Evaluation:**
   - Skip expensive strategies if document already high-ranked
   - Example: Skip topic boost if score > 0.95

## Tuning Framework

### Step 1: Baseline Measurement

```python
# Get baseline RRF results (no boosts)
baseline_results = calculate_hybrid_search_without_boosts(query)

# Measure: precision@10, user satisfaction, click-through rates
baseline_metrics = measure_quality(baseline_results)
```

### Step 2: Individual Strategy Testing

```python
for strategy_name in ["vendor", "doc_type", "recency", "entity", "topic"]:
    results = apply_single_strategy(
        baseline_results,
        query,
        strategy_name
    )
    metrics = measure_quality(results)
    print(f"{strategy_name}: +{metrics.precision_delta:.3f}")
```

### Step 3: Combination Testing

```python
# Test different strategy combinations
combinations = [
    ["vendor"],
    ["vendor", "recency"],
    ["vendor", "doc_type"],
    ["all 5 strategies"],
    # ... other combinations
]

best_combo = test_all_combinations(combinations)
```

### Step 4: Weight Tuning

```python
# For each strategy in best combo, fine-tune weight
for strategy in best_combo:
    for weight in [0.05, 0.10, 0.15, 0.20]:
        results = apply_with_weight(results, strategy, weight)
        metrics = measure_quality(results)
        # Track best weight
```

### Step 5: Monitor and Adjust

- Track user metrics weekly
- A/B test strategy changes
- Adjust weights gradually
- Re-evaluate quarterly

## Common Tuning Scenarios

### Scenario 1: Too Many Irrelevant Results in Top 10

**Diagnosis:** Quality boosts not strong enough

**Solution:**
```python
# Increase boost weights
custom_factors = {
    "vendor": 0.25,      # +25% (from 15%)
    "doc_type": 0.15,    # +15% (from 10%)
    "recency": 0.08,
    "entity": 0.15,      # +15% (from 10%)
    "topic": 0.12        # +12% (from 8%)
}
```

### Scenario 2: Missing Diverse Result Types

**Diagnosis:** Document type boost too strong

**Solution:**
```python
# Decrease document type boost
custom_factors = {
    "vendor": 0.15,
    "doc_type": 0.03,    # Reduced (from 10%)
    "recency": 0.05,
    "entity": 0.10,
    "topic": 0.08
}
```

### Scenario 3: Outdated Information Appearing

**Diagnosis:** Recency boost insufficient

**Solution:**
```python
# Dramatically increase recency boost
custom_factors = {
    "vendor": 0.15,
    "doc_type": 0.10,
    "recency": 0.20,     # Increased (from 5%)
    "entity": 0.10,
    "topic": 0.08
}
```

## Troubleshooting

### Boost Not Applied as Expected

**Check:**
1. Is `should_boost()` returning True?
2. Is `calculate_boost()` returning non-zero value?
3. Are strategies registered with factory?
4. Is strategy in strategies list?

**Debug:**
```python
# Test individual strategy
vendor_boost = BoostStrategyFactory.create_strategy("vendor")
print(vendor_boost.should_boost(query, result))
print(vendor_boost.calculate_boost(query, result))
```

### Score Not Changing After Boost

**Check:**
1. Is `boost_factor` > 0.0?
2. Is cumulative boost reaching score clamping (1.0)?
3. Are other strategies canceling the effect?

**Debug:**
```python
# Calculate before/after
before = result.hybrid_score
after = apply_boost_strategies([result], query)[0].hybrid_score
print(f"Before: {before}, After: {after}, Delta: {after - before}")
```

### Unexpected Strategy Combinations

**Check:**
1. Are strategies being applied in correct order?
2. Are boosts cumulative (as expected)?
3. Is score normalization working?

## Summary

Boost strategies provide a powerful, flexible system for ranking refinement:

**Key Points:**
1. Five built-in strategies for common use cases
2. Easy custom strategy creation via ABC
3. Factory pattern for dynamic registration and composition
4. Cumulative, clamped scoring prevents distortion
5. Minimal performance overhead (<10% of total search time)

**Best Practices:**
- Start with default strategy set
- Tune weights based on domain and metrics
- Add custom strategies for domain-specific signals
- Monitor and adjust based on user feedback
- Combine with cross-encoder re-ranking for accuracy

**Next Steps:**
- Review `rrf-algorithm-guide.md` for RRF details
- See `search-config-reference.md` for configuration
- Examine source code in `src/search/boost_strategies.py`
- Test custom strategies in your domain

---

**API Reference:** See inline docstrings in `src/search/boost_strategies.py`

**Example Implementation:** `src/search/boosting.py` - `apply_boost_strategies()` method
