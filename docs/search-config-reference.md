# Search Configuration Reference

## Overview

This reference guide covers all configuration options for the BMCIS knowledge base search system, including RRF parameters, boost weights, recency thresholds, and environment variable usage.

The search system supports configuration through:
1. **Environment variables** - Runtime configuration
2. **Programmatic APIs** - Direct configuration in code
3. **Factory patterns** - Extensible strategy creation
4. **Default values** - Sensible defaults for all parameters

## Environment Variables

### RRF Configuration

RRF (Reciprocal Rank Fusion) parameters control how vector and BM25 results are combined.

#### SEARCH_RRF_K

**Purpose:** RRF constant parameter

**Type:** Integer

**Default Value:** `60`

**Valid Range:** `1-1000`

**Description:**
Controls the RRF weighting curve. Smaller values emphasize top-ranked documents, larger values provide more democratic weighting.

**Common Values:**
- `30`: Heavy emphasis on top results (semantic-focused)
- `60`: Balanced (recommended default)
- `100`: More democratic weighting
- `150`: Minimal difference between ranks

**Example:**
```bash
# Semantic-heavy ranking
export SEARCH_RRF_K=40

# Balanced ranking (default)
export SEARCH_RRF_K=60

# Keyword-heavy ranking
export SEARCH_RRF_K=100
```

**Programmatic:**
```python
from src.search.config import SearchConfig

config = SearchConfig()
print(config.rrf_config.k)  # Access current k value

# Or set via environment
import os
os.environ["SEARCH_RRF_K"] = "75"
```

#### SEARCH_VECTOR_WEIGHT

**Purpose:** Weight for vector search score in hybrid calculation

**Type:** Float

**Default Value:** `0.6`

**Valid Range:** `0.0-1.0`

**Description:**
Controls the relative importance of vector (semantic) search in hybrid results. Note: RRF-based hybrid doesn't use this directly; use SEARCH_RRF_K instead for primary tuning.

**Example:**
```bash
# More semantic emphasis
export SEARCH_VECTOR_WEIGHT=0.7

# Balanced
export SEARCH_VECTOR_WEIGHT=0.6

# More keyword emphasis
export SEARCH_VECTOR_WEIGHT=0.5
```

#### SEARCH_BM25_WEIGHT

**Purpose:** Weight for BM25 search score in hybrid calculation

**Type:** Float

**Default Value:** `0.4`

**Valid Range:** `0.0-1.0`

**Description:**
Controls the relative importance of BM25 (keyword) search in hybrid results. Note: RRF-based hybrid doesn't use this directly; use SEARCH_RRF_K instead for primary tuning.

**Example:**
```bash
# More keyword emphasis
export SEARCH_BM25_WEIGHT=0.5

# Balanced
export SEARCH_BM25_WEIGHT=0.4

# More semantic emphasis
export SEARCH_BM25_WEIGHT=0.3
```

### Boost Configuration

Boost weights control content-aware ranking refinement applied after RRF.

#### SEARCH_BOOST_VENDOR

**Purpose:** Boost weight for vendor matching

**Type:** Float

**Default Value:** `0.15`

**Valid Range:** `0.0-1.0`

**Description:**
Boost applied when document vendor matches vendors mentioned in query.

**Example:**
```bash
# Vendor matches very important
export SEARCH_BOOST_VENDOR=0.25

# Default balanced
export SEARCH_BOOST_VENDOR=0.15

# Vendor less important
export SEARCH_BOOST_VENDOR=0.08

# Disable vendor boost
export SEARCH_BOOST_VENDOR=0.0
```

**Use Cases:**
- `0.25`: Multi-vendor knowledge base, vendor specificity critical
- `0.15`: Mixed vendor content, balanced importance
- `0.08`: Vendor less important than other factors
- `0.0`: Disable vendor matching entirely

#### SEARCH_BOOST_DOC_TYPE

**Purpose:** Boost weight for document type matching

**Type:** Float

**Default Value:** `0.10`

**Valid Range:** `0.0-1.0`

**Description:**
Boost applied when document type matches query intent (api docs, guides, kb articles, code samples, references).

**Example:**
```bash
# Document type very important
export SEARCH_BOOST_DOC_TYPE=0.18

# Balanced
export SEARCH_BOOST_DOC_TYPE=0.10

# Document type less important
export SEARCH_BOOST_DOC_TYPE=0.05

# Disable document type boost
export SEARCH_BOOST_DOC_TYPE=0.0
```

#### SEARCH_BOOST_RECENCY

**Purpose:** Boost weight for document recency

**Type:** Float

**Default Value:** `0.05`

**Valid Range:** `0.0-1.0`

**Description:**
Maximum boost applied for very recent documents (< 7 days old). Boost decreases for older documents (50% for 7-30 days, 0% for > 30 days).

**Example:**
```bash
# Recency critical (e.g., security updates, breaking changes)
export SEARCH_BOOST_RECENCY=0.20

# Default balanced
export SEARCH_BOOST_RECENCY=0.05

# Recency minimal (evergreen content)
export SEARCH_BOOST_RECENCY=0.02

# Disable recency boost
export SEARCH_BOOST_RECENCY=0.0
```

#### SEARCH_BOOST_ENTITY

**Purpose:** Boost weight for entity mention matching

**Type:** Float

**Default Value:** `0.10`

**Valid Range:** `0.0-1.0`

**Description:**
Boost applied when document contains proper nouns (entities) mentioned in query.

**Example:**
```bash
# Entity matching important
export SEARCH_BOOST_ENTITY=0.15

# Balanced
export SEARCH_BOOST_ENTITY=0.10

# Entity less important
export SEARCH_BOOST_ENTITY=0.05

# Disable entity boost
export SEARCH_BOOST_ENTITY=0.0
```

#### SEARCH_BOOST_TOPIC

**Purpose:** Boost weight for topic keyword matching

**Type:** Float

**Default Value:** `0.08`

**Valid Range:** `0.0-1.0`

**Description:**
Boost applied when document contains topic-related keywords matching query's primary topic (authentication, deployment, optimization, etc.).

**Example:**
```bash
# Topic matching important
export SEARCH_BOOST_TOPIC=0.12

# Balanced
export SEARCH_BOOST_TOPIC=0.08

# Topic less important
export SEARCH_BOOST_TOPIC=0.04

# Disable topic boost
export SEARCH_BOOST_TOPIC=0.0
```

### Recency Thresholds

#### SEARCH_RECENCY_VERY_RECENT

**Purpose:** Threshold for "very recent" documents

**Type:** Integer (days)

**Default Value:** `7`

**Valid Range:** `1-90`

**Description:**
Documents updated within this many days receive 100% recency boost. Beyond this threshold but within SEARCH_RECENCY_RECENT threshold, documents receive 70% boost.

**Example:**
```bash
# Define very recent as within 7 days
export SEARCH_RECENCY_VERY_RECENT=7

# Define very recent as within 14 days
export SEARCH_RECENCY_VERY_RECENT=14

# Define very recent as within 1 day (for critical domains)
export SEARCH_RECENCY_VERY_RECENT=1
```

#### SEARCH_RECENCY_RECENT

**Purpose:** Threshold for "recent" documents

**Type:** Integer (days)

**Default Value:** `30`

**Valid Range:** `1-365`

**Description:**
Documents updated between SEARCH_RECENCY_VERY_RECENT and this many days receive 70% recency boost. Beyond this threshold, no recency boost is applied.

**Example:**
```bash
# Recent = within 30 days (default)
export SEARCH_RECENCY_RECENT=30

# Recent = within 60 days
export SEARCH_RECENCY_RECENT=60

# Recent = within 7 days (strict recency)
export SEARCH_RECENCY_RECENT=7
```

### Search Defaults

#### SEARCH_TOP_K_DEFAULT

**Purpose:** Default number of results to return

**Type:** Integer

**Default Value:** `10`

**Valid Range:** `1-1000`

**Description:**
Default limit for search results if not specified in query.

**Example:**
```bash
# Return 10 results by default
export SEARCH_TOP_K_DEFAULT=10

# Return 20 results by default
export SEARCH_TOP_K_DEFAULT=20

# Return 50 results (comprehensive)
export SEARCH_TOP_K_DEFAULT=50
```

#### SEARCH_MIN_SCORE_DEFAULT

**Purpose:** Default minimum score threshold

**Type:** Float

**Default Value:** `0.0`

**Valid Range:** `0.0-1.0`

**Description:**
Default score threshold for filtering results. Results below this score are excluded from results.

**Example:**
```bash
# No minimum score filtering (return all results)
export SEARCH_MIN_SCORE_DEFAULT=0.0

# Require at least 0.3 score
export SEARCH_MIN_SCORE_DEFAULT=0.3

# Require at least 0.5 score (high confidence)
export SEARCH_MIN_SCORE_DEFAULT=0.5
```

## Configuration Patterns

### Pattern 1: Vendor-Focused Configuration

For multi-vendor knowledge base where vendor specificity is critical:

```bash
export SEARCH_RRF_K=60                 # Balanced RRF
export SEARCH_BOOST_VENDOR=0.25        # HIGH - vendor matches critical
export SEARCH_BOOST_DOC_TYPE=0.12      # HIGH - doc type matters
export SEARCH_BOOST_RECENCY=0.08       # MEDIUM - some freshness
export SEARCH_BOOST_ENTITY=0.08        # MEDIUM
export SEARCH_BOOST_TOPIC=0.05         # LOW
```

**Best For:**
- Multi-vendor platforms (OpenAI, Anthropic, Google, etc.)
- SaaS product documentation
- Integration guides

### Pattern 2: Freshness-Critical Configuration

For domains where fresh information is paramount (security, APIs, standards):

```bash
export SEARCH_RRF_K=60                 # Balanced RRF
export SEARCH_BOOST_VENDOR=0.10        # MEDIUM
export SEARCH_BOOST_DOC_TYPE=0.08      # MEDIUM
export SEARCH_BOOST_RECENCY=0.20       # VERY HIGH - freshness critical
export SEARCH_BOOST_ENTITY=0.10        # MEDIUM
export SEARCH_BOOST_TOPIC=0.08         # MEDIUM
export SEARCH_RECENCY_VERY_RECENT=7    # Very recent = < 7 days
export SEARCH_RECENCY_RECENT=30        # Recent = < 30 days
```

**Best For:**
- Security documentation
- API reference (versions evolve)
- Breaking change announcements
- Healthcare/medical information

### Pattern 3: Evergreen Content Configuration

For timeless knowledge where age matters less:

```bash
export SEARCH_RRF_K=60                 # Balanced RRF
export SEARCH_BOOST_VENDOR=0.15        # MEDIUM
export SEARCH_BOOST_DOC_TYPE=0.12      # MEDIUM
export SEARCH_BOOST_RECENCY=0.02       # VERY LOW - age irrelevant
export SEARCH_BOOST_ENTITY=0.12        # HIGH - entities important
export SEARCH_BOOST_TOPIC=0.10         # MEDIUM
```

**Best For:**
- Mathematics documentation
- Algorithms and data structures
- Core programming concepts
- History and background information

### Pattern 4: Semantic-Heavy Configuration

For cases where understanding query intent is critical:

```bash
export SEARCH_RRF_K=40                 # Smaller k = favor top results
export SEARCH_VECTOR_WEIGHT=0.7        # Higher vector weight
export SEARCH_BM25_WEIGHT=0.3          # Lower keyword weight
export SEARCH_BOOST_VENDOR=0.08        # LOW
export SEARCH_BOOST_DOC_TYPE=0.06      # LOW
export SEARCH_BOOST_RECENCY=0.04       # LOW
export SEARCH_BOOST_ENTITY=0.15        # HIGH - entities = concepts
export SEARCH_BOOST_TOPIC=0.15         # HIGH - topic understanding
```

**Best For:**
- Question answering systems
- General knowledge queries
- Exploratory search
- Academic knowledge bases

### Pattern 5: Keyword-Heavy Configuration

For exact keyword matching and technical precision:

```bash
export SEARCH_RRF_K=100                # Larger k = more democratic
export SEARCH_VECTOR_WEIGHT=0.4        # Lower vector weight
export SEARCH_BM25_WEIGHT=0.6          # Higher keyword weight
export SEARCH_BOOST_VENDOR=0.10        # MEDIUM
export SEARCH_BOOST_DOC_TYPE=0.15      # HIGH
export SEARCH_BOOST_RECENCY=0.08       # MEDIUM
export SEARCH_BOOST_ENTITY=0.08        # MEDIUM
export SEARCH_BOOST_TOPIC=0.05         # LOW
```

**Best For:**
- API documentation (exact endpoints)
- Configuration reference (exact parameters)
- Code examples with specific syntax
- Technical specifications

## Programmatic Configuration

### Using SearchConfig API

```python
from src.search.config import SearchConfig, RRFConfig, BoostConfig

# Create config from environment variables
config = SearchConfig.from_env()

# Access RRF settings
k = config.rrf_config.k
vector_weight = config.rrf_config.vector_weight

# Access boost settings
vendor_boost = config.boost_config.vendor
recency_boost = config.boost_config.recency

# Access recency thresholds
very_recent = config.recency_config.very_recent_days
recent = config.recency_config.recent_days
```

### Creating Custom Configuration

```python
from src.search.config import SearchConfig, RRFConfig, BoostConfig, RecencyConfig

# Create custom configuration
custom_config = SearchConfig(
    rrf_config=RRFConfig(
        k=75,
        vector_weight=0.5,
        bm25_weight=0.5
    ),
    boost_config=BoostConfig(
        vendor=0.20,
        doc_type=0.15,
        recency=0.10,
        entity=0.10,
        topic=0.08
    ),
    recency_config=RecencyConfig(
        very_recent_days=7,
        recent_days=30
    )
)

# Use custom configuration
results = hybrid_search.search(query, config=custom_config)
```

### Environment Variable Precedence

Configuration is loaded in order of precedence:

1. **Explicit parameters** (highest)
   ```python
   SearchConfig(rrf_config=RRFConfig(k=80))  # Explicit wins
   ```

2. **Environment variables**
   ```bash
   export SEARCH_RRF_K=80  # Env var used
   ```

3. **Default values** (lowest)
   ```python
   # If no env var, uses RRFConfig.k = 60 (default)
   ```

## Best Practices

### Configuration Strategy

1. **Start with defaults** - Use default values for initial deployment
2. **Monitor metrics** - Track user satisfaction, click-through rates, precision
3. **Identify bottlenecks** - Which results are failing? Which factors matter?
4. **Adjust incrementally** - Change one parameter at a time
5. **Measure impact** - Always A/B test configuration changes
6. **Document decisions** - Record why you chose specific values

### Parameter Tuning Workflow

```python
# Step 1: Get baseline
baseline_config = SearchConfig()
baseline_results = run_test_queries(baseline_config)
baseline_metrics = evaluate_quality(baseline_results)

# Step 2: Adjust one parameter
adjusted_config = SearchConfig(
    boost_config=BoostConfig(recency=0.15)  # Increase recency
)
adjusted_results = run_test_queries(adjusted_config)
adjusted_metrics = evaluate_quality(adjusted_results)

# Step 3: Compare
if adjusted_metrics.precision > baseline_metrics.precision:
    print("Improvement found! Use new config")
else:
    print("No improvement. Try different value")
```

### Configuration Validation

```python
# Validate configuration before use
from src.search.config import SearchConfig

try:
    config = SearchConfig.from_env()
    # Validation happens automatically via Pydantic
    print("Configuration valid")
except ValueError as e:
    print(f"Configuration error: {e}")
    # Fall back to defaults
    config = SearchConfig()
```

## Troubleshooting

### Issue: Results Haven't Changed After Config Update

**Possible Causes:**
1. Environment variable not loaded (restart app required)
2. Configuration parameter out of valid range (silently clamped)
3. Boost weight set to 0.0 (disabled)
4. Wrong parameter name (typo)

**Debug:**
```python
from src.search.config import SearchConfig

config = SearchConfig.from_env()
print(config)  # Print all settings to verify
```

### Issue: Too Many Low-Quality Results

**Solution:** Increase min score threshold or boost weights:
```bash
export SEARCH_MIN_SCORE_DEFAULT=0.4    # Filter low scores
export SEARCH_BOOST_VENDOR=0.25        # Increase vendor boost
```

### Issue: Too Few Results Returned

**Solution:** Decrease min score threshold or increase top_k:
```bash
export SEARCH_MIN_SCORE_DEFAULT=0.0    # No minimum
export SEARCH_TOP_K_DEFAULT=25         # Return more results
```

### Issue: Vendor Boost Not Applied

**Possible Causes:**
1. Vendor boost set to 0.0
2. Query doesn't contain vendor names
3. Document metadata missing vendor field

**Debug:**
```python
from src.search.boost_strategies import VendorBoostStrategy

strategy = VendorBoostStrategy(boost_factor=0.15)
print(strategy.should_boost(query, result))
print(strategy.calculate_boost(query, result))
```

## Configuration Validation Rules

All configuration values are validated with strict range checking:

| Parameter | Type | Min | Max | Constraint |
|-----------|------|-----|-----|-----------|
| SEARCH_RRF_K | int | 1 | 1000 | Required |
| SEARCH_VECTOR_WEIGHT | float | 0.0 | 1.0 | Optional |
| SEARCH_BM25_WEIGHT | float | 0.0 | 1.0 | Optional |
| SEARCH_BOOST_VENDOR | float | 0.0 | 1.0 | Optional |
| SEARCH_BOOST_DOC_TYPE | float | 0.0 | 1.0 | Optional |
| SEARCH_BOOST_RECENCY | float | 0.0 | 1.0 | Optional |
| SEARCH_BOOST_ENTITY | float | 0.0 | 1.0 | Optional |
| SEARCH_BOOST_TOPIC | float | 0.0 | 1.0 | Optional |
| SEARCH_RECENCY_VERY_RECENT | int | 1 | 90 | Optional |
| SEARCH_RECENCY_RECENT | int | 1 | 365 | Optional |
| SEARCH_TOP_K_DEFAULT | int | 1 | 1000 | Optional |
| SEARCH_MIN_SCORE_DEFAULT | float | 0.0 | 1.0 | Optional |

## Default Configuration

When no environment variables are set, the following defaults are used:

```python
SearchConfig(
    rrf_config=RRFConfig(
        k=60,
        vector_weight=0.6,
        bm25_weight=0.4
    ),
    boost_config=BoostConfig(
        vendor=0.15,        # +15%
        doc_type=0.10,      # +10%
        recency=0.05,       # +5%
        entity=0.10,        # +10%
        topic=0.08          # +8%
    ),
    recency_config=RecencyConfig(
        very_recent_days=7,
        recent_days=30
    ),
    search_config=SearchConfig(
        top_k_default=10,
        min_score_default=0.0
    )
)
```

## Environment Variable Examples

### Development Configuration

```bash
# .env.development
SEARCH_RRF_K=60
SEARCH_BOOST_VENDOR=0.15
SEARCH_BOOST_DOC_TYPE=0.10
SEARCH_BOOST_RECENCY=0.05
SEARCH_BOOST_ENTITY=0.10
SEARCH_BOOST_TOPIC=0.08
SEARCH_TOP_K_DEFAULT=10
SEARCH_MIN_SCORE_DEFAULT=0.0
```

### Production Configuration

```bash
# .env.production
SEARCH_RRF_K=70              # Slightly more democratic
SEARCH_BOOST_VENDOR=0.20     # Higher vendor emphasis
SEARCH_BOOST_DOC_TYPE=0.12
SEARCH_BOOST_RECENCY=0.08
SEARCH_BOOST_ENTITY=0.12
SEARCH_BOOST_TOPIC=0.10
SEARCH_TOP_K_DEFAULT=15      # Return more results
SEARCH_MIN_SCORE_DEFAULT=0.2 # Filter low confidence
```

### High-Security Configuration

```bash
# .env.security
SEARCH_RRF_K=60
SEARCH_BOOST_VENDOR=0.12
SEARCH_BOOST_DOC_TYPE=0.08
SEARCH_BOOST_RECENCY=0.25     # VERY HIGH - security critical
SEARCH_BOOST_ENTITY=0.10
SEARCH_BOOST_TOPIC=0.10
SEARCH_RECENCY_VERY_RECENT=3  # Very recent = < 3 days
SEARCH_RECENCY_RECENT=14      # Recent = < 2 weeks
SEARCH_TOP_K_DEFAULT=20
SEARCH_MIN_SCORE_DEFAULT=0.4  # High confidence only
```

## Performance Tuning

### Configuration Impact on Performance

Different configurations have different performance characteristics:

| Configuration | RRF Time | Boost Time | Total | Notes |
|---------------|----------|-----------|-------|-------|
| All strategies | ~5ms | ~55ms | ~60ms | All boosts applied |
| 3 strategies | ~5ms | ~33ms | ~38ms | Selective boosts |
| 1 strategy | ~5ms | ~11ms | ~16ms | Minimal boosts |
| No boosts | ~5ms | 0ms | ~5ms | RRF only |

### Optimization Tips

1. **Use selective strategies** - Apply only necessary strategies
2. **Cache configuration** - Load once, reuse across requests
3. **Batch process** - Process multiple queries to amortize overhead
4. **Monitor** - Track actual performance metrics in production

## Related Documentation

- **RRF Algorithm Guide:** `docs/rrf-algorithm-guide.md`
- **Boost Strategies Guide:** `docs/boost-strategies-guide.md`
- **Implementation:** `src/search/boost_strategies.py`, `src/search/config.py`

## Summary

Search configuration provides fine-grained control over:

1. **RRF Ranking** - Control balance of vector and keyword search
2. **Boost Weights** - Fine-tune content-aware refinement
3. **Recency Thresholds** - Adjust freshness requirements
4. **Result Defaults** - Set top_k, min_score globally

**Key Principles:**
- Start with defaults
- Measure impact of changes
- Adjust incrementally
- Monitor in production
- Document decisions

**Quick Start:**
```bash
# Use defaults
export SEARCH_RRF_K=60

# Or customize for your domain
export SEARCH_BOOST_RECENCY=0.20  # Emphasize freshness
export SEARCH_TOP_K_DEFAULT=20    # Return more results
```

---

For more information:
- Implementation details: `src/search/config.py`
- Boost strategies: `src/search/boost_strategies.py`
- Integration: `src/search/hybrid_search.py`
