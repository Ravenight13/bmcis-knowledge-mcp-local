# Task 6 Implementation Required - Action Plan

**Date**: 2025-11-08 13:50
**Status**: ⏸ **AWAITING IMPLEMENTATION**
**Priority**: HIGH

---

## Situation

Architecture review for Task 6 (Cross-Encoder Reranking System) was requested but **no implementation exists**. This document provides clear guidance for the development team on what needs to be built before review can proceed.

---

## What Needs to Be Built

### Core Components (Tasks 6.1-6.3)

#### 1. CrossEncoderReranker Class (Task 6.1)
**File**: `src/search/cross_encoder_reranker.py`

**Minimum Requirements**:
```python
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .results import SearchResult

class CrossEncoderReranker:
    """
    Reranks search results using ms-marco-MiniLM-L-6-v2 cross-encoder model.

    Accepts hybrid search results and produces top-5 reranked results based on
    query-document semantic similarity.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32
    ):
        """Initialize cross-encoder model and tokenizer."""
        pass

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query: Search query string
            results: List of SearchResult objects from hybrid search
            top_k: Number of top results to return (default: 5)

        Returns:
            List of top-k SearchResult objects with updated scores
        """
        pass
```

**Implementation Checklist**:
- [ ] Model loading with error handling
- [ ] GPU/CPU device detection and fallback
- [ ] Batch inference for efficiency
- [ ] SearchResult format preservation
- [ ] Score normalization
- [ ] Type annotations (mypy --strict compliant)
- [ ] Docstrings on all public methods
- [ ] Error handling for edge cases

#### 2. Candidate Selection Strategy (Task 6.2)
**Location**: Within `CrossEncoderReranker` class

**Minimum Requirements**:
```python
class CandidateSelector:
    """
    Selects optimal candidate pool size based on query complexity.

    Adaptive sizing: simple queries use smaller pools, complex queries use larger
    pools for better ranking quality.
    """

    def select_candidates(
        self,
        query: str,
        results: List[SearchResult],
        max_candidates: int = 50
    ) -> List[SearchResult]:
        """
        Select candidate pool from hybrid search results.

        Args:
            query: Search query for complexity analysis
            results: Full hybrid search result list
            max_candidates: Maximum pool size

        Returns:
            Filtered list of candidate SearchResult objects
        """
        pass

    def _analyze_query_complexity(self, query: str) -> int:
        """
        Analyze query complexity to determine pool size.

        Simple queries (1-2 words): 20 candidates
        Medium queries (3-5 words): 35 candidates
        Complex queries (6+ words): 50 candidates

        Returns:
            Recommended candidate pool size
        """
        pass
```

**Implementation Checklist**:
- [ ] Query complexity heuristics (word count, entities, etc.)
- [ ] Pool size calculation logic
- [ ] Integration with CrossEncoderReranker
- [ ] Test coverage for various query types

#### 3. Scoring & Selection (Task 6.3)
**Location**: Within `CrossEncoderReranker.rerank()` method

**Minimum Requirements**:
```python
def _score_pairs(
    self,
    query: str,
    documents: List[str]
) -> List[float]:
    """
    Score query-document pairs using cross-encoder model.

    Args:
        query: Search query
        documents: List of document text

    Returns:
        List of relevance scores (0-1 range)
    """
    pass

def _select_top_k(
    self,
    results: List[SearchResult],
    scores: List[float],
    top_k: int
) -> List[SearchResult]:
    """
    Select top-k results based on cross-encoder scores.

    Preserves all SearchResult metadata while updating scores.

    Args:
        results: Original SearchResult objects
        scores: Cross-encoder relevance scores
        top_k: Number of top results to return

    Returns:
        Top-k SearchResult objects with updated scores
    """
    pass
```

**Implementation Checklist**:
- [ ] Batch inference implementation
- [ ] Score normalization/calibration
- [ ] Top-k selection algorithm
- [ ] Metadata preservation
- [ ] SearchResult score field update

---

## Integration Requirements

### Input Contract (from HybridSearch)
```python
# HybridSearch produces this:
results: List[SearchResult] = hybrid_search.search(query, top_k=50)

# Each SearchResult has:
# - doc_id: str
# - content: str
# - score: float (BM25 or vector score)
# - source: str
# - metadata: Dict[str, Any]
# - context: str (optional)
```

### Output Contract (to API/MCP)
```python
# CrossEncoderReranker must produce:
reranked: List[SearchResult] = cross_encoder.rerank(query, results, top_k=5)

# Requirements:
# - Same SearchResult structure
# - Updated score field with cross-encoder scores
# - Preserved metadata (doc_id, source, context)
# - Top-5 results by relevance
```

---

## Testing Requirements

### Unit Tests (Minimum)
**File**: `tests/test_cross_encoder_reranker.py`

```python
def test_model_loading():
    """Test cross-encoder model loads successfully."""
    pass

def test_rerank_basic():
    """Test basic reranking with sample SearchResult objects."""
    pass

def test_candidate_selection():
    """Test adaptive pool sizing for different query complexities."""
    pass

def test_score_preservation():
    """Test SearchResult metadata preservation after reranking."""
    pass

def test_empty_results():
    """Test handling of empty result sets."""
    pass

def test_performance_basic():
    """Test reranking latency is within acceptable bounds."""
    pass
```

**Coverage Target**: 85%+

### Integration Tests
```python
def test_hybrid_to_reranker_integration():
    """Test full pipeline: HybridSearch → CrossEncoderReranker."""
    pass

def test_search_result_format_compatibility():
    """Test SearchResult objects pass through unchanged except scores."""
    pass
```

---

## Type Safety Requirements

### Type Stub File
**File**: `src/search/cross_encoder_reranker.pyi`

Must include:
- All public class definitions
- All public method signatures
- Generic types (List[SearchResult], etc.)
- Union types where applicable
- Literal types for enums

### Validation
```bash
# Must pass with 0 errors:
mypy --strict src/search/cross_encoder_reranker.py
```

---

## Performance Targets (Task 6.4 Preparation)

While Task 6.4 focuses on optimization, initial implementation should aim for:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model loading | < 5 seconds | First inference only |
| Batch inference (50 pairs) | < 100ms | Average over 10 runs |
| Overall reranking | < 200ms | End-to-end latency |
| Memory usage | < 500MB | Peak during inference |

**Profiling Code** (include in tests):
```python
import time

def test_reranking_latency():
    """Measure end-to-end reranking latency."""
    reranker = CrossEncoderReranker()
    query = "sample query"
    results = [...]  # 50 SearchResult objects

    start = time.perf_counter()
    reranked = reranker.rerank(query, results, top_k=5)
    latency = (time.perf_counter() - start) * 1000  # ms

    assert latency < 200, f"Reranking too slow: {latency:.2f}ms"
```

---

## Code Quality Checklist

Before requesting architecture review, verify:

### Static Analysis
- [ ] `ruff check src/search/cross_encoder_reranker.py` → 0 issues
- [ ] `mypy --strict src/search/cross_encoder_reranker.py` → 0 errors
- [ ] `pytest --cov=src.search.cross_encoder_reranker --cov-report=term` → 85%+

### Documentation
- [ ] Module docstring with overview
- [ ] Class docstrings with purpose
- [ ] Method docstrings with Args/Returns
- [ ] Type annotations on all public APIs
- [ ] Comments on complex algorithms

### Error Handling
- [ ] Model loading failures handled gracefully
- [ ] Device fallback (GPU → CPU) implemented
- [ ] Empty result set handling
- [ ] Invalid input validation
- [ ] Informative error messages

---

## Reference Implementations

### Similar Code in Codebase
- **HybridSearch**: `src/search/hybrid_search.py` (729 lines)
  - Shows pattern for combining multiple search methods
  - Type safety examples
  - SearchResult handling
  - Error handling patterns

- **SearchResult Model**: `src/search/results.py` (565 lines)
  - Data structure to preserve
  - Metadata handling
  - Score field usage

### External Examples
- **Sentence-Transformers Cross-Encoder**: https://www.sbert.net/examples/applications/cross-encoder/README.html
- **ms-marco-MiniLM-L-6-v2 Model**: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

---

## Timeline Estimate

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Task 6.1 (Model Loading) | 1.5-2 hours | Includes basic tests |
| Task 6.2 (Candidate Selection) | 1-1.5 hours | Heuristics + integration |
| Task 6.3 (Scoring & Selection) | 1-1.5 hours | Inference + top-k logic |
| Type stubs + documentation | 0.5-1 hour | mypy compliance |
| **Total** | **4-6 hours** | For experienced dev |

---

## Next Steps

### For Development Team
1. ✅ Start with Task 6.1 (model loading)
   ```bash
   task-master set-status --id=6.1 --status=in-progress
   ```

2. ✅ Create initial implementation file
   ```bash
   touch src/search/cross_encoder_reranker.py
   touch src/search/cross_encoder_reranker.pyi
   touch tests/test_cross_encoder_reranker.py
   ```

3. ✅ Implement + test incrementally
   - Model loading → test → commit
   - Candidate selection → test → commit
   - Scoring → test → commit

4. ✅ Validate before review request
   ```bash
   ruff check src/search/cross_encoder_reranker.py
   mypy --strict src/search/cross_encoder_reranker.py
   pytest tests/test_cross_encoder_reranker.py -v
   ```

5. ✅ Update Task Master
   ```bash
   task-master set-status --id=6.1 --status=done
   task-master set-status --id=6.2 --status=done
   task-master set-status --id=6.3 --status=done
   ```

### For Architecture Reviewer (This Agent)
1. ⏸ Wait for implementation completion notification
2. ⏸ Run automated checks (mypy, ruff, pytest)
3. ⏸ Conduct architecture review per checklist
4. ⏸ Document findings in `2025-11-08-HHMM-architecture-review.md`
5. ⏸ Provide optimization recommendations for Task 6.4

---

## Questions/Clarifications

If unclear on any requirements, consult:
- HybridSearch implementation for patterns
- SearchResult model for data contracts
- Task 5 test suite for testing patterns
- Session handoff docs for context

---

**Document Status**: Ready for implementation
**Review Status**: Blocked until Tasks 6.1-6.3 complete
**Expected Review Time**: 40-60 minutes once code available
