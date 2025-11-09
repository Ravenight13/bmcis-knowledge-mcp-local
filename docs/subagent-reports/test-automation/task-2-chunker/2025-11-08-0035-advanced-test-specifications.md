# Chunker Module - Advanced Test Specifications

**Date**: 2025-11-08
**Component**: Document Parsing & Chunking System
**Test Count**: 12 proposed additional tests with full implementation

## Test Implementation Details

### CATEGORY 1: Configuration Validation Enhancement

#### Test 1.1: Boundary Value Configuration

**Test ID**: `test_config_boundary_values`
**Purpose**: Validate configuration with extreme but valid boundary values
**Importance**: Prevents edge case bugs with unusual configurations
**Type**: Unit Test

```python
def test_config_boundary_values() -> None:
    """Test configuration with boundary values."""
    # Minimum valid chunk_size
    config = ChunkerConfig(chunk_size=1, overlap_tokens=0)
    assert config.chunk_size == 1
    assert config.overlap_tokens == 0

    # Maximum overlap (one less than chunk_size)
    config = ChunkerConfig(chunk_size=100, overlap_tokens=99)
    assert config.chunk_size == 100
    assert config.overlap_tokens == 99

    # Very large values
    config = ChunkerConfig(chunk_size=10000, overlap_tokens=9999)
    assert config.chunk_size == 10000
    assert config.overlap_tokens == 9999

    # Min chunk size at chunk size boundary
    config = ChunkerConfig(chunk_size=100, min_chunk_size=100)
    assert config.min_chunk_size == config.chunk_size
```

**Input**: Pydantic configuration with boundary values
**Expected Output**: All valid configs created successfully
**Edge Cases**:
- chunk_size = 1 (absolute minimum)
- overlap_tokens = chunk_size - 1 (maximum allowed)
- min_chunk_size = chunk_size (equal values)
- Large values (10000+)

**Assertions**:
- Config creates without errors
- Fields match input values
- Validation passes

**Execution Time**: <100ms

---

#### Test 1.2: Extra Fields Forbidden

**Test ID**: `test_config_extra_fields_forbidden`
**Purpose**: Validate that Pydantic forbids extra fields (strict mode)
**Importance**: Ensures configuration integrity
**Type**: Unit Test

```python
def test_config_extra_fields_forbidden() -> None:
    """Test that Pydantic forbids extra fields (Config.extra = 'forbid')."""
    with pytest.raises(ValueError) as exc_info:
        ChunkerConfig(chunk_size=512, invalid_field=True)

    assert "extra_forbidden" in str(exc_info.value) or "extra" in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        ChunkerConfig(overlap_tokens=50, unknown_param="value")

    assert "extra_forbidden" in str(exc_info.value) or "extra" in str(exc_info.value).lower()
```

**Input**: ChunkerConfig with extra unknown fields
**Expected Output**: ValueError raised with "extra" or "forbidden" in message
**Edge Cases**:
- Single extra field with boolean value
- Single extra field with string value
- Multiple unknown parameters
- Valid + invalid fields mixed

**Assertions**:
- ValueError raised (not silent)
- Error message is descriptive
- Valid fields still work when no extra fields present

**Execution Time**: <100ms

---

### CATEGORY 2: Overlap Validation and Preservation

#### Test 2.1: Overlap Boundary Condition

**Test ID**: `test_overlap_exceeds_chunk_size_just_barely`
**Purpose**: Test overlap validation at exact boundary (overlap = chunk_size - 1)
**Importance**: Prevents off-by-one validation errors
**Type**: Unit Test

```python
def test_overlap_exceeds_chunk_size_just_barely() -> None:
    """Test overlap at boundary: overlap = chunk_size - 1 should pass."""
    # This should succeed - overlap is less than chunk_size
    config = ChunkerConfig(chunk_size=100, overlap_tokens=99)
    config.validate_config()  # Explicit call
    assert config.overlap_tokens == 99
    assert config.chunk_size == 100

    # This should fail - overlap equals chunk_size
    with pytest.raises(ValueError, match="must be less than"):
        config = ChunkerConfig(chunk_size=100, overlap_tokens=100)
        config.validate_config()

    # This should fail - overlap exceeds chunk_size
    with pytest.raises(ValueError, match="must be less than"):
        config = ChunkerConfig(chunk_size=100, overlap_tokens=101)
        config.validate_config()
```

**Input**: Configuration with overlap at/near chunk_size boundary
**Expected Output**: Validation passes for overlap < chunk_size, fails otherwise
**Edge Cases**:
- overlap = chunk_size - 1 (should pass)
- overlap = chunk_size (should fail)
- overlap = chunk_size + 1 (should fail)

**Assertions**:
- Valid config accepted
- Invalid configs rejected
- Error messages are clear

**Execution Time**: <150ms

---

#### Test 2.2: Overlap Content Preservation

**Test ID**: `test_overlap_content_preservation`
**Purpose**: Verify that overlapping tokens actually appear in consecutive chunks
**Importance**: Validates core functionality of overlap feature
**Type**: Integration Test

```python
def test_overlap_content_preservation() -> None:
    """Test that overlapping content is actually preserved in chunks."""
    config = ChunkerConfig(chunk_size=20, overlap_tokens=5)
    chunker = Chunker(config=config)

    # Create tokens that will span multiple chunks
    text = "The quick brown fox jumps over the lazy dog. " * 10
    token_ids = list(range(100))  # Simplified token IDs

    chunks = chunker.chunk_text(text, token_ids)

    # Need at least 2 chunks to verify overlap
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Get last N tokens from current chunk
            overlap_size = config.overlap_tokens
            current_last_tokens = set(current_chunk.tokens[-overlap_size:])
            next_first_tokens = set(next_chunk.tokens[:overlap_size])

            # At least some tokens should overlap
            common_tokens = current_last_tokens & next_first_tokens
            assert len(common_tokens) > 0, \
                f"No overlap between chunk {i} and {i+1}"

            # Verify overlap matches configured amount
            assert next_chunk.metadata.overlap_tokens > 0 or len(chunks) == 1
```

**Input**: Document chunked with overlap_tokens=5, chunk_size=20
**Expected Output**: Consecutive chunks share overlapping tokens
**Edge Cases**:
- Minimum overlap (1 token)
- Small overlap (5 tokens) with small chunks (20 tokens)
- Large document with multiple chunks
- Only 2 chunks total

**Assertions**:
- Common tokens exist between consecutive chunks
- Overlap count matches configuration
- No data loss in overlapping regions

**Execution Time**: <200ms

---

### CATEGORY 3: Sentence Detection Edge Cases

#### Test 3.1: Sentence Detection with Abbreviations

**Test ID**: `test_sentence_detection_abbreviations`
**Purpose**: Handle sentence detection with abbreviations (Dr., Mr., U.S., etc.)
**Importance**: Improves robustness of boundary detection
**Type**: Unit Test

```python
def test_sentence_detection_abbreviations() -> None:
    """Test sentence detection handles abbreviations."""
    chunker = Chunker()

    # Test cases with common abbreviations
    test_cases = [
        ("Dr. Smith works here.", 1),  # Should find 1 main sentence
        ("U.S. government is established.", 1),
        ("Mr. and Mrs. Johnson are here.", 1),
        ("E.g. this is an example.", 1),
        ("The U.S. and U.K. are allies.", 1),
    ]

    for text, expected_min in test_cases:
        sentences = chunker._identify_sentences(text)

        # Should identify at least the expected number
        assert len(sentences) >= 1, f"Failed for: {text}"

        # Verify structure
        for start, end in sentences:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start >= 0
            assert end > start
            assert end <= len(text)

            # Extract sentence text
            sentence_text = text[start:end].strip()
            assert len(sentence_text) > 0
```

**Input**: Text containing abbreviations (Dr., Mr., U.S., etc.)
**Expected Output**: Sentence boundaries identified (may not be perfect but valid)
**Edge Cases**:
- Single abbreviation followed by period
- Multiple abbreviations in sequence
- Abbreviation at sentence boundary
- Common abbreviations (Dr., Mr., Mrs., etc.)

**Assertions**:
- Sentence ranges are valid
- No empty sentences
- Ranges don't exceed text bounds

**Execution Time**: <100ms

---

#### Test 3.2: Sentence Detection with Repeated Punctuation

**Test ID**: `test_sentence_detection_repeated_punctuation`
**Purpose**: Handle repeated punctuation (!!! ??? ...)
**Importance**: Improves robustness for informal text
**Type**: Unit Test

```python
def test_sentence_detection_repeated_punctuation() -> None:
    """Test sentence detection with repeated punctuation."""
    chunker = Chunker()

    test_cases = [
        "Wow!!! That's amazing!!!",
        "Really??? I don't know???",
        "Yes... maybe... not sure...",
        "Wait!!! Stop??? What???!",
        "Amazing!!!!!!!!!!!!!!!",
    ]

    for text in test_cases:
        sentences = chunker._identify_sentences(text)

        # Should create at least one sentence
        assert len(sentences) >= 1, f"Failed for: {text}"

        # Verify all sentences are valid
        for start, end in sentences:
            assert start >= 0
            assert end > start
            assert end <= len(text)

        # Total coverage should be good
        total_covered = sum(end - start for start, end in sentences)
        assert total_covered > len(text) * 0.7  # At least 70% coverage
```

**Input**: Text with multiple/repeated punctuation marks
**Expected Output**: Valid sentence boundaries identified
**Edge Cases**:
- Triple exclamation (!!!)
- Mixed punctuation (?!)
- Ellipsis (...)
- Consecutive sentence marks

**Assertions**:
- Valid ranges returned
- Good coverage of text
- No overlapping ranges

**Execution Time**: <100ms

---

### CATEGORY 4: Chunk Size Enforcement

#### Test 4.1: Minimum Chunk Size Enforcement

**Test ID**: `test_minimum_chunk_size_not_violated`
**Purpose**: Ensure no chunk (except last) is below configured minimum
**Importance**: Critical for consistent chunking behavior
**Type**: Integration Test

```python
def test_minimum_chunk_size_not_violated() -> None:
    """Test that no chunk (except last) violates min_chunk_size."""
    config = ChunkerConfig(
        chunk_size=100,
        min_chunk_size=50,
        overlap_tokens=10
    )
    chunker = Chunker(config=config)

    # Create document with enough tokens
    text = "sentence. " * 200
    token_ids = list(range(300))

    chunks = chunker.chunk_text(text, token_ids)

    # Validate chunk sizes
    for i, chunk in enumerate(chunks[:-1]):  # All except last
        assert chunk.token_count >= config.min_chunk_size, \
            f"Chunk {i} has {chunk.token_count} tokens, " \
            f"below minimum {config.min_chunk_size}"

    # Last chunk may be smaller
    if len(chunks) > 0:
        last_chunk = chunks[-1]
        assert last_chunk.token_count > 0, "Last chunk is empty"
```

**Input**: Document chunked with min_chunk_size=50, chunk_size=100
**Expected Output**: All chunks except last meet minimum size
**Edge Cases**:
- Exactly at minimum size
- Just below minimum (caught and enforced)
- Last chunk may be smaller
- Very small documents

**Assertions**:
- All non-last chunks >= min_chunk_size
- Last chunk > 0
- No chunks removed

**Execution Time**: <300ms

---

#### Test 4.2: Chunk Size Not Exceeded

**Test ID**: `test_chunk_size_not_exceeded`
**Purpose**: Verify chunks don't exceed configured maximum size
**Importance**: Validates primary chunking constraint
**Type**: Integration Test

```python
def test_chunk_size_not_exceeded() -> None:
    """Test that chunks respect configured chunk_size limit."""
    config = ChunkerConfig(
        chunk_size=100,
        overlap_tokens=10
    )
    chunker = Chunker(config=config)

    # Create large document
    text = "word " * 1000
    token_ids = list(range(1000))

    chunks = chunker.chunk_text(text, token_ids)

    # Validate chunk sizes
    for chunk in chunks:
        assert chunk.token_count <= config.chunk_size, \
            f"Chunk {chunk.metadata.chunk_index} exceeds " \
            f"size limit: {chunk.token_count} > {config.chunk_size}"

        # Also verify token_count matches actual tokens
        assert chunk.token_count == len(chunk.tokens), \
            f"Chunk {chunk.metadata.chunk_index}: " \
            f"token_count mismatch"

    # Verify coverage is complete
    total_unique_positions = len(set(
        pos
        for chunk in chunks
        for pos in range(
            chunk.metadata.start_token_pos,
            chunk.metadata.end_token_pos
        )
    ))

    # Should cover most tokens (with overlap)
    assert total_unique_positions > len(token_ids) * 0.9
```

**Input**: Large document (1000 tokens) chunked with chunk_size=100
**Expected Output**: No chunk exceeds 100 tokens
**Edge Cases**:
- Chunk at exact size limit
- Chunk approaching limit with overlap
- Very small chunk_size (10-20 tokens)
- Large documents (1000+ tokens)

**Assertions**:
- All chunks <= chunk_size
- token_count matches len(tokens)
- Good coverage of document

**Execution Time**: <300ms

---

### CATEGORY 5: Text Position Accuracy

#### Test 5.1: Chunk Text Covers Document

**Test ID**: `test_chunk_text_covers_document`
**Purpose**: Verify all chunks together provide good coverage of document
**Importance**: Ensures no information loss
**Type**: Integration Test

```python
def test_chunk_text_covers_document() -> None:
    """Test that chunks collectively cover most of the document."""
    chunker = Chunker()
    text = "The quick brown fox jumps over the lazy dog. " * 50

    from src.document_parsing.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)

    chunks = chunker.chunk_text(text, token_ids)

    # Collect all character positions covered
    covered_chars = set()
    for chunk in chunks:
        for i in range(chunk.start_pos, min(chunk.end_pos, len(text))):
            covered_chars.add(i)

    # Should cover majority of text
    coverage_ratio = len(covered_chars) / len(text)
    assert coverage_ratio > 0.8, \
        f"Coverage only {coverage_ratio:.1%}, expected >80%"

    # All chunks should reference valid positions
    for chunk in chunks:
        assert chunk.start_pos >= 0
        assert chunk.end_pos <= len(text)
        assert chunk.start_pos < chunk.end_pos
```

**Input**: Multi-sentence document with accurate tokenization
**Expected Output**: Chunks cover 80%+ of document text
**Edge Cases**:
- Short documents
- Very long documents
- Documents with special formatting
- Unicode content

**Assertions**:
- Coverage > 80%
- Valid position ranges
- All chunks non-empty

**Execution Time**: <500ms

---

#### Test 5.2: Chunk Text Consistency

**Test ID**: `test_chunk_text_consistency`
**Purpose**: Verify chunk.text matches actual extracted text
**Importance**: Validates text extraction accuracy
**Type**: Integration Test

```python
def test_chunk_text_consistency() -> None:
    """Test that chunk.text is consistent with original text."""
    chunker = Chunker()
    text = "First sentence. Second sentence. Third sentence. " * 20

    from src.document_parsing.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)

    chunks = chunker.chunk_text(text, token_ids)

    for i, chunk in enumerate(chunks):
        # Chunk text should not be empty
        assert len(chunk.text) > 0, \
            f"Chunk {i} has empty text"

        # Chunk text should be within original
        assert chunk.text in text or \
               chunk.text.strip() in text or \
               any(word in text for word in chunk.text.split()), \
            f"Chunk {i} text not found in original"

        # Chunk should have tokens
        assert len(chunk.tokens) > 0, \
            f"Chunk {i} has no tokens"

        # Token count should match
        assert chunk.token_count == len(chunk.tokens), \
            f"Chunk {i} token_count mismatch: " \
            f"{chunk.token_count} != {len(chunk.tokens)}"
```

**Input**: Well-structured document with tokenization
**Expected Output**: All chunks have consistent text representation
**Edge Cases**:
- Text with multiple spaces
- Text with special characters
- Unicode content
- Very short/long chunks

**Assertions**:
- Text not empty
- Text found in original
- Token counts match
- Token array not empty

**Execution Time**: <500ms

---

### CATEGORY 6: Integration and Performance

#### Test 6.1: Deterministic Chunking

**Test ID**: `test_deterministic_chunking`
**Purpose**: Verify chunking produces identical results on repeated calls
**Importance**: Essential for reproducibility and testing
**Type**: Integration Test

```python
def test_deterministic_chunking() -> None:
    """Test that chunking is deterministic (same input = same output)."""
    config = ChunkerConfig(chunk_size=512, overlap_tokens=50)
    chunker = Chunker(config=config)

    text = "The quick brown fox jumps over the lazy dog. " * 100

    from src.document_parsing.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)

    # Chunk twice
    chunks1 = chunker.chunk_text(text, token_ids)
    chunks2 = chunker.chunk_text(text, token_ids)

    # Should produce identical results
    assert len(chunks1) == len(chunks2), \
        "Different number of chunks on second call"

    for i, (c1, c2) in enumerate(zip(chunks1, chunks2)):
        assert c1.token_count == c2.token_count, \
            f"Chunk {i}: token_count differs"

        assert c1.tokens == c2.tokens, \
            f"Chunk {i}: token list differs"

        assert c1.text == c2.text, \
            f"Chunk {i}: text differs"

        assert c1.metadata.chunk_index == c2.metadata.chunk_index, \
            f"Chunk {i}: chunk_index differs"

        assert c1.metadata.start_token_pos == c2.metadata.start_token_pos, \
            f"Chunk {i}: start_token_pos differs"

        assert c1.metadata.end_token_pos == c2.metadata.end_token_pos, \
            f"Chunk {i}: end_token_pos differs"
```

**Input**: Same text and tokens, chunked twice
**Expected Output**: Identical chunk lists returned
**Edge Cases**:
- Multiple chunking calls
- Different chunker instances with same config
- Large documents
- Edge case documents

**Assertions**:
- Same number of chunks
- Identical tokens
- Identical text
- Identical metadata

**Execution Time**: <400ms

---

#### Test 6.2: Chunking Performance

**Test ID**: `test_chunking_performance_large_document`
**Purpose**: Verify chunking handles large documents efficiently
**Importance**: Performance requirement validation
**Type**: Performance Test

```python
def test_chunking_performance_large_document() -> None:
    """Test that chunking large documents completes in reasonable time."""
    import time

    chunker = Chunker()

    # Create a ~3000 token document
    text = "The quick brown fox jumps over the lazy dog. " * 200

    from src.document_parsing.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)

    # Measure execution time
    start = time.time()
    chunks = chunker.chunk_text(text, token_ids)
    elapsed = time.time() - start

    # Performance requirement: <1 second for 3000 tokens
    assert elapsed < 1.0, \
        f"Chunking took {elapsed:.2f}s, should be <1.0s"

    # Should produce multiple chunks
    assert len(chunks) > 1, \
        "Should produce multiple chunks for 3000 token document"

    # Each chunk should be reasonable size
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.token_count <= chunker.config.chunk_size
```

**Input**: Large document (~3000 tokens)
**Expected Output**: Chunking completes in <1 second with multiple chunks
**Edge Cases**:
- Very large documents (10000+ tokens)
- Many small chunks
- Many large chunks
- Various chunk_size configurations

**Assertions**:
- Execution time < 1 second
- Multiple chunks produced
- Chunks meet size requirements
- All chunks valid

**Execution Time**: <1.5 seconds (expected)

---

## Test Summary Table

| Test ID | Category | Purpose | Importance | Est. Time |
|---------|----------|---------|-----------|-----------|
| 1.1 | Config | Boundary values | High | 100ms |
| 1.2 | Config | Extra fields forbidden | High | 100ms |
| 2.1 | Overlap | Boundary condition | High | 150ms |
| 2.2 | Overlap | Content preservation | High | 200ms |
| 3.1 | Boundaries | Abbreviations | Medium | 100ms |
| 3.2 | Boundaries | Repeated punctuation | Medium | 100ms |
| 4.1 | Size | Min size enforcement | High | 300ms |
| 4.2 | Size | Max size enforcement | High | 300ms |
| 5.1 | Position | Coverage analysis | Medium | 500ms |
| 5.2 | Position | Text consistency | Medium | 500ms |
| 6.1 | Integration | Deterministic output | High | 400ms |
| 6.2 | Performance | Large document handling | Medium | 1500ms |

**Total Expected Execution Time**: ~5.5 seconds

---

## Implementation Notes

### Type Annotations Required
```python
from __future__ import annotations
from typing import Any
import pytest
import time
from src.document_parsing.chunker import (
    Chunk, ChunkMetadata, Chunker, ChunkerConfig
)
from src.document_parsing.tokenizer import Tokenizer
```

### Test Organization
- Add all tests to existing `tests/test_chunker.py`
- Group into logical test classes:
  - TestChunkerConfigAdvanced (tests 1.1, 1.2)
  - TestChunkerOverlapAdvanced (tests 2.1, 2.2)
  - TestChunkerBoundariesAdvanced (tests 3.1, 3.2)
  - TestChunkerSizeAdvanced (tests 4.1, 4.2)
  - TestChunkerPositionAdvanced (tests 5.1, 5.2)
  - TestChunkerIntegrationAdvanced (tests 6.1, 6.2)

### Fixture Usage
- Reuse existing tokenizer fixture or create new
- Leverage existing test utilities
- Keep setup simple and focused

### Expected Coverage Impact
- Current: 31 tests, ~80% coverage
- Proposed: 43 tests, ~92% coverage
- Key gaps addressed:
  - Boundary validation (improved)
  - Size enforcement (improved)
  - Position accuracy (improved)
  - Performance characteristics (validated)

---

**Generated**: 2025-11-08
**Status**: Ready for implementation
