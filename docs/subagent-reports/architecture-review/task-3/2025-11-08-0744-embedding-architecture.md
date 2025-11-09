# Task 3: Embedding Generation Pipeline - Architecture Review

**Date**: 2025-11-08
**Time**: 07:44
**Task**: Architecture review for sentence-transformers embedding generation pipeline
**Reviewer**: Claude Code (Architecture Review Agent)

---

## Executive Summary

This document provides a comprehensive architecture review for Task 3: Embedding Generation Pipeline. The pipeline will generate 768-dimensional embeddings for 2,600+ document chunks using the sentence-transformers `all-mpnet-base-v2` model and insert them into PostgreSQL with pgvector HNSW indexing.

**Key Architectural Decisions**:
1. **Single-process, batch-oriented design** for simplicity and memory control
2. **Model singleton pattern** with lazy initialization and CPU/MPS device auto-detection
3. **Batch size of 32 chunks** for optimal throughput without OOM risk
4. **Streaming database updates** using UPDATE queries with batch commits
5. **Type-safe vector handling** using `list[float]` throughout the stack

**Performance Targets**:
- Embedding generation: 10-50 chunks/second (CPU/MPS dependent)
- Total processing time: ~2-5 minutes for 2,600 chunks
- Memory footprint: <2GB RAM (model + batch overhead)

---

## 1. System Context & Integration Points

### 1.1 Phase 0 Infrastructure Integration

The embedding pipeline leverages all Phase 0 systems:

```python
# Configuration System
from src.core.config import get_settings
settings = get_settings()

# Database Pooling
from src.core.database import DatabasePool
with DatabasePool.get_connection() as conn:
    # ... embedding updates

# Structured Logging
from src.core.logging import StructuredLogger
logger = StructuredLogger.get_logger(__name__)
```

**Integration Benefits**:
- **Config**: Centralized settings for model paths, batch sizes, device selection
- **DatabasePool**: Automatic retry logic, health checks, connection management
- **StructuredLogger**: JSON logging with embedding metrics (chunks/sec, errors)

### 1.2 Phase 1 Document Parsing Integration

The embedding pipeline consumes `ProcessedChunk` objects from Phase 1:

```python
from src.document_parsing.models import ProcessedChunk

# Input: ProcessedChunk with NULL embeddings
chunk = ProcessedChunk(
    chunk_text="...",
    chunk_hash="abc123...",
    embedding=None,  # ← Phase 1 leaves this NULL
    # ... other fields
)

# Output: ProcessedChunk with 768-dim embeddings
chunk.embedding = [0.123, -0.456, ...]  # 768 floats
```

**Database State Assumptions**:
- `knowledge_base` table populated with 2,600+ rows (Phase 1 complete)
- All rows have `embedding IS NULL`
- `chunk_hash` unique constraint already enforced
- HNSW index created but empty (no vectors yet)

### 1.3 External Dependencies

**New Dependencies** (to add to `requirements.txt`):

```txt
# Embedding generation
sentence-transformers>=2.2.0
torch>=2.0.0                    # PyTorch (CPU or MPS on macOS)
transformers>=4.30.0            # HuggingFace transformers (transitive dependency)
```

**Dependency Rationale**:
- `sentence-transformers`: Provides `all-mpnet-base-v2` model with simple API
- `torch`: Required by sentence-transformers for model inference
- `transformers`: Transitive dependency, but pinned for stability

**Model Download**:
- First run: Downloads ~420MB model from HuggingFace Hub
- Cached to: `~/.cache/torch/sentence_transformers/`
- Subsequent runs: Load from cache (2-3 seconds)

---

## 2. Embedding Pipeline Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 EmbeddingGenerator (Class)                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Model Singleton                                     │   │
│  │  - SentenceTransformer('all-mpnet-base-v2')        │   │
│  │  - Device: auto-detect (MPS > CPU)                 │   │
│  │  - Cached in memory after first load               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Batch Processor                                     │   │
│  │  - Fetch chunks from DB (WHERE embedding IS NULL)  │   │
│  │  - Group into batches of 32                        │   │
│  │  - Generate embeddings (model.encode)              │   │
│  │  - Validate 768 dimensions                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Database Updater                                    │   │
│  │  - UPDATE knowledge_base SET embedding = %s        │   │
│  │  - WHERE chunk_hash = %s                           │   │
│  │  - Batch commit every 32 chunks                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Statistics Tracker                                  │   │
│  │  - Chunks processed / failed                       │   │
│  │  - Processing time / chunks per second             │   │
│  │  - Memory usage / model load time                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↑                                          ↓
         │                                          │
    DatabasePool                            StructuredLogger
    (Phase 0)                                  (Phase 0)
```

### 2.2 Data Flow

```
┌──────────────┐
│  knowledge_  │  SELECT chunk_hash, chunk_text
│     base     │  FROM knowledge_base
│   table      │  WHERE embedding IS NULL
│ (2,600 rows) │  ORDER BY chunk_hash
└──────┬───────┘
       │
       ↓ Fetch in batches of 32
┌──────────────────────────────────────┐
│ Batch 1: 32 chunks                   │
│  - chunk_hash_1 → "text content..."  │
│  - chunk_hash_2 → "more content..."  │
│  - ...                               │
└──────┬───────────────────────────────┘
       │
       ↓ model.encode(texts, batch_size=32)
┌──────────────────────────────────────┐
│ Embeddings: 32 × 768-dim vectors     │
│  - [0.123, -0.456, ...] (768 floats) │
│  - [0.789, 0.234, ...] (768 floats)  │
│  - ...                               │
└──────┬───────────────────────────────┘
       │
       ↓ Validate dimensions & UPDATE
┌──────────────────────────────────────┐
│ UPDATE knowledge_base                │
│ SET embedding = %s                   │
│ WHERE chunk_hash = %s                │
│                                      │
│ (Repeat for all 32 chunks)           │
│ COMMIT transaction                   │
└──────┬───────────────────────────────┘
       │
       ↓ Repeat until all chunks processed
┌──────────────┐
│  knowledge_  │  2,600 rows with
│     base     │  embedding NOT NULL
│   table      │  HNSW index populated
└──────────────┘
```

---

## 3. Design Decisions & Justifications

### 3.1 Model Loading Strategy

**Decision**: **Singleton pattern with lazy initialization**

```python
class EmbeddingGenerator:
    _model: SentenceTransformer | None = None
    _device: str | None = None

    @classmethod
    def _initialize_model(cls) -> None:
        """Initialize model singleton on first use."""
        if cls._model is not None:
            return

        # Auto-detect best device
        if torch.backends.mps.is_available():
            cls._device = "mps"  # Apple Silicon GPU
        else:
            cls._device = "cpu"

        cls._model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=cls._device,
        )
        logger.info(
            "Model loaded",
            extra={"model": "all-mpnet-base-v2", "device": cls._device}
        )
```

**Rationale**:
1. **Single instance**: Model weights (~420MB) loaded once, shared across all calls
2. **Lazy loading**: Model only loaded when first embedding request occurs
3. **Device auto-detection**: MPS (Apple Silicon) > CPU for optimal performance
4. **Memory efficiency**: Singleton prevents multiple copies of model in memory

**Alternatives Considered**:
- ❌ **Per-worker instances**: Too memory-intensive (420MB × workers)
- ❌ **Process pool**: IPC overhead + serialization complexity
- ❌ **GPU-only**: Not portable (requires CUDA setup)

### 3.2 Batch Processing Strategy

**Decision**: **Batch size of 32 chunks with streaming updates**

```python
def generate_embeddings_batch(self, batch_size: int = 32) -> EmbeddingStats:
    """Generate embeddings for all NULL chunks in batches."""

    with DatabasePool.get_connection() as conn:
        # Fetch chunks in batches
        while True:
            chunks = self._fetch_chunk_batch(conn, batch_size)
            if not chunks:
                break

            # Generate embeddings (32 texts → 32 × 768 vectors)
            texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,  # numpy for efficiency
            )

            # Update database
            self._update_chunk_embeddings(conn, chunks, embeddings)
            conn.commit()
```

**Rationale**:
1. **Batch size 32**: Optimal balance between throughput and memory (from sentence-transformers docs)
2. **Streaming fetch**: Avoids loading all 2,600 chunks into memory at once
3. **Batch commits**: Reduces transaction overhead (32 UPDATEs per commit)
4. **NumPy arrays**: More memory-efficient than Python lists for intermediate processing

**Performance Analysis**:
- **Memory**: 32 chunks × 500 chars × 2 bytes = ~32KB text + 32 × 768 × 4 bytes = ~98KB embeddings = **130KB per batch**
- **Throughput**: 32 chunks/batch × 0.5-2 sec/batch = **16-64 chunks/sec**
- **Total time**: 2,600 chunks ÷ 32 chunks/sec (avg) = **~81 seconds (~1.4 min)**

**Alternatives Considered**:
- ❌ **Batch size 1**: Too slow (10x more overhead)
- ❌ **Batch size 512**: OOM risk on CPU (memory scales linearly)
- ❌ **Load all chunks**: 2,600 chunks × 500 bytes = 1.3MB (acceptable, but streaming is safer)

### 3.3 Parallel Execution Pattern

**Decision**: **Single-process, sequential batches (no parallelism)**

**Rationale**:
1. **Model bottleneck**: sentence-transformers uses internal parallelism (PyTorch threads)
2. **GIL limitations**: Python multiprocessing adds overhead without benefit (CPU-bound but model is optimized)
3. **Simplicity**: Sequential processing easier to debug, log, and recover from errors
4. **Memory safety**: Avoids OOM from multiple model instances or large queues

**When to reconsider**:
- ✅ **Future GPU support**: Batch size could increase to 128-256 on CUDA
- ✅ **Multiple models**: If using ensemble of models, process pool could help
- ✅ **I/O-bound phase**: If fetching from S3 or network storage (not applicable here)

### 3.4 Error Handling & Recovery

**Decision**: **Chunk-level retry with batch isolation**

```python
def _update_chunk_embeddings(
    self,
    conn: Connection,
    chunks: list[dict],
    embeddings: np.ndarray
) -> None:
    """Update chunks with embeddings, handling errors gracefully."""

    for i, chunk in enumerate(chunks):
        try:
            # Validate embedding dimensions
            if embeddings[i].shape != (768,):
                raise ValueError(f"Invalid embedding shape: {embeddings[i].shape}")

            # Convert numpy to list for PostgreSQL
            embedding_list = embeddings[i].tolist()

            # Update single chunk
            cur.execute(
                "UPDATE knowledge_base SET embedding = %s WHERE chunk_hash = %s",
                (embedding_list, chunk["chunk_hash"])
            )

            self.stats.chunks_processed += 1

        except Exception as e:
            logger.error(
                "Failed to update chunk embedding",
                extra={"chunk_hash": chunk["chunk_hash"], "error": str(e)}
            )
            self.stats.chunks_failed += 1
            # Continue processing remaining chunks in batch
```

**Rationale**:
1. **Chunk isolation**: Single chunk failure doesn't block entire batch
2. **Dimension validation**: Catches model errors early (expected: 768)
3. **Structured logging**: Every failure logged with chunk_hash for debugging
4. **Graceful degradation**: Processing continues, failed chunks reported in stats

**Error Scenarios**:
- ✅ **Model output mismatch**: Validate 768 dimensions before UPDATE
- ✅ **Database deadlock**: DatabasePool retry logic handles transient failures
- ✅ **OOM during encoding**: Batch size 32 keeps memory bounded (worst case: skip batch, continue)

### 3.5 Type Safety & Vector Handling

**Decision**: **Use `list[float]` for embeddings throughout the stack**

```python
class ProcessedChunk(BaseModel):
    embedding: list[float] | None = Field(
        default=None,
        description="768-dimensional vector embedding"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimensions(cls, v: list[float] | None) -> list[float] | None:
        """Validate embedding is exactly 768 dimensions."""
        if v is not None and len(v) != 768:
            raise ValueError(f"Embedding must be 768-dimensional, got {len(v)}")
        return v
```

**Type Flow**:
```
NumPy Array (model output)
    ↓ .tolist()
Python list[float] (ProcessedChunk.embedding)
    ↓ psycopg parameter
PostgreSQL vector(768)
```

**Rationale**:
1. **Pydantic compatibility**: `list[float]` works natively with Pydantic v2
2. **PostgreSQL compatibility**: psycopg2 serializes lists to PostgreSQL arrays automatically
3. **Type safety**: mypy can validate `list[float]` (unlike `np.ndarray`)
4. **Simplicity**: No custom type converters or serialization logic needed

**Performance Consideration**:
- **Overhead**: NumPy → list conversion is O(768) = negligible (~1μs per chunk)
- **Memory**: Python list uses ~3x memory of NumPy array (768 floats × 16 bytes vs 768 × 4 bytes)
  - **Impact**: 32 chunks × 768 floats × 12 bytes overhead = **295KB per batch** (acceptable)

---

## 4. Configuration & Settings

### 4.1 Embedding Configuration

**New Configuration Class**:

```python
class EmbeddingConfig(BaseSettings):
    """Configuration for embedding generation."""

    model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence-transformers model name"
    )
    batch_size: int = Field(
        default=32,
        description="Embedding batch size",
        ge=1,
        le=512,
    )
    device: str | None = Field(
        default=None,  # Auto-detect
        description="Device for model (cpu, mps, cuda)"
    )
    embedding_dimensions: int = Field(
        default=768,
        description="Expected embedding dimensions",
        ge=1,
    )
    max_seq_length: int = Field(
        default=384,
        description="Maximum token sequence length for model"
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
    )
```

**Environment Variables**:
```bash
# Optional overrides
EMBEDDING_MODEL_NAME=all-mpnet-base-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu  # or mps, cuda
EMBEDDING_DIMENSIONS=768
EMBEDDING_MAX_SEQ_LENGTH=384
```

**Integration with `Settings`**:

```python
class Settings(BaseSettings):
    # Existing fields...
    database: DatabaseConfig
    logging: LoggingConfig
    application: ApplicationConfig

    # New field
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding generation configuration"
    )
```

### 4.2 Model Caching Strategy

**Cache Location**:
```
~/.cache/torch/sentence_transformers/sentence-transformers_all-mpnet-base-v2/
├── config.json                    # Model configuration
├── pytorch_model.bin             # Model weights (420MB)
├── tokenizer_config.json         # Tokenizer settings
├── vocab.txt                     # Vocabulary
└── modules.json                  # Model architecture
```

**Cache Behavior**:
1. **First run**: Downloads from HuggingFace Hub (~420MB, 30-60 sec on fast connection)
2. **Subsequent runs**: Loads from cache (~2-3 sec)
3. **Cache invalidation**: Manual deletion only (no automatic expiration)

**Offline Support**:
- ✅ If cache exists, works offline
- ❌ If cache missing, requires internet for first download

---

## 5. Database Integration

### 5.1 Chunk Fetching Strategy

**Query Pattern**:

```python
def _fetch_chunk_batch(
    self,
    conn: Connection,
    batch_size: int
) -> list[dict[str, Any]]:
    """Fetch next batch of chunks without embeddings."""

    query = """
        SELECT chunk_hash, chunk_text
        FROM knowledge_base
        WHERE embedding IS NULL
        ORDER BY chunk_hash  -- Consistent ordering for resumability
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(query, (batch_size,))
        rows = cur.fetchall()

    return [
        {"chunk_hash": row[0], "chunk_text": row[1]}
        for row in rows
    ]
```

**Design Decisions**:
- **`WHERE embedding IS NULL`**: Idempotent - reruns only process remaining chunks
- **`ORDER BY chunk_hash`**: Deterministic ordering enables resumability after crashes
- **`LIMIT` only**: Efficient, no need for OFFSET (WHERE clause filters completed chunks)

**Performance**:
- **Index usage**: Query uses `idx_knowledge_embedding` (HNSW index on embedding column)
- **Query cost**: O(log N) index scan + O(batch_size) fetch = **~1ms per batch**

### 5.2 Embedding Update Strategy

**Update Query**:

```python
def _update_chunk_embeddings(
    self,
    conn: Connection,
    chunks: list[dict],
    embeddings: np.ndarray
) -> None:
    """Update chunks with embeddings in batch transaction."""

    update_sql = """
        UPDATE knowledge_base
        SET embedding = %s::vector(768)
        WHERE chunk_hash = %s
    """

    with conn.cursor() as cur:
        for i, chunk in enumerate(chunks):
            embedding_list = embeddings[i].tolist()
            cur.execute(update_sql, (embedding_list, chunk["chunk_hash"]))

        # Commit batch
        conn.commit()
```

**Rationale**:
1. **Individual UPDATEs**: Simple, easy to debug, chunk-level error isolation
2. **Batch commits**: 32 UPDATEs per COMMIT reduces transaction overhead
3. **Explicit cast `::vector(768)`**: Ensures type safety at database level

**Alternatives Considered**:
- ❌ **executemany()**: psycopg2's executemany doesn't optimize for PostgreSQL (no real benefit)
- ❌ **COPY command**: Requires temp table + merge, adds complexity
- ❌ **Single UPDATE with unnest()**: Harder to debug, loses chunk-level error handling

### 5.3 HNSW Index Behavior

**Index Characteristics**:

```sql
CREATE INDEX idx_knowledge_embedding ON knowledge_base
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Build Strategy**:
- **Lazy build**: HNSW index built incrementally as embeddings added
- **Memory impact**: Index growth is O(N × log N) for HNSW
- **Query availability**: Index usable immediately (even partial)

**Performance Impact**:
- **Insert cost**: Each UPDATE triggers index update (~0.5-2ms overhead per chunk)
- **Total overhead**: 2,600 chunks × 1ms (avg) = **~2.6 seconds** index build time
- **Benefit**: No separate index rebuild step needed

---

## 6. Performance Considerations

### 6.1 Throughput Analysis

**Baseline Performance** (CPU, Apple M1/M2):

| Phase                    | Time per Batch (32 chunks) | Chunks/Sec |
|--------------------------|----------------------------|------------|
| Database fetch           | ~5ms                       | 6,400      |
| Model encoding (CPU)     | 500-2000ms                 | 16-64      |
| Embedding validation     | ~1ms                       | 32,000     |
| Database update + commit | ~50ms                      | 640        |
| **Total**                | **~556-2056ms**            | **16-58**  |

**Total Time Estimate**:
- **Best case** (MPS): 2,600 ÷ 58 = **~45 seconds**
- **Typical case** (CPU): 2,600 ÷ 32 = **~81 seconds**
- **Worst case** (slow CPU): 2,600 ÷ 16 = **~163 seconds (~2.7 min)**

**Bottleneck**: Model encoding (90-97% of total time)

### 6.2 Memory Usage Analysis

**Components**:

| Component                | Memory Usage          | Notes                          |
|--------------------------|-----------------------|--------------------------------|
| Model weights            | ~420MB                | Loaded once, shared            |
| Model runtime overhead   | ~200MB                | PyTorch buffers                |
| Text batch (32 chunks)   | ~32KB                 | 500 chars/chunk avg            |
| Embedding batch          | ~98KB                 | 32 × 768 × 4 bytes (numpy)     |
| Python list overhead     | ~295KB                | 32 × 768 × 12 bytes extra      |
| Database connection pool | ~5-10MB               | 5-20 connections               |
| **Total**                | **~620-730MB**        | Peak during batch processing   |

**Memory Safety**:
- ✅ Total usage <1GB on systems with 8GB+ RAM
- ✅ Batch size 32 prevents OOM even on 4GB systems
- ✅ Model caching in RAM avoids repeated loads

### 6.3 Optimization Opportunities

**Short-term** (Phase 2):
1. **MPS device support**: 3-5x speedup on Apple Silicon (already planned)
2. **Batch size tuning**: Test 64/128 on MPS (may improve throughput)
3. **Connection pooling tuning**: Increase pool size if update phase is slow

**Long-term** (Phase 3+):
1. **GPU support**: 10-50x speedup with CUDA (requires infrastructure)
2. **Model quantization**: Reduce model size to ~100MB with minimal accuracy loss
3. **Async database updates**: Pipeline fetch → encode → update for concurrency

---

## 7. Type Safety Guidelines

### 7.1 Type Annotations

**All public interfaces must have complete type hints**:

```python
from typing import Any
import numpy as np
from numpy.typing import NDArray
from psycopg2.extensions import connection as Connection

class EmbeddingGenerator:
    _model: SentenceTransformer | None = None

    def generate_embeddings_batch(
        self,
        batch_size: int = 32
    ) -> EmbeddingStats:
        """Generate embeddings for all NULL chunks."""
        ...

    def _fetch_chunk_batch(
        self,
        conn: Connection,
        batch_size: int
    ) -> list[dict[str, Any]]:
        """Fetch chunk batch from database."""
        ...

    def _encode_batch(
        self,
        texts: list[str]
    ) -> NDArray[np.float32]:
        """Encode texts to embeddings."""
        ...

    def _update_chunk_embeddings(
        self,
        conn: Connection,
        chunks: list[dict[str, Any]],
        embeddings: NDArray[np.float32]
    ) -> None:
        """Update database with embeddings."""
        ...
```

**Type Validation Strategy**:
```bash
# Must pass mypy --strict
mypy --strict src/embedding_generation/

# Expected result: 0 errors
```

### 7.2 Vector Type Handling

**Type Flow**:

```python
# 1. Model output (NumPy)
embeddings: NDArray[np.float32] = model.encode(texts)  # (32, 768)

# 2. Individual embedding extraction
embedding: NDArray[np.float32] = embeddings[i]  # (768,)

# 3. Convert to Python list for Pydantic
embedding_list: list[float] = embedding.tolist()

# 4. Assign to ProcessedChunk
chunk.embedding = embedding_list  # Pydantic validates len(embedding_list) == 768

# 5. Database insertion
cur.execute(
    "UPDATE ... SET embedding = %s::vector(768)",
    (embedding_list,)  # psycopg2 handles list → PostgreSQL array
)
```

**Edge Case Handling**:

```python
@field_validator("embedding")
@classmethod
def validate_embedding_dimensions(cls, v: list[float] | None) -> list[float] | None:
    """Validate embedding dimensions."""
    if v is None:
        return None

    if len(v) != 768:
        raise ValueError(f"Embedding must be 768-dimensional, got {len(v)}")

    # Validate all values are finite
    if not all(math.isfinite(x) for x in v):
        raise ValueError("Embedding contains NaN or Inf values")

    return v
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Model Loading Tests**:
```python
def test_model_singleton_initialization():
    """Verify model loaded once and cached."""
    gen1 = EmbeddingGenerator()
    gen1._initialize_model()
    model1_id = id(gen1._model)

    gen2 = EmbeddingGenerator()
    gen2._initialize_model()
    model2_id = id(gen2._model)

    assert model1_id == model2_id  # Same instance

def test_device_auto_detection():
    """Verify device selection (MPS > CPU)."""
    gen = EmbeddingGenerator()
    gen._initialize_model()

    if torch.backends.mps.is_available():
        assert gen._device == "mps"
    else:
        assert gen._device == "cpu"
```

**Embedding Generation Tests**:
```python
def test_embedding_dimensions():
    """Verify embeddings are 768-dimensional."""
    gen = EmbeddingGenerator()
    texts = ["test sentence one", "test sentence two"]

    embeddings = gen._encode_batch(texts)

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32

def test_embedding_batch_size():
    """Verify batch processing handles various sizes."""
    gen = EmbeddingGenerator()

    for size in [1, 16, 32, 64]:
        texts = [f"text {i}" for i in range(size)]
        embeddings = gen._encode_batch(texts)
        assert embeddings.shape == (size, 768)
```

### 8.2 Integration Tests

**Database Integration Tests**:
```python
@pytest.fixture
def sample_chunks_in_db(test_db_connection):
    """Populate test DB with chunks (no embeddings)."""
    chunks = [
        ProcessedChunk.create_from_chunk(
            chunk_text=f"Test content {i}",
            context_header=f"test.md > Section {i}",
            metadata=DocumentMetadata(
                title="Test Doc",
                source_file="test.md"
            ),
            chunk_index=i,
            total_chunks=10,
            token_count=50,
        )
        for i in range(10)
    ]

    # Insert without embeddings
    with test_db_connection.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "INSERT INTO knowledge_base (...) VALUES (...)",
                (chunk.chunk_text, chunk.chunk_hash, None, ...)  # embedding=None
            )
    test_db_connection.commit()

    return chunks

def test_end_to_end_embedding_generation(sample_chunks_in_db, test_db_connection):
    """Test complete pipeline: fetch → encode → update."""
    gen = EmbeddingGenerator()
    stats = gen.generate_embeddings_batch(batch_size=5)

    # Verify all chunks processed
    assert stats.chunks_processed == 10
    assert stats.chunks_failed == 0

    # Verify embeddings in database
    with test_db_connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL")
        count = cur.fetchone()[0]

    assert count == 10
```

### 8.3 Performance Tests

**Throughput Benchmarks**:
```python
def test_throughput_meets_target():
    """Verify embedding generation meets 10 chunks/sec minimum."""
    gen = EmbeddingGenerator()

    # Generate embeddings for 100 chunks
    texts = [f"chunk content {i}" for i in range(100)]

    start = time.time()
    embeddings = gen._encode_batch(texts)
    elapsed = time.time() - start

    chunks_per_sec = 100 / elapsed

    assert chunks_per_sec >= 10, f"Throughput too low: {chunks_per_sec:.2f} chunks/sec"
```

**Memory Leak Tests**:
```python
def test_no_memory_leak_in_batch_processing():
    """Verify memory doesn't grow unbounded."""
    import psutil
    import os

    gen = EmbeddingGenerator()
    process = psutil.Process(os.getpid())

    # Baseline memory
    baseline_mb = process.memory_info().rss / 1024 / 1024

    # Process 10 batches
    for _ in range(10):
        texts = [f"text {i}" for i in range(32)]
        embeddings = gen._encode_batch(texts)

    # Check memory growth
    final_mb = process.memory_info().rss / 1024 / 1024
    growth_mb = final_mb - baseline_mb

    # Allow 50MB growth for caches, etc.
    assert growth_mb < 50, f"Memory leak detected: {growth_mb:.2f}MB growth"
```

---

## 9. Implementation Checklist

### 9.1 Phase 2 Deliverables

**Task 3.1: Model Loading** ✓
- [ ] Add sentence-transformers to requirements.txt
- [ ] Implement EmbeddingConfig class
- [ ] Implement model singleton pattern
- [ ] Add device auto-detection (MPS > CPU)
- [ ] Write model loading tests
- [ ] Document model caching behavior

**Task 3.2: Batch Processing** ✓
- [ ] Implement chunk fetching query
- [ ] Implement batch encoding logic
- [ ] Add embedding dimension validation
- [ ] Implement database update logic
- [ ] Add batch commit strategy
- [ ] Write batch processing tests

**Task 3.3: Statistics & Logging** ✓
- [ ] Create EmbeddingStats model
- [ ] Add structured logging for all phases
- [ ] Track chunks processed/failed/skipped
- [ ] Measure throughput (chunks/sec)
- [ ] Log memory usage metrics
- [ ] Write statistics tests

**Task 3.4: Error Handling** ✓
- [ ] Add chunk-level error isolation
- [ ] Implement retry logic for transient failures
- [ ] Validate embeddings before database insert
- [ ] Handle model OOM gracefully
- [ ] Write error handling tests

**Task 3.5: Integration Testing** ✓
- [ ] End-to-end pipeline test
- [ ] Database integration test
- [ ] Performance benchmark test
- [ ] Memory leak test
- [ ] mypy --strict validation

### 9.2 File Structure

```
src/embedding_generation/
├── __init__.py
├── config.py              # EmbeddingConfig
├── generator.py           # EmbeddingGenerator class
├── models.py             # EmbeddingStats model
└── utils.py              # Helper functions

tests/
├── test_embedding_config.py
├── test_embedding_generator.py
├── test_embedding_integration.py
└── test_embedding_performance.py

requirements.txt          # Add sentence-transformers, torch
```

### 9.3 Configuration Updates

**requirements.txt**:
```diff
  # Tokenization using OpenAI's tiktoken for accurate token counting
  tiktoken>=0.7.0
+
+ # Embedding generation
+ sentence-transformers>=2.2.0
+ torch>=2.0.0
+ transformers>=4.30.0
```

**src/core/config.py**:
```diff
  class Settings(BaseSettings):
      database: DatabaseConfig
      logging: LoggingConfig
      application: ApplicationConfig
+     embedding: EmbeddingConfig
```

---

## 10. Architectural Trade-offs

### 10.1 Simplicity vs Performance

**Decision**: Prioritize simplicity for Phase 2

| Aspect              | Simple Approach          | Complex Approach        | Decision |
|---------------------|--------------------------|-------------------------|----------|
| Parallelism         | Single-process           | Multi-process pool      | Simple   |
| Batch size          | Fixed (32)               | Adaptive (16-512)       | Simple   |
| Database updates    | Sequential UPDATEs       | Bulk COPY + merge       | Simple   |
| Error handling      | Chunk-level retry        | Transaction retry       | Simple   |
| Progress tracking   | Logs only                | Progress bar + webhooks | Simple   |

**Rationale**:
- 2,600 chunks is small enough for simple approach (<3 minutes total)
- Complex optimizations add risk without meaningful benefit
- Future scalability (100K+ chunks) can iterate on proven simple design

### 10.2 Type Safety vs Flexibility

**Decision**: Maximize type safety with Pydantic + mypy

**Benefits**:
- ✅ Catch bugs at development time (mypy --strict)
- ✅ Clear contracts for embedding dimensions (768)
- ✅ Runtime validation prevents corrupted data

**Costs**:
- ❌ Slightly more boilerplate (validators, type hints)
- ❌ NumPy → list conversion overhead (~1μs/chunk)

**Verdict**: Type safety benefits outweigh costs for production system

### 10.3 Resumability vs Complexity

**Decision**: Built-in resumability via `WHERE embedding IS NULL`

**Benefits**:
- ✅ Idempotent - safe to rerun pipeline without duplicates
- ✅ Crash recovery - automatic resume from last checkpoint
- ✅ Incremental updates - add new chunks without re-embedding existing

**Implementation**:
- Query: `WHERE embedding IS NULL` filters completed chunks
- Order: `ORDER BY chunk_hash` ensures deterministic batch selection
- Zero additional complexity (no separate state tracking)

---

## 11. Security & Compliance

### 11.1 Model Provenance

**Model Source**: HuggingFace Hub (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

**Verification**:
- ✅ Official sentence-transformers organization
- ✅ 1M+ downloads, actively maintained
- ✅ MIT License (commercial use allowed)
- ✅ Model card with evaluation metrics

**Supply Chain Security**:
- Download over HTTPS (enforced by HuggingFace)
- Model hash verification (automatic via HuggingFace Hub)
- Pin model version in config (avoid automatic updates)

### 11.2 Data Privacy

**Chunk Text Handling**:
- Text sent to **local model only** (no external API calls)
- Embeddings computed on-device (CPU/MPS)
- No data leaves the server

**Logging Considerations**:
- ❌ Never log chunk_text (may contain PII)
- ✅ Log chunk_hash (non-sensitive identifier)
- ✅ Log aggregated metrics only

---

## 12. Future Scalability

### 12.1 Scaling to 100K+ Chunks

**Current Limitations**:
- Single-process: 30-60 chunks/sec → 30-55 minutes for 100K chunks
- Memory: 620MB peak (acceptable even at 100K chunks)

**Optimization Path**:
1. **GPU support** (CUDA): 10-50x speedup → 3-6 minutes for 100K
2. **Parallel processing**: 4 workers → 7-14 minutes for 100K (CPU)
3. **Async updates**: Pipeline fetch/encode/update → 20% improvement

### 12.2 Multiple Embedding Models

**Use Case**: Ensemble embeddings or A/B testing

**Architecture Change**:
```python
class MultiModelEmbeddingGenerator:
    models: dict[str, SentenceTransformer]

    def generate_embeddings(
        self,
        model_names: list[str]
    ) -> dict[str, NDArray]:
        """Generate embeddings from multiple models."""
        return {
            name: self.models[name].encode(texts)
            for name in model_names
        }
```

**Database Schema**:
```sql
ALTER TABLE knowledge_base
ADD COLUMN embedding_mpnet vector(768),
ADD COLUMN embedding_e5 vector(1024),
ADD COLUMN embedding_bge vector(768);
```

---

## 13. Recommendations

### 13.1 Immediate Actions (Phase 2)

1. **Install dependencies**:
   ```bash
   pip install sentence-transformers torch transformers
   ```

2. **Pre-download model** (optional, saves time in tests):
   ```python
   from sentence_transformers import SentenceTransformer
   SentenceTransformer("all-mpnet-base-v2")  # Downloads to cache
   ```

3. **Add EmbeddingConfig to Settings**:
   - Implement `src/embedding_generation/config.py`
   - Update `src/core/config.py` to include embedding config

4. **Implement singleton model loader**:
   - Create `src/embedding_generation/generator.py`
   - Add device auto-detection (MPS > CPU)
   - Write unit tests

### 13.2 Implementation Order

**Week 1: Core Infrastructure**
- Day 1-2: Config + model loading + tests
- Day 3-4: Batch fetching + encoding logic
- Day 5: Database update logic

**Week 2: Integration & Testing**
- Day 1-2: End-to-end integration tests
- Day 3: Performance benchmarks
- Day 4-5: Error handling + edge cases

**Week 3: Production Readiness**
- Day 1-2: Structured logging + metrics
- Day 3: Documentation + usage examples
- Day 4-5: Code review + refinement

### 13.3 Quality Gates

**Before merging to main**:
- [ ] mypy --strict passes (0 errors)
- [ ] pytest passes (100% tests green)
- [ ] Coverage ≥95% for new code
- [ ] All 2,600 chunks successfully embedded in dev environment
- [ ] Performance benchmark: ≥10 chunks/sec on CPU
- [ ] Memory usage <1GB peak
- [ ] Documentation complete (docstrings + README)

---

## 14. Conclusion

The proposed architecture for Task 3 (Embedding Generation Pipeline) prioritizes **simplicity, type safety, and production readiness** while maintaining excellent performance for the current scale (2,600 chunks).

**Key Strengths**:
1. ✅ **Simple design**: Single-process, batch-oriented, easy to debug
2. ✅ **Type-safe**: mypy --strict compliance, Pydantic validation
3. ✅ **Resilient**: Chunk-level error handling, automatic resumability
4. ✅ **Performant**: 16-58 chunks/sec, <3 min total time
5. ✅ **Integrated**: Seamless Phase 0/1 integration

**Architectural Philosophy**:
- Start simple, optimize when necessary
- Type safety prevents production bugs
- Resumability enables fault tolerance
- Structured logging enables observability

**Next Steps**:
1. Review this architecture document
2. Begin implementation following checklist (§9.1)
3. Write tests alongside implementation (TDD)
4. Iterate based on benchmark results

---

**Document Status**: READY FOR REVIEW
**Estimated Implementation Time**: 2-3 weeks (1 senior developer)
**Risk Level**: LOW (proven technologies, simple architecture)
**Recommendation**: APPROVE and proceed with implementation

