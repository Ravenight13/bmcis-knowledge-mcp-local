# Blocker 1 Fix Plan: Schema/Query Mismatch
**Date**: 2025-11-09
**Status**: Planning
**Priority**: Critical (Foundation Issue)
**Effort Estimate**: 3.5-4.5 hours
**Block Dependencies**: All Phase 1 fixes depend on this

---

## Executive Summary

### The Problem
The knowledge graph implementation has a **critical schema/query alignment mismatch** that will cause all 5 query patterns to fail at runtime:

| Component | Schema Definition | Query Reference | Type |
|-----------|------------------|-----------------|------|
| Entity name column | `text` (TEXT) | `entity_name` | Column Name |
| Confidence field | `confidence` (FLOAT) | `metadata->>'confidence'` (JSON extract) | Data Type |
| ID column | UUID (not INT) | Dataclass uses `int` | Type Mismatch |
| Mention table | `entity_mentions` | `chunk_entities` (doesn't exist) | Table Name |
| Metadata column | Doesn't exist | Referenced in 3 queries | Missing Schema |

### Impact Analysis
- **Query Execution**: All 5 methods fail on first column reference
- **Test Coverage**: No tests can pass while mismatches exist
- **Performance**: Cannot validate performance targets without correct queries
- **Architecture**: Cannot build cache layer until queries work

### Why This Blocks Everything
Tests are written to expect:
- Column names matching schema (`entity_name` vs `text`)
- Correct data types (`confidence: float` vs `metadata->>'confidence'`: string)
- Correct table names (`entity_mentions` exists in schema)
- Correct ID types (UUID, not int)

**Foundation Fix Required**: Schema/Query alignment must be 100% before other fixes can be validated.

---

## Mismatch Audit (Complete List)

### 1. Column Name Mismatches

#### Mismatch 1.1: Entity Name Column
- **Schema definition** (schema.sql:52): `text TEXT NOT NULL`
- **Query references**:
  - query_repository.py:134 (traverse_1hop): `e.entity_name AS text`
  - query_repository.py:216 (traverse_2hop): `e2.entity_name AS text`
  - query_repository.py:350 (traverse_bidirectional): `e.entity_name AS text`
  - query_repository.py:430 (traverse_with_type_filter): `e.entity_name AS text`
- **Problem**: Column `entity_name` doesn't exist in `knowledge_entities` table
- **Correct Reference**: `e.text`
- **Fix Type**: SQL column reference correction
- **Affected Methods**: 4 methods
- **Lines to Fix**: 134, 216, 350, 430 in query_repository.py

#### Mismatch 1.2: Confidence Field (JSONB vs Float)
- **Schema definition** (schema.sql:54): `confidence FLOAT NOT NULL DEFAULT 1.0`
- **Query references**:
  - query_repository.py:136 (traverse_1hop): `e.metadata->>'confidence'`
  - query_repository.py:218 (traverse_2hop): `e2.metadata->>'confidence'`
  - query_repository.py:352 (traverse_bidirectional): `e.metadata->>'confidence'`
  - query_repository.py:432 (traverse_with_type_filter): `e.metadata->>'confidence'`
- **Problem**: Column `metadata` doesn't exist; confidence is a native FLOAT column
- **Correct Reference**: `e.confidence`
- **Fix Type**: Remove JSONB extraction, use direct column
- **Affected Methods**: 4 methods
- **Lines to Fix**: 136, 218, 352, 432 in query_repository.py
- **Test Impact**: Lines 63, 72, 152, 208 expect string conversion (wrong)

### 2. Table Name Mismatches

#### Mismatch 2.1: Entity Mentions Table
- **Schema definition** (schema.sql:154): Table name is `entity_mentions`
- **Query reference** (query_repository.py:506): `FROM chunk_entities ce`
- **Problem**: Query references `chunk_entities` which doesn't exist in schema
- **Correct Reference**: `entity_mentions`
- **Fix Type**: Table name correction
- **Affected Methods**: 1 method (get_entity_mentions)
- **Lines to Fix**: 506 in query_repository.py

#### Mismatch 2.2: Knowledge Base Table
- **Schema definition**: No table named `knowledge_base` exists
- **Query references** (query_repository.py:507-508):
  - `JOIN knowledge_base kb ON kb.id = ce.chunk_id`
  - `kb.source_file AS document_id`
  - `kb.source_category AS document_category`
  - `kb.chunk_text AS chunk_text`
  - `kb.chunk_index`
  - `kb.created_at AS indexed_at`
- **Problem**: References non-existent table with non-existent columns
- **Correct Mapping**: All mention data comes from `entity_mentions` schema
- **Fix Type**: Complete query rewrite for mentions
- **Lines to Fix**: 497-510 in query_repository.py

### 3. ID Type Mismatches

#### Mismatch 3.1: Entity ID Type
- **Schema definition** (schema.sql:51): `id UUID PRIMARY KEY`
- **Models.py** (line 65): `id: Mapped[UUID]` (correct)
- **Query repository** (line 28): `id: int` in RelatedEntity dataclass
- **Query repository** (line 40): `id: int` in TwoHopEntity dataclass
- **Query repository** (line 55): `id: int` in BidirectionalEntity dataclass
- **Query repository** (line 69): `id: int` in EntityMention dataclass
- **Tests** (line 60): `uuid4()` passed to test, but dataclass expects int
- **Problem**: Type mismatch between schema (UUID) and code (int)
- **Correct Type**: UUID
- **Fix Type**: Update dataclass definitions and test expectations
- **Lines to Fix**: 28, 40, 55, 69 in query_repository.py

### 4. Missing Schema Elements

#### Mismatch 4.1: Metadata Column
- **Schema definition**: No `metadata` JSONB column in `knowledge_entities`
- **Query references**:
  - query_repository.py:126 (traverse_1hop): `r.metadata AS relationship_metadata`
  - query_repository.py:139 (traverse_1hop): `re.relationship_metadata`
  - query_repository.py:435 (traverse_with_type_filter): `r.metadata AS relationship_metadata`
  - query_repository.py:469 (traverse_with_type_filter): `relationship_metadata=row[6]`
- **Available Alternative**: `relationship_weight` FLOAT column exists (line 107 in schema.sql)
- **Fix Type**: Either add metadata column to schema OR remove from queries
- **Decision Needed**: Architecture decision - does relationship metadata need JSON storage?
- **Current Recommendation**: Remove metadata from queries; use weight instead

---

## Task Breakdown

### Task 1.1: Audit Column Name Mismatches (FOUNDATION TASK)
**Status**: To Do
**Effort**: 30-45 minutes
**Dependencies**: None
**Blocks**: Tasks 1.2, 1.3, 1.4

#### Objective
Create definitive mapping of every schema column to every query reference, identifying all mismatches with precision.

#### Detailed Steps

1. **Extract schema column inventory** (schema.sql):
   ```
   knowledge_entities table:
   - id (UUID)
   - text (TEXT)
   - entity_type (VARCHAR 50)
   - confidence (FLOAT)
   - canonical_form (TEXT nullable)
   - mention_count (INTEGER)
   - created_at (TIMESTAMP)
   - updated_at (TIMESTAMP)

   entity_relationships table:
   - id (UUID)
   - source_entity_id (UUID FK)
   - target_entity_id (UUID FK)
   - relationship_type (VARCHAR 50)
   - confidence (FLOAT)
   - relationship_weight (FLOAT)
   - is_bidirectional (BOOLEAN)
   - created_at (TIMESTAMP)
   - updated_at (TIMESTAMP)

   entity_mentions table:
   - id (UUID)
   - entity_id (UUID FK)
   - document_id (VARCHAR 255)
   - chunk_id (INTEGER)
   - mention_text (TEXT)
   - offset_start (INTEGER nullable)
   - offset_end (INTEGER nullable)
   - created_at (TIMESTAMP)
   ```

2. **Extract query column references** (query_repository.py):
   - Line 134: `e.entity_name` (traverse_1hop)
   - Line 136: `e.metadata->>'confidence'` (traverse_1hop)
   - Line 126: `r.metadata` (traverse_1hop)
   - Line 216: `e2.entity_name` (traverse_2hop)
   - Line 218: `e2.metadata->>'confidence'` (traverse_2hop)
   - Line 350: `e.entity_name` (traverse_bidirectional)
   - Line 352: `e.metadata->>'confidence'` (traverse_bidirectional)
   - Line 430: `e.entity_name` (traverse_with_type_filter)
   - Line 432: `e.metadata->>'confidence'` (traverse_with_type_filter)
   - Line 506-508: `FROM chunk_entities` (get_entity_mentions) - table doesn't exist
   - Line 500: `kb.source_file` - column doesn't exist
   - Line 501: `kb.chunk_text` - column doesn't exist
   - Line 502: `kb.source_category` - column doesn't exist

3. **Create mismatch mapping** (output to audit file):
   ```
   Format:
   Query Line | Method | Column Ref | Schema Column | Status
   134 | traverse_1hop | entity_name | text | MISMATCH
   136 | traverse_1hop | metadata->>'confidence' | confidence | TYPE_MISMATCH
   506 | get_entity_mentions | chunk_entities | entity_mentions | TABLE_MISMATCH
   ```

4. **Document ID type mismatches**:
   - Check dataclass definitions (lines 28, 40, 55, 69)
   - Compare to schema definitions (all UUIDs)
   - Document each type mismatch with line numbers

5. **Output deliverable**:
   - File: `docs/subagent-reports/task-planning/2025-11-09-audit-column-mismatches.md`
   - Contents:
     - Complete column inventory (schema)
     - Complete column reference list (queries)
     - Mismatch mapping table with 100% coverage
     - Type mismatch table for dataclasses
     - Severity classification for each mismatch

#### Files to Check
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/src/knowledge_graph/schema.sql` (lines 50-162)
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/src/knowledge_graph/query_repository.py` (lines 1-570)

#### Verification Checklist
- [ ] All 8 schema columns documented with exact types
- [ ] All column references extracted from queries (find all references using grep)
- [ ] Mismatch table has 15+ mismatches identified
- [ ] ID type mismatches documented (4 dataclasses)
- [ ] Table name mismatches documented
- [ ] Output file created and committed

---

### Task 1.2: Fix Query Repository Column References
**Status**: To Do
**Effort**: 1-1.5 hours
**Dependencies**: Task 1.1
**Blocks**: Task 1.3, 1.4

#### Objective
Correct all SQL column references in query_repository.py to match schema definitions exactly.

#### Detailed Changes Required

##### Change 2.1: traverse_1hop method (lines 120-144)
**Current Code**:
```sql
SELECT
    e.id,
    e.entity_name AS text,           -- LINE 134: WRONG
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,  -- LINE 136: WRONG
    re.relationship_type,
    re.relationship_confidence,
    re.relationship_metadata        -- LINE 139: metadata column doesn't exist
```

**Corrected Code**:
```sql
SELECT
    e.id,
    e.text,                          -- LINE 134: Fixed
    e.entity_type,
    e.confidence AS entity_confidence,  -- LINE 136: Fixed (direct column, not JSONB)
    re.relationship_type,
    re.relationship_confidence,
    re.relationship_weight AS relationship_metadata  -- LINE 139: Use weight instead
```

**Changes**:
- Line 134: `e.entity_name AS text` → `e.text`
- Line 136: `e.metadata->>'confidence'` → `e.confidence`
- Line 139: `re.relationship_metadata` → `re.relationship_weight AS relationship_metadata`
- Line 126: Remove reference to `r.metadata` in CTE (line 126 stays same, just output differs)

**Test Impact**:
- Line 63 expects string confidence; needs to handle float
- Line 92 expects correct metadata structure

##### Change 2.2: traverse_2hop method (lines 202-248)
**Current Code**:
```sql
SELECT
    r2.target_entity_id AS entity_id,
    e2.entity_name AS text,         -- LINE 216: WRONG
    e2.entity_type,
    e2.metadata->>'confidence' AS entity_confidence,  -- LINE 218: WRONG
    ...
    ei.entity_name AS intermediate_entity_name,  -- LINE 222: WRONG
```

**Corrected Code**:
```sql
SELECT
    r2.target_entity_id AS entity_id,
    e2.text,                        -- LINE 216: Fixed
    e2.entity_type,
    e2.confidence AS entity_confidence,  -- LINE 218: Fixed
    ...
    ei.text AS intermediate_entity_name,  -- LINE 222: Fixed
```

**Changes**:
- Line 216: `e2.entity_name AS text` → `e2.text`
- Line 218: `e2.metadata->>'confidence'` → `e2.confidence`
- Line 222: `ei.entity_name` → `ei.text`

**Test Impact**:
- Lines 152, 156 need to change from expecting string confidence to float

##### Change 2.3: traverse_bidirectional method (lines 313-362)
**Current Code**:
```sql
SELECT
    c.entity_id,
    e.entity_name AS text,          -- LINE 350: WRONG
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,  -- LINE 352: WRONG
```

**Corrected Code**:
```sql
SELECT
    c.entity_id,
    e.text,                         -- LINE 350: Fixed
    e.entity_type,
    e.confidence AS entity_confidence,  -- LINE 352: Fixed
```

**Changes**:
- Line 350: `e.entity_name AS text` → `e.text`
- Line 352: `e.metadata->>'confidence'` → `e.confidence`

**Test Impact**:
- Line 208 needs to change from expecting string confidence to float

##### Change 2.4: traverse_with_type_filter method (lines 427-444)
**Current Code**:
```sql
SELECT
    e.id,
    e.entity_name AS text,          -- LINE 430: WRONG
    e.entity_type,
    e.metadata->>'confidence' AS entity_confidence,  -- LINE 432: WRONG
    r.relationship_type,
    r.confidence AS relationship_confidence,
    r.metadata AS relationship_metadata  -- LINE 435: WRONG
```

**Corrected Code**:
```sql
SELECT
    e.id,
    e.text,                         -- LINE 430: Fixed
    e.entity_type,
    e.confidence AS entity_confidence,  -- LINE 432: Fixed
    r.relationship_type,
    r.confidence AS relationship_confidence,
    r.relationship_weight AS relationship_metadata  -- LINE 435: Use weight
```

**Changes**:
- Line 430: `e.entity_name AS text` → `e.text`
- Line 432: `e.metadata->>'confidence'` → `e.confidence`
- Line 435: `r.metadata` → `r.relationship_weight AS relationship_metadata`

**Test Impact**:
- Lines 251-252 need confidence as float, not string

##### Change 2.5: get_entity_mentions method (lines 497-511)
**Current Code**:
```sql
SELECT
    ce.chunk_id AS chunk_id,
    kb.source_file AS document_id,
    kb.chunk_text AS chunk_text,
    kb.source_category AS document_category,
    kb.chunk_index,
    ce.confidence AS mention_confidence,
    kb.created_at AS indexed_at
FROM chunk_entities ce
JOIN knowledge_base kb ON kb.id = ce.chunk_id
WHERE ce.entity_id = %s
```

**Corrected Code**:
```sql
SELECT
    em.chunk_id AS chunk_id,
    em.document_id,
    em.mention_text AS chunk_text,
    NULL AS document_category,
    0 AS chunk_index,
    1.0 AS mention_confidence,
    em.created_at AS indexed_at
FROM entity_mentions em
WHERE em.entity_id = %s
```

**Analysis**:
- `chunk_entities` table doesn't exist; should be `entity_mentions`
- `knowledge_base` table doesn't exist; document info is in `entity_mentions`
- `entity_mentions` doesn't have `confidence` column (only in relationships)
- `entity_mentions` doesn't have category/chunk_index (only document_id, chunk_id)
- Simplified query based on available schema columns

**Changes**:
- Line 499: `ce.chunk_id` → `em.chunk_id`
- Line 500: `kb.source_file` → `em.document_id` (direct column, no alias needed)
- Line 501: `kb.chunk_text` → `em.mention_text` (as chunk_text)
- Line 502: `kb.source_category` → `NULL` (column doesn't exist)
- Line 503: `kb.chunk_index` → `0` (column doesn't exist)
- Line 504: `ce.confidence` → `1.0` (confidence not tracked per mention)
- Line 506: `FROM chunk_entities ce` → `FROM entity_mentions em`
- Line 507: `JOIN knowledge_base...` → Remove (table doesn't exist)
- Line 508: `WHERE ce.entity_id` → `WHERE em.entity_id`

**Test Impact**:
- Lines 289-298 need major rewrite
- Confidence changed from expected 0.92 to hardcoded 1.0
- Document category, chunk_index will be NULL/0

#### Files to Modify
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/src/knowledge_graph/query_repository.py`

#### Verification Strategy

**SQL Syntax Validation**:
1. After each method fix, run `sqlparse` to validate syntax
2. Check that all column references exist in schema
3. Verify aliases match SELECT clause outputs
4. Confirm JOIN conditions are valid

**Column Mapping Validation**:
```python
# For each method, verify:
# 1. All columns referenced exist in schema
import schema_parser
columns = schema_parser.get_columns('knowledge_entities')
assert 'text' in columns  # Not 'entity_name'
assert 'confidence' in columns  # Type is FLOAT, not JSONB
```

**Test Compatibility**:
1. Review test expectations (test_query_repository.py)
2. Update test fixtures to match corrected queries
3. Verify dataclass field types match query outputs

#### Acceptance Criteria
- [ ] All 5 methods have corrected column references
- [ ] All references to non-existent columns removed
- [ ] All JSONB extractions replaced with direct column access
- [ ] Table names match schema (chunk_entities → entity_mentions)
- [ ] SQL syntax validated and correct
- [ ] Query outputs align with dataclass field definitions
- [ ] Changes committed with clear message

---

### Task 1.3: Validate Schema vs ORM Models Alignment
**Status**: To Do
**Effort**: 30-45 minutes
**Dependencies**: Task 1.1 (informational only)
**Blocks**: Task 1.4

#### Objective
Ensure SQLAlchemy ORM models in models.py match schema.sql definitions exactly.

#### Detailed Checks

##### Check 3.1: KnowledgeEntity Model (models.py:41-104)
**Schema Columns** (schema.sql:50-60):
- id: UUID PRIMARY KEY
- text: TEXT NOT NULL
- entity_type: VARCHAR(50) NOT NULL
- confidence: FLOAT NOT NULL DEFAULT 1.0 CHECK [0.0, 1.0]
- canonical_form: TEXT nullable
- mention_count: INTEGER NOT NULL DEFAULT 0
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- updated_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

**Model Fields** (models.py:65-74):
```python
id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ...) ✓
text: Mapped[str] = mapped_column(Text, nullable=False, index=True) ✓
entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True) ✓
confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0) ✓
canonical_form: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True) ✓
mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True) ✓
created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow) ✓
updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, ..., onupdate=datetime.utcnow) ✓
```

**Status**: ✅ ALIGNED - All fields match schema exactly

**Validation Checklist**:
- [ ] id field: UUID type, primary key
- [ ] text field: String type, not nullable, indexed
- [ ] entity_type field: String(50), not nullable, indexed
- [ ] confidence field: Float, default 1.0, constraint [0.0, 1.0]
- [ ] canonical_form field: Optional string, indexed
- [ ] mention_count field: Integer, default 0, indexed
- [ ] Timestamps correct: datetime.utcnow used

##### Check 3.2: EntityRelationship Model (models.py:106-185)
**Schema Columns** (schema.sql:101-113):
- id: UUID PRIMARY KEY
- source_entity_id: UUID FK NOT NULL
- target_entity_id: UUID FK NOT NULL
- relationship_type: VARCHAR(50) NOT NULL
- confidence: FLOAT NOT NULL DEFAULT 1.0 CHECK [0.0, 1.0]
- relationship_weight: FLOAT NOT NULL DEFAULT 1.0
- is_bidirectional: BOOLEAN NOT NULL DEFAULT FALSE
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- updated_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

**Model Fields** (models.py:134-148):
```python
id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ...) ✓
source_entity_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True) ✓
target_entity_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True) ✓
relationship_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True) ✓
confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0) ✓
relationship_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0) ✓
is_bidirectional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True) ✓
created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow) ✓
updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, ..., onupdate=datetime.utcnow) ✓
```

**Status**: ✅ ALIGNED - All fields match schema exactly

**Validation Checklist**:
- [ ] id field: UUID type, primary key
- [ ] source_entity_id: UUID, FK to knowledge_entities.id, ON DELETE CASCADE, indexed
- [ ] target_entity_id: UUID, FK to knowledge_entities.id, ON DELETE CASCADE, indexed
- [ ] relationship_type: String(50), not nullable, indexed
- [ ] confidence: Float, default 1.0, constraint [0.0, 1.0]
- [ ] relationship_weight: Float, default 1.0
- [ ] is_bidirectional: Boolean, default FALSE, indexed
- [ ] Timestamps correct: datetime.utcnow used
- [ ] Unique constraint on (source, target, type): Present in __table_args__
- [ ] No self-loops constraint: Present in __table_args__

##### Check 3.3: EntityMention Model (models.py:187-245)
**Schema Columns** (schema.sql:154-162):
- id: UUID PRIMARY KEY
- entity_id: UUID FK NOT NULL
- document_id: VARCHAR(255) NOT NULL
- chunk_id: INTEGER NOT NULL
- mention_text: TEXT NOT NULL
- offset_start: INTEGER nullable
- offset_end: INTEGER nullable
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

**Model Fields** (models.py:216-225):
```python
id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ...) ✓
entity_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                       ForeignKey("knowledge_entities.id", ondelete="CASCADE"), index=True) ✓
document_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True) ✓
chunk_id: Mapped[int] = mapped_column(Integer, nullable=False) ✓
mention_text: Mapped[str] = mapped_column(Text, nullable=False) ✓
offset_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True) ✓
offset_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True) ✓
created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow) ✓
```

**Status**: ✅ ALIGNED - All fields match schema exactly

**Validation Checklist**:
- [ ] id field: UUID type, primary key
- [ ] entity_id: UUID, FK to knowledge_entities.id, ON DELETE CASCADE, indexed
- [ ] document_id: String(255), not nullable, indexed
- [ ] chunk_id: Integer, not nullable
- [ ] mention_text: Text, not nullable
- [ ] offset_start: Optional integer, nullable
- [ ] offset_end: Optional integer, nullable
- [ ] created_at: datetime with utcnow default
- [ ] Composite index on (document_id, chunk_id): Present
- [ ] Composite index on (entity_id, document_id): Present

##### Check 3.4: Relationship Configuration
**Check backref consistency** (models.py:77-93):
```python
# KnowledgeEntity relationships
relationships_from: relationship(..., back_populates="source_entity")  # ✓
relationships_to: relationship(..., back_populates="target_entity")    # ✓
mentions: relationship(..., back_populates="entity")                   # ✓

# EntityRelationship relationships
source_entity: relationship(..., back_populates="relationships_from")  # ✓
target_entity: relationship(..., back_populates="relationships_to")    # ✓

# EntityMention relationships
entity: relationship(..., back_populates="mentions")                   # ✓
```

**Status**: ✅ ALIGNED - All relationship backrefs match

#### Files to Check
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/src/knowledge_graph/models.py`
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/src/knowledge_graph/schema.sql`

#### Output Deliverable
- File: `docs/subagent-reports/task-planning/2025-11-09-orm-schema-alignment-validation.md`
- Contents:
  - Alignment matrix for all 3 models
  - Column-by-column comparison (schema vs ORM)
  - Type compatibility verification
  - Relationship configuration validation
  - Status: PASSED or FAILED with specific issues

#### Acceptance Criteria
- [ ] All 3 models validated against schema
- [ ] Column types match exactly (UUID, String, Float, Integer, Boolean, DateTime)
- [ ] Nullable constraints match
- [ ] Default values match
- [ ] Foreign key references correct
- [ ] Relationship backrefs consistent
- [ ] Validation report generated
- [ ] No model modifications needed (likely) OR modifications identified with rationale

---

### Task 1.4: Create Schema Alignment Test Suite
**Status**: To Do
**Effort**: 1-1.5 hours
**Dependencies**: Tasks 1.2, 1.3 (verification only)
**Blocks**: None (enables Phase 1 testing)

#### Objective
Create comprehensive test suite to prevent future schema/query misalignment and validate all fixes.

#### Test File Location
- **Path**: `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/tests/knowledge_graph/test_schema_alignment.py`
- **Purpose**: Schema validation tests (separate from query tests)
- **Scope**: 10-15 tests covering schema, ORM, and query validation

#### Test Coverage Structure

##### Category 1: Schema Structure Tests (3 tests)

**Test 1.4.1: Schema Tables Exist**
```python
def test_schema_required_tables_exist(database_inspector):
    """Verify all required tables exist in database schema."""
    inspector = database_inspector
    tables = inspector.get_table_names()

    required_tables = [
        'knowledge_entities',
        'entity_relationships',
        'entity_mentions'
    ]

    for table in required_tables:
        assert table in tables, f"Required table '{table}' not found"
```

**Test 1.4.2: Knowledge Entities Columns**
```python
def test_knowledge_entities_columns(database_inspector):
    """Verify knowledge_entities table has correct columns and types."""
    inspector = database_inspector
    columns = {col['name']: col['type'] for col in
               inspector.get_columns('knowledge_entities')}

    expected_columns = {
        'id': 'UUID',
        'text': 'TEXT',
        'entity_type': 'VARCHAR',
        'confidence': 'FLOAT',
        'canonical_form': 'TEXT',
        'mention_count': 'INTEGER',
        'created_at': 'TIMESTAMP',
        'updated_at': 'TIMESTAMP'
    }

    for col_name, col_type in expected_columns.items():
        assert col_name in columns, f"Missing column: {col_name}"
        # Verify type (PostgreSQL representation)
        assert col_type.upper() in str(columns[col_name]).upper()
```

**Test 1.4.3: Entity Relationships Columns**
```python
def test_entity_relationships_columns(database_inspector):
    """Verify entity_relationships table structure."""
    inspector = database_inspector
    columns = {col['name']: col['type'] for col in
               inspector.get_columns('entity_relationships')}

    expected = {
        'id': 'UUID',
        'source_entity_id': 'UUID',
        'target_entity_id': 'UUID',
        'relationship_type': 'VARCHAR',
        'confidence': 'FLOAT',
        'relationship_weight': 'FLOAT',
        'is_bidirectional': 'BOOLEAN',
        'created_at': 'TIMESTAMP',
        'updated_at': 'TIMESTAMP'
    }

    for col_name in expected:
        assert col_name in columns, f"Missing column: {col_name}"
```

##### Category 2: ORM Model Tests (3 tests)

**Test 1.4.4: KnowledgeEntity Model Matches Schema**
```python
def test_knowledge_entity_model_fields():
    """Verify KnowledgeEntity ORM model fields match schema."""
    from src.knowledge_graph.models import KnowledgeEntity
    from sqlalchemy import inspect

    mapper = inspect(KnowledgeEntity)
    columns = {col.name: col.type for col in mapper.columns}

    assert 'id' in columns
    assert 'text' in columns
    assert columns['text'].python_type == str
    assert columns['confidence'].python_type == float
    assert 'entity_name' not in columns  # Should not exist
    assert 'metadata' not in columns  # Should not exist
```

**Test 1.4.5: EntityRelationship Model Matches Schema**
```python
def test_entity_relationship_model_fields():
    """Verify EntityRelationship ORM model matches schema."""
    from src.knowledge_graph.models import EntityRelationship
    from sqlalchemy import inspect

    mapper = inspect(EntityRelationship)
    columns = {col.name: col.type for col in mapper.columns}

    # Verify correct columns exist
    assert 'relationship_weight' in columns

    # Verify incorrect columns don't exist
    assert 'metadata' not in columns
```

**Test 1.4.6: EntityMention Model Matches Schema**
```python
def test_entity_mention_model_fields():
    """Verify EntityMention ORM model matches schema."""
    from src.knowledge_graph.models import EntityMention
    from sqlalchemy import inspect

    mapper = inspect(EntityMention)
    columns = {col.name: col.type for col in mapper.columns}

    # Verify correct table and columns
    assert 'entity_id' in columns
    assert 'document_id' in columns
    assert 'mention_text' in columns

    # Verify incorrect table not referenced
    # (no way to directly test, but query tests will catch)
```

##### Category 3: Query Validation Tests (4 tests)

**Test 1.4.7: Query Column References Valid**
```python
def test_query_repository_column_references():
    """Verify all query_repository SQL references valid columns."""
    import re
    from pathlib import Path

    query_file = Path('src/knowledge_graph/query_repository.py')
    content = query_file.read_text()

    # Extract SQL queries
    sql_blocks = re.findall(r'query = """(.*?)"""', content, re.DOTALL)

    # Known valid columns
    valid_entity_cols = ['id', 'text', 'entity_type', 'confidence',
                         'canonical_form', 'mention_count', 'created_at', 'updated_at']
    valid_rel_cols = ['id', 'source_entity_id', 'target_entity_id', 'relationship_type',
                      'confidence', 'relationship_weight', 'is_bidirectional', 'created_at', 'updated_at']

    for sql in sql_blocks:
        # These should NOT appear
        assert 'entity_name' not in sql, "Query references non-existent entity_name column"
        assert 'metadata' not in sql, "Query references non-existent metadata column"
        assert 'chunk_entities' not in sql, "Query references non-existent chunk_entities table"
        assert 'knowledge_base' not in sql, "Query references non-existent knowledge_base table"
```

**Test 1.4.8: No References to Non-existent Columns**
```python
def test_no_jsonb_confidence_extraction(database_connection):
    """Verify queries don't use JSONB extraction for confidence."""
    import re
    from pathlib import Path

    query_file = Path('src/knowledge_graph/query_repository.py')
    content = query_file.read_text()

    # metadata->>'confidence' should not appear
    assert "metadata->>'confidence'" not in content, \
        "Query still uses JSONB extraction for non-existent metadata column"
```

**Test 1.4.9: Query Result Types Match Dataclasses**
```python
def test_dataclass_id_types():
    """Verify dataclass id fields match UUID schema type."""
    from src.knowledge_graph.query_repository import (
        RelatedEntity, TwoHopEntity, BidirectionalEntity, EntityMention
    )
    from uuid import UUID
    from typing import get_type_hints

    # Check RelatedEntity
    hints = get_type_hints(RelatedEntity)
    # Note: This test may need adjustment based on actual dataclass implementation
    # The point is to catch type mismatches (int vs UUID)

    # Current issue: dataclasses have id: int, but schema uses UUID
    # After fix: should be id: UUID
```

**Test 1.4.10: Confidence Field Types**
```python
def test_confidence_types_float():
    """Verify confidence fields are float, not string."""
    from src.knowledge_graph.query_repository import RelatedEntity
    from typing import get_type_hints

    hints = get_type_hints(RelatedEntity)

    # entity_confidence should be Optional[float]
    assert hints['entity_confidence'] == Optional[float], \
        "entity_confidence should be Optional[float], not string"

    # relationship_confidence should be float
    assert hints['relationship_confidence'] == float, \
        "relationship_confidence should be float"
```

##### Category 4: Integration Tests (2 tests)

**Test 1.4.11: Schema Constraints Enforced**
```python
def test_confidence_constraint_enforced(database_connection):
    """Verify confidence CHECK constraint is enforced."""
    from src.knowledge_graph.models import KnowledgeEntity
    from sqlalchemy.orm import Session

    with Session(database_connection) as session:
        # Try to create entity with invalid confidence
        invalid_entity = KnowledgeEntity(
            text='Test',
            entity_type='TEST',
            confidence=1.5  # Outside [0.0, 1.0]
        )

        session.add(invalid_entity)

        with pytest.raises(Exception):  # Database constraint violation
            session.commit()
```

**Test 1.4.12: Foreign Key Constraints**
```python
def test_foreign_key_referential_integrity(database_connection):
    """Verify foreign key constraints prevent orphaned references."""
    from src.knowledge_graph.models import EntityRelationship
    from uuid import uuid4

    # Try to create relationship with non-existent entity
    bad_rel = EntityRelationship(
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type='test'
    )

    with Session(database_connection) as session:
        session.add(bad_rel)
        with pytest.raises(Exception):  # Foreign key violation
            session.commit()
```

##### Category 5: Regression Tests (3 tests)

**Test 1.4.13: Known Mismatch 1: entity_name Column**
```python
def test_entity_name_column_not_used():
    """REGRESSION: Verify entity_name column is not referenced."""
    from pathlib import Path
    content = Path('src/knowledge_graph/query_repository.py').read_text()

    # This specific error was in traverse_1hop at line 134
    assert 'entity_name' not in content, \
        "Regression: query still references non-existent entity_name column"
```

**Test 1.4.14: Known Mismatch 2: Metadata JSONB Extraction**
```python
def test_no_metadata_jsonb_extraction():
    """REGRESSION: Verify metadata JSONB extraction removed."""
    from pathlib import Path
    content = Path('src/knowledge_graph/query_repository.py').read_text()

    # This was in traverse_1hop, traverse_2hop, traverse_bidirectional
    assert "metadata->>" not in content, \
        "Regression: query still uses JSONB extraction on non-existent metadata"
```

**Test 1.4.15: Known Mismatch 3: chunk_entities Table**
```python
def test_chunk_entities_table_not_referenced():
    """REGRESSION: Verify chunk_entities table not referenced."""
    from pathlib import Path
    content = Path('src/knowledge_graph/query_repository.py').read_text()

    # This was in get_entity_mentions
    assert 'chunk_entities' not in content, \
        "Regression: query still references non-existent chunk_entities table"
```

#### Test Fixtures Required

**Fixture 1: database_inspector**
```python
@pytest.fixture(scope="function")
def database_inspector(test_database):
    """Provide database inspector for schema validation."""
    from sqlalchemy import inspect
    inspector = inspect(test_database)
    return inspector
```

**Fixture 2: database_connection**
```python
@pytest.fixture(scope="function")
def database_connection(test_database):
    """Provide database connection for constraint testing."""
    return test_database
```

#### Files to Create
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/tests/knowledge_graph/test_schema_alignment.py`

#### Files to Modify
- `/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local/tests/conftest.py` (add test_database fixture if missing)

#### Acceptance Criteria
- [ ] 15 tests created with clear, specific assertions
- [ ] Tests validate schema structure (3 tests)
- [ ] Tests validate ORM models (3 tests)
- [ ] Tests validate queries (4 tests)
- [ ] Tests validate constraints (2 tests)
- [ ] Regression tests for known mismatches (3 tests)
- [ ] All tests pass with corrected code from Task 1.2
- [ ] All tests fail with original code (to verify they catch issues)
- [ ] Test file documented with test purposes
- [ ] Pytest markers used for slow/database tests
- [ ] Test execution integrated with CI/CD

---

## Implementation Sequence

### Execution Order (SEQUENTIAL - each depends on previous)

```
Task 1.1: Audit Mismatches (30-45 min)
    ↓
Task 1.2: Fix Queries (1-1.5 hours)
    ↓
Task 1.3: Validate ORM Models (30-45 min) [parallel with 1.2 acceptable]
    ↓
Task 1.4: Create Tests (1-1.5 hours)
    ↓
Complete & Validate Blocker 1 Fix
```

### Parallel Opportunities
- Task 1.1 and Task 1.3 can run in parallel (both independent)
- Task 1.4 should wait for Tasks 1.2-1.3 to identify test scenarios

### Total Timeline
- **Sequential path**: 3.5-4.5 hours
- **With parallelization**: 3-4 hours (if 1.1 and 1.3 run simultaneously)

---

## Test Strategy

### How to Verify Blocker 1 is Fixed

#### Pre-Fix Validation (Tasks 1.1-1.2)
1. **Parse all SQL queries** for references to:
   - `entity_name` (should be `text`)
   - `metadata` (should not exist)
   - `chunk_entities` (should be `entity_mentions`)
   - `knowledge_base` (should not exist)

2. **Verify column type matches**:
   - All confidence fields are FLOAT, not JSONB string
   - All IDs are UUID, not integer
   - All string fields are TEXT/VARCHAR, not numeric

3. **Execute syntax validation**:
   ```bash
   python -m sqlparse query_repository.py
   ```

#### Post-Fix Validation (Task 1.4)
1. **Run test suite**:
   ```bash
   pytest tests/knowledge_graph/test_schema_alignment.py -v
   ```

2. **All 15 tests pass** ✓
   - Schema structure tests (3): PASS
   - ORM model tests (3): PASS
   - Query validation tests (4): PASS
   - Constraint tests (2): PASS
   - Regression tests (3): PASS

3. **Integration test passes**:
   ```bash
   pytest tests/knowledge_graph/test_query_repository.py -v
   ```

4. **Verify specific fixes**:
   - traverse_1hop uses `e.text` not `e.entity_name`
   - traverse_2hop uses `e2.confidence` not `e2.metadata->>'confidence'`
   - traverse_bidirectional uses `e.text` not `e.entity_name`
   - traverse_with_type_filter uses `e.text` not `e.entity_name`
   - get_entity_mentions uses `entity_mentions` not `chunk_entities`

---

## Rollback Plan

### If Schema Changes Needed During Fix

#### Scenario 1: Metadata Column Required
**Decision Point**: If Task 1.2 finds relationship metadata is essential
- Add `metadata JSONB` column to `entity_relationships` table
- Run migration in schema.sql
- Update models.py with new column
- Revert query changes for metadata references
- Note: Avoid this if possible; `relationship_weight` is simpler

#### Scenario 2: Query Changes Invalid
**Decision Point**: If corrected queries don't work in practice
- Revert all changes in query_repository.py to original
- Re-audit mismatches with actual error messages
- Investigate why schema differs from expected
- Adjust Task 1.2 approach based on real errors

#### Scenario 3: Test Database Issues
**Decision Point**: If test suite can't connect to database
- Mock database layer for syntax validation only
- Focus on static analysis (column references, table names)
- Defer constraint/integration tests until database available
- Validate fixes via code review instead

### Rollback Commands
```bash
# Revert query_repository.py to latest commit
git checkout HEAD src/knowledge_graph/query_repository.py

# Revert models.py if modified
git checkout HEAD src/knowledge_graph/models.py

# Remove test file if created
git rm tests/knowledge_graph/test_schema_alignment.py
```

---

## Sign-Off Criteria: Blocker 1 Complete

Blocker 1 is considered **RESOLVED** when:

1. **Column Reference Audit Complete**
   - [ ] Mismatch audit document created (Task 1.1)
   - [ ] All mismatches documented with line numbers
   - [ ] Mapping table created (query ref → schema column)

2. **Queries Corrected**
   - [ ] Task 1.2 completed
   - [ ] All 5 methods in query_repository.py fixed
   - [ ] No references to `entity_name`, `metadata` JSONB, or non-existent tables
   - [ ] Column references match schema exactly

3. **ORM Models Validated**
   - [ ] Task 1.3 completed
   - [ ] Validation report generated
   - [ ] All models aligned with schema
   - [ ] No model modifications needed (or documented)

4. **Test Suite Created**
   - [ ] Task 1.4 completed
   - [ ] 15+ tests created
   - [ ] All tests pass with corrected code
   - [ ] Regression tests prevent future misalignment

5. **Integration Verification**
   - [ ] Existing test_query_repository.py tests can run (with updated fixtures)
   - [ ] No SQL syntax errors in any query
   - [ ] Column type conversions work (UUID parsing, float handling)
   - [ ] Mock database tests pass

6. **Code Quality**
   - [ ] Code follows PEP 8 style
   - [ ] Docstrings updated to reflect correct columns
   - [ ] Type hints are accurate
   - [ ] All changes committed with clear messages

---

## Dependencies & Blockers

### External Dependencies
- PostgreSQL database schema (required for runtime validation)
- pytest framework (required for test execution)
- SQLAlchemy (already in project)
- sqlparse (for syntax validation) - may need pip install

### Internal Dependencies
- schema.sql (reference for correct columns)
- models.py (reference for ORM alignment)
- test_query_repository.py (reference for test expectations)

### Known Blockers
- **Schema inspection**: May need database running to validate constraints
- **Metadata design**: Decision needed on whether relationship metadata needed
- **ID type change**: May require test fixture updates if dataclasses change from int to UUID

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Mismatches identified | 15+ | Pending (Task 1.1) |
| Queries corrected | 5 methods | Pending (Task 1.2) |
| ORM models validated | 3 models | Pending (Task 1.3) |
| Tests created | 15+ tests | Pending (Task 1.4) |
| Test pass rate | 100% | Pending (Task 1.4) |
| Regression prevention | 3+ tests | Pending (Task 1.4) |
| Phase 1 unblocked | Yes | Pending |

---

## Notes & Observations

### Key Insights from Code Review
1. **Schema is correct**: Entity and relationship tables properly normalized
2. **Queries have wrong references**: Systematic use of non-existent columns
3. **ORM models are correct**: Already properly aligned with schema
4. **Tests are well-structured**: Just need fixture updates, not rewrite

### Architecture Decisions Made
1. **Removed metadata JSONB**: Use `relationship_weight` instead for relationship metadata
2. **Simplified mentions query**: Removed non-existent joins, use available columns only
3. **UUID vs int**: Schema uses UUID (correct for sharding), dataclasses need update

### Future Prevention
1. Add schema validation tests to CI/CD
2. Use SQLAlchemy migration tools (Alembic) for schema changes
3. Automatically sync docstrings with actual query columns
4. Consider GraphQL schema generation from SQL schema

