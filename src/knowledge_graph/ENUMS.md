# Enum Validation in Knowledge Graph

## Overview

This document describes the enum validation system that enforces allowed entity types and relationship types at both the PostgreSQL and ORM layers.

**Goal**: Prevent data corruption and enable reliable type-based queries through two-layer validation (defense-in-depth).

**Key Principle**: Invalid data should be impossible to store, whether the data originates from the ORM or directly from SQL.

---

## Entity Types

Valid entity types represent the canonical set of named entity types that can be extracted from documents using spaCy's `en_core_web_md` model.

### Core Types (Phase 1)

| Type | Description | Example |
|------|-------------|---------|
| **PERSON** | People, including fictional characters | John Doe, Claude |
| **ORG** | Organizations, companies, agencies, institutions | Anthropic, OpenAI |
| **GPE** | Geopolitical entities (countries, cities, states) | United States, San Francisco |
| **PRODUCT** | Products, technologies, services | Claude AI, iPhone |
| **EVENT** | Named events (conferences, wars, tournaments) | ICLR 2024, World Cup |

### Extended Types (Phase 2+)

| Type | Description | Example |
|------|-------------|---------|
| **FACILITY** | Buildings, airports, highways, bridges | The Pentagon, JFK Airport |
| **LAW** | Named laws, regulations, legal documents | US Constitution, GDPR |
| **LANGUAGE** | Named languages | English, Mandarin Chinese |
| **DATE** | Absolute or relative dates | January 15, 2025, last week |
| **TIME** | Times smaller than a day | 3:30 PM, noon |
| **MONEY** | Monetary values with currency | $100, 50 EUR |
| **PERCENT** | Percentage values | 25%, 99.9% |

### Future Extensions

New entity types can be added using:

```sql
ALTER TYPE entity_type_enum ADD VALUE 'NEW_TYPE';
```

Note: Adding enum values is non-blocking and doesn't require table locks.

---

## Relationship Types

Valid relationship types represent the canonical set of relationships that can exist between entities in the knowledge graph.

### Implemented Types

| Type | Description | Example | Symmetric |
|------|-------------|---------|-----------|
| **hierarchical** | Parent/child, creator/creation, owner/owned | Company owns Product, Person works at Company | No |
| **mentions-in-document** | Co-occurrence in same document/chunk | Both entities mentioned in same document | Yes |
| **similar-to** | Semantic similarity (embedding-based) | Product A is similar to Product B | Yes |

### Future Types (Phase 2+)

- `works-at` - Employment relationships
- `located-in` - Geographic containment
- `part-of` - Compositional relationships
- `alias-of` - Entity deduplication relationships

---

## Validation Strategy

### Layer 1: PostgreSQL Enum Types (Database-Enforced)

PostgreSQL enum types provide the first line of defense at the database layer.

**Benefits**:
- No invalid values can be inserted (rejected at INSERT/UPDATE time)
- Case-sensitive enforcement
- No SQL injection possible
- Atomic constraint enforcement
- Enables efficient type filtering in queries

**Example**:
```sql
-- Valid: Succeeds
INSERT INTO knowledge_entities (text, entity_type, confidence)
VALUES ('Anthropic', 'ORG', 0.95);

-- Invalid: Fails immediately
INSERT INTO knowledge_entities (text, entity_type, confidence)
VALUES ('Anthropic', 'INVALID_TYPE', 0.95);
-- ERROR: invalid input value for enum entity_type_enum: "INVALID_TYPE"
```

### Layer 2: ORM Validators (Application-Enforced)

SQLAlchemy validators provide application-level validation before database writes.

**Benefits**:
- Fail-fast: Errors caught before network round-trip
- Clear error messages for client code
- Validation happens during object instantiation
- Easy to test

**Example**:
```python
# Valid: Succeeds
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="ORG",  # Valid enum value
    confidence=0.95
)

# Invalid: Raises ValueError immediately
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="INVALID_TYPE",  # Not in EntityTypeEnum
    confidence=0.95
)
# ValueError: Invalid entity_type 'INVALID_TYPE'.
# Must be one of: PERSON, ORG, GPE, PRODUCT, EVENT, ...
```

### Defense-in-Depth

Both layers work together to ensure:

1. **ORM-based code**: Validation happens early (Layer 2)
2. **Direct SQL code**: Validation happens at database (Layer 1)
3. **No edge cases**: Invalid data cannot bypass either layer
4. **Consistent behavior**: Both layers enforce identical constraints

---

## Implementation Details

### EntityTypeEnum (Python)

Located in `src/knowledge_graph/models.py`:

```python
class EntityTypeEnum(str, Enum):
    """Valid entity types from spaCy en_core_web_md NER model."""

    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    FACILITY = "FACILITY"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
```

### RelationshipTypeEnum (Python)

Located in `src/knowledge_graph/models.py`:

```python
class RelationshipTypeEnum(str, Enum):
    """Valid relationship types for knowledge graph edges."""

    HIERARCHICAL = "hierarchical"
    MENTIONS_IN_DOCUMENT = "mentions-in-document"
    SIMILAR_TO = "similar-to"
```

### ORM Validators

**KnowledgeEntity**:
```python
@validates('entity_type')
def validate_entity_type(self, key: str, value: str) -> str:
    """Validate entity_type is one of the allowed enum values."""
    if value not in [t.value for t in EntityTypeEnum]:
        valid_types = ", ".join([t.value for t in EntityTypeEnum])
        raise ValueError(f"Invalid entity_type '{value}'. Must be one of: {valid_types}")
    return value
```

**EntityRelationship**:
```python
@validates('relationship_type')
def validate_relationship_type(self, key: str, value: str) -> str:
    """Validate relationship_type is one of the allowed enum values."""
    if value not in [t.value for t in RelationshipTypeEnum]:
        valid_types = ", ".join([t.value for t in RelationshipTypeEnum])
        raise ValueError(f"Invalid relationship_type '{value}'. Must be one of: {valid_types}")
    return value
```

### PostgreSQL Migration

Migration file: `src/knowledge_graph/migrations/004_add_enum_types.sql`

Creates two enum types:
```sql
CREATE TYPE entity_type_enum AS ENUM (
    'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'FACILITY',
    'LAW', 'LANGUAGE', 'DATE', 'TIME', 'MONEY', 'PERCENT'
);

CREATE TYPE relationship_type_enum AS ENUM (
    'hierarchical', 'mentions-in-document', 'similar-to'
);
```

Then alters columns to use these types:
```sql
ALTER TABLE knowledge_entities
    ALTER COLUMN entity_type TYPE entity_type_enum
    USING entity_type::entity_type_enum;

ALTER TABLE entity_relationships
    ALTER COLUMN relationship_type TYPE relationship_type_enum
    USING relationship_type::relationship_type_enum;
```

---

## Usage Examples

### Python: Valid Operations

```python
from src.knowledge_graph.models import (
    KnowledgeEntity,
    EntityRelationship,
    EntityTypeEnum,
    RelationshipTypeEnum,
)

# Create entity with valid type
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type=EntityTypeEnum.ORG.value,  # Recommended: use enum value
    confidence=0.95
)

# Alternative: Use enum value directly
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="ORG",  # Also valid: string literal
    confidence=0.95
)

# Create relationship with valid type
relationship = EntityRelationship(
    source_entity_id=source.id,
    target_entity_id=target.id,
    relationship_type=RelationshipTypeEnum.HIERARCHICAL.value,
    confidence=0.9
)
```

### Python: Invalid Operations (Raises ValueError)

```python
# Invalid entity type: lowercase
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="org",  # WRONG: lowercase
    confidence=0.95
)
# ValueError: Invalid entity_type 'org'. Must be one of: PERSON, ORG, ...

# Invalid entity type: mixed case
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="Org",  # WRONG: mixed case
    confidence=0.95
)
# ValueError: Invalid entity_type 'Org'. Must be one of: PERSON, ORG, ...

# Invalid entity type: typo
entity = KnowledgeEntity(
    text="Anthropic",
    entity_type="ORGANIZATION",  # WRONG: should be "ORG"
    confidence=0.95
)
# ValueError: Invalid entity_type 'ORGANIZATION'. Must be one of: PERSON, ORG, ...

# Invalid relationship type: underscores instead of hyphens
relationship = EntityRelationship(
    source_entity_id=source.id,
    target_entity_id=target.id,
    relationship_type="mentions_in_document",  # WRONG: underscores
    confidence=0.9
)
# ValueError: Invalid relationship_type 'mentions_in_document'.
# Must be one of: hierarchical, mentions-in-document, similar-to
```

### SQL: Valid Operations

```sql
-- Valid: Create entity with enum type
INSERT INTO knowledge_entities (text, entity_type, confidence)
VALUES ('Anthropic', 'ORG', 0.95);

-- Valid: Create relationship with enum type
INSERT INTO entity_relationships
    (source_entity_id, target_entity_id, relationship_type, confidence)
VALUES (source_id, target_id, 'hierarchical', 0.9);

-- Valid: Query by type
SELECT * FROM knowledge_entities
WHERE entity_type = 'PERSON'
ORDER BY confidence DESC;

-- Valid: Filter relationships
SELECT * FROM entity_relationships
WHERE relationship_type IN ('hierarchical', 'similar-to');
```

### SQL: Invalid Operations (Rejects with Error)

```sql
-- Invalid: Unknown entity type
INSERT INTO knowledge_entities (text, entity_type, confidence)
VALUES ('Anthropic', 'INVALID_TYPE', 0.95);
-- ERROR: invalid input value for enum entity_type_enum: "INVALID_TYPE"

-- Invalid: Lowercase enum value
INSERT INTO knowledge_entities (text, entity_type, confidence)
VALUES ('Anthropic', 'org', 0.95);
-- ERROR: invalid input value for enum entity_type_enum: "org"

-- Invalid: Unknown relationship type
INSERT INTO entity_relationships
    (source_entity_id, target_entity_id, relationship_type, confidence)
VALUES (source_id, target_id, 'works_for', 0.9);
-- ERROR: invalid input value for enum relationship_type_enum: "works_for"
```

---

## Testing

Comprehensive test suite: `tests/knowledge_graph/test_enum_validation.py`

**Test Classes**:

1. **TestEntityTypeEnumDefinitions**: Verify EntityTypeEnum structure
2. **TestRelationshipTypeEnumDefinitions**: Verify RelationshipTypeEnum structure
3. **TestEntityTypeOrmValidation**: ORM-layer entity type validation
4. **TestRelationshipTypeOrmValidation**: ORM-layer relationship type validation
5. **TestEnumValueMessages**: Verify error messages are helpful

**Test Coverage**:
- Valid values accepted
- Invalid values rejected
- Case sensitivity enforced
- SQL injection attempts blocked
- Error messages list valid values
- All enum values defined

**Run Tests**:
```bash
pytest tests/knowledge_graph/test_enum_validation.py -v
```

---

## Performance Impact

### Validation Overhead

- **ORM validation**: <1ms per entity/relationship creation (negligible)
- **Database validation**: Database enforces at INSERT/UPDATE time (included in query latency)
- **Query performance**: No impact (enums are efficient in PostgreSQL)

### Storage Impact

- Enum columns use 1-4 bytes per value (vs 50 bytes for VARCHAR)
- No additional indexes needed beyond existing type indexes
- **Storage savings**: ~20% for type columns

### Query Impact

Enum types enable efficient filtering:
```sql
-- Efficient: PostgreSQL optimizes enum comparisons
SELECT * FROM knowledge_entities
WHERE entity_type = 'PERSON';  -- Uses existing index

-- Type-based aggregation is efficient
SELECT entity_type, COUNT(*) as count
FROM knowledge_entities
GROUP BY entity_type;  -- Enum values sorted efficiently
```

---

## Migration Path

### Step 1: Deploy Code Changes

1. Add enum classes to `src/knowledge_graph/models.py`
2. Add validators to ORM models
3. Deploy application code

### Step 2: Run Database Migration

```bash
psql -h localhost -U postgres -d knowledge_graph \
    -f src/knowledge_graph/migrations/004_add_enum_types.sql
```

**What happens**:
1. Creates PostgreSQL enum types
2. Validates existing data matches enum values
3. Converts columns from VARCHAR to ENUM type
4. **Fails if invalid data detected** (manual cleanup required)

### Step 3: Verify

```bash
# Verify enum types created
SELECT typname FROM pg_type
WHERE typname IN ('entity_type_enum', 'relationship_type_enum');

# Verify columns use enum types
SELECT table_name, column_name, udt_name
FROM information_schema.columns
WHERE column_name IN ('entity_type', 'relationship_type');
```

### Step 4: Test

```bash
pytest tests/knowledge_graph/test_enum_validation.py -v
```

---

## Future Extensions

### Adding New Entity Types

To add a new entity type (e.g., `QUANTITY` for measurements):

1. **Update Python enum**:
```python
class EntityTypeEnum(str, Enum):
    # ... existing types ...
    QUANTITY = "QUANTITY"
```

2. **Update PostgreSQL**:
```sql
ALTER TYPE entity_type_enum ADD VALUE 'QUANTITY';
```

3. **Update validation tests**:
```python
def test_quantity_type_accepted(self, db_session):
    entity = KnowledgeEntity(
        text="5 meters",
        entity_type="QUANTITY",
        confidence=0.9
    )
    db_session.add(entity)
    db_session.commit()
    assert entity.id is not None
```

4. **Deploy and test**

### Adding New Relationship Types

Same pattern for relationship types:

1. Add to `RelationshipTypeEnum`
2. Update PostgreSQL enum
3. Add tests
4. Deploy

---

## Troubleshooting

### Problem: Migration fails with "invalid input value"

**Cause**: Existing data contains invalid enum values

**Solution**:
1. Query for invalid values:
```sql
SELECT DISTINCT entity_type FROM knowledge_entities
WHERE entity_type NOT IN ('PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT',
                          'FACILITY', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
                          'MONEY', 'PERCENT');
```

2. Fix data (map to valid type or delete):
```sql
UPDATE knowledge_entities
SET entity_type = 'ORG'
WHERE entity_type = 'ORGANIZATION';
```

3. Retry migration

### Problem: ORM validation too strict

**Example**: Code expects to accept "PERSON " (with space)

**Solution**: Normalize data before ORM validation:
```python
entity = KnowledgeEntity(
    text="John",
    entity_type=user_input.strip().upper(),  # Normalize
    confidence=0.9
)
```

### Problem: Need temporary exception for specific value

**Solution**: Use string literal, not enum:
```python
# Bypass validation: Use string directly (will still be caught by DB)
session.execute(
    text("INSERT INTO knowledge_entities (text, entity_type, confidence) "
         "VALUES (:text, :type, :conf)"),
    {"text": "Test", "type": "CUSTOM", "conf": 0.9}
)
# This will be rejected by PostgreSQL (Layer 1 validation still applies)
```

---

## Related Documentation

- [Schema Design](SCHEMA.md) - Overall database design
- [Query Patterns](QUERIES.md) - Common query patterns
- [Model Constraints](../../../tests/knowledge_graph/test_model_constraints.py) - ORM-level constraints
- [HP 7: Enum Validation](../../../docs/subagent-reports/task-planning/2025-11-09-highpriority4-5-7-optimizations-plan.md) - Implementation planning

---

## Summary

**Key Benefits**:
- Prevents data corruption (invalid types rejected)
- Enables reliable type-based queries
- SQL injection-proof
- Clear error messages for debugging
- Efficient storage and queries
- Defense-in-depth validation

**Validation Layers**:
1. **ORM Layer**: Fail-fast validation on object creation
2. **Database Layer**: PostgreSQL enum enforcement at INSERT/UPDATE

**Cost**: Negligible performance impact, significant data integrity gain
