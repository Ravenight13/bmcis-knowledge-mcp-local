# Task 2.2 PostgreSQL Schema Constraint Tests - Quick Start Guide

## Overview

This guide helps you quickly set up and run the PostgreSQL schema constraint tests for the Knowledge Graph.

## What's Being Tested

30 comprehensive tests validating that invalid data cannot enter the database:

- **CHECK constraints**: Confidence ranges [0.0, 1.0], no self-loops
- **UNIQUE constraints**: Prevents duplicate entities and relationships
- **Foreign keys**: Referential integrity across tables
- **CASCADE deletes**: Automatic cleanup of related records
- **Triggers**: Auto-updating timestamps
- **Indexes**: All 11+ required indexes exist
- **Column types**: Correct data types (UUID, FLOAT, TEXT)
- **NULL constraints**: Required fields cannot be NULL
- **Precision**: Numeric accuracy for production use

## Quick Start (5 minutes)

### 1. Ensure PostgreSQL is Running

```bash
# Check if PostgreSQL is running
psql --version

# Start PostgreSQL if needed (macOS with Homebrew)
brew services start postgresql

# Or for Linux
sudo service postgresql start
```

### 2. Create Test Database

```bash
# Create test database (one-time setup)
createdb test_kg

# Verify it exists
psql -l | grep test_kg
```

### 3. Run Tests

```bash
# Run all 30 constraint tests
cd /Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local

pytest tests/knowledge_graph/test_schema_constraints.py -v

# Expected output:
# ================================ 30 passed in X.XXs ================================
```

## Running Specific Tests

```bash
# Run just CHECK constraint tests
pytest tests/knowledge_graph/test_schema_constraints.py::TestCheckConstraints -v

# Run just UNIQUE constraint tests
pytest tests/knowledge_graph/test_schema_constraints.py::TestUniqueConstraints -v

# Run just one test
pytest tests/knowledge_graph/test_schema_constraints.py::TestCheckConstraints::test_entity_confidence_check_lower_bound -v

# Run with detailed output and print statements
pytest tests/knowledge_graph/test_schema_constraints.py -vv -s
```

## Test Coverage by Category

### CHECK Constraints (4 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestCheckConstraints -v
```

### UNIQUE Constraints (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestUniqueConstraints -v
```

### Foreign Keys (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestForeignKeyConstraints -v
```

### CASCADE Delete (2 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestCascadeDelete -v
```

### Triggers (2 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestTriggers -v
```

### Indexes (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestIndexes -v
```

### Column Types (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestColumnTypes -v
```

### Constraint Naming (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestConstraintNaming -v
```

### NULL Constraints (3 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestNullConstraints -v
```

### Numeric Precision (2 tests)
```bash
pytest tests/knowledge_graph/test_schema_constraints.py::TestDataTypePrecision -v
```

## Test Coverage Report

```bash
# Generate coverage report
pytest tests/knowledge_graph/test_schema_constraints.py \
  --cov=src.knowledge_graph \
  --cov-report=term-missing \
  --cov-report=html

# View HTML report
open htmlcov/index.html
```

## Troubleshooting

### Error: "test_kg" database does not exist

```bash
# Create the test database
createdb test_kg

# Verify creation
psql -c "SELECT datname FROM pg_database WHERE datname='test_kg'"
```

### Error: "connection refused" to localhost:5432

PostgreSQL is not running. Start it:

```bash
# macOS
brew services start postgresql

# Linux
sudo service postgresql start

# Or verify connection directly
psql -h localhost -U postgres -d postgres -c "SELECT version()"
```

### Error: "permission denied" on test_kg

Ensure PostgreSQL user has access:

```bash
# Connect as superuser and grant permissions
psql -U postgres -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE test_kg TO postgres"
```

### Error: Test fixtures timeout

Increase pytest timeout or check PostgreSQL response:

```bash
# Run with longer timeout
pytest tests/knowledge_graph/test_schema_constraints.py --timeout=30 -v

# Check PostgreSQL status
psql -c "SELECT 1"
```

## Performance Expectations

- **Full test suite**: ~5-10 seconds
- **Single test**: ~100-500ms
- **Index tests**: ~50-200ms (queries pg_indexes)
- **Trigger tests**: ~100-300ms (includes time.sleep(0.1))

## Files Involved

- **Test File**: `tests/knowledge_graph/test_schema_constraints.py` (1,339 lines)
- **Database Schema**: `src/knowledge_graph/schema.sql`
- **ORM Models**: `src/knowledge_graph/models.py`
- **Implementation Report**: `docs/subagent-reports/task-implementation/2025-11-09-1300-task-2-2-schema-constraints-tests.md`

## Key Testing Patterns

### Test Pattern 1: CHECK Constraint

```python
def test_entity_confidence_check_lower_bound(self, db_connection, cleanup_entities):
    """Test CHECK constraint rejects invalid values."""
    cursor = db_connection.cursor()

    with pytest.raises(IntegrityError) as exc_info:
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Test", "PERSON", -0.1)  # Invalid
        )
        db_connection.commit()

    error_msg = str(exc_info.value).lower()
    assert "confidence" in error_msg
```

### Test Pattern 2: UNIQUE Constraint

```python
def test_entity_unique_text_type_constraint(self, db_connection, cleanup_entities):
    """Test UNIQUE constraint prevents duplicates."""
    cursor = db_connection.cursor()

    # First insert should succeed
    cursor.execute(
        """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
           VALUES (%s, %s, %s, %s)""",
        (str(uuid4()), "Lutron", "VENDOR", 0.95)
    )
    db_connection.commit()

    # Second insert should fail
    with pytest.raises(IntegrityError):
        cursor.execute(
            """INSERT INTO knowledge_entities (id, text, entity_type, confidence)
               VALUES (%s, %s, %s, %s)""",
            (str(uuid4()), "Lutron", "VENDOR", 0.90)  # Duplicate!
        )
        db_connection.commit()
```

### Test Pattern 3: FK Constraint

```python
def test_relationship_fk_source_entity_required(self, db_connection, cleanup_entities, valid_entity_id):
    """Test FK constraint requires valid source entity."""
    cursor = db_connection.cursor()

    fake_source_id = uuid4()

    with pytest.raises(IntegrityError):
        cursor.execute(
            """INSERT INTO entity_relationships
               (id, source_entity_id, target_entity_id, relationship_type, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (str(uuid4()), str(fake_source_id), str(valid_entity_id), "test", 0.8)
        )
        db_connection.commit()
```

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Run PostgreSQL Constraint Tests
  run: |
    pytest tests/knowledge_graph/test_schema_constraints.py \
      --tb=short \
      --cov=src.knowledge_graph \
      --junitxml=test-results.xml
```

## Understanding Test Output

### Successful Run
```
test_schema_constraints.py::TestCheckConstraints::test_entity_confidence_check_lower_bound PASSED
test_schema_constraints.py::TestCheckConstraints::test_entity_confidence_check_upper_bound PASSED
test_schema_constraints.py::TestCheckConstraints::test_entity_confidence_check_boundary_values PASSED
test_schema_constraints.py::TestCheckConstraints::test_relationship_confidence_check_constraint PASSED
test_schema_constraints.py::TestCheckConstraints::test_no_self_loops_check_constraint PASSED
test_schema_constraints.py::TestUniqueConstraints::test_entity_unique_text_type_constraint PASSED
...
================================ 30 passed in 5.23s ================================
```

### Failed Test Example
```
FAILED test_schema_constraints.py::TestCheckConstraints::test_entity_confidence_check_lower_bound
AssertionError: assert 'confidence' in 'database constraint violation'

The constraint name 'confidence' was not in the error message.
This typically means the constraint in PostgreSQL is named differently.
```

## Next Steps

1. **Run the tests**: `pytest tests/knowledge_graph/test_schema_constraints.py -v`
2. **Review results**: Check output for any failures
3. **Implement Task 2.1**: ORM model constraint tests
4. **Set up CI/CD**: Add tests to pre-commit and GitHub Actions
5. **Monitor**: Run tests regularly to catch data integrity issues

## Resources

- **Test File**: `tests/knowledge_graph/test_schema_constraints.py`
- **Documentation**: `docs/subagent-reports/task-implementation/2025-11-09-1300-task-2-2-schema-constraints-tests.md`
- **Summary**: `TASK_2_2_IMPLEMENTATION_SUMMARY.md`
- **Schema**: `src/knowledge_graph/schema.sql`
- **Models**: `src/knowledge_graph/models.py`

## Contact & Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test file docstrings for detailed explanations
3. Check implementation report for design decisions
4. Verify PostgreSQL is running and test database exists
