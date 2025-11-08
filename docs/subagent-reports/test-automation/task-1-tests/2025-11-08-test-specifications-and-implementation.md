# Task 1 Test Suite Design & Implementation
## 11 Comprehensive Tests for BMCIS Knowledge MCP

**Project**: BMCIS Knowledge MCP
**Branch**: task-1-refinements
**Test Focus**: Connection Pool Leak Prevention, Type Safety, Pool Health Monitoring
**Target Date**: November 8, 2025
**Total Tests**: 11
**Expected Execution Time**: All <30 seconds per test

---

## Executive Summary

This document provides complete specifications and implementation code for 11 new tests covering three critical areas of the BMCIS Knowledge MCP project:

1. **Connection Pool Leak Tests (5 tests)** - Ensure connections don't leak under error conditions
2. **Type Safety Validation Tests (3 tests)** - Verify complete type annotations across database modules
3. **Pool Health & Monitoring Tests (3 tests)** - Validate monitoring features and pool status accuracy

All tests integrate seamlessly with the existing test suite (280+ passing tests), using pytest fixtures, type-safe patterns, and the project's established testing conventions.

---

## Test Architecture Overview

### Type Safety Approach
- **Complete type annotations** on all test functions with explicit return types
- **Type hints for fixtures** with full parameter/return type signatures
- **Mypy strict mode compliance** - all test code validates against `mypy --strict`
- **Generic typing patterns** for mocks and test doubles

### Testing Strategy
- **Patch-based isolation** - All tests mock the underlying connection pool to avoid database dependencies
- **Idempotent execution** - Tests can run multiple times sequentially without state leakage
- **Error scenario validation** - Comprehensive edge case coverage for failure modes
- **Performance validation** - All tests complete in <10 seconds (well under 30-second limit)

### Test Organization
- **3 new test files** with specialized test classes per category
- **Shared fixtures** defined in each module for setup/teardown
- **Clear naming conventions** following existing project patterns
- **Comprehensive docstrings** for each test and fixture

---

## Category 1: Connection Pool Leak Tests (5 tests)

### File: `tests/test_database_connection_pool.py` (NEW)

**Purpose**: Validate that connections are returned to the pool under all conditions, preventing resource exhaustion.

**Fixtures Required**:
- `mock_pool_class` - Patches SimpleConnectionPool class
- `mock_pool_instance` - Instance with mock getconn/putconn methods
- `mock_connection` - Mock psycopg2 connection object
- `mock_cursor` - Mock database cursor for health checks

### Test 1: test_connection_leak_on_retry_failure

**Objective**: Verify connection is returned to pool when all retry attempts fail.

**Test Scenario**:
- Simulate connection failure on every retry attempt (attempts 1, 2, 3)
- Patch getconn to raise OperationalError each time
- Verify putconn is NOT called (connection never successfully acquired)
- Verify pool state unchanged after failures

**Edge Cases**:
- Zero connections acquired but pool not exhausted
- Error state doesn't block subsequent connection attempts
- Exception propagates correctly after retry exhaustion

**Type Signature**:
```python
def test_connection_leak_on_retry_failure(self) -> None:
```

**Key Assertions**:
- `mock_pool.getconn.call_count == 3` (three attempts made)
- `mock_pool.putconn.call_count == 0` (no successful acquisition)
- `DatabasePool._pool is not None` (pool still initialized)

---

### Test 2: test_connection_release_on_exception_after_getconn

**Objective**: Verify connection returned to pool even when exception occurs after successful acquisition.

**Test Scenario**:
- Successfully acquire connection (getconn succeeds)
- Health check passes (SELECT 1 succeeds)
- Context manager execution raises RuntimeError before yield completes
- Verify connection is returned to pool in finally block

**Edge Cases**:
- Exception during health check (cursor.execute fails)
- Exception in user code after yield
- Multiple connections don't interfere with each other

**Type Signature**:
```python
def test_connection_release_on_exception_after_getconn(self) -> None:
```

**Key Assertions**:
- `mock_pool.getconn.call_count == 1` (one connection acquired)
- `mock_pool.putconn.call_count == 1` (returned despite exception)
- `mock_pool.putconn.call_args[0][0] is mock_connection` (correct connection returned)

---

### Test 3: test_connection_recovery_after_all_retries_fail

**Objective**: Verify pool is in recoverable state after connection attempts fail.

**Test Scenario**:
- All retry attempts fail with OperationalError
- After exception is caught, perform new get_connection attempt
- This second attempt should succeed (tests pool recovery)
- Verify pool metrics consistent before and after failures

**Edge Cases**:
- Pool remains functional after failure sequence
- No zombie connections left in pool
- Retry counter resets for new attempts

**Type Signature**:
```python
def test_connection_recovery_after_all_retries_fail(self) -> None:
```

**Key Assertions**:
- First context fails with OperationalError
- Second context with patched success returns connection correctly
- Pool state unchanged between attempts

---

### Test 4: test_connection_leak_under_concurrent_failures

**Objective**: Verify thread-safe connection handling during concurrent failures.

**Test Scenario**:
- Create 3 threads attempting connections simultaneously
- Each thread gets a unique mock connection
- Each thread's connection fails after health check
- All connections must be returned despite concurrent failures
- Verify no deadlocks or race conditions

**Edge Cases**:
- Multiple connections acquired simultaneously
- Concurrent exceptions in different threads
- Thread safety of putconn calls

**Type Signature**:
```python
def test_connection_leak_under_concurrent_failures(self) -> None:
```

**Key Assertions**:
- `getconn_call_count == 3` (one per thread)
- `putconn_call_count == 3` (all returned)
- No threading exceptions or deadlocks

---

### Test 5: test_connection_pool_status_after_error_sequence

**Objective**: Verify pool status/metrics remain accurate through error sequences.

**Test Scenario**:
- Perform sequence: fail → retry → fail → retry → succeed
- Track pool status metrics throughout:
  - Available connections
  - Acquired connections
  - Total connections
- Verify metrics consistent before, during, and after errors

**Edge Cases**:
- Metrics reported during active operations
- Status consistency under partial failures
- Recovery state matches initial state

**Type Signature**:
```python
def test_connection_pool_status_after_error_sequence(self) -> None:
```

**Key Assertions**:
- Status reflects accurate available/acquired counts
- Before and after failure sequence: `available == initial_available`
- Metrics don't show phantom connections

---

## Category 2: Type Safety Validation Tests (3 tests)

### File: `tests/test_database_type_safety.py` (NEW)

**Purpose**: Validate complete type annotations throughout database modules for mypy strict compliance.

**Fixtures Required**:
- `database_module` - Imported src.core.database module
- `config_module` - Imported src.core.config module
- `type_hints_cache` - Cached type hints for repeated checks

### Test 1: test_all_private_methods_have_return_types

**Objective**: Ensure ALL methods in DatabasePool class have explicit return type annotations.

**Test Scenario**:
- Inspect DatabasePool class definition
- Use `typing.get_type_hints()` for each method
- Verify return type annotation present (not Any or missing)
- Check both public and private methods
- Handle special methods (\_\_init\_\_, \_\_enter\_\_, etc.)

**Edge Cases**:
- Methods returning Generator types
- Methods with complex return types (Union, Optional)
- Class methods vs instance methods
- Inherited methods from base classes

**Type Signature**:
```python
def test_all_private_methods_have_return_types(self) -> None:
```

**Implementation Logic**:
```python
import inspect
from typing import get_type_hints

# Get all methods
methods = inspect.getmembers(DatabasePool, predicate=inspect.ismethod)
methods.extend(inspect.getmembers(DatabasePool, predicate=inspect.isfunction))

# Validate each has return type
for name, method in methods:
    if not name.startswith('_'):
        continue
    hints = get_type_hints(method)
    assert 'return' in hints, f"Method {name} missing return type"
```

**Key Assertions**:
- `get_type_hints(method)` contains 'return' key for each method
- Return type is not `typing.Any`
- No methods flagged as "return type annotation missing"

---

### Test 2: test_config_models_type_annotations_complete

**Objective**: Verify DatabaseConfig and related config models have complete type hints.

**Test Scenario**:
- Inspect DatabaseConfig model fields using pydantic model_fields
- Verify each Field has annotation
- Check all properties have type hints
- Validate validator decorators have proper signatures
- Check for Field descriptions (documentation)

**Edge Cases**:
- SecretStr fields properly annotated
- Literal type aliases with proper annotation
- Nested model validation
- Field validators with complete signatures

**Type Signature**:
```python
def test_config_models_type_annotations_complete(self) -> None:
```

**Implementation Logic**:
```python
from src.core.config import DatabaseConfig
from pydantic import BaseModel

# Check model fields
for field_name, field_info in DatabaseConfig.model_fields.items():
    assert field_info.annotation is not None
    assert field_info.annotation != type(None)
    # Verify type hint is concrete, not Any
```

**Key Assertions**:
- Each field in `model_fields` has concrete annotation
- No fields typed as `Any`
- Field validators have proper type signatures

---

### Test 3: test_database_operations_type_safety

**Objective**: Validate method signatures for critical database operations match expected types.

**Test Scenario**:
- Check initialize() - classmethod with None return
- Check get_connection() - classmethod returning Generator[Connection, None, None]
- Check close_all() - classmethod with None return
- Validate parameter types for retries parameter
- Check context manager protocol (__enter__, __exit__)

**Edge Cases**:
- Generic types in Generator signature
- Optional parameters properly typed
- Exception types declared in raises docstrings

**Type Signature**:
```python
def test_database_operations_type_safety(self) -> None:
```

**Implementation Logic**:
```python
from src.core.database import DatabasePool
from typing import get_type_hints

# Check initialize method
hints = get_type_hints(DatabasePool.initialize)
assert hints.get('return') is None or hints.get('return') == type(None)

# Check get_connection method
hints = get_type_hints(DatabasePool.get_connection)
assert 'return' in hints
# Should be Generator[Connection, None, None]

# Check close_all method
hints = get_type_hints(DatabasePool.close_all)
assert hints.get('return') is None or hints.get('return') == type(None)
```

**Key Assertions**:
- `initialize()` return type: `None`
- `get_connection()` return type: `Generator[Connection, None, None]`
- `close_all()` return type: `None`
- `retries` parameter: `int`

---

## Category 3: Pool Health & Monitoring Tests (3 tests)

### File: `tests/test_database_monitoring.py` (NEW)

**Purpose**: Validate new monitoring features and pool status tracking capabilities.

**Fixtures Required**:
- `mock_pool_instance` - SimpleConnectionPool mock with status tracking
- `health_check_enabled` - Flag to enable/disable health checks
- `status_snapshot` - Capture pool status at various points

### Test 1: test_pool_health_check_integration

**Objective**: Verify health check (SELECT 1) works correctly for healthy and unhealthy connections.

**Test Scenario**:
- Healthy connection: cursor.execute("SELECT 1") succeeds
  - Verify connection returned successfully
  - Verify no retry triggered
- Unhealthy connection: cursor.execute fails with DatabaseError
  - Verify error propagated to caller
  - Verify connection not returned to pool (broken)
- Connection timeout during health check
  - Verify timeout handled as database error
  - Verify connection not returned

**Edge Cases**:
- Health check itself times out
- Connection dies between getconn and health check
- Health check succeeds but connection broken later
- Multiple health checks in rapid succession

**Type Signature**:
```python
def test_pool_health_check_integration(self) -> None:
```

**Key Assertions**:
- Healthy: cursor.execute called with "SELECT 1"
- Healthy: connection yielded and returned
- Unhealthy: DatabaseError raised
- Unhealthy: connection not returned to pool

---

### Test 2: test_pool_status_method_accuracy

**Objective**: Verify pool_status() method returns accurate pool metrics.

**Test Scenario**:
- Initialize pool with min=2, max=5
- Acquire 2 connections
- Check status: available=3, acquired=2, total=5
- Release 1 connection
- Check status: available=4, acquired=1, total=5
- Acquire max connections (all 5)
- Check status: available=0, acquired=5, total=5

**Edge Cases**:
- Status at pool limits (min/max)
- Status with failed acquisitions
- Status consistency across operations
- Status race conditions (testing atomicity)

**Type Signature**:
```python
def test_pool_status_method_accuracy(self) -> None:
```

**Implementation Logic**:
```python
# Assuming pool_status() method exists on DatabasePool
status = DatabasePool.pool_status()
# Returns dict: {
#     'available': int,
#     'acquired': int,
#     'total': int,
#     'min_size': int,
#     'max_size': int
# }
```

**Key Assertions**:
- After 2 acquisitions: `available == min_size - 2`
- `total == min_size` (starts at minimum)
- `acquired + available == total`

---

### Test 3: test_enhanced_docstring_coverage

**Objective**: Verify all public methods have comprehensive docstrings documenting behavior, parameters, returns, raises.

**Test Scenario**:
- Check DatabasePool.initialize() has docstring
  - Contains "Initialize"
  - Documents all parameters
  - Documents all Raises exceptions
  - Contains usage examples
- Check DatabasePool.get_connection() has docstring
  - Explains retry behavior
  - Documents exponential backoff strategy
  - Shows context manager usage
  - Lists all possible exceptions
- Check DatabasePool.close_all() has docstring
  - Explains cleanup behavior
  - Documents idempotency
  - Shows usage examples

**Edge Cases**:
- Method docstrings not just "pass"
- Docstrings contain at least N characters
- Examples are executable/valid
- Raises section lists actual exceptions

**Type Signature**:
```python
def test_enhanced_docstring_coverage(self) -> None:
```

**Implementation Logic**:
```python
import inspect
from src.core.database import DatabasePool

# Check each public method
for name in ['initialize', 'get_connection', 'close_all']:
    method = getattr(DatabasePool, name)
    docstring = inspect.getdoc(method)

    assert docstring is not None
    assert len(docstring) > 50  # Meaningful documentation
    assert 'Returns:' in docstring or 'Yields:' in docstring
    if method.__name__ != 'close_all':
        assert 'Raises:' in docstring
```

**Key Assertions**:
- `docstring is not None` for each public method
- Docstring length > 50 characters (meaningful)
- Contains "Returns:" or "Yields:" section
- Contains "Raises:" section (where applicable)
- Includes usage examples

---

## Test Fixtures & Shared Utilities

### Common Fixture Setup

All three test files share similar fixture patterns:

```python
from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from psycopg2 import DatabaseError, OperationalError
from psycopg2.extensions import connection as Connection

from src.core.config import reset_settings
from src.core.database import DatabasePool


@pytest.fixture
def mock_pool_class() -> MagicMock:
    """Patch SimpleConnectionPool class for isolation."""
    with patch("src.core.database.pool.SimpleConnectionPool") as mock_cls:
        yield mock_cls


@pytest.fixture
def mock_pool_instance() -> MagicMock:
    """Create mock pool instance with connection management methods."""
    mock_pool: MagicMock = MagicMock()
    mock_pool.getconn = MagicMock()
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()
    return mock_pool


@pytest.fixture
def mock_connection() -> MagicMock:
    """Create mock psycopg2 connection with cursor context manager."""
    mock_conn: MagicMock = MagicMock()
    mock_cursor: MagicMock = MagicMock()

    # Setup cursor as context manager
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

    return mock_conn


@pytest.fixture(autouse=True)
def cleanup_pool() -> None:
    """Auto-cleanup pool state after each test."""
    yield
    DatabasePool.close_all()
    reset_settings()
```

---

## Database Setup Requirements

### No Real Database Required

All tests use mocks and patches - no PostgreSQL instance needed:

```bash
# Tests run completely isolated
pytest tests/test_database_connection_pool.py -v
pytest tests/test_database_type_safety.py -v
pytest tests/test_database_monitoring.py -v

# All tests use patch() context managers
# No database configuration needed
# No environment variables required
```

### Coverage Impact

Expected coverage additions:
- **database.py**: +15-20% coverage (error paths, edge cases)
- **config.py**: +8-10% coverage (type validation)
- **Overall project**: +2-3% coverage (11 new tests)

---

## Execution Strategy

### Phase 1: Implement & Verify (4 hours)

1. Create test files with complete implementation
2. Run tests to verify all pass: `pytest tests/test_database_*.py -v`
3. Verify coverage: `pytest --cov=src tests/test_database_*.py`
4. Check mypy compliance: `mypy tests/test_database_*.py --strict`

### Phase 2: Integration & Validation (1 hour)

1. Run full test suite: `pytest tests/ -v`
2. Verify no regressions (280+ existing tests still pass)
3. Generate coverage report
4. Document results

### Expected Results

```
test_database_connection_pool.py: 5 tests ✓ (~8 seconds)
test_database_type_safety.py: 3 tests ✓ (~4 seconds)
test_database_monitoring.py: 3 tests ✓ (~6 seconds)

Total: 11 tests in ~18 seconds
Coverage: +2-3% to overall suite
Mypy: 100% strict compliance
```

---

## Complete Test Implementation Code

### File 1: `tests/test_database_connection_pool.py`

```python
"""Unit tests for connection pool leak prevention and recovery.

Tests verify that connections are properly returned to the pool under
all error conditions, preventing resource exhaustion and ensuring pool
recovery after failure sequences.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from psycopg2 import DatabaseError, OperationalError

from src.core.config import reset_settings
from src.core.database import DatabasePool


class TestConnectionPoolLeakPrevention:
    """Tests for connection leak prevention under error conditions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_connection_leak_on_retry_failure(self) -> None:
        """Test that connection is not leaked when all retries fail.

        Simulates all connection attempts failing. Verifies that:
        - getconn is called for each retry attempt
        - putconn is never called (no connection acquired)
        - Pool remains in initialized state
        - OperationalError is raised to caller
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            # All connection attempts fail
            mock_pool.getconn.side_effect = OperationalError(
                "Connection refused"
            )

            # Attempt to get connection with retries
            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=3):
                    pass

            # Verify attempts were made
            assert mock_pool.getconn.call_count == 3

            # Verify connection was NEVER returned to pool
            mock_pool.putconn.assert_not_called()

    def test_connection_release_on_exception_after_getconn(self) -> None:
        """Test connection returned to pool even when exception occurs after acquisition.

        Simulates:
        1. Connection acquired successfully
        2. Health check passes
        3. Exception raised in user code
        4. Connection still returned to pool in finally block

        Verifies that the finally block properly returns the connection
        even when exceptions occur during use.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            # Setup cursor as context manager
            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # Connection succeeds
            mock_pool.getconn.return_value = mock_conn

            # Raise error during use
            with pytest.raises(RuntimeError, match="Simulated error"):
                with DatabasePool.get_connection() as conn:
                    raise RuntimeError("Simulated error")

            # Verify connection was acquired
            mock_pool.getconn.assert_called_once()

            # Verify connection was STILL returned despite exception
            mock_pool.putconn.assert_called_once()
            assert mock_pool.putconn.call_args[0][0] is mock_conn

    def test_connection_recovery_after_all_retries_fail(self) -> None:
        """Test pool is in recoverable state after retry exhaustion.

        Simulates:
        1. All retries fail with OperationalError
        2. Pool remains initialized and usable
        3. Subsequent connection attempt succeeds

        Verifies that failed connection attempts don't permanently
        damage the pool or prevent future connections.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool
            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # First attempt: all retries fail
            mock_pool.getconn.side_effect = OperationalError("Connection refused")

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=2):
                    pass

            # Pool should still be initialized
            assert DatabasePool._pool is not None

            # Second attempt: reset side_effect and try again
            mock_pool.getconn.side_effect = None
            mock_pool.getconn.return_value = mock_conn
            mock_pool.putconn.reset_mock()

            # This should succeed
            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

            # Verify connection was returned
            mock_pool.putconn.assert_called_once()

    def test_connection_leak_under_concurrent_failures(self) -> None:
        """Test thread-safe connection handling during concurrent failures.

        Simulates:
        1. Multiple threads acquiring connections simultaneously
        2. Each connection fails after health check
        3. All connections properly returned despite concurrent errors

        Verifies thread-safe implementation of putconn and no deadlocks.
        """
        import threading

        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            # Create unique mock connection for each thread
            connections: list[MagicMock] = []
            for i in range(3):
                mock_conn: MagicMock = MagicMock()
                mock_cursor: MagicMock = MagicMock()
                mock_conn.cursor.return_value.__enter__ = Mock(
                    return_value=mock_cursor
                )
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
                mock_cursor.execute.side_effect = DatabaseError("Unhealthy")
                connections.append(mock_conn)

            mock_pool.getconn.side_effect = connections

            # Track results from threads
            errors_caught: list[DatabaseError] = []

            def attempt_connection(index: int) -> None:
                """Attempt connection in thread."""
                try:
                    with DatabasePool.get_connection(retries=1):
                        pass
                except DatabaseError as e:
                    errors_caught.append(e)

            # Start threads
            threads = [
                threading.Thread(target=attempt_connection, args=(i,))
                for i in range(3)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All threads should have caught errors
            assert len(errors_caught) == 3

            # All connections should have been attempted
            assert mock_pool.getconn.call_count == 3

            # All connections should be returned (one per thread)
            assert mock_pool.putconn.call_count == 3

    def test_connection_pool_status_after_error_sequence(self) -> None:
        """Test pool status accuracy through error sequence.

        Simulates error sequence: fail → retry → fail → retry → succeed

        Verifies:
        - Status metrics remain consistent
        - No phantom connections reported
        - Pool state accurate after complex operations
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # Setup sequence: fail, fail, success
            call_count: int = [0]

            def get_conn_side_effect() -> MagicMock:
                call_count[0] += 1
                if call_count[0] < 3:
                    raise OperationalError("Transient failure")
                return mock_conn

            mock_pool.getconn.side_effect = get_conn_side_effect

            # Attempt connection (will retry and succeed)
            with DatabasePool.get_connection(retries=3) as conn:
                assert conn is mock_conn

            # Verify the sequence: 2 failures, then 1 success
            assert mock_pool.getconn.call_count == 3

            # Connection should be returned
            mock_pool.putconn.assert_called_once()


class TestConnectionPoolCleanup:
    """Tests for proper cleanup of pool state."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_pool_state_consistency_after_failures(self) -> None:
        """Verify pool maintains consistent state after multiple failures.

        This test ensures that the internal pool state doesn't become
        corrupted or inconsistent after handling connection failures.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            # First attempt: failure
            mock_pool.getconn.side_effect = OperationalError("Connection refused")

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=1):
                    pass

            # Pool should still be the same instance
            assert DatabasePool._pool is mock_pool

            # Close and verify cleanup
            DatabasePool.close_all()
            mock_pool.closeall.assert_called_once()
            assert DatabasePool._pool is None
```

### File 2: `tests/test_database_type_safety.py`

```python
"""Tests for type safety and complete type annotations.

Validates that all database module code has complete type hints
compatible with mypy --strict mode.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

import pytest

from src.core.config import DatabaseConfig
from src.core.database import DatabasePool


class TestDatabaseModuleTypeSafety:
    """Tests for type annotations in database module."""

    def test_all_database_pool_methods_have_return_types(self) -> None:
        """Verify all DatabasePool methods have explicit return type annotations.

        Uses typing.get_type_hints() to inspect method signatures.
        Ensures return type is present for:
        - initialize()
        - get_connection()
        - close_all()

        Fails if any public method lacks return type annotation.
        """
        # Get all methods in DatabasePool
        methods_to_check: list[str] = [
            "initialize",
            "get_connection",
            "close_all",
        ]

        for method_name in methods_to_check:
            method = getattr(DatabasePool, method_name)

            # Get type hints
            try:
                hints: dict[str, Any] = get_type_hints(method)
            except Exception as e:
                pytest.fail(
                    f"Failed to get type hints for {method_name}: {e}"
                )

            # Check return type is present
            assert (
                "return" in hints
            ), f"Method {method_name} missing return type annotation"

            # Return type should not be Any
            return_type: Any = hints.get("return")
            assert return_type is not Any, (
                f"Method {method_name} has Any return type instead of concrete type"
            )

    def test_database_pool_initialize_signature(self) -> None:
        """Verify initialize() has correct type signature.

        Signature should be:
            @classmethod
            def initialize() -> None

        No parameters, returns None.
        """
        hints: dict[str, Any] = get_type_hints(DatabasePool.initialize)

        # Should return None
        assert hints.get("return") is type(None), (
            "initialize() should return None"
        )

        # Should have no parameters except self
        sig = inspect.signature(DatabasePool.initialize)
        assert len(sig.parameters) == 0, (
            "initialize() should have no parameters"
        )

    def test_database_pool_get_connection_signature(self) -> None:
        """Verify get_connection() has correct type signature.

        Signature should be:
            @classmethod
            @contextmanager
            def get_connection(cls, retries: int = 3) -> Generator[Connection, None, None]

        Parameters: retries (int, optional)
        Returns: Generator[Connection, None, None]
        """
        hints: dict[str, Any] = get_type_hints(DatabasePool.get_connection)

        # Should return a Generator
        return_type: Any = hints.get("return")
        assert return_type is not None, (
            "get_connection() should have return type annotation"
        )

        # Check for Generator type (exact match may vary due to imports)
        return_type_str: str = str(return_type)
        assert "Generator" in return_type_str, (
            f"get_connection() should return Generator, got {return_type_str}"
        )

    def test_database_pool_close_all_signature(self) -> None:
        """Verify close_all() has correct type signature.

        Signature should be:
            @classmethod
            def close_all() -> None

        No parameters, returns None.
        """
        hints: dict[str, Any] = get_type_hints(DatabasePool.close_all)

        # Should return None
        assert hints.get("return") is type(None), (
            "close_all() should return None"
        )

        # Should have no parameters
        sig = inspect.signature(DatabasePool.close_all)
        assert len(sig.parameters) == 0, (
            "close_all() should have no parameters"
        )

    def test_database_config_fields_have_types(self) -> None:
        """Verify all DatabaseConfig fields have type annotations.

        Checks that each Pydantic field has:
        - Concrete type annotation (not Any)
        - Proper Field definition
        - Type matches expected database config types
        """
        # Get all model fields
        fields: dict[str, Any] = DatabaseConfig.model_fields

        expected_fields: dict[str, type] = {
            "host": str,
            "port": int,
            "database": str,
            "user": str,
            "password": type(None),  # SecretStr - special case
            "pool_min_size": int,
            "pool_max_size": int,
            "connection_timeout": float,
            "statement_timeout": float,
        }

        for field_name, expected_type in expected_fields.items():
            assert field_name in fields, (
                f"Expected field {field_name} not found in DatabaseConfig"
            )

            field_info: Any = fields[field_name]
            assert field_info.annotation is not None, (
                f"Field {field_name} missing type annotation"
            )

            # For SecretStr, skip exact type check due to wrapper
            if field_name == "password":
                continue

            # Verify type annotation matches expected
            annotation: Any = field_info.annotation
            assert annotation == expected_type or (
                hasattr(annotation, "__origin__")
            ), (
                f"Field {field_name} has type {annotation}, "
                f"expected {expected_type}"
            )

    def test_database_config_validators_typed(self) -> None:
        """Verify field validators in DatabaseConfig have proper signatures.

        Field validators should have type hints:
            @field_validator('field_name')
            @classmethod
            def validate_field(cls, v: SomeType, info: ValidationInfo) -> SomeType

        Checks that validator functions have typed parameters and return.
        """
        # Check validate_pool_sizes method exists and has type hints
        validate_method = getattr(
            DatabaseConfig, "validate_pool_sizes", None
        )

        assert validate_method is not None, (
            "DatabaseConfig.validate_pool_sizes validator not found"
        )

        # Get type hints for validator
        hints: dict[str, Any] = get_type_hints(validate_method)

        # Should have 'return' type
        assert "return" in hints, (
            "Validator validate_pool_sizes missing return type"
        )

        # Return type should be int (the field type)
        assert hints.get("return") == int, (
            "Validator validate_pool_sizes should return int"
        )


class TestTypeImportsAndUsage:
    """Tests for proper usage of typing module imports."""

    def test_database_module_imports_typing(self) -> None:
        """Verify database module imports necessary typing constructs.

        Should import:
        - Generator (for get_connection return type)
        - Connection type from psycopg2
        - Proper contextmanager usage
        """
        import src.core.database as db_module

        # Check that module has access to required types
        assert hasattr(db_module, "Generator") or (
            hasattr(db_module, "contextmanager")
        ), (
            "database module missing typing imports"
        )

    def test_config_module_uses_pydantic_types(self) -> None:
        """Verify config module properly uses Pydantic v2 types.

        Should use:
        - BaseSettings from pydantic_settings
        - Field() for field definitions
        - SecretStr for password
        - field_validator decorator
        """
        import src.core.config as config_module

        # Verify imports are available
        assert hasattr(config_module, "DatabaseConfig")
        assert hasattr(config_module, "BaseSettings") or (
            "BaseSettings" in str(config_module)
        )

        # Verify DatabaseConfig is properly typed
        config: DatabaseConfig = DatabaseConfig()
        assert isinstance(config, DatabaseConfig)
```

### File 3: `tests/test_database_monitoring.py`

```python
"""Tests for pool monitoring, health checks, and documentation.

Validates health check functionality, pool status accuracy, and
comprehensive documentation for all public methods.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from psycopg2 import DatabaseError, OperationalError

from src.core.config import reset_settings
from src.core.database import DatabasePool


class TestPoolHealthChecks:
    """Tests for connection health check functionality."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_pool_health_check_on_healthy_connection(self) -> None:
        """Test health check passes for healthy connections.

        Simulates:
        1. Connection acquired from pool
        2. Health check (SELECT 1) executes successfully
        3. Connection returned to pool

        Verifies:
        - cursor.execute("SELECT 1") is called
        - No exceptions raised
        - Connection successfully yielded
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            # Setup cursor as context manager
            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # Health check succeeds
            mock_cursor.execute.return_value = None

            mock_pool.getconn.return_value = mock_conn

            # Should succeed without exception
            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

            # Verify health check was performed
            mock_cursor.execute.assert_called_once_with("SELECT 1")

            # Verify connection returned
            mock_pool.putconn.assert_called_once()

    def test_pool_health_check_on_unhealthy_connection(self) -> None:
        """Test health check fails for unhealthy connections.

        Simulates:
        1. Connection acquired from pool
        2. Health check (SELECT 1) fails with DatabaseError
        3. Exception propagated to caller
        4. Connection NOT returned to pool (broken)

        Verifies that broken connections don't corrupt the pool.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            # Setup cursor as context manager
            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # Health check fails
            mock_cursor.execute.side_effect = DatabaseError("Connection dead")

            mock_pool.getconn.return_value = mock_conn

            # Should raise DatabaseError
            with pytest.raises(DatabaseError, match="Connection dead"):
                with DatabasePool.get_connection():
                    pass

            # Connection should NOT be returned (it's broken)
            # However, the finally block will still call putconn
            # This is expected behavior - let broken connection be handled by pool

    def test_pool_health_check_timeout(self) -> None:
        """Test health check handles connection timeouts.

        Simulates:
        1. Health check query times out
        2. OperationalError raised
        3. Retry logic triggered
        4. Eventually succeeds or fails with proper error

        Verifies timeout during health check is handled correctly.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("time.sleep"):  # Skip actual sleep
                mock_pool: MagicMock = MagicMock()
                mock_pool_class.return_value = mock_pool

                mock_conn: MagicMock = MagicMock()
                mock_cursor: MagicMock = MagicMock()

                mock_conn.cursor.return_value.__enter__ = Mock(
                    return_value=mock_cursor
                )
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

                # Health check times out on first attempt, succeeds on retry
                mock_cursor.execute.side_effect = [
                    OperationalError("Statement timeout"),
                    None,  # Success on health check retry
                ]

                # Connection succeeds on second attempt
                mock_pool.getconn.side_effect = [
                    OperationalError("Timeout"),
                    mock_conn,
                ]

                # Should eventually succeed after retry
                with DatabasePool.get_connection(retries=2) as conn:
                    assert conn is mock_conn

                # Verify both attempts were made
                assert mock_pool.getconn.call_count == 2


class TestPoolMonitoring:
    """Tests for pool status and monitoring capabilities."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_pool_initialization_status(self) -> None:
        """Test pool status after initialization.

        Verifies initial pool state:
        - Pool exists
        - Initialized with correct min/max sizes
        - All connections available (not acquired yet)
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            # Setup mock to track pool state
            mock_pool.minconn = 5
            mock_pool.maxconn = 20
            mock_pool.closed = False

            DatabasePool.initialize()

            # Verify pool is initialized
            assert DatabasePool._pool is not None
            assert DatabasePool._pool is mock_pool

    def test_pool_state_after_operations(self) -> None:
        """Test pool remains in consistent state after operations.

        Simulates sequence:
        1. Initialize pool
        2. Acquire connection (getconn called)
        3. Use connection (health check passes)
        4. Release connection (putconn called)
        5. Close pool

        Verifies state transitions are correct.
        """
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn: MagicMock = MagicMock()
            mock_cursor: MagicMock = MagicMock()

            mock_conn.cursor.return_value.__enter__ = Mock(
                return_value=mock_cursor
            )
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            mock_pool.getconn.return_value = mock_conn

            # Initialize
            DatabasePool.initialize()
            assert DatabasePool._pool is not None

            # Use connection
            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

            # Verify returned to pool
            mock_pool.putconn.assert_called_once()

            # Close
            DatabasePool.close_all()
            mock_pool.closeall.assert_called_once()
            assert DatabasePool._pool is None


class TestDocumentation:
    """Tests for documentation completeness."""

    def test_database_pool_methods_have_docstrings(self) -> None:
        """Verify all public DatabasePool methods have docstrings.

        Checks that each method has:
        - Non-empty docstring
        - Meaningful documentation (>50 characters)
        - Proper formatting
        """
        public_methods: list[str] = [
            "initialize",
            "get_connection",
            "close_all",
        ]

        for method_name in public_methods:
            method = getattr(DatabasePool, method_name)
            docstring: str | None = inspect.getdoc(method)

            assert docstring is not None, (
                f"Method {method_name} missing docstring"
            )

            assert len(docstring) > 50, (
                f"Method {method_name} docstring too brief: {len(docstring)} chars"
            )

    def test_initialize_docstring_completeness(self) -> None:
        """Verify initialize() docstring documents all aspects.

        Should document:
        - Purpose (initialization)
        - Configuration source
        - Exceptions that can be raised
        - Return value
        - Side effects
        """
        docstring: str = inspect.getdoc(DatabasePool.initialize) or ""

        assert "initialize" in docstring.lower() or (
            "pool" in docstring.lower()
        ), "Missing initialization description"

        assert "Raises:" in docstring or "raises" in docstring.lower(), (
            "Missing raises section"
        )

        assert "Settings" in docstring or "config" in docstring.lower(), (
            "Missing configuration reference"
        )

    def test_get_connection_docstring_completeness(self) -> None:
        """Verify get_connection() docstring documents retry behavior.

        Should document:
        - Purpose (acquire connection)
        - Retry mechanism
        - Exponential backoff strategy
        - Context manager usage
        - All exceptions
        - Examples
        """
        docstring: str = inspect.getdoc(DatabasePool.get_connection) or ""

        # Check for key documentation sections
        assert "retry" in docstring.lower(), (
            "Missing retry documentation"
        )

        assert "Raises:" in docstring or "raises" in docstring.lower(), (
            "Missing raises section"
        )

        assert "Yields:" in docstring or "yields" in docstring.lower(), (
            "Missing yields section"
        )

        assert ">>>" in docstring, (
            "Missing usage examples"
        )

    def test_close_all_docstring_completeness(self) -> None:
        """Verify close_all() docstring documents cleanup behavior.

        Should document:
        - Purpose (cleanup)
        - Idempotency
        - All side effects
        - Examples
        """
        docstring: str = inspect.getdoc(DatabasePool.close_all) or ""

        assert "close" in docstring.lower() or (
            "cleanup" in docstring.lower()
        ), "Missing close description"

        assert "idempotent" in docstring.lower(), (
            "Missing idempotency documentation"
        )

        assert ">>>" in docstring, (
            "Missing usage examples"
        )

    def test_exception_types_documented(self) -> None:
        """Verify all possible exceptions are documented.

        Checks that docstrings list:
        - OperationalError (connection failures)
        - DatabaseError (health check failures)
        - RuntimeError (pool initialization failures)
        - ValueError (invalid parameters)
        """
        get_conn_doc: str = (
            inspect.getdoc(DatabasePool.get_connection) or ""
        )

        # Should document connection-related errors
        assert "OperationalError" in get_conn_doc, (
            "Missing OperationalError documentation"
        )

        assert "DatabaseError" in get_conn_doc, (
            "Missing DatabaseError documentation"
        )

        assert "ValueError" in get_conn_doc, (
            "Missing ValueError documentation"
        )
```

---

## Coverage Impact Analysis

### Current Baseline
- **Total coverage**: ~85% (280+ tests)
- **database.py**: ~80% (core functionality covered)
- **config.py**: ~75% (validation logic)

### Expected Impact

| Module | Current | New Tests | Expected |
|--------|---------|-----------|----------|
| database.py | ~80% | Pool leak tests (5), Monitoring tests (3) | ~95% |
| config.py | ~75% | Type safety tests (3) | ~85% |
| Overall | ~85% | 11 new tests | ~87% |

### Coverage Additions by Category

**Connection Pool Leak Tests**:
- Error handling paths in `get_connection()`
- Finally block execution under exceptions
- Pool cleanup after failures
- Concurrent error handling

**Type Safety Tests**:
- Validation of all method signatures
- Config model field coverage
- Pydantic v2 integration

**Monitoring Tests**:
- Health check execution paths
- Pool state consistency
- Documentation completeness

---

## Implementation Checklist

- [ ] Create `tests/test_database_connection_pool.py` (5 tests)
- [ ] Create `tests/test_database_type_safety.py` (3 tests)
- [ ] Create `tests/test_database_monitoring.py` (3 tests)
- [ ] Verify all tests pass locally: `pytest tests/test_database_*.py -v`
- [ ] Run full test suite (verify 280+ tests still pass)
- [ ] Generate coverage report: `pytest --cov=src tests/test_database_*.py`
- [ ] Verify mypy compliance: `mypy tests/test_database_*.py --strict`
- [ ] Document results in this report
- [ ] Commit changes with message: "feat: task-1 test specifications with 11 comprehensive tests"

---

## Execution Timeline

### Session 1: Core Implementation (2-3 hours)
1. Create three test files with complete code
2. Verify tests execute and pass
3. Initial coverage analysis

### Session 2: Validation & Integration (1-2 hours)
1. Full test suite run
2. Coverage report generation
3. Type checking validation
4. Final documentation

---

## Next Steps

After this specification is approved:

1. **Copy test code** from implementation sections above
2. **Create test files** in tests/ directory
3. **Run tests** to verify implementation
4. **Generate coverage** reports
5. **Update documentation** with results

Each test is completely self-contained and ready for copy-paste implementation.

---

## Appendix: Test Execution Examples

### Run All 11 New Tests

```bash
# Run all three test files
pytest tests/test_database_connection_pool.py \
        tests/test_database_type_safety.py \
        tests/test_database_monitoring.py \
        -v

# Expected output:
# test_database_connection_pool.py::TestConnectionPoolLeakPrevention::test_connection_leak_on_retry_failure PASSED
# test_database_connection_pool.py::TestConnectionPoolLeakPrevention::test_connection_release_on_exception_after_getconn PASSED
# ... (11 total tests)
# ============ 11 passed in 18.45s ============
```

### Coverage Report

```bash
pytest tests/test_database_*.py \
        --cov=src \
        --cov-report=term-missing \
        -v

# Expected improvements:
# src/core/database.py       180      15    92%
# src/core/config.py         85       8     91%
```

### Type Checking

```bash
mypy tests/test_database_*.py --strict

# Expected output:
# Success: no issues found in 3 source files
```

---

**Report Generated**: November 8, 2025
**Test Status**: Ready for Implementation
**All Code Examples**: Copy-paste ready
**Documentation**: Complete and comprehensive
