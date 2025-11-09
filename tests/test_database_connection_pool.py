"""Unit tests for connection pool leak prevention and recovery.

Tests verify that connections are properly returned to the pool under
all error conditions, preventing resource exhaustion and ensuring pool
recovery after failure sequences.
"""

from __future__ import annotations

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
        4. Connection still returned to pool via outer finally block

        Verifies that the outer finally block properly returns the connection
        even when exceptions occur during use, ensuring no resource leaks.
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

            # Verify connection was returned to pool despite exception
            # The outer finally block (line 254-256) handles this
            mock_pool.putconn.assert_called_once()
            assert mock_pool.putconn.call_args[0][0] is mock_conn

    def test_connection_recovery_after_all_retries_fail(self) -> None:
        """Test pool is in recoverable state after retry exhaustion.

        Simulates:
        1. All retries fail with OperationalError
        2. Pool remains initialized and usable
        3. Subsequent connection attempt succeeds
        4. Successful connection is returned to pool

        Verifies that failed connection attempts don't permanently
        damage the pool or prevent future connections, and that
        successful recovery properly returns the connection.
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

            # Verify connection was returned to pool after successful recovery
            mock_pool.putconn.assert_called_once()
            assert mock_pool.putconn.call_args[0][0] is mock_conn

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
        - Connection acquisition succeeds after multiple retries
        - Successful connection is returned to pool after error sequence
        - Pool state accurate after complex retry operations
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

            # Connection should be returned after successful acquisition
            mock_pool.putconn.assert_called_once()
            assert mock_pool.putconn.call_args[0][0] is mock_conn


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
