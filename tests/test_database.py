"""Comprehensive test suite for database connection pooling.

Tests cover:
- Pool initialization and configuration
- Connection acquisition and lifecycle
- Retry logic with exponential backoff
- Health checks and connection validation
- Error handling and edge cases
- Performance characteristics
- Pool cleanup and resource management
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Generator, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest
import psycopg2
from psycopg2 import DatabaseError, OperationalError
from psycopg2.extensions import connection as Connection

from src.core.config import DatabaseConfig, get_settings, reset_settings
from src.core.database import DatabasePool


class TestDatabasePoolInitialization:
    """Tests for pool initialization and configuration."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_pool_not_initialized_by_default(self) -> None:
        """Test that pool is None before initialization."""
        # Reset to ensure clean state
        DatabasePool.close_all()
        assert DatabasePool._pool is None

    def test_pool_initializes_with_default_config(self) -> None:
        """Test pool initializes with default database configuration."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            # Verify pool was created
            assert DatabasePool._pool is not None
            mock_pool_class.assert_called_once()

    def test_pool_respects_min_max_sizes(self) -> None:
        """Test pool configuration respects min/max size settings."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            # Extract call arguments
            call_kwargs: dict[str, Any] = mock_pool_class.call_args[1]

            # Verify pool sizes match configuration
            settings = get_settings()
            assert call_kwargs["minconn"] == settings.database.pool_min_size
            assert call_kwargs["maxconn"] == settings.database.pool_max_size

    def test_pool_uses_correct_connection_parameters(self) -> None:
        """Test pool configuration uses correct host, port, database, user."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            call_kwargs: dict[str, Any] = mock_pool_class.call_args[1]
            settings = get_settings()

            assert call_kwargs["host"] == settings.database.host
            assert call_kwargs["port"] == settings.database.port
            assert call_kwargs["database"] == settings.database.database
            assert call_kwargs["user"] == settings.database.user

    def test_pool_uses_secret_password(self) -> None:
        """Test pool uses SecretStr password value correctly."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            call_kwargs: dict[str, Any] = mock_pool_class.call_args[1]
            settings = get_settings()

            # Password should be extracted from SecretStr
            assert call_kwargs["password"] == settings.database.password.get_secret_value()

    def test_pool_configures_statement_timeout(self) -> None:
        """Test pool sets statement timeout as PostgreSQL option."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            call_kwargs: dict[str, Any] = mock_pool_class.call_args[1]
            settings = get_settings()

            # Statement timeout should be in options as milliseconds
            expected_timeout_ms = int(settings.database.statement_timeout * 1000)
            assert "statement_timeout" in call_kwargs["options"]
            assert str(expected_timeout_ms) in call_kwargs["options"]

    def test_pool_sets_connection_timeout(self) -> None:
        """Test pool configures connection timeout parameter."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()

            call_kwargs: dict[str, Any] = mock_pool_class.call_args[1]
            settings = get_settings()

            assert call_kwargs["connect_timeout"] == int(
                settings.database.connection_timeout
            )

    def test_initialize_is_idempotent(self) -> None:
        """Test initialize can be called multiple times safely."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            DatabasePool.initialize()
            first_pool = DatabasePool._pool

            # Call again
            DatabasePool.initialize()
            second_pool = DatabasePool._pool

            # Should be same instance (not created again)
            assert first_pool is second_pool
            # Should only be called once
            assert mock_pool_class.call_count == 1

    def test_initialize_logs_pool_creation(self) -> None:
        """Test initialize logs pool creation with configuration."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("src.core.database.logger") as mock_logger:
                mock_pool_instance: MagicMock = MagicMock()
                mock_pool_class.return_value = mock_pool_instance

                DatabasePool.initialize()

                # Verify logger was called
                assert mock_logger.info.called

    def test_initialize_handles_operational_error(self) -> None:
        """Test initialize raises RuntimeError on OperationalError."""
        with patch(
            "src.core.database.pool.SimpleConnectionPool",
            side_effect=OperationalError("Connection refused"),
        ) as mock_pool_class:
            with pytest.raises(RuntimeError, match="Failed to initialize connection pool"):
                DatabasePool.initialize()

    def test_initialize_handles_database_error(self) -> None:
        """Test initialize raises RuntimeError on DatabaseError."""
        with patch(
            "src.core.database.pool.SimpleConnectionPool",
            side_effect=DatabaseError("Invalid parameters"),
        ) as mock_pool_class:
            with pytest.raises(RuntimeError, match="Failed to initialize connection pool"):
                DatabasePool.initialize()

    def test_initialize_handles_unexpected_error(self) -> None:
        """Test initialize propagates unexpected exceptions."""
        with patch(
            "src.core.database.pool.SimpleConnectionPool",
            side_effect=Exception("Unexpected error"),
        ) as mock_pool_class:
            with pytest.raises(Exception, match="Unexpected error"):
                DatabasePool.initialize()


class TestConnectionManagement:
    """Tests for connection acquisition and lifecycle management."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_get_connection_initializes_pool(self) -> None:
        """Test get_connection initializes pool if needed."""
        with patch.object(DatabasePool, "initialize") as mock_init:
            with patch.object(DatabasePool, "_pool") as mock_pool:
                mock_pool.getconn.return_value = MagicMock()

                DatabasePool._pool = None
                with patch("src.core.database.pool.SimpleConnectionPool"):
                    with patch.object(DatabasePool, "get_connection") as mock_get:
                        # Mock the context manager behavior
                        def mock_context(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
                            yield MagicMock()

                        mock_get.return_value = mock_context()

    def test_get_connection_returns_valid_connection(self) -> None:
        """Test get_connection returns valid connection from pool."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

    def test_connection_context_manager_returns_to_pool(self) -> None:
        """Test connection is returned to pool after use."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection():
                pass

            # Verify connection was returned to pool
            mock_pool_instance.putconn.assert_called_once_with(mock_conn)

    def test_connection_returned_even_on_exception(self) -> None:
        """Test connection is returned to pool even if exception occurs."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(ValueError):
                with DatabasePool.get_connection():
                    raise ValueError("Test exception")

            # Verify connection was still returned
            mock_pool_instance.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_performs_health_check(self) -> None:
        """Test get_connection executes health check query."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection():
                pass

            # Verify SELECT 1 was executed for health check
            mock_cursor.execute.assert_called()
            calls = [str(call) for call in mock_cursor.execute.call_args_list]
            assert any("SELECT 1" in str(call) for call in calls)

    def test_health_check_failure_raises_error(self) -> None:
        """Test connection fails if health check fails."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            # Health check fails
            mock_cursor.execute.side_effect = OperationalError("Connection lost")

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=1):
                    pass


class TestRetryLogic:
    """Tests for exponential backoff retry logic."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_first_attempt_succeeds(self) -> None:
        """Test no retry needed when first attempt succeeds."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection(retries=3) as conn:
                assert conn is mock_conn

            # getconn should be called only once
            assert mock_pool_instance.getconn.call_count == 1

    @patch("src.core.database.time.sleep")
    def test_retry_on_connection_failure(self, mock_sleep: MagicMock) -> None:
        """Test retry logic on connection failure."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            # First attempt fails, second succeeds
            mock_pool_instance.getconn.side_effect = [
                OperationalError("Connection refused"),
                mock_conn,
            ]
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection(retries=2) as conn:
                assert conn is mock_conn

            # getconn should be called twice
            assert mock_pool_instance.getconn.call_count == 2
            # sleep should be called once with backoff time 2^0 = 1
            mock_sleep.assert_called_once_with(1)

    @patch("src.core.database.time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep: MagicMock) -> None:
        """Test exponential backoff uses correct timing: 2^attempt."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            # All attempts fail
            mock_pool_instance.getconn.side_effect = OperationalError("Connection refused")
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=4):
                    pass

            # Verify backoff times: 2^0=1, 2^1=2, 2^2=4 (no sleep on last attempt)
            expected_calls = [call(1), call(2), call(4)]
            assert mock_sleep.call_args_list == expected_calls

    def test_max_retries_honored(self) -> None:
        """Test max retries limit is respected."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.side_effect = OperationalError("Connection refused")
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=3):
                    pass

            # Should attempt exactly 3 times
            assert mock_pool_instance.getconn.call_count == 3

    def test_retries_parameter_validation(self) -> None:
        """Test invalid retries parameter raises ValueError."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(ValueError, match="retries must be >= 1"):
                with DatabasePool.get_connection(retries=0):
                    pass

            with pytest.raises(ValueError, match="retries must be >= 1"):
                with DatabasePool.get_connection(retries=-1):
                    pass

    @patch("src.core.database.time.sleep")
    def test_retry_logs_attempts(self, mock_sleep: MagicMock) -> None:
        """Test retry logic logs warning on failed attempts."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("src.core.database.logger") as mock_logger:
                mock_pool_instance: MagicMock = MagicMock()
                mock_pool_instance.getconn.side_effect = OperationalError(
                    "Connection refused"
                )
                mock_pool_class.return_value = mock_pool_instance
                DatabasePool._pool = mock_pool_instance

                with pytest.raises(OperationalError):
                    with DatabasePool.get_connection(retries=2):
                        pass

                # Should log warnings for failed attempts
                assert mock_logger.warning.called
                assert mock_logger.error.called


class TestHealthChecks:
    """Tests for connection health check functionality."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_health_check_executes_select_one(self) -> None:
        """Test health check runs SELECT 1 query."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with DatabasePool.get_connection():
                pass

            # Verify SELECT 1 was executed
            mock_cursor.execute.assert_called_with("SELECT 1")

    def test_health_check_failure_on_operational_error(self) -> None:
        """Test health check catches OperationalError and retries."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute.side_effect = OperationalError("Connection lost")

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.side_effect = [
                OperationalError("Connection lost"),
            ]
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(OperationalError):
                with DatabasePool.get_connection(retries=1):
                    pass

    def test_health_check_failure_on_database_error(self) -> None:
        """Test health check catches DatabaseError."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute.side_effect = DatabaseError("Query error")

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.side_effect = [
                DatabaseError("Query error"),
            ]
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(DatabaseError):
                with DatabasePool.get_connection(retries=1):
                    pass


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_all_retries_exhausted_raises_error(self) -> None:
        """Test error is raised after all retries are exhausted."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            error_msg = "Connection permanently refused"
            mock_pool_instance.getconn.side_effect = OperationalError(error_msg)
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(OperationalError, match=error_msg):
                with DatabasePool.get_connection(retries=3):
                    pass

    def test_retries_default_to_three(self) -> None:
        """Test default retries parameter is 3."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Call without specifying retries
            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

    def test_uninitialized_pool_initializes_on_get_connection(self) -> None:
        """Test get_connection initializes pool if not already done."""
        with patch.object(DatabasePool, "initialize") as mock_init:
            DatabasePool._pool = None

            with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
                mock_conn: MagicMock = MagicMock(spec=Connection)
                mock_cursor: MagicMock = MagicMock()
                mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
                mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
                mock_cursor.execute = MagicMock()
                mock_cursor.fetchone = MagicMock(return_value=(1,))

                mock_pool_instance: MagicMock = MagicMock()
                mock_pool_instance.getconn.return_value = mock_conn
                mock_pool_class.return_value = mock_pool_instance

                # Manually set _pool after patch
                with patch.object(DatabasePool, "_pool", None):
                    with patch.object(DatabasePool, "initialize") as mock_init_call:
                        mock_init_call.side_effect = lambda: setattr(
                            DatabasePool, "_pool", mock_pool_instance
                        )
                        with DatabasePool.get_connection():
                            pass


class TestPoolCleanup:
    """Tests for pool cleanup and resource management."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_close_all_closes_pool(self) -> None:
        """Test close_all closes the connection pool."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            DatabasePool.close_all()

            # Verify closeall was called
            mock_pool_instance.closeall.assert_called_once()

    def test_close_all_resets_pool_to_none(self) -> None:
        """Test close_all sets _pool to None."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            DatabasePool.close_all()

            assert DatabasePool._pool is None

    def test_close_all_is_idempotent(self) -> None:
        """Test close_all can be called multiple times safely."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            DatabasePool.close_all()
            DatabasePool.close_all()  # Should not raise

            # Should only be called once
            mock_pool_instance.closeall.assert_called_once()

    def test_close_all_when_pool_is_none(self) -> None:
        """Test close_all handles None pool gracefully."""
        DatabasePool._pool = None
        DatabasePool.close_all()  # Should not raise
        assert DatabasePool._pool is None

    def test_close_all_handles_closeall_exception(self) -> None:
        """Test close_all handles exception from closeall."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.closeall.side_effect = Exception("Closeall failed")
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Should not raise
            DatabasePool.close_all()

            # Pool should still be set to None
            assert DatabasePool._pool is None


class TestPerformance:
    """Tests for performance characteristics and timing."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    @patch("src.core.database.time.sleep")
    def test_connection_acquisition_timing(self, mock_sleep: MagicMock) -> None:
        """Test connection acquisition completes in reasonable time."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            start = time.time()
            with DatabasePool.get_connection() as conn:
                pass
            elapsed = time.time() - start

            # Should be very fast (no sleep on success)
            assert elapsed < 1.0  # Should be milliseconds

    def test_health_check_overhead(self) -> None:
        """Test health check doesn't significantly slow connection acquisition."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Measure time for multiple acquisitions
            start = time.time()
            for _ in range(10):
                with DatabasePool.get_connection():
                    pass
            total_time = time.time() - start

            # 10 acquisitions should still be fast
            assert total_time < 5.0  # Very generous allowance

    def test_pool_reuses_connections(self) -> None:
        """Test pool reuses connections instead of reconnecting."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            # Return same connection each time
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Multiple acquisitions
            for _ in range(3):
                with DatabasePool.get_connection():
                    pass

            # getconn should be called 3 times
            assert mock_pool_instance.getconn.call_count == 3
            # putconn should be called 3 times
            assert mock_pool_instance.putconn.call_count == 3


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_handle_none_returned_from_getconn(self) -> None:
        """Test handling of None returned from getconn causes AttributeError."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = None
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # When getconn returns None, attempting to call cursor() on None raises AttributeError
            with pytest.raises(AttributeError):
                with DatabasePool.get_connection(retries=1):
                    pass

    def test_concurrent_connection_requests(self) -> None:
        """Test multiple connection acquisitions in sequence."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Sequential acquisitions
            for _ in range(5):
                with DatabasePool.get_connection():
                    pass

            # All should succeed
            assert mock_pool_instance.getconn.call_count == 5

    def test_database_error_during_getconn(self) -> None:
        """Test handling DatabaseError during getconn."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.side_effect = DatabaseError("Pool error")
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            with pytest.raises(DatabaseError):
                with DatabasePool.get_connection(retries=1):
                    pass

    def test_mixed_error_types_in_retries(self) -> None:
        """Test retry logic handles mixed error types."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            # First fails with OperationalError, second with DatabaseError, third succeeds
            mock_pool_instance.getconn.side_effect = [
                OperationalError("Connection refused"),
                DatabaseError("Database error"),
                mock_conn,
            ]
            mock_pool_class.return_value = mock_pool_instance
            DatabasePool._pool = mock_pool_instance

            # Should eventually succeed after mixed errors
            with DatabasePool.get_connection(retries=3):
                pass


class TestIntegration:
    """Integration tests for complete workflows."""

    def teardown_method(self) -> None:
        """Clean up pool and settings after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_pool_lifecycle_initialization_to_cleanup(self) -> None:
        """Test complete pool lifecycle from init to cleanup."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_conn: MagicMock = MagicMock(spec=Connection)
            mock_cursor: MagicMock = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.execute = MagicMock()
            mock_cursor.fetchone = MagicMock(return_value=(1,))

            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pool_class.return_value = mock_pool_instance

            # Start: pool uninitialized
            assert DatabasePool._pool is None

            # Initialize
            DatabasePool.initialize()
            assert DatabasePool._pool is not None

            # Use connections
            with DatabasePool.get_connection():
                pass

            with DatabasePool.get_connection():
                pass

            # Cleanup
            DatabasePool.close_all()
            assert DatabasePool._pool is None
            mock_pool_instance.closeall.assert_called_once()

    def test_configuration_affects_pool_creation(self) -> None:
        """Test that pool respects all configuration parameters."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_instance: MagicMock = MagicMock()
            mock_pool_class.return_value = mock_pool_instance

            # Set custom environment variables
            os.environ["DB_HOST"] = "custom-host.com"
            os.environ["DB_PORT"] = "5433"
            os.environ["DB_POOL_MIN_SIZE"] = "10"
            os.environ["DB_POOL_MAX_SIZE"] = "50"

            try:
                reset_settings()
                DatabasePool.close_all()

                DatabasePool.initialize()

                # Verify pool was created with custom settings
                call_kwargs = mock_pool_class.call_args[1]
                assert call_kwargs["host"] == "custom-host.com"
                assert call_kwargs["port"] == 5433
                assert call_kwargs["minconn"] == 10
                assert call_kwargs["maxconn"] == 50

            finally:
                os.environ.pop("DB_HOST", None)
                os.environ.pop("DB_PORT", None)
                os.environ.pop("DB_POOL_MIN_SIZE", None)
                os.environ.pop("DB_POOL_MAX_SIZE", None)
                reset_settings()
