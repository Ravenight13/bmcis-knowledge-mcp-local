"""Unit tests for database connection pooling.

Tests connection pool initialization, health checks, retry logic,
and proper resource cleanup. Uses test fixtures and mocking to avoid
requiring actual database connections.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from psycopg2 import DatabaseError, OperationalError

from src.core.config import reset_settings
from src.core.database import DatabasePool


class TestDatabasePoolInitialization:
    """Tests for DatabasePool.initialize() method."""

    def setup_method(self) -> None:
        """Reset database pool before each test."""
        DatabasePool._pool = None
        reset_settings()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_initialize_creates_pool(self) -> None:
        """Test that initialize() creates a SimpleConnectionPool."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool:
            DatabasePool.initialize()

            # Verify pool was created with correct parameters
            mock_pool.assert_called_once()
            call_kwargs = mock_pool.call_args.kwargs

            assert call_kwargs["minconn"] == 5  # Default from config
            assert call_kwargs["maxconn"] == 20  # Default from config
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 5432
            assert call_kwargs["database"] == "bmcis_knowledge_dev"

    def test_initialize_idempotent(self) -> None:
        """Test that calling initialize() twice doesn't create two pools."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool:
            DatabasePool.initialize()
            DatabasePool.initialize()  # Second call should be skipped

            # Pool should only be created once
            mock_pool.assert_called_once()

    def test_initialize_handles_database_error(self) -> None:
        """Test that initialize() raises RuntimeError on DatabaseError."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool:
            mock_pool.side_effect = DatabaseError("Connection failed")

            with pytest.raises(RuntimeError, match="Failed to initialize connection pool"):
                DatabasePool.initialize()

    def test_initialize_statement_timeout_conversion(self) -> None:
        """Test that statement_timeout is converted to milliseconds."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool:
            DatabasePool.initialize()

            call_kwargs = mock_pool.call_args.kwargs
            # Default statement_timeout is 30 seconds = 30000 milliseconds
            assert "statement_timeout=30000" in call_kwargs["options"]


class TestDatabasePoolConnection:
    """Tests for DatabasePool.get_connection() method."""

    def setup_method(self) -> None:
        """Reset database pool before each test."""
        DatabasePool._pool = None
        reset_settings()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_get_connection_initializes_pool(self) -> None:
        """Test that get_connection initializes pool if needed."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_pool.getconn.return_value = mock_conn

            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

            # Verify health check was performed
            mock_cursor.execute.assert_called_once_with("SELECT 1")

    def test_get_connection_returns_valid_connection(self) -> None:
        """Test that get_connection returns a working connection."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_pool.getconn.return_value = mock_conn

            with DatabasePool.get_connection() as conn:
                assert conn is mock_conn

            # Verify connection was returned to pool
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_retries_on_operational_error(self) -> None:
        """Test that get_connection retries on OperationalError."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("time.sleep") as mock_sleep:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

                # First call fails, second succeeds
                mock_pool.getconn.side_effect = [
                    OperationalError("Connection refused"),
                    mock_conn,
                ]

                with DatabasePool.get_connection(retries=3) as conn:
                    assert conn is mock_conn

                # Verify retry with exponential backoff
                mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second

    def test_get_connection_exhausts_retries(self) -> None:
        """Test that get_connection raises after exhausting retries."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("time.sleep") as mock_sleep:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                # All attempts fail
                mock_pool.getconn.side_effect = OperationalError("Connection refused")

                with pytest.raises(OperationalError):
                    with DatabasePool.get_connection(retries=3):
                        pass

                # Verify exponential backoff attempts
                assert mock_sleep.call_count == 2  # 2^0 and 2^1

    def test_get_connection_health_check_fails(self) -> None:
        """Test that get_connection raises on health check failure."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = DatabaseError("Health check failed")
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_pool.getconn.return_value = mock_conn

            with pytest.raises(DatabaseError):
                with DatabasePool.get_connection():
                    pass

    def test_get_connection_cleans_up_on_exception(self) -> None:
        """Test that connection is returned to pool even on exception."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_pool.getconn.return_value = mock_conn

            with pytest.raises(RuntimeError):
                with DatabasePool.get_connection() as conn:
                    raise RuntimeError("Simulated error")

            # Connection should still be returned to pool
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_invalid_retries(self) -> None:
        """Test that get_connection raises ValueError for invalid retry count."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool_class.return_value = MagicMock()

            with pytest.raises(ValueError, match="retries must be >= 1"):
                with DatabasePool.get_connection(retries=0):
                    pass


class TestDatabasePoolCleanup:
    """Tests for DatabasePool.close_all() method."""

    def setup_method(self) -> None:
        """Reset database pool before each test."""
        DatabasePool._pool = None
        reset_settings()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_close_all_closes_pool(self) -> None:
        """Test that close_all() closes the connection pool."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            DatabasePool.initialize()
            DatabasePool.close_all()

            # Verify pool was closed
            mock_pool.closeall.assert_called_once()
            assert DatabasePool._pool is None

    def test_close_all_idempotent(self) -> None:
        """Test that calling close_all() multiple times is safe."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            DatabasePool.initialize()
            DatabasePool.close_all()
            DatabasePool.close_all()  # Second call should not raise

            # Pool should only be closed once
            mock_pool.closeall.assert_called_once()

    def test_close_all_when_pool_not_initialized(self) -> None:
        """Test that close_all() handles uninitialized pool gracefully."""
        # Should not raise any exception
        DatabasePool.close_all()
        assert DatabasePool._pool is None


class TestDatabasePoolExponentialBackoff:
    """Tests for exponential backoff retry strategy."""

    def setup_method(self) -> None:
        """Reset database pool before each test."""
        DatabasePool._pool = None
        reset_settings()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DatabasePool.close_all()
        reset_settings()

    def test_exponential_backoff_progression(self) -> None:
        """Test that backoff times follow 2^n pattern."""
        with patch("src.core.database.pool.SimpleConnectionPool") as mock_pool_class:
            with patch("time.sleep") as mock_sleep:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                # All attempts fail to trigger retries
                mock_pool.getconn.side_effect = OperationalError("Connection refused")

                with pytest.raises(OperationalError):
                    with DatabasePool.get_connection(retries=4):
                        pass

                # Verify backoff sequence: 2^0=1, 2^1=2, 2^2=4
                expected_sleeps = [call(1), call(2), call(4)]
                mock_sleep.assert_has_calls(expected_sleeps)
