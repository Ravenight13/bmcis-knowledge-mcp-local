"""Type stubs for database connection pooling module.

Defines the complete type interface for connection pool management,
including connection lifecycle, retry logic, and health checks.
"""

from typing import Generator, Optional, Any
from contextlib import contextmanager
from psycopg2.extensions import connection as Connection
import logging

logger: logging.Logger

class DatabasePool:
    """Connection pool management for PostgreSQL databases.

    Manages a SimpleConnectionPool with automatic initialization,
    connection health checks, and exponential backoff retry logic.
    All connections are validated before use and properly cleaned up.
    """

    _pool: Optional[Any]

    @classmethod
    def initialize(cls) -> None:
        """Initialize the connection pool from application settings.

        Reads pool configuration from DatabaseConfig (min/max size, timeouts)
        and creates a SimpleConnectionPool with statement timeout settings.
        Connection pool is created lazily on first use if not already initialized.

        Raises:
            RuntimeError: If pool initialization fails (invalid config, DB unreachable).
            DatabaseError: If PostgreSQL connection parameters are invalid.

        Returns:
            None
        """
        ...

    @classmethod
    @contextmanager
    def get_connection(
        cls,
        retries: int = 3
    ) -> Generator[Connection, None, None]:
        """Acquire a database connection from the pool with retry logic.

        Automatically initializes the pool if needed. Performs exponential backoff
        retry on connection failures (up to specified retries). Validates each
        connection with a SELECT 1 health check before returning. Ensures proper
        cleanup by returning connection to pool after use.

        Args:
            retries: Maximum number of connection attempts (default: 3).
                    Each retry uses exponential backoff: 2^attempt seconds.

        Yields:
            Connection: Valid psycopg2 connection object from the pool.

        Raises:
            OperationalError: If all connection attempts fail (after retries exhausted).
            DatabaseError: If health check fails on acquired connection.
            RuntimeError: If pool initialization fails.

        Examples:
            >>> with DatabasePool.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("SELECT * FROM users")
            ...         users = cur.fetchall()
        """
        ...

    @classmethod
    def close_all(cls) -> None:
        """Close all connections in the pool and reset pool state.

        Closes all idle and active connections in the pool gracefully.
        Sets _pool to None, forcing reinitialization on next get_connection().
        Safe to call multiple times (idempotent operation).

        Used for cleanup during application shutdown or testing.

        Returns:
            None
        """
        ...
