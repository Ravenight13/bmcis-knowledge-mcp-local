"""Database connection pooling with retry logic and health checks.

Provides connection pool management for PostgreSQL using psycopg2 with
automatic health checks, exponential backoff retry logic, and graceful
error handling. Integrates with the application configuration system
for pool sizing and timeout settings.

Module manages the complete lifecycle of database connections from
acquisition to release, ensuring proper resource cleanup and connection
validation at every step.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import psycopg2
from psycopg2 import DatabaseError, OperationalError, pool
from psycopg2.extensions import connection as Connection

from src.core.config import get_settings
from src.core.logging import StructuredLogger

# Module logger for connection pool operations
# Uses StructuredLogger for automatic initialization
logger: logging.Logger = StructuredLogger.get_logger(__name__)


class DatabasePool:
    """Connection pool management for PostgreSQL databases.

    Manages a SimpleConnectionPool with automatic initialization from
    application settings, connection health checks, and exponential backoff
    retry logic for resilient database access.

    The pool maintains a configurable minimum and maximum size, with all
    connections validated via health checks before use. Connection failures
    are retried with exponential backoff (2^attempt seconds).

    Class attributes:
        _pool: The underlying SimpleConnectionPool instance or None if not
               yet initialized. Lazily created on first get_connection() call.

    Example:
        >>> DatabasePool.initialize()
        >>> with DatabasePool.get_connection() as conn:
        ...     with conn.cursor() as cur:
        ...         cur.execute("SELECT version()")
        ...         version = cur.fetchone()
    """

    _pool: pool.SimpleConnectionPool | None = None

    @classmethod
    def initialize(cls) -> None:
        """Initialize the connection pool from application settings.

        Reads pool configuration from DatabaseConfig in application settings:
        - Pool size: pool_min_size, pool_max_size
        - Connection params: host, port, database, user, password
        - Timeouts: connection_timeout, statement_timeout

        Creates a SimpleConnectionPool configured with the statement timeout
        as a PostgreSQL session parameter for server-side enforcement.

        Raises:
            RuntimeError: If pool initialization fails due to invalid config
                         or database unreachable.
            DatabaseError: If PostgreSQL connection parameters are invalid.
            ValueError: If pool configuration is invalid (max < min).

        Returns:
            None

        Side effects:
            Sets cls._pool to the initialized SimpleConnectionPool instance.
            Logs connection pool initialization with configured parameters.
        """
        if cls._pool is not None:
            logger.debug("Connection pool already initialized, skipping reinit")
            return

        try:
            settings = get_settings()
            db = settings.database

            # Calculate statement timeout in milliseconds for PostgreSQL
            # PostgreSQL expects statement_timeout in milliseconds
            statement_timeout_ms = int(db.statement_timeout * 1000)

            # Create connection pool with configured parameters
            cls._pool = pool.SimpleConnectionPool(
                minconn=db.pool_min_size,
                maxconn=db.pool_max_size,
                host=db.host,
                port=db.port,
                database=db.database,
                user=db.user,
                password=db.password.get_secret_value(),
                connect_timeout=int(db.connection_timeout),
                # PostgreSQL session parameter for statement timeout
                options=f"-c statement_timeout={statement_timeout_ms}",
            )

            logger.info(
                "Connection pool initialized: min_size=%d, max_size=%d, "
                "host=%s, port=%d, database=%s, statement_timeout=%d ms",
                db.pool_min_size,
                db.pool_max_size,
                db.host,
                db.port,
                db.database,
                statement_timeout_ms,
            )

        except (DatabaseError, psycopg2.Error) as e:
            logger.error("Failed to initialize connection pool: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to initialize connection pool: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during pool initialization: %s", e, exc_info=True)
            raise

    @classmethod
    @contextmanager
    def get_connection(cls, retries: int = 3) -> Generator[Connection, None, None]:
        """Acquire a database connection from the pool with retry logic.

        Automatically initializes the pool on first call. Implements exponential
        backoff retry strategy with configurable retry attempts. Each acquired
        connection is validated with a health check (SELECT 1) before returning.

        The context manager ensures proper cleanup: connections are returned
        to the pool even if an exception occurs during use.

        Args:
            retries: Maximum number of connection attempts (must be > 0,
                    default: 3). Each attempt uses exponential backoff:
                    wait_time = 2^attempt seconds.

        Yields:
            Connection: Valid, health-checked psycopg2 connection from pool.

        Raises:
            OperationalError: If all connection attempts fail after retries
                             are exhausted (connection refused, network error).
            DatabaseError: If connection health check fails.
            RuntimeError: If pool initialization fails.
            ValueError: If retries < 1.

        Side effects:
            Logs retry attempts, backoff waits, and final failure at WARNING level.
            Initializes pool on first call if not already done.
            Returns connection to pool in finally block.

        Examples:
            >>> # Basic usage with default retries
            >>> with DatabasePool.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("SELECT * FROM users WHERE id = %s", (1,))
            ...         user = cur.fetchone()

            >>> # Custom retry count for transient failures
            >>> with DatabasePool.get_connection(retries=5) as conn:
            ...     conn.commit()

            >>> # Nested context managers for transactions
            >>> with DatabasePool.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("BEGIN")
            ...         cur.execute("INSERT INTO logs VALUES (%s)", (msg,))
            ...         cur.execute("COMMIT")
        """
        if retries < 1:
            raise ValueError("retries must be >= 1")

        # Initialize pool on first call
        if cls._pool is None:
            cls.initialize()

        yielded_conn: Connection | None = None

        try:
            # Retry loop with exponential backoff
            for attempt in range(retries):
                # CRITICAL FIX: Move connection variable inside loop to prevent leaks
                # Each retry attempt has its own connection lifecycle with guaranteed cleanup
                conn: Connection | None = None

                try:
                    # Acquire connection from pool
                    conn = cls._pool.getconn()  # type: ignore[union-attr]
                    logger.debug(
                        "Acquired connection from pool (attempt %d/%d)", attempt + 1, retries
                    )

                    # Health check: verify connection is alive
                    # SELECT 1 is a lightweight query that doesn't modify state
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")

                    logger.debug("Connection health check passed")

                    # CRITICAL: Track yielded connection for outer finally cleanup
                    yielded_conn = conn
                    yield conn
                    return

                except (OperationalError, DatabaseError) as e:
                    # CRITICAL FIX: Return failed attempt connection immediately before retry
                    # This prevents leaking connections when retries occur
                    if conn is not None:
                        try:
                            cls._pool.putconn(conn)
                            logger.debug(
                                "Returned failed attempt connection to pool (attempt %d)",
                                attempt + 1,
                            )
                        except Exception as pool_error:
                            logger.warning(
                                "Error returning failed connection to pool: %s",
                                pool_error,
                            )

                    # Connection failed, log and potentially retry
                    if attempt < retries - 1:
                        # Calculate exponential backoff wait time
                        wait_time = 2**attempt  # 1, 2, 4, 8... seconds
                        logger.warning(
                            "Connection attempt %d failed: %s. Retrying in %d seconds...",
                            attempt + 1,
                            e,
                            wait_time,
                        )
                        time.sleep(wait_time)
                    else:
                        # All retries exhausted, log final failure
                        logger.error(
                            "All %d connection attempts failed. Last error: %s",
                            retries,
                            e,
                            exc_info=True,
                        )
                        raise

            # This line should never be reached due to return/raise above
            msg = "Unexpected control flow: no retry attempt succeeded"
            logger.error(msg)
            raise RuntimeError(msg)

        finally:
            # Always return yielded connection, even if exception occurred during use
            # This handles exceptions raised in user code (inside the with block)
            if yielded_conn is not None and cls._pool is not None:
                try:
                    cls._pool.putconn(yielded_conn)
                    logger.debug("Connection returned to pool")
                except Exception as pool_error:
                    logger.warning(
                        "Error returning connection to pool in finally: %s",
                        pool_error,
                    )

    @classmethod
    def close_all(cls) -> None:
        """Close all connections in the pool and reset pool state.

        Closes all idle and active connections in the pool gracefully.
        Closes the SimpleConnectionPool completely. Sets _pool to None
        to force reinitialization on next get_connection() call.

        This method is idempotent and safe to call multiple times.
        Used during application shutdown, testing, or connection reset.

        Returns:
            None

        Side effects:
            Closes all pooled connections.
            Sets cls._pool to None.
            Logs pool closure with connection count.

        Example:
            >>> # During application shutdown
            >>> try:
            ...     # ... application code ...
            ... finally:
            ...     DatabasePool.close_all()
        """
        if cls._pool is not None:
            try:
                logger.info("Closing all connections in pool")
                cls._pool.closeall()
                logger.info("Pool closed successfully")
            except Exception as e:
                logger.error("Error closing pool: %s", e, exc_info=True)
            finally:
                cls._pool = None
        else:
            logger.debug("Pool not initialized, no connections to close")
