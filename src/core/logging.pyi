"""Type stubs for structured logging module.

Provides complete type definitions for the StructuredLogger class and
structured logging utilities for database operations and API calls.
"""

import logging

class StructuredLogger:
    """Centralized logging configuration and management."""

    _configured: bool

    @classmethod
    def initialize(cls) -> None:
        """Initialize logging system from configuration.

        Reads logging configuration from application settings, configures
        root logger with appropriate handlers (console and/or file), and
        sets up log formatting (JSON or text).

        Returns:
            None

        Side effects:
            Configures root logger with handlers from LoggingConfig.
            Sets _configured to True after initialization.
            Idempotent: subsequent calls do nothing.

        Raises:
            RuntimeError: If log file directory cannot be created.
        """
        ...

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get named logger with structured logging support.

        Automatically initializes the logging system if not already done.
        Returns a logger with the specified name for structured logging.

        Args:
            name: Logger name (typically __name__).

        Returns:
            logging.Logger: Configured logger instance.

        Side effects:
            Calls initialize() if not already configured.
        """
        ...

def log_database_operation(
    operation: str,
    duration_ms: float,
    rows_affected: int,
    error: str | None = None,
) -> None:
    """Log database operation with structured fields.

    Logs database operation completion or failure with structured context
    including operation type, execution duration, affected rows, and any
    error information.

    Args:
        operation: Type of database operation (SELECT, INSERT, UPDATE, DELETE, etc).
        duration_ms: Operation duration in milliseconds.
        rows_affected: Number of rows affected by operation.
        error: Error message if operation failed (default: None for success).

    Returns:
        None

    Side effects:
        Writes structured log entry at INFO or ERROR level depending on success.
    """
    ...

def log_api_call(
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log API call with structured fields.

    Logs API call completion with structured context including HTTP method,
    endpoint, status code, and execution duration.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc).
        endpoint: API endpoint path (e.g., /api/v1/users).
        status_code: HTTP response status code.
        duration_ms: Request duration in milliseconds.

    Returns:
        None

    Side effects:
        Writes structured log entry at INFO level.
    """
    ...
