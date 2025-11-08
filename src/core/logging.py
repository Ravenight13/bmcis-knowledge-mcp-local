"""Structured logging system with JSON formatting and log rotation.

Provides centralized logging configuration integrated with the application's
configuration system. Supports JSON and text formatting, console and file output,
and automatic log rotation based on file size.

Implements a singleton-like pattern for logging initialization with support
for both simple text logs and structured JSON logs suitable for production
environments and log aggregation systems.

Example:
    >>> from src.core.logging import StructuredLogger, log_database_operation
    >>> StructuredLogger.initialize()
    >>> logger = StructuredLogger.get_logger(__name__)
    >>> logger.info("Application started")
    >>> log_database_operation("SELECT", duration_ms=45.2, rows_affected=10)
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Any

from src.core.config import get_settings


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging.

    Formats log records as JSON objects with timestamp, level, logger name,
    and message. Additional fields can be passed via the extra parameter
    in log calls.

    Example:
        >>> logger.info("User login", extra={"user_id": 123, "ip": "192.168.1.1"})
        # Produces: {"timestamp": "...", "level": "INFO", "logger": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log entry as string.
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields passed via the extra parameter
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                # Skip standard logging fields
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "asctime",
                }:
                    log_data[key] = value

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class StructuredLogger:
    """Centralized logging configuration and management.

    Manages application-wide logging setup with support for multiple handlers,
    formats, and log rotation. Implements initialization pattern to ensure
    logging is configured exactly once per application instance.

    The logger respects application configuration for:
    - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Output format (JSON or text)
    - Console output (enabled/disabled)
    - File output (enabled/disabled)
    - Log rotation (max file size and backup count)

    Class attributes:
        _configured: Flag indicating if logging has been initialized.
    """

    _configured: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize logging system from configuration.

        Reads logging configuration from application settings and configures
        the root logger with appropriate handlers (console and/or file).
        Automatically suppresses debug logs from third-party libraries in
        production environments.

        Configuration is read from LoggingConfig in application settings:
        - level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - format: Output format ("json" or "text")
        - console_enabled: Enable console handler
        - file_enabled: Enable file handler
        - file_path: Path to log file
        - max_file_size: Maximum bytes before rotation
        - backup_count: Number of rotated files to keep

        Returns:
            None

        Side effects:
            Configures root logger with handlers.
            Sets _configured to True.
            Creates log directory if file output is enabled.
            Idempotent: subsequent calls do nothing.

        Raises:
            RuntimeError: If log file directory cannot be created.
        """
        if cls._configured:
            return

        settings = get_settings()
        log_config = settings.logging

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_config.level)

        # Remove any existing handlers to avoid duplication
        root_logger.handlers.clear()

        # Create formatter based on configuration
        if log_config.format == "json":
            formatter: logging.Formatter = JsonFormatter(
                fmt="%(timestamp)s %(level)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Console handler for stdout
        if log_config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler with rotation
        if log_config.file_enabled:
            log_file = Path(log_config.file_path)
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                msg = f"Failed to create log directory: {log_file.parent}"
                raise RuntimeError(msg) from e

            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count,
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Suppress debug logs from third-party libraries in production
        if settings.environment == "production":
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("psycopg").setLevel(logging.WARNING)
            logging.getLogger("psycopg2").setLevel(logging.WARNING)
        else:
            # Still suppress some verbose third-party loggers in non-prod
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("psycopg").setLevel(logging.INFO)
            logging.getLogger("psycopg2").setLevel(logging.INFO)

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get named logger with structured logging support.

        Automatically initializes the logging system if not already done.
        Returns a logger instance with the specified name.

        Args:
            name: Logger name (typically __name__ or a component name).

        Returns:
            logging.Logger: Configured logger instance for structured logging.

        Side effects:
            Calls initialize() if not already configured.

        Example:
            >>> logger = StructuredLogger.get_logger(__name__)
            >>> logger.info("Process started", extra={"process_id": 123})
        """
        if not cls._configured:
            cls.initialize()
        return logging.getLogger(name)


# Module-level logger for this module
logger: logging.Logger = logging.getLogger(__name__)


def log_database_operation(
    operation: str,
    duration_ms: float,
    rows_affected: int,
    error: Optional[str] = None,
) -> None:
    """Log database operation with structured fields.

    Logs database operation completion or failure with structured context
    including operation type, execution duration, affected rows, and any
    error information. Automatically initializes logging if needed.

    Args:
        operation: Type of database operation (SELECT, INSERT, UPDATE, DELETE, etc).
        duration_ms: Operation duration in milliseconds.
        rows_affected: Number of rows affected by operation (0 for SELECT-only).
        error: Error message if operation failed (default: None for success).

    Returns:
        None

    Side effects:
        Writes structured log entry at INFO level for success or ERROR for failure.
        Calls StructuredLogger.initialize() if logging not yet configured.

    Example:
        >>> log_database_operation("INSERT", duration_ms=15.3, rows_affected=1)
        >>> log_database_operation(
        ...     "UPDATE",
        ...     duration_ms=45.2,
        ...     rows_affected=5,
        ...     error="Connection timeout"
        ... )
    """
    if error:
        logger.error(
            "Database operation failed",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "error": error,
            },
        )
    else:
        logger.info(
            "Database operation completed",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "rows_affected": rows_affected,
            },
        )


def log_api_call(
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log API call with structured fields.

    Logs API call completion with structured context including HTTP method,
    endpoint, status code, and execution duration. Automatically initializes
    logging if needed.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc).
        endpoint: API endpoint path (e.g., /api/v1/users or /docs).
        status_code: HTTP response status code (200, 404, 500, etc).
        duration_ms: Request duration in milliseconds.

    Returns:
        None

    Side effects:
        Writes structured log entry at INFO level.
        Calls StructuredLogger.initialize() if logging not yet configured.

    Example:
        >>> log_api_call("GET", "/api/v1/users", 200, 45.2)
        >>> log_api_call("POST", "/api/v1/users", 201, 123.5)
        >>> log_api_call("GET", "/api/v1/users/999", 404, 5.3)
    """
    logger.info(
        "API call completed",
        extra={
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
        },
    )
