"""Tests for structured logging system.

Tests the StructuredLogger initialization, configuration integration,
JSON/text formatting, file rotation, and structured logging utilities.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest

from src.core.logging import (
    StructuredLogger,
    JsonFormatter,
    log_database_operation,
    log_api_call,
)
from src.core.config import LoggingConfig, get_settings, reset_settings


# Test fixtures


@pytest.fixture
def temp_log_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_logger() -> Generator[None, None, None]:
    """Reset StructuredLogger and settings before and after each test."""
    StructuredLogger._configured = False
    reset_settings()

    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    yield

    StructuredLogger._configured = False
    reset_settings()

    # Clear all handlers from root logger after test
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)


# Tests for StructuredLogger initialization


def test_structured_logger_initialize(reset_logger: None) -> None:
    """Test StructuredLogger.initialize() configures root logger."""
    StructuredLogger.initialize()

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO  # Default level
    assert len(root_logger.handlers) > 0  # At least console handler
    assert StructuredLogger._configured is True


def test_structured_logger_initialize_idempotent(reset_logger: None) -> None:
    """Test StructuredLogger.initialize() is idempotent."""
    StructuredLogger.initialize()
    first_handler_count = len(logging.getLogger().handlers)

    StructuredLogger.initialize()
    second_handler_count = len(logging.getLogger().handlers)

    assert first_handler_count == second_handler_count


def test_structured_logger_get_logger_initializes_automatically(
    reset_logger: None,
) -> None:
    """Test StructuredLogger.get_logger() initializes if needed."""
    assert StructuredLogger._configured is False

    logger = StructuredLogger.get_logger("test_module")

    assert StructuredLogger._configured is True
    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)


def test_structured_logger_with_console_enabled(reset_logger: None) -> None:
    """Test StructuredLogger with console handler enabled."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)


def test_structured_logger_with_file_enabled(
    reset_logger: None, temp_log_dir: Path
) -> None:
    """Test StructuredLogger with file handler enabled."""
    log_file = temp_log_dir / "test.log"

    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=False,
            file_enabled=True,
            file_path=str(log_file),
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers
        )


def test_structured_logger_creates_log_directory(
    reset_logger: None, temp_log_dir: Path
) -> None:
    """Test StructuredLogger creates log directory if it doesn't exist."""
    log_file = temp_log_dir / "subdir" / "test.log"
    assert not log_file.parent.exists()

    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=False,
            file_enabled=True,
            file_path=str(log_file),
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        assert log_file.parent.exists()


def test_structured_logger_fails_if_directory_creation_fails(
    reset_logger: None,
) -> None:
    """Test StructuredLogger raises RuntimeError if log directory creation fails."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=False,
            file_enabled=True,
            file_path="/invalid/path/that/cannot/be/created/test.log",
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(RuntimeError, match="Failed to create log directory"):
                StructuredLogger.initialize()


# Tests for JSON formatter


def test_json_formatter_formats_basic_log_record() -> None:
    """Test JsonFormatter formats log record as valid JSON."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.module",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    result = formatter.format(record)
    data = json.loads(result)

    assert data["level"] == "INFO"
    assert data["logger"] == "test.module"
    assert data["message"] == "Test message"
    assert "timestamp" in data


def test_json_formatter_includes_extra_fields() -> None:
    """Test JsonFormatter includes extra fields in JSON output."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.module",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.user_id = 123  # type: ignore[attr-defined]
    record.request_id = "abc-def"  # type: ignore[attr-defined]

    result = formatter.format(record)
    data = json.loads(result)

    assert data["user_id"] == 123
    assert data["request_id"] == "abc-def"


def test_json_formatter_handles_exception_info() -> None:
    """Test JsonFormatter includes exception information."""
    formatter = JsonFormatter()

    try:
        raise ValueError("Test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]


# Tests for log level configuration


def test_structured_logger_respects_log_level(reset_logger: None) -> None:
    """Test StructuredLogger respects configured log level."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="WARNING",
            format="text",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING


def test_structured_logger_suppresses_third_party_debug_in_production(
    reset_logger: None,
) -> None:
    """Test StructuredLogger suppresses third-party debug logs in production."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="DEBUG",
            format="text",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "production"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        urllib3_logger = logging.getLogger("urllib3")
        psycopg_logger = logging.getLogger("psycopg")

        assert urllib3_logger.level == logging.WARNING
        assert psycopg_logger.level == logging.WARNING


def test_structured_logger_allows_info_third_party_in_development(
    reset_logger: None,
) -> None:
    """Test StructuredLogger allows third-party INFO logs in development."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="DEBUG",
            format="text",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        psycopg_logger = logging.getLogger("psycopg")
        assert psycopg_logger.level == logging.INFO


# Tests for text vs JSON formatting


def test_structured_logger_uses_text_formatter(reset_logger: None) -> None:
    """Test StructuredLogger uses text formatter when configured."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)


def test_structured_logger_uses_json_formatter(reset_logger: None) -> None:
    """Test StructuredLogger uses JSON formatter when configured."""
    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="json",
            console_enabled=True,
            file_enabled=False,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)


# Tests for structured logging utilities


def test_log_database_operation_success() -> None:
    """Test log_database_operation logs successful database operation."""
    with patch("src.core.logging.logger") as mock_logger:
        log_database_operation(
            operation="SELECT",
            duration_ms=45.2,
            rows_affected=10,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Database operation completed"
        assert call_args[1]["extra"]["operation"] == "SELECT"
        assert call_args[1]["extra"]["duration_ms"] == 45.2
        assert call_args[1]["extra"]["rows_affected"] == 10


def test_log_database_operation_failure() -> None:
    """Test log_database_operation logs failed database operation."""
    with patch("src.core.logging.logger") as mock_logger:
        log_database_operation(
            operation="INSERT",
            duration_ms=15.3,
            rows_affected=0,
            error="Connection timeout",
        )

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[0][0] == "Database operation failed"
        assert call_args[1]["extra"]["operation"] == "INSERT"
        assert call_args[1]["extra"]["error"] == "Connection timeout"


def test_log_api_call() -> None:
    """Test log_api_call logs API request completion."""
    with patch("src.core.logging.logger") as mock_logger:
        log_api_call(
            method="GET",
            endpoint="/api/v1/users",
            status_code=200,
            duration_ms=45.2,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "API call completed"
        assert call_args[1]["extra"]["method"] == "GET"
        assert call_args[1]["extra"]["endpoint"] == "/api/v1/users"
        assert call_args[1]["extra"]["status_code"] == 200
        assert call_args[1]["extra"]["duration_ms"] == 45.2


def test_log_api_call_with_error_status() -> None:
    """Test log_api_call logs API call with error status."""
    with patch("src.core.logging.logger") as mock_logger:
        log_api_call(
            method="POST",
            endpoint="/api/v1/users",
            status_code=500,
            duration_ms=123.5,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[1]["extra"]["status_code"] == 500


# Integration tests


def test_structured_logger_integration_with_database_pool(reset_logger: None) -> None:
    """Test StructuredLogger integration with database module."""
    from src.core.database import logger as db_logger

    StructuredLogger.initialize()

    # Verify database module's logger is properly initialized
    assert isinstance(db_logger, logging.Logger)
    assert db_logger.name == "src.core.database"


def test_log_rotation_configuration(reset_logger: None, temp_log_dir: Path) -> None:
    """Test log rotation is configured with correct parameters."""
    log_file = temp_log_dir / "test.log"

    with patch("src.core.logging.get_settings") as mock_settings:
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig(
            level="INFO",
            format="text",
            console_enabled=False,
            file_enabled=True,
            file_path=str(log_file),
            max_file_size=5242880,  # 5 MB
            backup_count=3,
        )
        mock_config.environment = "development"
        mock_settings.return_value = mock_config

        StructuredLogger.initialize()

        root_logger = logging.getLogger()
        file_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        assert len(file_handlers) > 0
        handler = file_handlers[0]
        assert handler.maxBytes == 5242880
        assert handler.backupCount == 3
